import logging

import torch.nn.functional as F
from huggingface_hub import snapshot_download
from pre_train.model import GPTConfig, TransformerLM, load_checkpoint
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutput


DEFAULT_REPO_ID = "itskoma/GPT2.5"
DEFAULT_PRETRAINING_CHECKPOINT_PATTERN = "pretraining_checkpoint/*"
DEFAULT_PRETRAINING_CHECKPOINT_PATH = "checkpoints/pretraining_checkpoint/"
DEFAULT_CHECKPOINT_LOCAL_DIR = "checkpoints"
DEFAULT_CHECKPOINT_SUBFOLDER = "pretraining_checkpoint"
logger = logging.getLogger(__name__)


class MyConfig(PretrainedConfig):
    model_type = "gpt2.5"

    def __init__(
        self,
        vocab_size=GPTConfig.vocab_size,
        context_length=GPTConfig.context_length,
        num_layers=GPTConfig.num_layers,
        num_heads=GPTConfig.num_heads,
        d_model=GPTConfig.d_model,
        d_ff=GPTConfig.d_ff,
        theta=GPTConfig.theta,
        device=str(GPTConfig.device),
        tie_word_embeddings=True,
        **kwargs,
    ):
        """Mirror the base GPT config in a HF-compatible configuration object."""
        super().__init__(**kwargs, tie_word_embeddings=tie_word_embeddings)
        self.vocab_size = int(vocab_size)
        self.context_length = int(context_length)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.d_model = int(d_model)
        self.d_ff = int(d_ff)
        self.theta = float(theta)
        self.device = device if isinstance(device, str) else str(device)


class HFTransformerLM(PreTrainedModel):
    config_class = MyConfig
    _tied_weights_keys = {"model.linear.weight": "model.emb.weight"}
    _supports_flash_attn = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    def __init__(self, config):
        """Expose TransformerLM through the minimal HF interface used by Trainer."""
        super().__init__(config)
        self.model = TransformerLM(
            config.vocab_size,
            config.context_length,
            config.num_layers,
            config.d_model,
            config.num_heads,
            config.d_ff,
            config.theta,
            config.device,
        )
        self._sync_attn_implementation()
        self.post_init()

    def _check_and_adjust_attn_implementation(
        self, attn_implementation: str | None, is_init_check: bool = False
    ) -> str:
        """Fall back to SDPA when flash attention is requested but unavailable."""
        try:
            return super()._check_and_adjust_attn_implementation(
                attn_implementation, is_init_check=is_init_check
            )
        except Exception as exc:
            if attn_implementation is None:
                raise

            is_paged = attn_implementation.startswith("paged|")
            requested_implementation = attn_implementation.removeprefix("paged|")
            if requested_implementation not in {
                "flash_attention_2",
                "flash_attention_3",
            }:
                raise

            fallback_implementation = "paged|sdpa" if is_paged else "sdpa"
            logger.warning(
                "FlashAttention requested but unavailable (%s). Falling back to %s.",
                exc,
                fallback_implementation,
            )
            return super()._check_and_adjust_attn_implementation(
                fallback_implementation, is_init_check=is_init_check
            )

    def _get_runtime_attn_implementation(self) -> str:
        """Return the backend name understood by the wrapped runtime model."""
        return self.config._attn_implementation.removeprefix("paged|")

    def _sync_attn_implementation(self) -> None:
        """Push the config-selected attention backend into the wrapped model."""
        self.model.set_attn_implementation(self._get_runtime_attn_implementation())

    def get_input_embeddings(self):
        """Expose the token embedding layer through the HF API."""
        return self.model.emb

    def get_output_embeddings(self):
        """Expose the tied LM head through the HF API."""
        return self.model.linear

    def set_input_embeddings(self, new_embeddings):
        """Replace the wrapped token embeddings."""
        self.model.emb = new_embeddings

    def set_output_embeddings(self, new_embeddings):
        """Replace the wrapped output head."""
        self.model.linear = new_embeddings

    def set_attn_implementation(self, attn_implementation: str | dict):
        """Update the configured attention backend and sync it to runtime."""
        super().set_attn_implementation(attn_implementation)
        self._sync_attn_implementation()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        labels=None,
        **kwargs,
    ):
        """Run the wrapped model and optionally compute next-token loss."""
        self._sync_attn_implementation()
        logits, _ = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return CausalLMOutput(loss=loss, logits=logits)


def _build_base_model():
    """Construct the raw pretrained TransformerLM with the project defaults."""
    return TransformerLM(
        GPTConfig.vocab_size,
        GPTConfig.context_length,
        GPTConfig.num_layers,
        GPTConfig.d_model,
        GPTConfig.num_heads,
        GPTConfig.d_ff,
        GPTConfig.theta,
        GPTConfig.device,
    )


def _load_pretraining_model(
    repo_id: str = DEFAULT_REPO_ID,
    checkpoint_pattern: str = DEFAULT_PRETRAINING_CHECKPOINT_PATTERN,
    checkpoint_path: str = DEFAULT_PRETRAINING_CHECKPOINT_PATH,
    local_dir: str = DEFAULT_CHECKPOINT_LOCAL_DIR,
):
    """Download and load a base checkpoint into a fresh TransformerLM."""
    base_model = _build_base_model()
    snapshot_download(
        repo_id,
        allow_patterns=checkpoint_pattern,
        repo_type="model",
        local_dir=local_dir,
    )
    load_checkpoint(checkpoint_path, base_model)
    return base_model


def _build_hf_model(base_model):
    """Wrap a loaded base model for HF Trainer while keeping tied weights exposed."""
    model = HFTransformerLM(MyConfig())
    model.model.load_state_dict(base_model.state_dict())
    return model
