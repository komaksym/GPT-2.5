from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from contextlib import nullcontext
from deepeval.models.base_model import DeepEvalBaseLLM
from typing import List
from torch.nn.utils.rnn import pad_sequence
import torch

class MyGPT(DeepEvalBaseLLM):
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device=device
    
    def load_model(self):
        return self.model
    
    def generate(self, prompt):
        return self.batch_generate(prompt)[0]
    
    def batch_generate(self, prompts: List[str]) -> List[str]:
        model = self.load_model()

        # Convert to tokens
        encoded_tokens = self.tokenizer.encode_ordinary_batch(prompts)
        # Convert to a list of tensors
        tensors = [torch.tensor(prompt) for prompt in encoded_tokens]
        # Pad tensors and convert to a single tensor
        model_inputs = pad_sequence(tensors, batch_first=True).to(device=self.device)
        
        # Prepare context for FSDP
        if isinstance(model, FSDP):
             context = FSDP.summon_full_params(model, recurse=True, writeback=False)
        else:
             context = nullcontext()

        # Retrieve context length safely
        try:
            context_length = model.tblocks[0].mhsa.rope.max_seq_len
        except Exception:
            context_length = 1024

        # Truncate inputs
        if model_inputs.shape[1] > context_length:
            model_inputs = model_inputs[:, -context_length:]

        # Generate
        with context:
            generated_ids = model.generate(self.tokenizer, model_inputs, context_length, max_new_tokens=100)
        # Decode
        return self.tokenizer.decode_batch(generated_ids.tolist())

    async def a_generate(self, prompt):
        return self.generate(prompt)
    
    def get_model_name(self):
        return "MyGPT"