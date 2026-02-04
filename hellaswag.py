from deepeval.models.base_model import DeepEvalBaseLLM
from typing import List
from torch.nn.utils.rnn import pad_sequence
import torch
from model.model import top_p_sampling, softmax


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
        
        # Retrieve context length safely
        try:
            # Try to get it from the model blocks if possible
            context_length = model.tblocks[0].mhsa.rope.max_seq_len
        except Exception:
            context_length = 1024

        # Truncate inputs to context length
        if model_inputs.shape[1] > context_length:
            model_inputs = model_inputs[:, -context_length:]

        # Generate loop (Avoiding model.generate to ensure we go through FSDP wrapper)
        model.eval()
        with torch.no_grad():
            for _ in range(100): # max_new_tokens
                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                    # Call model directly to trigger FSDP forward
                    logits, _ = model(model_inputs)
                
                # Take the last token's logits
                next_token_logits = logits[:, -1, :]

                # Pick next tokens
                probs = softmax(next_token_logits, dim=-1, temp=0.8)
                next_tokens = top_p_sampling(probs)
                
                model_inputs = torch.cat([model_inputs, next_tokens], dim=1)
                
                # Check for end of text (simplified)
                if (next_tokens == 50256).any(): # 50256 is <|endoftext|> for gpt2
                    break
                
                if model_inputs.shape[1] >= context_length:
                    model_inputs = model_inputs[:, -context_length:]

        model.train()
        # Decode and return
        return self.tokenizer.decode_batch(model_inputs.tolist())

    async def a_generate(self, prompt):
        return self.generate(prompt)
    
    def get_model_name(self):
        return "MyGPT"