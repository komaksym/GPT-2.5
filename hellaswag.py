from deepeval.models.base_model import DeepEvalBaseLLM
import tiktoken
from .train import VOCAB_SIZE
from model.model import *

class MyGPT(DeepEvalBaseLLM):
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def load_model(self):
        return self.model
    
    def generate(self, prompt):
        model = self.load_model()

        device = self.model.device

        model_outputs = model.generate(self.tok, prompt, max_new_tokens=100)
        return self.enc.decode(model_outputs)
       

    async def a_generate(self, prompt):
        return self.generate(prompt)


if __name__ == "__main__":
    model = TransformerLM(VOCAB_SIZE, context_length, num_layers,
                          d_model, num_heads, d_ff, theta, device=device)