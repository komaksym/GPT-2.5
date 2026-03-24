from transformers import AutoConfig, AutoModel 
from vllm import LLM

#AutoConfig.register("gpt2.5", MyConfig)
#AutoModel.register(MyConfig, HFTransformerLM)


llm = LLM(model="itskoma/MyGPT")  # Name or path of your model
llm.apply_model(lambda model: print(type(model)))