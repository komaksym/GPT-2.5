from vllm import LLM

llm = LLM(model="openai-community/gpt2")  # Name or path of your model
llm.apply_model(lambda model: print(type(model)))