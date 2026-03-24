from vllm import LLM, SamplingParams


def main() -> None:
    llm = LLM(
        model="itskoma/MyGPT",
        trust_remote_code=True,
        model_impl="transformers",
    )
    llm.apply_model(lambda model: print(type(model)))
    outputs = llm.generate("Hello, my name is", SamplingParams(max_tokens=32))
    print(outputs[0].outputs[0].text)


if __name__ == "__main__":
    main()
