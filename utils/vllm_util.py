from vllm import LLM,SamplingParams

def get_api_response(prompt, model_path, device="auto"):
    ### config
    print("WARNING:using device: ",device)
    model = LLM(
        model=model_path,
        dtype=torch.bfloat16,
        tensor_parallel_size=4
    )
    if device=='auto':device='cuda'
    gen_kwargs = {'top_p': 0.95,'temperature': 1.0,'max_tokens': 2048}
    gen_kwargs=SamplingParams(**gen_kwargs)
    ### config

    response=model.generate(prompt,gen_kwargs)
    # prompt=response.prompt
    # print("prompt:\n",prompt)
    response=response.outputs[0].text
    # print("response:\n",response)
    return response