from .model_generation import GenerationMixin
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_api_response(prompt, model_path, device="auto"):
    ### config
    print("WARNING:using device: ",device)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)
    if device=='auto':device='cuda'
    gen_kwargs = {'top_p': 0.95,'temperature': 1.0,'max_tokens': 2048}
    input_ids = tokenizer(prompt,return_tensors='pt',padding=True).input_ids.to(device)
    output_ids = model.generate(input_ids, do_sample=True,**gen_kwargs)
    response = tokenizer.decode(output_ids[i][input_ids[i].shape[0]:], skip_special_tokens=True)  # Skip the input part
    print(response)
    return response