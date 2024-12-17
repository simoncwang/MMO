from openai import OpenAI
import torch
from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer, AutoModel, Qwen2VLForConditionalGeneration

# initialize all of the models for inference, returns dict of initialized models
def initializeModels(model_types: dict) -> dict:
    initialized_models = {}

    for model_name, model_type in model_types:
        if model_type == "openai":
            api_key = input("Please enter your OpenAI API Key: ")

            client = OpenAI(api_key=api_key, timeout=60)
        
            initialized_models[model_name] = client
        elif model_type == "internvl":
            if torch.cuda.is_available():
                model = AutoModel.from_pretrained(
                    model_name,
                    torch_dtype="auto",
                    device_map="auto",
                    low_cpu_mem_usage=True,
                    use_flash_attn=True,
                    trust_remote_code=True).eval()
            else:
                model = AutoModel.from_pretrained(
                    model_name,
                    torch_dtype="auto",
                    device_map="auto",
                    low_cpu_mem_usage=True,
                    use_flash_attn=False,
                    trust_remote_code=True).eval()
            
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)

            initialized_models[model_name] = (model,tokenizer)
        elif model_type == "qwenvl":
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name, torch_dtype="auto", device_map="auto"
            )
            processor = AutoProcessor.from_pretrained(model_name)

            initialized_models[model_name] = (model,processor)
        elif model_type == "qwen":   # basic qwen model from hf w no vision
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto"
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name)

            initialized_models[model_name] = (model,tokenizer)

    return initialized_models