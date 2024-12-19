from openai import OpenAI
import torch
from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer, AutoModel, Qwen2VLForConditionalGeneration
import base64
from io import BytesIO
from qwen_vl_utils import process_vision_info
from .mmo_intern_util import *
import re
from pydantic import BaseModel
import ast

# a single subtask
class Subtask(BaseModel):
    subtask: str
    assigned_model: str

# list of all subtasks
class Subtasks(BaseModel):
    original_question: str
    subtasks: list[Subtask]

# initialize all of the models for inference, returns dict of initialized models
def initializeModels(model_types: dict) -> dict:
    initialized_models = {}

    for model_name, model_type in model_types.items():
        if model_type == "openai":
            api_key = input("Please enter your OpenAI API Key: ")

            client = OpenAI(api_key=api_key, timeout=60)
        
            initialized_models[model_name] = client
        elif model_type == "internvl":
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
        elif model_type == "hftext":   # basic text model from hf w no vision
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto"
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name)

            initialized_models[model_name] = (model,tokenizer)

    return initialized_models

def parseScienceQA(answer, answer_choice_tokens):
    """
    copied from GPQA Github (https://github.com/klukosiute/gpqa-eval/blob/main/run_gpqa.py)
    updated for ScienceQA numerical choices (1-4).
    """
    # Update patterns to focus on numerical choices (1-4) d is a digit
    patterns = [
        r"Answer: (\d)",         # Matches "Answer: d"
        r"answer: (\d)",         # Matches "answer: d"
        r"answer is \((\d)\)",     # Matches "answer is (d)"
        r"Answer: \((\d)\)",       # Matches "Answer: (d)"
        r"answer: \((\d)\)",       # Matches "answer: (d)"
        r"answer \((\d)\)",        # Matches "answer (d)"
        r"answer is (\d)\.",       # Matches "answer is d."
        r"Answer: (\d)\.",         # Matches "Answer: d."
        r"answer: (\d)\.",         # Matches "answer: d."
        r"answer (\d)\.",          # Matches "answer d."
        r"answer is choice \((\d)\)", # Matches "answer is choice (d)"
        r"answer is Choice \((\d)\)", # Matches "answer is Choice (d)"
        r"answer is choice (\d)",     # Matches "answer is choice d"
        r"answer is Choice (\d)",     # Matches "answer is Choice d"
        r"Answer: Choice (\d)",
        r"answer: choice (\d)",
        r"Answer: choice (\d)",
        r"answer: Choice (\d)",
        r"Answer: Choice \((\d)\)",
        r"answer: choice \((\d)\)",
        r"Answer: choice \((\d)\)",
        r"answer: Choice \((\d)\)",
        r"(\d)",
    ]
    for pattern in patterns:
        match = re.search(pattern, answer)
        # Check if the match is valid and within the choice tokens
        if match and match.group(1) in answer_choice_tokens:
            return match.group(1)
    return "nothing"

def parse_MMMU(answer, answer_choice_tokens):
    """
    Adapted for MMMU where answers are A, B, C, D, or E (case-insensitive).
    """
    # Patterns for alphabetical answers (A, B, C, D, E) both uppercase and lowercase
    patterns = [
        r"Answer: ([A-Ea-e])",            # Matches "Answer: A" or "Answer: a"
        r"answer: ([A-Ea-e])",            # Matches "answer: A" or "answer: a"
        r"answer is \(([A-Ea-e])\)",      # Matches "answer is (A)" or "answer is (a)"
        r"Answer: \(([A-Ea-e])\)",        # Matches "Answer: (A)" or "Answer: (a)"
        r"answer: \(([A-Ea-e])\)",        # Matches "answer: (A)" or "answer: (a)"
        r"answer \(([A-Ea-e])\)",         # Matches "answer (A)" or "answer (a)"
        r"answer is ([A-Ea-e])\.",        # Matches "answer is A." or "answer is a."
        r"Answer: ([A-Ea-e])\.",          # Matches "Answer: A." or "Answer: a."
        r"answer: ([A-Ea-e])\.",          # Matches "answer: A." or "answer: a."
        r"answer ([A-Ea-e])\.",           # Matches "answer A." or "answer a."
        r"answer is choice \(([A-Ea-e])\)",  # Matches "answer is choice (A)" or "answer is choice (a)"
        r"answer is Choice \(([A-Ea-e])\)",  # Matches "answer is Choice (A)" or "answer is Choice (a)"
        r"answer is choice ([A-Ea-e])",      # Matches "answer is choice A" or "answer is choice a"
        r"answer is Choice ([A-Ea-e])",      # Matches "answer is Choice A" or "answer is Choice a"
        r"Answer: Choice ([A-Ea-e])",
        r"answer: choice ([A-Ea-e])",
        r"Answer: choice ([A-Ea-e])",
        r"answer: Choice ([A-Ea-e])",
        r"Answer: Choice \(([A-Ea-e])\)",
        r"answer: choice \(([A-Ea-e])\)",
        r"Answer: choice \(([A-Ea-e])\)",
        r"answer: Choice \(([A-Ea-e])\)",
        r"([A-Ea-e])"                      # Matches just a single letter (A, B, C, D, E)
    ]
    for pattern in patterns:
        match = re.search(pattern, answer)

        # Check if the match is valid and within the choice tokens (case-insensitive)
        if match and match.group(1).upper() in answer_choice_tokens:
            return match.group(1).upper()  # Return the answer in uppercase format
    
    return "invalid"

def getInternVLReponse(model, tokenizer, text, image):    # if internVL
    # set the max number of tiles in `max_num`
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")

    if image:
        pixel_values = load_image(image, max_num=12).to(torch.bfloat16).to(device)
    else:
        pixel_values = None
        
    generation_config = dict(max_new_tokens=1024, do_sample=True, pad_token_id=tokenizer.eos_token_id)

    response = model.chat(tokenizer, pixel_values, text, generation_config)

    return response

def getQwenVLResponse(model, processor, messages):
    # specifying to avoid issues where tensors are on different GPUs
    device="cuda:0"
    
    # moving model to same device as inputs
    # model = model.to(device)

    # Prepare text input
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    # Prepare vision inputs
    image_inputs, video_inputs = process_vision_info(messages)

    # Prepare processor inputs
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    # Move inputs to the same device
    inputs = inputs.to("cuda")

    # Generate the response
    generated_ids = model.generate(**inputs, max_new_tokens=128)

    # Trim and decode the output
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    response = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    return response


def getHFTextResponse(model, tokenizer, text):   
    messages = [
        {"role": "user", "content": text}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response

def build_commander_prompt_ScienceQA(problem):
    # Extract relevant fields
    question = problem['question']
    choices = problem['choices']

    # Construct a prompt (you may format this as needed)
    prompt = f"Question: {question}\nChoices:\n"
    for idx, choice in enumerate(choices):
        prompt += f"Choice {idx + 1}: {choice}\n"
    prompt += "\nUse the image as context if present."
    return prompt

def build_commander_prompt_MMMU(problem):
    # Extract relevant fields
    question = problem['question']
    # converting options string to a list
    options_str = problem['options']
    options = ast.literal_eval(options_str)

    # mapping from indices to letter choices
    mapping = ['A', 'B', 'C', 'D', 'E']

    # if there are more than 5 options, just trim the options down
    if len(options)>5:
        options = options[0:4]

    # Construct a prompt (you may format this as needed)
    prompt = f"Question: {question}\n Choices:\n"
    for idx, choice in enumerate(options):
        prompt += f"Choice {mapping[idx]}: {choice}\n"
    prompt += "\nUse the image as context if present."
    return prompt

def build_ScienceQA_prompt(problem):
    # Extract relevant fields
    question = problem['question']
    choices = problem['choices']

    # Construct a prompt (you may format this as needed)
    prompt = f"Question: {question}\n Choices:\n"
    for idx, choice in enumerate(choices):
        prompt += f"Choice {idx + 1}: {choice}\n"
    prompt += "\nUse the image as context if present. Please choose the correct choice by returning the single number absolutely in the following format: Answer: <single digit>"
    return prompt

def build_MMMU_prompt(problem):
    # Extract relevant fields
    question = problem['question']
    # converting options string to a list
    options_str = problem['options']
    options = ast.literal_eval(options_str)

    # mapping from indices to letter choices
    mapping = ['A', 'B', 'C', 'D', 'E']

    # if there are more than 5 options, just trim the options down
    if len(options)>5:
        options = options[0:4]

    # Construct a prompt (you may format this as needed)
    prompt = f"Question: {question}\n Choices:\n"
    for idx, choice in enumerate(options):
        prompt += f"Choice {mapping[idx]}: {choice}\n"
    prompt += "\nUse the image as context if present. Please choose the correct choice absolutely in the following format: Answer: <single letter>"
    return prompt


# Function to encode the image for openai
def encode_image(image):
    if image.mode != "RGB":
        image = image.convert("RGB")

    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)
    base64_string = base64.b64encode(buffer.read()).decode('utf-8')

    # Ensure proper padding
    missing_padding = len(base64_string) % 4
    if missing_padding:
        base64_string += '=' * (4 - missing_padding)
    
    return base64_string.strip()

