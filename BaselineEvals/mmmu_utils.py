"""Utils for data load, save, and process (e.g., prompt construction)"""

import os
import json
import yaml
import re
from intern_util import *
import base64
from io import BytesIO
from qwen_vl_utils import process_vision_info
import ast


DOMAIN_CAT2SUB_CAT = {
  'Art and Design': ['Art', 'Art_Theory', 'Design', 'Music'],
  'Business': ['Accounting', 'Economics', 'Finance', 'Manage','Marketing'],
  'Science': ['Biology', 'Chemistry', 'Geography', 'Math', 'Physics',],
  'Health and Medicine': ['Basic_Medical_Science', 'Clinical_Medicine', 'Diagnostics_and_Laboratory_Medicine', 'Pharmacy', 'Public_Health'],
  'Humanities and Social Science': ['History', 'Literature', 'Sociology', 'Psychology'],
  'Tech and Engineering': ['Agriculture', 'Architecture_and_Engineering', 'Computer_Science', 'Electronics', 'Energy_and_Power', 'Materials', 'Mechanical_Engineering'],
}


CAT_SHORT2LONG = {
    'acc': 'Accounting',
    'agri': 'Agriculture',
    'arch': 'Architecture_and_Engineering',
    'art': 'Art',
    'art_theory': 'Art_Theory',
    'bas_med': 'Basic_Medical_Science',
    'bio': 'Biology',
    'chem': 'Chemistry',
    'cli_med': 'Clinical_Medicine',
    'cs': 'Computer_Science',
    'design': 'Design',
    'diag_med': 'Diagnostics_and_Laboratory_Medicine',
    'econ': 'Economics',
    'elec': 'Electronics',
    'ep': 'Energy_and_Power',
    'fin': 'Finance',
    'geo': 'Geography',
    'his': 'History',
    'liter': 'Literature',
    'manage': 'Manage',
    'mark': 'Marketing',
    'mate': 'Materials',
    'math': 'Math',
    'mech': 'Mechanical_Engineering',
    'music': 'Music',
    'phar': 'Pharmacy',
    'phys': 'Physics',
    'psy': 'Psychology',
    'pub_health': 'Public_Health',
    'socio': 'Sociology'
}

def parse_answer(answer, answer_choice_tokens):
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



def getInternReponse(model, tokenizer, text, image):
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

def getQwenResponse(model, processor, messages):
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    response = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    return response

def getHFResponse(model, tokenizer, text):
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

def build_prompt(problem):
    # Extract relevant fields
    question = problem['question']
    # converting options string to a list
    options_str = problem['options']
    options = ast.literal_eval(options_str)

    # if there are more than 5 options, just trim the options down
    if len(options)>5:
        options = options[0:4]

    # mapping from indices to letter choices
    mapping = ['A', 'B', 'C', 'D', 'E']

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
