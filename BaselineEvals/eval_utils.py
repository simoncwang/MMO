from pydantic import BaseModel
import re
from intern_util import *
import base64
from io import BytesIO
from qwen_vl_utils import process_vision_info


def parse_answer(answer, answer_choice_tokens):
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
    choices = problem['choices']

    # Construct a prompt (you may format this as needed)
    prompt = f"Question: {question}\n Choices:\n"
    for idx, choice in enumerate(choices):
        prompt += f"Choice {idx + 1}: {choice}\n"
    prompt += "\nUse the image as context if present. Please choose the correct choice by returning the single number absolutely in the following format: Answer: <single digit>"
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

# def build_evaluation_prompt(problem):
#     # Extract relevant fields
#     question = problem['question']
#     context = problem['image']  # or any relevant textual context
#     choices = problem['choices']

#     # Construct a prompt (you may format this as needed)
#     prompt = f"Question: {question}\nContext: {context}\nChoices:\n"
#     for idx, choice in enumerate(choices):
#         prompt += f"Choice {idx + 1}: {choice}\n"
#     prompt += "\nPlease choose the correct answer by returning the single number of the correct choice."
#     return prompt
