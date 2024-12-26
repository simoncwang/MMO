#####################################################
'''
base evaluation copied and modified from official MMMU GitHub
https://github.com/MMMU-Benchmark/MMMU/blob/main/mmmu/run_llava.py
'''
#####################################################

import torch
import os
import random
import numpy as np
import time
from tqdm import tqdm
from openai import OpenAI
from datasets import load_dataset, concatenate_datasets
from argparse import ArgumentParser
from mmmu_utils import *
from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer, AutoModel, Qwen2VLForConditionalGeneration
from sklearn.metrics import accuracy_score

# getting command line arguments
parser = ArgumentParser()
parser.add_argument(
    "--samples",
    type=int,
    default=None,
    help="Number of samples from dataset"
)

parser.add_argument(
    "--model",
    type=str,
    default="gpt-4o",
    help="Enter the name of the OpenAI model (e.g. gpt4o), or enter the path to your HuggingFace model (e.g. Qwen/Qwen2.5-0.5B-Instruct)"
)

parser.add_argument(
    "--model_type",
    type=str,
    default="openai",
    help="The provider of your model, either openai or hf"
)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--subset', type=str, default="Physics")


args = parser.parse_args()

samples = args.samples
model_name = args.model
model_type = args.model_type
subset = args.subset

# load the specified subset of the dataset
print(f"------loading {subset} subset of the MMMU dataset------")
data = load_dataset("MMMU/MMMU", subset)

# Set your OpenAI API key (make sure to replace 'your_api_key' with your actual key)
if model_type == "openai":
    api_key = input("Please enter your OpenAI API Key: ")

    client = OpenAI(api_key=api_key, timeout=60)

# building models
if model_type == "intern":
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        low_cpu_mem_usage=True,
        use_flash_attn=False,
        trust_remote_code=True).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
elif model_type == "hf":
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
elif model_type == "qwen":
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_name)

# prompts
answer_choices = {'A','B','C','D','E'}

answer_prompt = "Absolutely provide your answer in the following format: Answer: <single letter>"

def get_model_answer(problem):
    correct_answer = problem["answer"]
    prompt = build_prompt(problem)
    image = problem["image_1"]   # inspecting the dataset, the vast majority of the questions only use 1 image

    if (model_type == "openai"):
        # Getting the base64 string
        if image:   # if context image given
            base64_image = encode_image(image)
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt + answer_prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url":  f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]
        else:   # only text given
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt + answer_prompt,
                        }
                    ]
                }
            ]

        response = client.chat.completions.create(
            model=model_name,
            messages = messages
        )
        output = response.choices[0].message.content.strip()

    elif (model_type == "hf"):   # model from huggingface transformers
        output = getHFResponse(model=model, tokenizer=tokenizer, text=prompt + answer_prompt)
    
    elif (model_type == "intern"):   # intern VL
        output = getInternReponse(model, tokenizer, text=prompt+answer_prompt, image=image)
    
    elif (model_type == "qwen"):   # Qwen VL
        if image:
            base64_image = encode_image(image)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt+answer_prompt},
                    ],
                }
            ]
        else:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt+answer_prompt},
                    ],
                }
            ]
        
        output = getQwenResponse(model, processor, messages)

    output = parse_answer(output, answer_choices)
    model_answer = output.strip()


    return correct_answer, model_answer

def set_seed(seed_value):
    """
    Set the seed for PyTorch (both CPU and CUDA), Python, and NumPy for reproducible results.

    :param seed_value: An integer value to be used as the seed.
    """
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # For multi-GPU setups
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)

    multiple_choice_data = data["validation"].filter(lambda problem: problem["question_type"] == "multiple-choice")

    if samples: 
        multiple_choice_data = multiple_choice_data.shuffle(seed=51).select(range(samples))

    # Initialize variables for testing
    predictions = []
    ground_truth = []
    total_questions = len(multiple_choice_data)
    num_invalid = 0

    start_time = time.time()
    for problem in tqdm(multiple_choice_data, total=total_questions, desc="Evaluating", unit="question"):
        correct_answer, model_answer = get_model_answer(problem)

        if model_answer == "invalid":
            num_invalid+=1
        
        # Compare model's answer index to the correct index
        predictions.append(model_answer)
        ground_truth.append(correct_answer)
    
    end_time = time.time()
    runtime = end_time - start_time
    print(f"\nRuntime: {runtime:.2f} seconds")
    
    # Calculate accuracy
    accuracy = accuracy_score(ground_truth, predictions)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Number of invalid responses: {num_invalid}")

if __name__ == '__main__':
    main()