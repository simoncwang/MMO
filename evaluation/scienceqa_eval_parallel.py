from datasets import load_dataset
from openai import OpenAI
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from util import *
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import argparse
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool, Manager, set_start_method
import os


# getting command line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--samples",
    type=int,
    default=100,
    help="Number of samples from evaluation dataset"
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

args = parser.parse_args()

samples = args.samples
model_name = args.model
model_type = args.model_type

# checking if Cuda is available
cuda = False
device = "cpu"
if torch.cuda.is_available():
    cuda = True
    device = "cuda"
    print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
else:
    print("Using CPU")

# prompting
answer_choices = ['1', '2', '3', '4']
answer_prompt = "Absolutely provide your answer in the following format: Answer: <single digit of correct choice>"

# Load the ScienceQA dataset
data = load_dataset('derek-thomas/ScienceQA', split='test')  # choose the test set

# Set your OpenAI API key (make sure to replace 'your_api_key' with your actual key)
if model_type == "openai":
    api_key = input("Please enter your OpenAI API Key: ")

    client = OpenAI(api_key=api_key, timeout=60)

# Sample a few entries for testing
sample_size = samples  # Adjust the sample size as needed
data = data.shuffle(seed=51).select(range(sample_size))

# Function to get the model's response (assuming you are using GPT-4)
def get_model_answer(problem):
    correct_answer_index = problem['answer']
    prompt = build_prompt(problem)
    image = problem['image']

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
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        output = getHFResponse(model=model, tokenizer=tokenizer, text=prompt + answer_prompt)
    
    elif (model_type == "intern"):   # intern VL
        if cuda:
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
        
        output = getInternReponse(model, tokenizer, text=prompt+answer_prompt, image=image)


    output = parse_answer(output, answer_choices)

    try:
        model_answer_index = int(output.strip()) - 1  # Convert to 0-based index
    except ValueError:
        model_answer_index = -1  # Handle invalid responses

    return correct_answer_index, model_answer_index


# Initialize variables for testing
predictions = []
ground_truth = []
total_questions = len(data)
num_invalid = 0
iterations = 0

# Multiprocessing evaluation
if __name__ == "__main__":
    try:
        # Set the multiprocessing start method to 'spawn' for CUDA support
        set_start_method('spawn')
    except RuntimeError:
        pass

    with Manager() as manager:
        ground_truth = manager.list()
        predictions = manager.list()
        num_invalid = manager.Value("i", 0)

        with Pool(processes=os.cpu_count()) as pool:
            print(f"Number of cpus: {os.cpu_count()}")
            results = list(tqdm(pool.imap(get_model_answer, data), total=len(data), desc="Evaluating", unit="question"))

        for correct_answer, model_answer in results:
            ground_truth.append(correct_answer)
            if model_answer == -1:
                num_invalid.value += 1
            predictions.append(model_answer)

        # Calculate accuracy
        accuracy = accuracy_score(list(ground_truth), list(predictions))
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print(f"Number of invalid responses: {num_invalid.value}")