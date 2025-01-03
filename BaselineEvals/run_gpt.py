#####################################################
'''
base evaluation copied from official MMLU-Pro Huggingface page
https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/blob/main/run_gpt4o.py
'''
#####################################################

from openai import OpenAI
import datasets
import json
import re
import random
from tqdm import tqdm
import argparse

api_key = input("Please enter your OpenAI API key: ")
client = OpenAI(api_key=api_key)
random.seed(42)

# getting command line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--samples",
    type=int,
    default=100,
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

args = parser.parse_args()

samples = args.samples
model_name = args.model
model_type = args.model_type

def run_one_question(question: str):
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": "You are an knowledge expert, you are supposed to answer the multi-choice question to derive your final answer as `The answer is ...`."
            },
            {
                "role": "user",
                "content": [
                    {
                    "type": "text",
                    "text": question
                    }
                ]
            }
        ],
        temperature=0.1,
        max_tokens=4096,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )

    return response.choices[0].message.content


def form_options(options: list):
    option_str = 'Options are:\n'
    opts = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    for opt, o in zip(options, opts):
        option_str += f'({o}): {opt}' + '\n'
    return option_str

def get_prediction(output):
    pattern = r"answer is \(?([ABCDEFGHIJ])\)?"
    match = re.search(pattern, output)
    if match:
        return match.group(1)
    else:
        print("extraction failed, do a random guess")
        return random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'])



if __name__ == "__main__":
    dataset = datasets.load_dataset('TIGER-Lab/MMLU-Pro')

    # doing a random subset of test split
    data = dataset['test'].shuffle(seed=42).select(range(samples))

    categories = ['computer science', 'math', 'chemistry', 'engineering', 'law', 'biology',
                  'health', 'physics', 'business', 'philosophy', 'economics', 'other',
                  'psychology', 'history']

    # load 5-shot prompts for each category
    prompts = {c: '' for c in categories}
    for d in dataset['validation']:
        prompts[d['category']] += 'Q:' + ' ' + d['question'] + '\n' + form_options(d['options']) + '\n' + d['cot_content'] + '\n\n'

    per_category_accuracy = {c: [0, 0] for c in categories}
    success, fail = 0, 0
    answers = []
    total_questions = len(data)

    print('----------------- Start Answering -------------------')
    for entry in tqdm(data, total=total_questions, desc="Evaluating", unit="question"):
        # print(entry)
        prefix = prompts[entry['category']]
        query = prefix + 'Q: ' + entry['question'] + '\n' + form_options(entry['options']) + '\n'
        answer = run_one_question(query)
        # print(answer)
        entry['solution'] = answer
        answers.append(entry)

        prediction = get_prediction(answer)
        if entry["answer"] == prediction:
            success += 1
            per_category_accuracy[entry['category']][0] += 1
        else:
            fail += 1
            per_category_accuracy[entry['category']][1] += 1

        # print(success / (success + fail))

    # with open('outputs.json', 'w') as f:
    #     json.dump(answers, f, indent=2)
    print(f"Overall accuracy: {success/success+fail * 100:.2f}")

    for k, v in per_category_accuracy.items():
        print('accuracy: ', k, v[0] / (v[0] + v[1]))