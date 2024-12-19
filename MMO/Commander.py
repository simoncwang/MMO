###############################################################################################
'''
The commander of your specialized team of agents. This agent is the most capable and can
utilize your team's strengths to complete the mission in the most effective and efficient
way possible.

Provide the commander with the roster of agents available, and watch as complex tasks
are broken down into sub-tasks that each agent is most proficient as.
'''
###############################################################################################
from .Roster import Roster
from tqdm import tqdm
from .mmo_utils import *
from .mmo_utils import Subtasks
from openai import OpenAI
from multiprocessing import Pool, cpu_count
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool, Manager
import time

# global variables
api_key = input("\nPlease enter your OpenAI API Key: ")
client = OpenAI(api_key=api_key, timeout=60)
benchmark = input("\nPlease enter the name of the benchmark you want to run, mmmu or scienceqa (case sensitive): ")
assert benchmark in {"mmmu", "scienceqa"}, "Invalid benchmark. Please enter either 'mmmu' or 'scienceqa' (case sensitive)"

class Commander():
    def __init__(self, roster: Roster, commander_model: str, mode: str, model_name: str, model_type: str, multithreading: bool, dataset) -> None:
        self.roster = roster
        self.dataset = dataset
        self.commander_model = commander_model
        self.mode = mode

        # for single model experiments
        self.model_name = model_name
        self.model_type = model_type

        self.multithreading = multithreading
        
        # # initializing OpenAI client once
        # api_key = input("\nPlease enter your OpenAI API Key: ")
        # client = OpenAI(api_key=api_key, timeout=60)

        return
    
    def getRoster(self) -> Roster:
        return self.roster
    
    # iterate over all problems in the dataset and get answers depending on the organization mode
    def getResponses(self) -> list:
        mode = self.mode
        data = self.dataset
        total_questions = len(data)
        
        predictions = []
        ground_truth = []
        num_invalid = 0

        if mode == "auto-subtask":   # automatically assigning subtasks using gpt-4o (or another commander model)
            print(f"Running auto subtasks mode with")
            start_time = time.time()
            if self.multithreading:
                print("multithreading!")
                
                with ThreadPoolExecutor(max_workers=4) as executor:  # Adjust workers as needed
                    futures = []
                    for problem in data:
                        futures.append(
                            executor.submit(self.processSubtasks, problem)
                        )

                    for future in tqdm(as_completed(futures), total=total_questions, desc="Evaluating", unit="question"):
                        final_answer, correct_answer = future.result()

                        print(f"TEST FINAL ANSWER: {final_answer}, CORRECT ANSWER: {correct_answer}")
                        
                        if benchmark == "scienceqa":
                            try:
                                final_answer  = int(final_answer) - 1
                            except ValueError:
                                num_invalid += 1
                                final_answer = -1 

                        predictions.append(final_answer)
                        ground_truth.append(correct_answer)

            else:   # single threaded
                print("no multithreading!")
                for problem in tqdm(data, total=total_questions, desc="Evaluating", unit="question"):
                    correct_answer = problem['answer']
                    subtasks = assignSubtasks(client, self.commander_model, problem, self.roster,benchmark)
                    # printSubtasks(subtasks)
                    subtask_answers = self.roster.getSubtaskAnswers(problem, subtasks)

                    # for task,answer in subtask_answers.items():
                    #     print(f"TASK DESCRIPTION:\n{task}\nANSWER:\n{answer}")

                    final_answer = finalAnswerFromSubtasks(client, self.commander_model, problem, subtask_answers,benchmark)

                    # convert to 0-index to match ground truth
                    if benchmark == "scienceqa":
                        try:
                            final_answer  = int(final_answer) - 1
                        except ValueError:
                            num_invalid += 1
                            final_answer = -1 

                    predictions.append(final_answer)
                    ground_truth.append(correct_answer)
            
            end_time = time.time()
            runtime = end_time - start_time
            print(f"\nRuntime: {runtime:.2f} seconds")
            print(f"Number of invalid answers: {num_invalid}")

        elif mode == "majority-vote-single":
            start_time = time.time()
            num_votes = 5
            model_name = self.model_name
            model_type = self.model_type

            print(f"\nRunning majority vote with single model! num_votes: {num_votes}, model_name: {model_name}\n")

            if model_type == "openai":   # if openai can use multiprocessing
                with Pool(processes=os.cpu_count()) as pool:
                    print(f"Number of CPUs: {os.cpu_count()}")
                    results = list(tqdm(pool.imap(majorityVoteSingleParallel, data), total=len(data), desc="Evaluating", unit="question"))

                ground_truth = [res[0] for res in results]
                predictions = [res[1] for res in results]
                # num_invalid = sum(1 for _, model_answer in results if model_answer == -1)
            else:
                for problem in tqdm(data, total=total_questions, desc="Evaluating", unit="question"):
                    correct_answer,model_answer = majorityVoteSingleModel(problem, num_votes, self.roster, model_name, model_type, benchmark)

                    predictions.append(model_answer)
                    ground_truth.append(correct_answer)

                    # if model_answer_index == -1:
                    #     num_invalid+=1
        
            end_time = time.time()
            runtime = end_time - start_time
            print(f"\nRuntime: {runtime:.2f} seconds")
            print(f"Number of invalid answers: {num_invalid}")

        return predictions,ground_truth
    
    # wrapper function to process a single problem and its subtask functions, for multithreading
    def processSubtasks(self, problem):
        correct_answer_index = problem['answer']
        subtasks = assignSubtasks(client, self.commander_model, problem, self.roster)
        subtask_answers = self.roster.getSubtaskAnswers(problem, subtasks, benchmark)
        final_answer = finalAnswerFromSubtasks(client, self.commander_model, problem, subtask_answers)

        return final_answer, correct_answer_index


# class for openai structured outputs so answer always single int
class ScienceQAAnswer(BaseModel):
    answer: int

class MMMUAnswer(BaseModel):
    answer: str

# use commander model to analyze subtasks answers then provide a final answer
def finalAnswerFromSubtasks(client,commander_model,problem,subtask_answers: dict) -> int:
    
    # collect all subtask descriptions and answers as context
    subtask_context = ""
    i = 0
    for task,answer in subtask_answers.items():
        subtask_context += f"Subtask {i} description: {task}\nSubtask {i} answer:\n{answer}"
        i+=1
    
    answer_format = "single digit"

    # prompts
    if benchmark == "scienceqa":
        question_prompt = build_commander_prompt_ScienceQA(problem)
        image = problem['image']
        Answer = ScienceQAAnswer
    elif benchmark == "mmmu":
        question_prompt = build_commander_prompt_MMMU(problem)
        image = problem["image_1"]
        Answer = MMMUAnswer
        answer_format = "single letter"

    system_prompt = "You are an expert at answering multiple choice questions given answers to subtasks as context"
    prompt = f"Answer the following question: {question_prompt} by providing the {answer_format} of the correct choice. Use the image if provided as well as the following answers to subtasks to inform your final answer: {subtask_context}"
    
    # setting up messages depending on if a context image was provided
    if image:
        base64_image = encode_image(image)
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
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
    else:
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]

    # getting the parsed response
    completion = client.beta.chat.completions.parse(
        model=commander_model,
        messages=messages,
        response_format=Answer,
    )

    return completion.choices[0].message.parsed.answer

# function to print subtasks in a nice format just for testing
def printSubtasks(subtasks: Subtasks):
    print(f"Orignal Question:\n-----------------\n\n{subtasks.original_question}\n")
    
    i = 1
    for subtask in subtasks.subtasks:
        print(f"Subtask {i}\n-----------------")
        print(f"\nAssigned model: {subtask.assigned_model}")
        print(f"Subtask Description: {subtask.subtask}\n")
        i+=1

# formates the codenames(skills of each model) in a way the model can understand
def formatCodenames(roster: Roster):
    codenames = roster.getCodenames()
    output = "\nModels availablewith their best strengths:\n"

    for model_name,strength in codenames.items():
        output += f"Model name: {model_name}, Strength: {strength}"

    return output

# create subtasks and assign them to the best model
def assignSubtasks(client,commander_model,problem,roster: Roster):
    if benchmark == "scienceqa":
        question_prompt = build_commander_prompt_ScienceQA(problem)
        image = problem['image']
    elif benchmark == "mmmu":
        question_prompt = build_commander_prompt_MMMU(problem)
        image = problem["image_1"]

    # getting the model names and their strengths
    model_strengths = formatCodenames(roster)

    # system instructions
    system_prompt = f"You are an expert manager, based on the following list of avialable model names and their strengths: {model_strengths}, assign subtasks to each model that will help solve the overall question and can be solved INDEPENDENTLY. Do NOT answer the question yourself, and return the model name EXACTLY as provided. You can create more subtasks than the number of models, and/or assign the same type of model to multiple subtasks. If there is no image provided, do NOT assign any tasks involving analyzing an image."

    # currently assuming commander model is openai
    # setting up messages depending on if a context image was provided
    if image:
        base64_image = encode_image(image)
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": question_prompt,
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
    else:
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question_prompt}
        ]

    # getting the parsed response
    completion = client.beta.chat.completions.parse(
        model=commander_model,
        messages=messages,
        response_format=Subtasks,
    )

    return completion.choices[0].message.parsed


# do majority voting with multiple calls to a single model
def majorityVoteSingleModel(problem, num_votes: int, roster: Roster, model_name: str, model_type: str, benchmark):
    if benchmark == "scienceqa":
        answer_choices = {'1', '2', '3', '4'}
        prompt = build_ScienceQA_prompt(problem)
        image = problem['image']
        votes = {
            -1: 0,
            0: 0,
            1: 0,
            2: 0,
            3: 0
        }
    elif benchmark == "mmmu":
        answer_choices = {'A','B','C','D','E'}
        prompt = build_MMMU_prompt(problem)
        image = problem["image_1"]
        votes = {
            "invalid": 0,
            'A': 0,
            'B': 0,
            'C': 0,
            'D': 0,
            'E': 0
        }

    answer_prompt = "Absolutely provide your answer in the following format: Answer: <single digit of correct choice>"

    initialized_models = roster.initialized_models
    correct_answer = problem['answer']
    
    if model_type == "internvl":
        model,tokenizer = initialized_models[model_name]
    elif model_type == "qwenvl":
        model,processor = initialized_models[model_name]
    elif model_type == "hftext":
        model,tokenizer = initialized_models[model_name]
    
    # call model num_votes
    for i in range(num_votes):
        if (model_type == "hftext"):   # model from huggingface transformers
            output = getHFTextResponse(model=model, tokenizer=tokenizer, text=prompt + answer_prompt)
        
        elif (model_type == "internvl"):   # intern VL
            output = getInternVLReponse(model, tokenizer, text=prompt+answer_prompt, image=image)
        
        elif (model_type == "qwenvl"):   # Qwen VL
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
            
            output = getQwenVLResponse(model, processor, messages)
        elif model_type == "openai":
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

        # parsing the output
        if benchmark == "scienceqa":
            output = parseScienceQA(output, answer_choices)
            try:
                model_answer = int(output.strip()) - 1  # Convert to 0-based index
            except ValueError:
                model_answer = -1  # Handle invalid responses
        else:
            output = parse_MMMU(output, answer_choices)
            model_answer = output.strip()
        
        votes[model_answer]+=1
    
    max_vote_choice = max(votes, key=votes.get)

    return correct_answer, max_vote_choice

# multiprocessing enabled majority vote for openai models
def majorityVoteSingleParallel(problem):
    if benchmark == "scienceqa":
        answer_choices = {'1', '2', '3', '4'}
        prompt = build_ScienceQA_prompt(problem)
        image = problem['image']
        votes = {
            -1: 0,
            0: 0,
            1: 0,
            2: 0,
            3: 0
        }
    elif benchmark == "mmmu":
        answer_choices = {'A','B','C','D','E'}
        prompt = build_MMMU_prompt(problem)
        image = problem["image_1"]
        votes = {
            "invalid": 0,
            'A': 0,
            'B': 0,
            'C': 0,
            'D': 0,
            'E': 0
        }

    answer_prompt = "Absolutely provide your answer in the following format: Answer: <single digit of correct choice>"

    correct_answer = problem['answer']
    num_votes = 5
    model_name = "gpt-4o-mini"

    for i in range(num_votes):
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

        # parsing the output
        if benchmark == "scienceqa":
            output = parseScienceQA(output, answer_choices)
            try:
                model_answer = int(output.strip()) - 1  # Convert to 0-based index
            except ValueError:
                model_answer = -1  # Handle invalid responses
        else:
            output = parse_MMMU(output, answer_choices)
            model_answer = output.strip()
        
        votes[model_answer]+=1
    
    max_vote_choice = max(votes, key=votes.get)

    return correct_answer, max_vote_choice

# do majority voting with calls to multiple models
def majorityVoteMultiModel():
    return