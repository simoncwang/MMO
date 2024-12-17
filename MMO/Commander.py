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
from openai import OpenAI

class Commander():
    def __init__(self, roster: Roster, commander_model: str, organization_mode: str, dataset) -> None:
        self.roster = roster
        self.dataset = dataset
        self.commander_model = commander_model
        self.organization_mode = organization_mode
        
        # initializing OpenAI client once
        api_key = input("Please enter your OpenAI API Key: ")
        self.client = OpenAI(api_key=api_key, timeout=60)

        return
    
    def getRoster(self) -> Roster:
        return self.roster
    
    # iterate over all problems in the dataset and get answers depending on the organization mode
    def getResponses(self):
        mode = self.organization_mode
        data = self.dataset
        total_questions = len(data)

        if mode == "auto-subtask":
            for problem in tqdm(data, total=total_questions, desc="Evaluating", unit="question"):
                subtasks = assignSubtasks(self.client, self.commander_model, problem, self.roster)
                printSubtasks(subtasks)
        elif mode == "majority-vote":
            for problem in tqdm(data, total=total_questions, desc="Evaluating", unit="question"):
                majorityVote()
    

    
# given a problem from the dataset, use the commander model to generate subtasks (without solving the problem)
                
# a single subtask
class Subtask(BaseModel):
    subtask: str
    assigned_model: str

# list of all subtasks
class Subtasks(BaseModel):
    original_question: str
    subtasks: list[Subtask]

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
    # for creating subtasks, only need question and image if given
    question = problem['question']
    image = problem['image']

    # getting the model names and their strengths
    model_strengths = formatCodenames(roster)

    # generating question and system prompts
    question_prompt = f"Question: {question}\n"
    question_prompt += "Use the provided image as context if present."
    system_prompt = f"You are an expert manager, based on the following list of avialable model names and their strengths: {model_strengths}, assign subtasks to each model that will help solve the overall question. Do NOT answer the question yourself, and return the model name EXACTLY as provided."

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

# do majority voting
def majorityVote():
    return