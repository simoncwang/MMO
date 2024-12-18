from .mmo_utils import *

class Roster():
    def __init__(self, agents: dict):
        # what each model in our system is best at
        all_codenames = {
            "Qwen/Qwen2-VL-2B-Instruct": "vision-medquality-fast",
            "OpenGVLab/InternVL2_5-2B": "vision-highquality-slow",
            "gpt-4o-mini": "multi",
            "Qwen/Qwen2.5-1.5B-Instruct": "only-text-no-vision"
        }
        
        # assigning codenames to our agents
        self.codenames = {}
        for agent in agents:
            self.codenames[agent] = all_codenames[agent]
        
        # storing the model types of each model
        model_types = {}
        for model,model_type in agents.items():
            model_types[model] = model_type
        
        # initialize all models and store here
        num_models = len(agents)
        print(f"Initializing {num_models} models...please wait a moment!\n")
        self.initialized_models = initializeModels(model_types)
        return
    
    # returns codenames dict
    def getCodenames(self):
        return self.codenames
    
    def printCodenames(self):
        print(f"Roster Codenames:\n")
        for agent,codename in self.codenames.items():
            print(f"Agent name: {agent}, Codename: {codename}\n")

    
    # get answers from each model for their subtask
    def getSubtaskAnswers(self, problem, subtasks: Subtasks) -> dict:
        question = build_prompt(problem)
        image = problem['image']
        initialized_models = self.initialized_models

        image_provided = False
        if image:
            image_provided = True
            base64_image = encode_image(image)
        
        responses = {}

        # iterate through the subtask list
        for subtask in subtasks.subtasks:
            model_name = subtask.assigned_model
            task_description = subtask.subtask

            prompt = f"Given the original question:\n{question}\nAnd the image attached if present, complete the following subtask description: {task_description}"

            if model_name == "Qwen/Qwen2.5-1.5B-Instruct":
                model,tokenizer = initialized_models[model_name]
                response = getHFTextResponse(model,tokenizer,prompt)
            elif model_name == "Qwen/Qwen2-VL-2B-Instruct":
                model,processor = initialized_models[model_name]
                if image_provided:
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": image},
                                {"type": "text", "text": prompt},
                            ],
                        }
                    ]
                else:
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                            ],
                        }
                    ]

                response = getQwenVLResponse(model,processor,messages)
            elif model_name == "OpenGVLab/InternVL2_5-2B":
                model,tokenizer = initialized_models[model_name]
                response = getInternVLReponse(model,tokenizer,prompt,image)

            responses[task_description] = response

        return responses
    
    


