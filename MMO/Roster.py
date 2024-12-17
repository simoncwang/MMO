from .mmo_utils import *

class Roster():
    def __init__(self, agents: dict):
        # what each model in our system is best at
        all_codenames = {
            "Qwen/Qwen2-VL-2B-Instruct": "vision-fast",
            "OpenGVLab/InternVL2_5-2B": "vision-slow",
            "gpt-4o-mini": "multi",
            "Qwen/Qwen2.5-1.5B-Instruct": "text"
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

    
    
    


