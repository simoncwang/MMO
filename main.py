from MMO import Commander, Roster

def main():

    # list of models we have access to, with their identifying model types
    agents = {
        "gpt-4o-mini": "openai",
        "OpenGVLab/InternVL2_5-2B": "internvl",
        "Qwen/Qwen2-VL-2B-Instruct": "qwenvl",
        "Qwen/Qwen2.5-1.5B-Instruct": "qwen"
    }

    roster = Roster(agents)
    
    # default commander model is gpt-4o
    commander_model = "gpt-4o"
    commander = Commander(roster, commander_model)

    # test
    commander.roster.printCodenames()
    
    return ""


if __name__ == "__main__":
    main()