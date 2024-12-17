from MMO import Commander, Roster
from datasets import load_dataset
import argparse

def main():
    # getting command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Number of samples from evaluation dataset"
    )

    args = parser.parse_args()
    samples = args.samples

    # list of models we have access to, with their identifying model types
    agents = {
        # "gpt-4o-mini": "openai",
        "OpenGVLab/InternVL2_5-2B": "internvl",
        "Qwen/Qwen2-VL-2B-Instruct": "qwenvl",
        "Qwen/Qwen2.5-1.5B-Instruct": "hftext"
    }

    # creating a Roster object from the list of models
    roster = Roster(agents)

    # loading dataset (currently scienceqa)
    dataset = load_dataset('derek-thomas/ScienceQA', split='test')

    # Sample a few entries for testing
    sample_size = samples  # Adjust the sample size as needed
    dataset = dataset.shuffle(seed=51).select(range(sample_size))
    
    # default commander model is gpt-4o
    commander_model = "gpt-4o"
    commander = Commander(roster=roster, commander_model=commander_model, organization_mode="auto-subtask", dataset=dataset)

    # tests
    commander.roster.printCodenames()
    commander.getResponses()

    
    return


if __name__ == "__main__":
    main()