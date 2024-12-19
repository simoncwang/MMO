from MMO import Commander, Roster
from datasets import load_dataset
import argparse
from sklearn.metrics import accuracy_score

def main():
    # getting command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Number of samples from evaluation dataset"
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="majority-vote-single",
        help="The experiment to run"
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt-4o-mini",
        help="The name of the model to run the experiment on"
    )

    parser.add_argument(
        "--model_type",
        type=str,
        default="openai",
        help="The model provider or type"
    )

    parser.add_argument(
        "--benchmark",
        type=str,
        default="scienceqa",
        help="The benchmark to test on."
    )

    parser.add_argument(
        "--subset",
        type=str,
        default="Physics",
        help="The subset of MMMU"
    )


    args = parser.parse_args()
    samples = args.samples
    mode = args.mode
    model_name = args.model_name
    model_type = args.model_type
    benchmark = args.benchmark
    subset = args.subset

    # list of models we have access to, with their identifying model types
    agents = {
        # "gpt-4o-mini": "openai",
        "OpenGVLab/InternVL2_5-2B": "internvl",
        # "Qwen/Qwen2-VL-2B-Instruct": "qwenvl",
        "Qwen/Qwen2.5-1.5B-Instruct": "hftext",
        # "gpt-4o-mini": "openai"
    }

    # creating a Roster object from the list of models
    roster = Roster(agents)

    if benchmark == "scienceqa":
        # loading dataset (currently scienceqa)
        dataset = load_dataset('derek-thomas/ScienceQA', split='test')

        # Sample a few entries for testing
        sample_size = samples  # Adjust the sample size as needed
        dataset = dataset.shuffle(seed=51).select(range(sample_size))
    elif benchmark == "mmmu":
        dataset = load_dataset("MMMU/MMMU", subset)
        # filtering only multiple choice
        dataset = dataset["validation"].filter(lambda problem: problem["question_type"] == "multiple-choice")
    
    # default commander model is gpt-4o
    commander_model = "gpt-4o"
    # commander = Commander(roster=roster, commander_model=commander_model, organization_mode="auto-subtask", multithreading=True, dataset=dataset)
    commander = Commander(roster=roster, commander_model=commander_model, mode=mode, model_name=model_name, model_type=model_type, multithreading=True, dataset=dataset)

    # tests
    commander.roster.printCodenames()
    predictions,ground_truth = commander.getResponses()

    # Printing out metrics
    if benchmark == "scienceqa":
        print(f"Number of samples: {samples}")
        
    accuracy = accuracy_score(ground_truth, predictions)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    # print(f"Number of invalid responses: {num_invalid}")
    
    return


if __name__ == "__main__":
    main()