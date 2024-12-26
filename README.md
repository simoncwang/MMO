# MMO: An Investigation of Multi-modal Multi-agent Organization & Robust Benchmarking of MLLMs
A course project for CMSC848K - Multimodal Foundation Models @ University of Maryland

## Introduction

## Setup

First, **clone this repository** to your local machine:

    git clone https://github.com/simoncwang/MMO.git

Then, **create a Python environment**

Conda:

    conda create -n MMO python=3.12

**activate the environment** with

Conda:

    conda activate MMO

Next, **install the required packages** with:

    pip install -r requirements.txt


## Baselines

Navigate to the BaselineEvals folder with

    cd BaselineEvals

Then for **ScienceQA** baselines run the following command (edit for desired model and sample size (max = 4241)):

    python3 scienceqa_eval.py --samples <SAMPLESIZE> --model <MODELNAME> --model_type <MODELTYPE>

Examples of benchmark runs:

    # example of standard benchmark with InternVL
    python3 scienceqa_eval.py --samples 1000 --model OpenGVLab/InternVL2_5-2B --model_type intern

    # example using parallelized code with gpt-4o
    python3 scienceqa_eval_parallel.py --samples 1000 --model gpt-4o --model_type openai

**NOTES:**
* For models installed through Huggingface, provide the path to the model as shown on the model card (e.g. "Qwen/Qwen2-VL-2B-Instruct"), for OpenAI models simply write the model name (e.g. "gpt-4o-mini")
* The --model_type argument specifies the general type of the model, here "qwen" indicates QwenVL, "intern" InternVL, "openai" any OpenAI model, and "hf" general text-only models from Huggingface (only Qwen2.5 tested)
* Instead of "scienceqa_eval.py", you can use "scieceqa_eval_parallel.py" then run as normal for faster runtimes. It is **recommended to use this option for only OpenAI models** currently, as during my testing the parallelized runs for other open-source models were slower.

For **MMMU** baselines run the following command (example shown for gpt-4o, but model and model_type can be changed like before):

        python3 mmmu_eval.py --model gpt-4o --model_type openai --subset Physics

**NOTES:**
* General usage is same as ScienceQA, but for MMMU specify the subset desired (e.g. Physics, Computer Science, Chemistry, etc.), for full documentation visit the official MMMU page for all subsets available https://mmmu-benchmark.github.io/
* By default, no sample size is set since I am using the validation partition of the MMMU dataset, and therefore subset sizes are relatively small and we can use the entire dataset. However, if you run into GPU memory issues, use the optional --samples argument to specify a smaller sample size (refer to MMMU documentation for the size of your subset)
 
## Multi-agent Benchmarking

Navigate back to the root directory if you are in the BaselineEvals directory:

        cd ..

There are currently two multi-agent experiments implemented; **Auto-Subtask** and **Majority Vote**. You may specify the type of experiment using the --mode argument, and the benchmark to test on using --benchmark as follows:

For **Auto-Subtask:**:

        python3 main.py --mode auto-subtask --benchmark mmmu --subset Physics

For **Majority Vote**:

        python3 main.py --samples 1000 --mode majority-vote-single --model_name gpt-4o-mini --model_type openai --benchmark scienceqa

**NOTES:**
* For the Auto-Subtask mode, do NOT provide --model_name, --samples, or --model_type arguments. The Auto-Subtask system is already preset with a Commander model and a Roster of agents. To edit the models in the system you must change them within the main.py file and MMO/Commander.py file (**a more easy to use configuration system will be implemented in the near future**).
* For the Majority Vote experiment, be careful setting the sample size when using the mmmu benchmark so as to not exceed the size of the subset you are using (also be s ure to include the --subset argument when using --benchmark mmmu)


## Future Work

This codebase is a work-in-progress, and I intend to extend the system to be compatible with many other common models and top benchmarking datasets. I am working on consolidating the code to run from a single execution script, and allow multi-agent configurations to be organized in separate configuration files for ease-of-use.

Please let me know if there are any issues, and any suggestions for improvements would be greatly appreciated!


