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


## Usage

### Baselines

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
* 
 
###


