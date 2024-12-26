# MMO: An Investigation of Multi-modal Multi-agent Organization & Robust Benchmarking of MLLMs
A course project for CMSC848K - Multimodal Foundation Models @ University of Maryland

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

Then for ScienceQA baselines run the following command (edit for desired model and sample size (max = 4241)):

    python3 scienceqa_eval.py --samples <SAMPLESIZE> --model <MODELNAME> --model_type <MODELTYPE>
    python3 scienceqa_eval.py --samples 1000 --model Qwen/Qwen2.5-1.5B-Instruct --model_type hf

**NOTES:**
* For models installed through Huggingface, provide the path to the model as shown on the model card (e.g. "Qwen/Qwen2-VL-2B-Instruct"), for OpenAI models simply write the model name (e.g. "gpt-4o-mini")
* The --model_type argument specifies the general type of the model, here "qwen" indicates QwenVL, "intern" InternVL, "openai" any OpenAI model, and "hf" general text-only models from Huggingface (only Qwen2.5 tested)
* Instead of "scienceqa_eval.py", you can use "scieceqa_eval_parallel.py" then run as normal for faster runtimes. It is **recommended to use this option for only OpenAI models** currently, as during my testing the parallelized runs for other open-source models were slower.


