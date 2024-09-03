# LLM4HWDesign Starting Toolkit
## Introduction
This repository provides a starting toolkit for participants of the [LLM4HWDesign Contest at ICCAD 2024](https://nvlabs.github.io/LLM4HWDesign/). The toolkit includes scripts and utilities for deduplicating training data, fine-tuning models, and evaluating model performance. Participants can use this toolkit to kickstart their work and streamline their development process.

## Base Dataset
The base dataset used in the contest is the [MG-Verilog dataset](https://huggingface.co/datasets/GaTech-EIC/MG-Verilog). For your submitted data, please follow the same format as the [MG-Verilog dataset](https://huggingface.co/datasets/GaTech-EIC/MG-Verilog). Please note that you can either provide multiple levels or a single level of description for each code sequence, but we will **concatenate all descriptions at different levels into one string** for each code sequence following the script below.

```python
instructions_dict = {
    "summary": "xxx",
    "detailed explanation": "yyy"
}

result = ";\n".join([f"{key}: {value}" for key, value in instructions_dict.items()]) + "\n"

'''
result should be
summary: xxx;
detailed explanation: yyy
'''
```
## Toolkit Release Progress
- [x] **Deduplication**: Scripts to identify and remove duplicate samples from the dataset.
- [x] **Fine-tuning**: Scripts to fine-tune a pretrained language model on the MG-Verilog dataset.
- [x] **Evaluation**: Tools to evaluate the performance of the fine-tuned model using standard metrics.


## Setup Environment

We assume CUDA 12.1. (Only needed if you want to do fine-tuning and evaluation on your own.)

`conda env create -f environment.yml`


## Deduplication
The toolkit includes a deduplication script, which will be used to deduplicate each participant's data against the base dataset during the evaluation of Phase I.
To run the deduplication script:
```bash
python minhash.py
```


## Evaluation

The following shows an example on how to evaluate your fine-tuned model.

**Prerequisites**:

`export HF_TOKEN=your_huggingface_token`

Prepare your fine-tuned model and tokenizer in HuggingFace format.

Install [Iverilog](https://steveicarus.github.io/iverilog/usage/installation.html) 

Install VerilogEval as the benchmark:

**Please read the WARNINGS in the [VerilogEval](https://github.com/NVlabs/verilog-eval/tree/main?tab=readme-ov-file#usage) before proceeding**

**Only support 1.0 Version** https://github.com/NVlabs/verilog-eval/tree/release/1.0.0 

Pay attention to the "verilog-eval" or "verilog_eval" which is used in mg-verilog's own midified VerilogEval

```bash
git clone https://github.com/NVlabs/verilog-eval.git
pip install -e verilog-eval
```

**Evaluation Scripts**:

```bash
cd model_eval
./gen.sh <path_to_folder_with_your_model_and_config> <your_huggingface_token>
#example: ./gen.sh "finetuned_model/" "hf-xxxxxxxxxx"
```

NOTE: The folder with your model and config should include two files (1) the generated pytorch_model.bin and (2) the [model config](https://huggingface.co/meta-llama/CodeLlama-7b-Instruct-hf/blob/main/config.json) of CodeLlama-7B-Instruct from HuggingFace

The results will be printed and logged in `./model_eval/data/gen.jsonl`
