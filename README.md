# LLM4HWDesign_Starting_Toolkit
## Introduction
This repository provides a starting toolkit for participants of the [LLM4HWDesign Contest at ICCAD 2024](https://nvlabs.github.io/LLM4HWDesign/). The toolkit includes scripts and utilities for deduplicating training data, fine-tuning models, and evaluating model performance. Participants can use this toolkit to kickstart their work and streamline their development process.

## Base Dataset
The base dataset used in the contest is the [MG-Verilog dataset](https://huggingface.co/datasets/GaTech-EIC/MG-Verilog). For your submitted data, please follow the same format as the [MG-Verilog dataset](https://huggingface.co/datasets/GaTech-EIC/MG-Verilog).

## Toolkit Release Progress
- [x] **Deduplication**: Scripts to identify and remove duplicate samples from the dataset.
- [ ] **Fine-tuning**: Scripts to fine-tune a pretrained language model on the MG-Verilog dataset.
- [ ] **Evaluation**: Tools to evaluate the performance of the fine-tuned model using standard metrics.


## Setup Environment

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

```bash
git clone https://github.com/NVlabs/verilog-eval.git
pip install -e verilog-eval
```

**Evaluation Scripts**:

```bash
cd model_eval
./gen.sh <path_to_your_model> <path_to_your_tokenizer>
#example: ./gen.sh "codellama/CodeLlama-7b-Instruct-hf" "code_llama"
```

The results will be printed and logged in `./data/gen.jsonl`