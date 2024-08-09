# Fine-Tunning in NeMo

In our evaluation process for each submission, we will conduct fine-tuning with the following steps. To reproduce this process, we suggest using a machine with 4x A100 GPUs (or any NVIDIA GPU card with more than 24GB VRAM). If you plan to use a different number of GPUs other than 4, please adjust the `TP_SIZE` and `GPU_COUNT` parameters in the `run_peft.sh` script accordingly.


## NeMO docker setup

```bash
# In Host Machine
docker pull nvcr.io/nvidia/nemo:24.05
```

## Run NeMO docker
```bash
# In Host Machine
docker run --gpus all --shm-size=2g --net=host --ulimit memlock=-1 --rm -it -v ${PWD}:/workspace -w /workspace -v ${PWD}/results:/results nvcr.io/nvidia/nemo:24.05 bash
```

## Prepare dataset

Please preapre the dataset to be summited in .jsonl format. Here we provided an example of converting [MG-Verilog dataset](https://huggingface.co/datasets/GaTech-EIC/MG-Verilog) in huggingface into .jsonl format and used it as the fine-tuning dataset.

```bash
# In Host Machine
cd MG-Verilog
python data_format_conversion.py
```

## Download Codellama-7b weight
```bash
# In Docker Container
huggingface-cli login
huggingface-cli download codellama/CodeLlama-7b-Instruct-hf --local-dir CodeLlama-7b
```

## Convert huggingface model to NeMO format
```bash
# In Docker Container
python /opt/NeMo/scripts/checkpoint_converters/convert_llama_hf_to_nemo.py --input_name_or_path=./CodeLlama-7b/ --output_path=CodeLlama-7b.nemo
```

## Run LoRA fine-tunning
```bash
# In Docker Container
mkdir /workspace/results/MG-Verilog
bash run_peft.sh
```

## Convert NeMO model to huggingface format
```bash
# In Docker Container
cd /opt/NeMo/
python scripts/nlp_language_modeling/merge_lora_weights/merge.py \
    trainer.accelerator=cpu \
    gpt_model_file=/workspace/CodeLlama-7b.nemo \
    lora_model_path=/workspace/results/MG-Verilog/checkpoints/megatron_gpt_peft_lora_tuning.nemo \
    merged_model_path=/workspace/results/MG-Verilog/checkpoints/CodeLlama-7b-lora-merged.nemo

python /opt/NeMo/scripts/checkpoint_converters/convert_llama_nemo_to_hf.py \
    --input_name_or_path /workspace/results/MG-Verilog/checkpoints/CodeLlama-7b-lora-merged.nemo \
    --output_path /workspace/results/MG-Verilog/checkpoints/CodeLlama-7b-lora-merged/pytorch_model.bin \
    --cpu-only
```