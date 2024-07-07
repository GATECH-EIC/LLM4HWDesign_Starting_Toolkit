#should be from huggingface hub;
#base_model="codellama/CodeLlama-7b-Instruct-hf"
base_model=$1
#tokenizer_type="code_llama"
tokenizer_type=$2


work_dir=$(pwd)

export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64'

accelerate launch --multi_gpu generate.py \
    --checkpoint_dir ./result_ckpt \
    --model_type "qlora" \
    --base_model $base_model \
    --tokenizer_type $tokenizer_type \
    --cache_dir $work_dir/.cache \
    --hf_token $HF_TOKEN \
    --max_new_tokens 1024 \
    --temperature 0.7 \
    --desc_file ../verilog-eval/descriptions/VerilogDescription_Machine.jsonl \
    --desc_key "detail_description" \
    --prompt_type "baseline" \
    --eval_file ../verilog-eval/data/VerilogEval_Machine.jsonl \
    --output_file ./data/gen.jsonl \
    --fp16 \
    --sample_k 20 \
    --result_name Test \
    --batch_size 2 
