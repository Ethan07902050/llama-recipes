torchrun --nnodes 1 --nproc_per_node 4 examples/finetuning.py \
    --enable_fsdp \
    --use_peft \
    --peft_method lora \
    --model_name meta-llama/Llama-2-7b-hf \
    --fsdp_config.pure_bf16 \
    --output_dir path/to/save/PEFT/model \
    --dataset pororo_dataset
