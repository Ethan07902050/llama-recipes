# python -m llama_recipes.finetuning \
#     --use_peft \
#     --peft_method lora \
#     --model_name meta-llama/Llama-2-7b-hf \
#     --quantization \
#     --output_dir outputs \
#     --dataset pororo_dataset \
#     --save_metrics

torchrun --nnodes 1 --nproc_per_node 4 examples/finetuning.py \
    --enable_fsdp \
    --use_peft \
    --peft_method lora \
    --model_name meta-llama/Llama-2-7b-hf \
    --fsdp_config.pure_bf16 \
    --output_dir outputs/10-epoch \
    --num_epochs 10 \
    --dataset pororo_dataset \
    --save_metrics
