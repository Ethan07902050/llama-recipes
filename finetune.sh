# python -m llama_recipes.finetuning \
#     --use_peft \
#     --peft_method lora \
#     --model_name meta-llama/Llama-2-13b-chat-hf \
#     --quantization \
#     --output_dir outputs/vwp-13b-chat-3-epoch \
#     --dataset vwp_dataset \
#     --save_metrics

torchrun --nnodes 1 --nproc_per_node 2 examples/finetuning.py \
    --enable_fsdp \
    --use_peft \
    --peft_method lora \
    --model_name meta-llama/Llama-2-13b-chat-hf \
    --fsdp_config.pure_bf16 \
    --output_dir outputs/vwp-13b-chat-1-epoch \
    --num_epochs 1 \
    --batch_size_training 1 \
    --dataset vwp_dataset \
    --save_metrics
