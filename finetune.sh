python -m llama_recipes.finetuning \
    --use_peft \
    --peft_method lora \
    --model_name meta-llama/Llama-2-7b-hf \
    --quantization \
    --output_dir outputs \
    --dataset pororo_dataset \
    --save_metrics
