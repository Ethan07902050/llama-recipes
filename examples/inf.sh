python pororo_inference.py \
    --data_path /work/u1509343/storydalle/data/pororo_png \
    --out_path background_llama_test_10_epoch.json \
    --model_name meta-llama/Llama-2-7b-hf \
    --peft_model ../outputs/10-epoch \
    --quantization \
    --max_new_tokens 1024 \
    --max_padding_length 1024 \
    --top_p 0.9 \
    --temperature 0.6 \
    --repetition_penalty 1.2