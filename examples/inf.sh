export CUDA_VISIBLE_DEVICES=0,1
python vwp_inference.py \
    --data_path ../../vwp/data \
    --out_path ../outputs/background_llama_test_1_epoch.json \
    --model_name meta-llama/Llama-2-13b-chat-hf \
    --peft_model ../outputs/vwp-13b-chat-1-epoch \
    --quantization \
    --max_new_tokens 1024 \
    --max_padding_length 1024 \
    --top_p 0.9 \
    --temperature 0.6 \
    --repetition_penalty 1.2
