# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# from accelerate import init_empty_weights, load_checkpoint_and_dispatch

import fire
import os
import sys
import time
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
from transformers import LlamaTokenizer

from llama_recipes.inference.safety_utils import get_safety_checker, AgentType
from llama_recipes.inference.model_utils import load_model, load_peft_model

from accelerate.utils import is_xpu_available

instruction = (
    'The paragraph below describes a series of frames in a cartoon. Provide concise and coherent descriptions of the background layout for each frame. Focus solely on objects in the background.\n\n'
    'Input:\nLoopy thinks Loopy is not good at exercising.\nPoby suggests Loopy to dance. Eddy Crong Pororo and Loopy are surprised.\nEddy Crong Pororo and Loopy are looking at Poby dancing.\nPoby is standing in front of his friends. Poby explains that Poby dances as and exercise.\n\n'
    'Response:\n'
    'The paragraph below describes a series of frames in a cartoon. Provide concise and coherent descriptions of the background layout for each frame. Focus solely on objects in the background.\n\n'
    'Input:\n{caption}\n\nResponse:\n'
)

def main(
    model_name,
    data_path: str='',
    out_path: str='',
    video_len: int=4,
    peft_model: str=None,
    quantization: bool=False,
    max_new_tokens =100, #The maximum numbers of tokens to generate
    prompt_file: str=None,
    seed: int=42, #seed value for reproducibility
    do_sample: bool=True, #Whether or not to use sampling ; use greedy decoding otherwise.
    min_length: int=None, #The minimum length of the sequence to be generated, input prompt + min_new_tokens
    use_cache: bool=True,  #[optional] Whether or not the model should use the past last key/values attentions Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
    top_p: float=1.0, # [optional] If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
    temperature: float=1.0, # [optional] The value used to modulate the next token probabilities.
    top_k: int=50, # [optional] The number of highest probability vocabulary tokens to keep for top-k-filtering.
    repetition_penalty: float=1.0, #The parameter for repetition penalty. 1.0 means no penalty.
    length_penalty: int=1, #[optional] Exponential penalty to the length that is used with beam-based generation. 
    enable_azure_content_safety: bool=False, # Enable safety check with Azure content safety api
    enable_sensitive_topics: bool=False, # Enable check for sensitive topics using AuditNLG APIs
    enable_salesforce_content_safety: bool=True, # Enable safety check with Salesforce safety flan t5
    enable_llamaguard_content_safety: bool=False,
    max_padding_length: int=None, # the max padding length to be used with tokenizer padding the prompts.
    use_fast_kernels: bool = False, # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    **kwargs
):
    # Set the seeds for reproducibility
    if is_xpu_available():
        torch.xpu.manual_seed(seed)
    else:
        torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    
    model = load_model(model_name, quantization)
    if peft_model:
        model = load_peft_model(model, peft_model)

    model.eval()
    
    if use_fast_kernels:
        """
        Setting 'use_fast_kernels' will enable
        using of Flash Attention or Xformer memory-efficient kernels 
        based on the hardware being used. This would speed up inference when used for batched inputs.
        """
        try:
            from optimum.bettertransformer import BetterTransformer
            model = BetterTransformer.transform(model)    
        except ImportError:
            print("Module 'optimum' not found. Please install 'optimum' it before proceeding.")

    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    data_path = Path(data_path)
    id_path = data_path / 'train_seen_unseen_ids.npy'
    following_path = data_path  / 'following_cache4.npy'
    description_path = data_path / 'descriptions.npy'

    _, _, test_ids = np.load(id_path, allow_pickle=True)
    followings = np.load(following_path)
    descriptions_original = np.load(description_path, allow_pickle=True, encoding='latin1').item()

    examples = {}
    for src_img_id in tqdm(test_ids):
        # load pororo captions
        tgt_img_paths = [str(followings[src_img_id][i])[2:-1] for i in range(video_len)]
        tgt_img_ids = [str(tgt_img_path).replace(str(data_path), '').replace('.png', '') for tgt_img_path in tgt_img_paths]
        captions = [descriptions_original[tgt_img_id][0].strip() for tgt_img_id in tgt_img_ids]
        prompt = instruction.format(caption='\n'.join(captions))

        batch = tokenizer(prompt, padding='max_length', truncation=True, max_length=max_padding_length, return_tensors="pt")
        if is_xpu_available():
            batch = {k: v.to("xpu") for k, v in batch.items()}
        else:
            batch = {k: v.to("cuda") for k, v in batch.items()}

        with torch.no_grad():
            outputs = model.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                top_p=top_p,
                temperature=temperature,
                min_length=min_length,
                use_cache=use_cache,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                **kwargs 
            )

        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        background = output_text[len(prompt):].strip('\n').split('\n')
        examples[str(src_img_id)] = background

    with open(data_path / out_path, 'w') as f:
        json.dump(examples, f, indent=2)
                

if __name__ == "__main__":
    fire.Fire(main)
