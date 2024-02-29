# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://crfm.stanford.edu/2023/03/13/alpaca.html

import copy
import json
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import Dataset

instruction = (
    'The paragraph below describes a series of frames in a film.\n' 
    'Provide concise and coherent descriptions of the background layout for each frame.\n' 
    'Focus solely on objects in the background.\n' 
    'Do not mention character names.\n' 
    'Each input should correspond to a response.\n\n'
    'Input:\n{caption}\n\nResponse:\n'
)

class InstructionDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train"):
        data_path = Path(dataset_config.data_path)
        background_path = data_path / f'background_llava_{partition}.json'
        metadata_path = data_path / 'metadata.json'
        self.data_path = data_path
        self.metadata = json.load(open(metadata_path))[partition]
        self.background = json.load(open(background_path))
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss

        # load pororo captions
        captions = []
        for i, caption in enumerate(self.metadata[index]['text']):
            captions.append(f'{i+1}. {caption}')

        # load llava generated prompts
        _id = self.metadata[index]['assignment_id']
        background_prompts = []
        for i, prompt in enumerate(self.background[_id]):
            background_prompts.append(f'{i+1}. {prompt}')

        prompt = instruction.format(caption='\n'.join(captions))
        example = prompt + '\n'.join(background_prompts)
        prompt = torch.tensor(
            self.tokenizer.encode(prompt), dtype=torch.int64
        )
        example = self.tokenizer.encode(example)
        example.append(self.tokenizer.eos_token_id)
        example = torch.tensor(
            example, dtype=torch.int64
        )
        labels = copy.deepcopy(example)
        labels[: len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = IGNORE_INDEX

        return {
            "input_ids": example.tolist(),
            "labels": labels.tolist(),
            "attention_mask":example_mask.tolist(),
        }
