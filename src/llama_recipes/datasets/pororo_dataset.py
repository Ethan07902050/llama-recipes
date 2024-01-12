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
    'The paragraph below describes a series of frames in a cartoon. Provide concise and coherent descriptions of the background layout for each frame. Focus solely on objects in the background.\n\n'
    'Input:\n{caption}\n\nResponse:\n'
)

class InstructionDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train"):
        data_path = Path(dataset_config.data_path)
        background_path = data_path / f'background_prompt_{partition}.json'
        id_path = data_path / 'train_seen_unseen_ids.npy'
        following_path = data_path  / 'following_cache4.npy'
        description_path = data_path / 'descriptions.npy'
        self.data_path = dataset_config.data_path
        
        # original pororo captions
        self.video_len = dataset_config.video_len
        train_ids, val_ids, test_ids = np.load(id_path, allow_pickle=True)
        self.ids = train_ids if partition == 'train' else val_ids
        self.followings = np.load(following_path)
        self.descriptions_original = np.load(description_path, allow_pickle=True, encoding='latin1').item()
        
        # llava generated background
        self.background = json.load(open(background_path))
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss

        # load pororo captions
        src_img_id = self.ids[index]
        tgt_img_paths = [str(self.followings[src_img_id][i])[2:-1] for i in range(self.video_len)]
        tgt_img_ids = [str(tgt_img_path).replace(self.data_path, '').replace('.png', '') for tgt_img_path in tgt_img_paths]
        captions = [self.descriptions_original[tgt_img_id][0].strip() for tgt_img_id in tgt_img_ids]

        # load llava generated prompts
        background_prompts = self.background[str(src_img_id)]

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
