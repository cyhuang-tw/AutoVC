import json
import os
import random

import torch
from torch.nn.functional import pad
from torch.utils.data import Dataset


class SpeakerDataset(Dataset):
    def __init__(self, data_dir, seg_len):
        self.data_dir = data_dir
        self.meta_data = json.load(open(os.path.join(data_dir, "metadata.json")))
        self.seg_len = seg_len
        self.idx2spk = list(self.meta_data.keys())

    def __len__(self):
        return len(self.meta_data.keys())

    def __getitem__(self, index):
        # sample utterances
        index = self.idx2spk[index]
        emb_path = self.meta_data[index]["embed"]
        embed = torch.load(os.path.join(self.data_dir, emb_path))
        mel = torch.load(
            os.path.join(self.data_dir, random.choice(self.meta_data[index]["uttrs"]))
        )
        if len(mel) < self.seg_len:
            len_pad = self.seg_len - len(mel)
            mel = pad(mel, (0, 0, 0, len_pad), "constant", 0)
        else:
            start = random.randint(0, max(len(mel) - self.seg_len, 0))
            mel = mel[start : (start + self.seg_len)]
        return mel, embed
