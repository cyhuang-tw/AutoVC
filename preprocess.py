import argparse
import json
import os
import random
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from typing import List
from uuid import uuid4

import torch
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm

from modules.Audioprocessor import AudioProcessor


def embed_uttrs(encoder: nn.Module, uttrs: List[Tensor], seg_len: int):
    with torch.no_grad():
        uttrs = list(filter(lambda x: len(x) > seg_len, uttrs))
        random.shuffle(uttrs)
        uttrs = uttrs[: min(len(uttrs), 10)]
        starts = [random.randint(0, len(x) - seg_len) for x in uttrs]
        uttrs = torch.stack(
            [x[start : start + seg_len] for (x, start) in zip(uttrs, starts)]
        )
        embeds = encoder(uttrs)
        embed = embeds.mean(dim=0)

    return embed


def main(data_dir: str, save_dir: str, encoder_path: str, seg_len: int, n_workers: int):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = torch.jit.load(encoder_path).eval().to(device)
    meta_data = defaultdict(list)
    os.makedirs(os.path.join(save_dir, "uttrs"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "embed"), exist_ok=True)
    spk_dirs = os.listdir(data_dir)
    for spk in tqdm(spk_dirs):
        wav_files = [
            os.path.join(data_dir, spk, file)
            for file in os.listdir(os.path.join(data_dir, spk))
        ]
        with Pool(n_workers) as p:
            mels = p.map(AudioProcessor.file2spectrogram, wav_files)
        mels = list(filter(lambda x: x is not None, mels))
        mels = [torch.from_numpy(mel).to(device) for mel in mels]
        embed = embed_uttrs(encoder, mels, seg_len)
        rnd_paths = [
            os.path.join("uttrs", f"{uuid4().hex}.pt") for _ in range(len(mels))
        ]
        dummy = [
            torch.save(mel.cpu(), os.path.join(save_dir, path))
            for path, mel in zip(rnd_paths, mels)
        ]
        emb_path = os.path.join("embed", f"{spk}.pt")
        torch.save(embed.cpu(), os.path.join(save_dir, emb_path))
        meta_data[spk] = {"embed": emb_path, "uttrs": rnd_paths}
    json.dump(meta_data, open(os.path.join(save_dir, "metadata.json"), "w"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str)
    parser.add_argument("save_dir", type=str)
    parser.add_argument("encoder_path", type=str)
    parser.add_argument("--seg_len", type=int, default=128)
    parser.add_argument("--n_workers", type=int, default=cpu_count())
    main(**vars(parser.parse_args()))
