import argparse
import json
import random
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import List
from uuid import uuid4

import torch
import torch.nn as nn
import torchaudio
from torch import Tensor
from torch.multiprocessing import Pool, cpu_count, set_start_method
from tqdm import tqdm

from data import Wav2Mel


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


def process_file(wav_path: Path, wav2mel: nn.Module) -> Tensor:
    wav, sr = torchaudio.load(wav_path)
    mel = wav2mel(wav, sr)
    return mel


def main(
    data_dir: Path, save_dir: Path, encoder_path: Path, seg_len: int, n_workers: int
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_start_method("spawn")
    encoder = torch.jit.load(encoder_path).eval().to(device)
    meta_data = defaultdict(list)
    (save_dir / "uttrs").mkdir(exist_ok=True)
    (save_dir / "embed").mkdir(exist_ok=True)
    spk_dirs = data_dir.iterdir()
    wav2mel = Wav2Mel()
    file2mel = partial(process_file, wav2mel=wav2mel)
    for spk in tqdm(spk_dirs):
        wav_files = list(spk.iterdir())
        with Pool(n_workers) as p:
            mels = p.map(file2mel, wav_files)
        mels = list(filter(lambda x: x is not None, mels))
        mels = [mel.to(device) for mel in mels]
        embed = embed_uttrs(encoder, mels, seg_len)
        rnd_paths = [f"uttrs/{uuid4().hex}.pt" for _ in range(len(mels))]
        dummy = [
            torch.save(mel.cpu(), save_dir / path) for path, mel in zip(rnd_paths, mels)
        ]
        emb_path = f"embed/{spk}.pt"
        torch.save(embed.cpu(), save_dir / emb_path)
        meta_data[spk] = {"embed": emb_path, "uttrs": rnd_paths}
    json.dump(meta_data, (save_dir / "metadata.json").open(mode="w"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("save_dir", type=Path)
    parser.add_argument("encoder_path", type=Path)
    parser.add_argument("--seg_len", type=int, default=128)
    parser.add_argument("--n_workers", type=int, default=cpu_count())
    main(**vars(parser.parse_args()))
