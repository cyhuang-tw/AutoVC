import argparse
from math import ceil
from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torchaudio
from torch import Tensor
from torch.nn.functional import pad

from data import Wav2Mel


def chunks(lst: List, n: int) -> List[List]:
    for i in range(0, len(lst), n):
        yield lst[i : (i + n)]


def pad_seq(x: Tensor, base: int = 32) -> Tuple[Tensor, int]:
    len_out = int(base * ceil(float(len(x)) / base))
    len_pad = len_out - len(x)
    assert len_pad >= 0
    return pad(x, (0, 0, 0, len_pad), "constant", 0), len_pad


def get_embed(encoder: nn.Module, mel: Tensor) -> Tensor:
    emb = encoder(mel[None, :])
    return emb


def main(
    model_path: Path,
    vocoder_path: Path,
    source: Path,
    target: Path,
    output: Path,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.jit.load(model_path).to(device)
    vocoder = torch.jit.load(vocoder_path).to(device)
    wav2mel = Wav2Mel()

    src, src_sr = torchaudio.load(source)
    tgt, tgt_sr = torchaudio.load(target)
    src_mel = wav2mel(src, src_sr).to(device)
    tgt_mel = wav2mel(tgt, tgt_sr).to(device)
    src_emb = get_embed(model.speaker_encoder, src_mel)
    tgt_emb = get_embed(model.speaker_encoder, tgt_mel)
    src_mel, len_pad = pad_seq(src_mel)
    src_mel = src_mel[None, :]

    with torch.no_grad():
        _, mel, _ = model(src_mel, src_emb, tgt_emb)
    mel = mel[0, :, :] if len_pad == 0 else mel[0, :-len_pad, :]

    with torch.no_grad():
        wav = vocoder.generate([mel])[0].data.cpu().numpy()
    sf.write(output, wav.astype(np.float32), wav2mel.sample_rate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=Path)
    parser.add_argument("vocoder_path", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("output", type=Path)
    main(**vars(parser.parse_args()))
