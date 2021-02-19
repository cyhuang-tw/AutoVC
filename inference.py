import argparse
import json
import os
import random
from math import ceil
from typing import List, Tuple

import matplotlib.pyplot as plt
import soundfile as sf
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.functional import pad

from modules.Audioprocessor import AudioProcessor


def chunks(lst: List, n: int) -> List[List]:
    for i in range(0, len(lst), n):
        yield lst[i : (i + n)]


def pad_seq(x: Tensor, base: int = 32) -> Tuple[Tensor, int]:
    len_out = int(base * ceil(float(len(x)) / base))
    len_pad = len_out - len(x)
    assert len_pad >= 0
    return pad(x, (0, 0, 0, len_pad), "constant", 0), len_pad


def get_embed(encoder: nn.Module, mel: Tensor) -> Tensor:
    embs = encoder(mel[None, :])
    emb = embs.mean(dim=0, keepdim=True)
    return emb


def main(
    data_path: str, output_path: str, model_path: str, vocoder_path: str, n_samples: int
):
    random.seed(531)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(output_path, exist_ok=True)
    model = torch.jit.load(model_path).to(device)
    vocoder = torch.jit.load(vocoder_path).to(device)

    speakers = sorted(os.listdir(data_path))
    mels = []
    infos = []

    for i in range(n_samples):
        # sample speakers
        spks = random.sample(speakers, k=2)
        src_spk = os.path.join(data_path, spks[0])
        tgt_spk = os.path.join(data_path, spks[1])

        # sample utterances
        src_uttr = os.path.join(src_spk, random.choice(sorted(os.listdir(src_spk))))
        tgt_uttr = os.path.join(tgt_spk, random.choice(sorted(os.listdir(tgt_spk))))

        # file to mel
        src_mel = AudioProcessor.file2spectrogram(src_uttr)
        tgt_mel = AudioProcessor.file2spectrogram(tgt_uttr)

        # np array to torch tensor
        src_mel = torch.from_numpy(src_mel).to(device)
        tgt_mel = torch.from_numpy(tgt_mel).to(device)
        src_emb = get_embed(model.speaker_encoder, src_mel)
        tgt_emb = get_embed(model.speaker_encoder, tgt_mel)
        src_mel, len_pad = pad_seq(src_mel)
        src_mel = src_mel[None, :]

        # conversion
        with torch.no_grad():
            _, mel, _ = model(src_mel, src_emb, tgt_emb)
        mel = mel[0, :, :] if len_pad == 0 else mel[0, :-len_pad, :]

        mels.append(mel)
        infos.append({"src": src_uttr, "tgt": tgt_uttr})
        plt.imshow(mel.squeeze(0).cpu().numpy())
        plt.savefig(os.path.join(output_path, f"{i:03d}.png"))
        plt.clf()

    mel_chunks = list(chunks(mels, 50))
    wavs = []

    for mel_chunk in mel_chunks:
        with torch.no_grad():
            wavs.extend(vocoder.generate(mel_chunk))

    for i, (wav, info) in enumerate(zip(wavs, infos)):
        wav_path = os.path.join(output_path, f"{i:03d}.wav")
        sf.write(wav_path, wav.data.cpu().numpy(), AudioProcessor.sample_rate)
        cfg_path = os.path.join(output_path, f"{i:03d}.json")
        json.dump(info, open(cfg_path, "w"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=str)
    parser.add_argument("output_path", type=str)
    parser.add_argument("model_path", type=str)
    parser.add_argument("vocoder_path", type=str)
    parser.add_argument("--n_samples", type=int, default=100)
    main(**vars(parser.parse_args()))
