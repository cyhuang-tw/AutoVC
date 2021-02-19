import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import yaml
from tensorboardX import SummaryWriter
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import trange

from data.dataset import SpeakerDataset
from modules.models import AutoVC


def main(
    config: str,
    data_path: str,
    save_path: str,
    n_steps: int,
    save_steps: int,
    log_steps: int,
    batch_size: int,
    seg_len: int,
):
    torch.backends.cudnn.benchmark = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(save_path, exist_ok=True)
    config = yaml.load(open(config, "r"), Loader=yaml.FullLoader)
    writer = SummaryWriter(save_path)

    model = AutoVC(config)
    model = torch.jit.script(model).to(device)
    train_set = SpeakerDataset(data_path, seg_len=seg_len)
    data_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        worker_init_fn=lambda x: np.random.seed((torch.initial_seed()) % (2 ** 32)),
    )

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4
    )
    MSELoss = nn.MSELoss()
    L1Loss = nn.L1Loss()
    lambda_cnt = 1.0

    pbar = trange(n_steps)
    for step in pbar:
        try:
            mels, embs = next(data_iter)
        except:
            data_iter = iter(data_loader)
            mels, embs = next(data_iter)
        mels = mels.to(device)
        embs = embs.to(device)
        rec_org, rec_pst, codes = model(mels, embs)

        fb_codes = torch.cat(model.content_encoder(rec_pst, embs), dim=-1)

        # reconstruction loss
        org_loss = MSELoss(rec_org, mels)
        pst_loss = MSELoss(rec_pst, mels)
        # content consistency
        cnt_loss = L1Loss(fb_codes, codes)

        loss = org_loss + pst_loss + lambda_cnt * cnt_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % save_steps == 0:
            model.save(os.path.join(save_path, f"model-{step + 1}.pt"))
            torch.save(
                optimizer.state_dict(),
                os.path.join(save_path, f"optimizer-{step + 1}.pt"),
            )

        if (step + 1) % log_steps == 0:
            writer.add_scalar("loss/org_rec", org_loss.item(), step + 1)
            writer.add_scalar("loss/pst_rec", pst_loss.item(), step + 1)
            writer.add_scalar("loss/content", cnt_loss.item(), step + 1)
        pbar.set_postfix(
            {
                "org_rec": org_loss.item(),
                "pst_rec": pst_loss.item(),
                "cnt": cnt_loss.item(),
            }
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("data_path", type=str)
    parser.add_argument("save_path", type=str)
    parser.add_argument("--n_steps", type=int, default=int(1e7))
    parser.add_argument("--save_steps", type=int, default=10000)
    parser.add_argument("--log_steps", type=int, default=250)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seg_len", type=int, default=128)
    main(**vars(parser.parse_args()))
