from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor


class LinearLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = True,
        w_init_gain: str = "linear",
    ):
        super(LinearLayer, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight, gain=torch.nn.init.calculate_gain(w_init_gain)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear_layer(x)
        return x


class ConvLayer(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: Optional[int] = None,
        dilation: int = 1,
        bias: bool = True,
        w_init_gain: str = "linear",
    ):
        super(ConvLayer, self).__init__()
        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        return x


class SpeakerEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        dim_input: int,
        dim_cell: int,
        dim_emb: int,
    ):
        super(SpeakerEncoder, self).__init__()
        self.lstm = nn.LSTM(
            input_size=dim_input,
            hidden_size=dim_cell,
            num_layers=num_layers,
            batch_first=True,
        )
        self.embedding = nn.Linear(dim_cell, dim_emb)

    def forward(self, x: Tensor) -> Tensor:
        lstm_out, _ = self.lstm(x)
        embeds = self.embedding(lstm_out[:, -1, :])
        norm = embeds.norm(p=2, dim=-1, keepdim=True)
        embeds = embeds.div(norm)
        return embeds


class ContentEncoder(nn.Module):
    def __init__(
        self,
        dim_input: int,
        dim_hidden: int,
        dim_neck: int,
        dim_emb: int,
        lstm_stride: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        n_conv_layers: int,
        n_lstm_layers: int,
    ):
        super(ContentEncoder, self).__init__()
        self.dim_neck = dim_neck
        self.stride = lstm_stride

        self.conv_layers = nn.ModuleList(
            [
                nn.Sequential(
                    ConvLayer(
                        dim_input + dim_emb if i == 0 else dim_hidden,
                        dim_hidden,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=kernel_size // 2,
                        dilation=dilation,
                        w_init_gain="relu",
                    ),
                    nn.BatchNorm1d(dim_hidden),
                )
                for i in range(n_conv_layers)
            ]
        )

        self.lstm = nn.LSTM(
            dim_hidden, dim_neck, n_lstm_layers, batch_first=True, bidirectional=True
        )

        self.act = nn.ReLU()

    def forward(self, x: Tensor, emb: Tensor) -> List[Tensor]:
        x = x.squeeze(1).transpose(2, 1)
        emb = emb.unsqueeze(-1).expand(-1, -1, x.size(-1))
        x = torch.cat((x, emb), dim=1)

        for conv in self.conv_layers:
            x = self.act(conv(x))
        x = x.transpose(1, 2)

        outputs, _ = self.lstm(x)
        out_forward = outputs[:, :, : self.dim_neck]
        out_backward = outputs[:, :, self.dim_neck :]

        codes = [
            torch.cat(
                (out_forward[:, i + self.stride - 1, :], out_backward[:, i, :]), dim=-1
            )
            for i in range(0, outputs.size(1), self.stride)
        ]

        return codes


class Decoder(nn.Module):
    def __init__(
        self,
        dim_input: int,
        dim_hidden: int,
        dim_out: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        n_conv_layers: int,
        n_lstm_layers: int,
    ):
        super(Decoder, self).__init__()

        self.conv_layers = nn.ModuleList(
            [
                nn.Sequential(
                    ConvLayer(
                        dim_input,
                        dim_input,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=kernel_size // 2,
                        dilation=dilation,
                        w_init_gain="relu",
                    ),
                    nn.BatchNorm1d(dim_input),
                )
                for _ in range(n_conv_layers)
            ]
        )

        self.lstm2 = nn.LSTM(dim_input, dim_hidden, n_lstm_layers, batch_first=True)

        self.out_layer = LinearLayer(dim_hidden, dim_out)

        self.act = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        for conv in self.conv_layers:
            x = self.act(conv(x))
        x = x.transpose(1, 2)

        outputs, _ = self.lstm2(x)
        decoder_output = self.out_layer(outputs)

        return decoder_output


class PreNet(nn.Module):
    def __init__(self, dim_neck: int, dim_emb: int, dim_out: int, n_layers: int):
        super(PreNet, self).__init__()
        self.lstm = nn.LSTM(dim_neck * 2 + dim_emb, dim_out, n_layers, batch_first=True)

    def forward(self, x: Tensor) -> Tensor:
        x, _ = self.lstm(x)
        x = x.transpose(1, 2)
        return x


class PostNet(nn.Module):
    def __init__(
        self,
        dim_input: int,
        dim_hidden: int,
        dim_out: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        n_layers: int,
    ):
        super(PostNet, self).__init__()
        self.conv_layers = nn.ModuleList(
            [
                nn.Sequential(
                    ConvLayer(
                        dim_input if i == 0 else dim_hidden,
                        dim_hidden if i != n_layers - 1 else dim_out,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=kernel_size // 2,
                        dilation=dilation,
                        w_init_gain="tanh" if i != n_layers - 1 else "linear",
                    ),
                    nn.BatchNorm1d(dim_hidden if i != n_layers - 1 else dim_out),
                )
                for i in range(n_layers)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        for idx, conv in enumerate(self.conv_layers):
            x = conv(x)
            if idx != len(self.conv_layers) - 1:
                x = x.tanh()
        return x


class AutoVC(nn.Module):
    def __init__(self, config: Dict):
        super(AutoVC, self).__init__()
        self.speaker_encoder = torch.jit.load(config["SpeakerEncoder"]["path"])
        self.content_encoder = ContentEncoder(**config["ContentEncoder"])
        self.decoder = Decoder(**config["Decoder"])
        self.prenet = PreNet(**config["PreNet"])
        self.postnet = PostNet(**config["PostNet"])

        for p in self.speaker_encoder.parameters():
            p.requires_grad_(False)

    def forward(
        self, src_mels: Tensor, src_embs: Tensor, tgt_embs: Optional[Tensor] = None
    ):
        if tgt_embs is None:
            tgt_embs = src_embs

        src_codes = self.content_encoder(src_mels, src_embs)

        tmp = [
            code.unsqueeze(1).expand(-1, int(src_mels.size(1) / len(src_codes)), -1)
            for code in src_codes
        ]

        code_exp = torch.cat(tmp, dim=1)

        encoder_outputs = torch.cat(
            (code_exp, tgt_embs.unsqueeze(1).expand(-1, src_mels.size(1), -1)), dim=-1
        )

        prenet_outputs = self.prenet(encoder_outputs)

        mel_outputs = self.decoder(prenet_outputs)

        postnet_residual = self.postnet(mel_outputs.transpose(2, 1))
        mel_outputs_postnet = mel_outputs + postnet_residual.transpose(2, 1)

        return mel_outputs, mel_outputs_postnet, torch.cat(src_codes, dim=-1)
