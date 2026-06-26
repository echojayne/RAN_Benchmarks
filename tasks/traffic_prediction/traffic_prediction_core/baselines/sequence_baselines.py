"""Trainable sequence baselines for traffic forecasting."""

from __future__ import annotations

import torch
from torch import nn


class LSTMForecaster(nn.Module):
    def __init__(
        self,
        *,
        num_regions: int,
        hidden_size: int,
        num_layers: int,
        horizon: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.num_regions = num_regions
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.encoder = nn.LSTM(
            input_size=num_regions,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, horizon * num_regions)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.encoder(inputs)
        last_hidden = hidden[-1]
        forecast = self.head(last_hidden)
        return forecast.view(inputs.shape[0], self.horizon, self.num_regions)


class _CausalConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1)
        self.net = nn.Sequential(
            nn.ConstantPad1d((padding, 0), 0.0),
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.ConstantPad1d((padding, 0), 0.0),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, dilation=dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.downsample is None else self.downsample(x)
        return self.net(x) + residual


class TCNForecaster(nn.Module):
    def __init__(
        self,
        *,
        num_regions: int,
        hidden_channels: int,
        num_layers: int,
        kernel_size: int,
        horizon: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.num_regions = num_regions
        blocks = []
        in_channels = num_regions
        for index in range(num_layers):
            dilation = 2**index
            blocks.append(
                _CausalConvBlock(
                    in_channels=in_channels,
                    out_channels=hidden_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
            in_channels = hidden_channels
        self.encoder = nn.Sequential(*blocks)
        self.head = nn.Linear(hidden_channels, horizon * num_regions)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs.transpose(1, 2)
        x = self.encoder(x)
        last_hidden = x[:, :, -1]
        forecast = self.head(last_hidden)
        return forecast.view(inputs.shape[0], self.horizon, self.num_regions)

