"""Minimal iTransformer-style backbone for traffic forecasting."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ITransformerConfig:
    num_regions: int
    context_len: int
    horizon: int
    d_model: int = 96
    depth: int = 4
    num_heads: int = 4
    ffn_dim: int = 192
    dropout: float = 0.1


class ITransformerModel(nn.Module):
    """Invert time/variates so attention happens over region tokens."""

    def __init__(self, config: ITransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.token_proj = nn.Linear(config.context_len, config.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.ffn_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.depth)
        self.output_head = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.horizon),
        )

    def forward(self, inputs: torch.Tensor, *, active_layers: int | None = None) -> torch.Tensor:
        if active_layers is None:
            active_layers = self.config.depth
        if active_layers < 1 or active_layers > self.config.depth:
            raise ValueError(f"active_layers must be in [1, {self.config.depth}], got {active_layers}")
        tokens = inputs.transpose(1, 2).contiguous()
        tokens = self.token_proj(tokens)
        encoded = tokens
        for layer in list(self.encoder.layers)[:active_layers]:
            encoded = layer(encoded)
        forecast = self.output_head(encoded)
        return forecast.transpose(1, 2).contiguous()
