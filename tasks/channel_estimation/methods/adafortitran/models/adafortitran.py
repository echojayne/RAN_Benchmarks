"""AdaFortiTran/FortiTran backbones aligned with the official public repository."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class AdaFortiTranConfig:
    num_subcarriers: int = 120
    num_symbols: int = 14
    input_channels: int = 4
    output_channels: int = 2
    pilot_vector_length: int = 80
    d_enc: int = 32
    encoder_layers: int = 6
    num_heads: int = 4
    ffn_dim: int = 64
    dropout: float = 0.1
    patch_subcarriers: int = 3
    patch_symbols: int = 2
    shallow_channels: int = 8
    hidden_channels: int = 32
    activation: str = "gelu"
    max_seq_len: int = 512
    pos_encoding_type: str = "learnable"
    use_channel_adaptation: bool = False
    channel_adaptivity_hidden_sizes: tuple[int, int, int] | None = None
    adaptive_token_length: int | None = None

    @property
    def num_tokens(self) -> int:
        return self.num_subcarriers * self.num_symbols

    @property
    def patch_dim(self) -> int:
        return self.patch_subcarriers * self.patch_symbols

    @property
    def patch_count(self) -> int:
        return (self.num_subcarriers // self.patch_subcarriers) * (self.num_symbols // self.patch_symbols)


class ConvStack(nn.Module):
    """Backward-compatible conv stack kept for legacy checkpoints."""

    def __init__(self, in_channels: int, mid_channels: int, out_channels: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout2d(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvEnhancer(nn.Module):
    """Official FortiTran convolutional enhancement block."""

    def __init__(self, shallow_channels: int = 8, hidden_channels: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, shallow_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(shallow_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, shallow_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(shallow_channels, 1, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LearnedLinearUpsampler(nn.Module):
    """Map pilot-position LS vectors into a dense full-grid estimate."""

    def __init__(self, pilot_vector_length: int, full_grid_size: int) -> None:
        super().__init__()
        self.linear = nn.Linear(pilot_vector_length, full_grid_size)

    def forward(self, pilot_vector: torch.Tensor) -> torch.Tensor:
        batch_size = pilot_vector.shape[0]
        return self.linear(pilot_vector.reshape(batch_size, -1))


class ChannelAdapter(nn.Module):
    """Encode SNR / delay-spread / Doppler priors into adaptive transformer tokens."""

    def __init__(self, hidden_sizes: tuple[int, int, int]) -> None:
        super().__init__()
        self.snr_encoder = self._create_mlp(hidden_sizes)
        self.ds_encoder = self._create_mlp(hidden_sizes)
        self.dop_encoder = self._create_mlp(hidden_sizes)

    @staticmethod
    def _create_mlp(hidden_sizes: tuple[int, int, int]) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(1, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], hidden_sizes[2]),
        )

    def forward(
        self,
        snr: torch.Tensor,
        delay_spread: torch.Tensor,
        doppler_shift: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = snr.shape[0]
        snr_emb = self.snr_encoder(snr).reshape(batch_size, -1, 2)
        ds_emb = self.ds_encoder(delay_spread).reshape(batch_size, -1, 2)
        dop_emb = self.dop_encoder(doppler_shift).reshape(batch_size, -1, 2)
        return torch.cat((snr_emb, ds_emb, dop_emb), dim=2)


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, max_len: int, d_model: int) -> None:
        super().__init__()
        self.position_embeddings = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.position_embeddings, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.position_embeddings[:, : x.size(1), :]


class TransformerEncoderForChannels(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        model_dim: int,
        num_heads: int,
        activation: str,
        dropout: float,
        num_layers: int,
        max_len: int,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, model_dim)
        self.position = LearnablePositionalEncoding(max_len, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=2 * model_dim,
            activation=activation,
            dropout=dropout,
            batch_first=True,
        )
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.output_proj = nn.Linear(model_dim, output_dim)

    def forward(self, x: torch.Tensor, *, active_layers: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if active_layers is None:
            active_layers = len(self.layers)
        hidden = self.position(self.input_proj(x))
        for layer in self.layers[:active_layers]:
            hidden = layer(hidden)
        return self.output_proj(hidden), hidden


class PatchEmbedding(nn.Module):
    def __init__(self, patch_size: tuple[int, int]) -> None:
        super().__init__()
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = self.unfold(x.unsqueeze(1))
        return patches.permute(0, 2, 1)


class InversePatchEmbedding(nn.Module):
    def __init__(self, output_size: tuple[int, int], patch_size: tuple[int, int]) -> None:
        super().__init__()
        self.fold = nn.Fold(output_size=output_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fold(x.permute(0, 2, 1)).squeeze(1)


class AdaFortiTranBackbone(nn.Module):
    """FortiTran / AdaFortiTran backbone aligned with the official public repository."""

    def __init__(self, config: AdaFortiTranConfig) -> None:
        super().__init__()
        if config.d_enc % config.num_heads != 0:
            raise ValueError("d_enc must be divisible by num_heads")
        if config.num_subcarriers % config.patch_subcarriers != 0:
            raise ValueError("num_subcarriers must be divisible by patch_subcarriers")
        if config.num_symbols % config.patch_symbols != 0:
            raise ValueError("num_symbols must be divisible by patch_symbols")
        if config.use_channel_adaptation:
            if config.channel_adaptivity_hidden_sizes is None:
                raise ValueError("channel_adaptivity_hidden_sizes is required when channel adaptation is enabled")
            if config.adaptive_token_length is None:
                raise ValueError("adaptive_token_length is required when channel adaptation is enabled")
            if config.channel_adaptivity_hidden_sizes[-1] != 2 * config.patch_count:
                raise ValueError(
                    "channel_adaptivity_hidden_sizes[-1] must equal 2 * patch_count to match the official token layout"
                )

        self.config = config
        self.use_channel_adaptation = bool(config.use_channel_adaptation)
        self.upsampler = LearnedLinearUpsampler(config.pilot_vector_length, config.num_tokens)
        self.initial_enhancer = ConvEnhancer(config.shallow_channels, config.hidden_channels)
        self.patch_embedder = PatchEmbedding((config.patch_subcarriers, config.patch_symbols))
        transformer_input_dim = config.patch_dim + (int(config.adaptive_token_length or 0) if self.use_channel_adaptation else 0)
        if self.use_channel_adaptation:
            self.channel_adapter = ChannelAdapter(tuple(int(v) for v in config.channel_adaptivity_hidden_sizes or ()))
        self.transformer_encoder = TransformerEncoderForChannels(
            input_dim=transformer_input_dim,
            output_dim=config.patch_dim,
            model_dim=config.d_enc,
            num_heads=config.num_heads,
            activation=config.activation,
            dropout=config.dropout,
            num_layers=config.encoder_layers,
            max_len=config.max_seq_len,
        )
        self.patch_reconstructor = InversePatchEmbedding(
            (config.num_subcarriers, config.num_symbols),
            (config.patch_subcarriers, config.patch_symbols),
        )
        self.final_refiner = ConvEnhancer(config.shallow_channels, config.hidden_channels)
        self.max_representation_dim = 2 * config.d_enc

    def _resolve_channel_conditions(
        self,
        conditioning: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        if not self.use_channel_adaptation:
            return None
        if conditioning is None:
            raise ValueError("conditioning=(snr, delay_spread, doppler) is required for AdaFortiTran")
        snr, delay_spread, doppler_shift = conditioning
        snr = snr.to(device=device, dtype=torch.float32).reshape(batch_size, 1)
        delay_spread = delay_spread.to(device=device, dtype=torch.float32).reshape(batch_size, 1)
        doppler_shift = doppler_shift.to(device=device, dtype=torch.float32).reshape(batch_size, 1)
        return snr, delay_spread, doppler_shift

    def _forward_real(
        self,
        x: torch.Tensor,
        *,
        conditioning: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
        active_encoder_layers: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        upsampled = self.upsampler(x).view(
            batch_size,
            1,
            self.config.num_subcarriers,
            self.config.num_symbols,
        )
        conv_enhanced = self.initial_enhancer(upsampled).squeeze(1)
        patch_embeddings = self.patch_embedder(conv_enhanced)
        if self.use_channel_adaptation:
            channel_tokens = self.channel_adapter(*self._resolve_channel_conditions(conditioning, batch_size, x.device))
            transformer_input = torch.cat((patch_embeddings, channel_tokens), dim=2)
        else:
            transformer_input = patch_embeddings
        transformer_output, hidden = self.transformer_encoder(
            transformer_input,
            active_layers=active_encoder_layers,
        )
        reconstructed = self.patch_reconstructor(transformer_output)
        residual_combined = conv_enhanced + reconstructed
        refined_output = self.final_refiner(residual_combined.unsqueeze(1)).squeeze(1)
        return refined_output, hidden

    def forward(
        self,
        inputs: torch.Tensor | None = None,
        *,
        sparse_ls_grid: torch.Tensor | None = None,
        pilot_vector: torch.Tensor | None = None,
        ls_coarse_grid: torch.Tensor | None = None,
        conditioning: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
        active_encoder_layers: int | None = None,
        return_representation: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        del inputs, sparse_ls_grid, ls_coarse_grid
        if pilot_vector is None:
            raise ValueError("pilot_vector is required")
        if active_encoder_layers is None:
            active_encoder_layers = self.config.encoder_layers
        if active_encoder_layers < 1 or active_encoder_layers > self.config.encoder_layers:
            raise ValueError(
                f"active_encoder_layers must be in [1, {self.config.encoder_layers}], got {active_encoder_layers}"
            )

        if pilot_vector.ndim == 4:
            real_input = pilot_vector[:, 0].reshape(pilot_vector.shape[0], -1)
            imag_input = pilot_vector[:, 1].reshape(pilot_vector.shape[0], -1)
        elif pilot_vector.ndim == 3:
            real_input = pilot_vector[:, 0, :].reshape(pilot_vector.shape[0], -1)
            imag_input = pilot_vector[:, 1, :].reshape(pilot_vector.shape[0], -1)
        else:
            raise ValueError(f"unexpected pilot_vector shape: {pilot_vector.shape}")

        real_output, real_hidden = self._forward_real(
            real_input,
            conditioning=conditioning,
            active_encoder_layers=active_encoder_layers,
        )
        imag_output, imag_hidden = self._forward_real(
            imag_input,
            conditioning=conditioning,
            active_encoder_layers=active_encoder_layers,
        )
        prediction = torch.stack((real_output, imag_output), dim=1)
        if not return_representation:
            return prediction
        representation = torch.cat((real_hidden.mean(dim=1), imag_hidden.mean(dim=1)), dim=1)
        return prediction, representation


AdaFortiTranModel = AdaFortiTranBackbone


class LegacyConvRefiner(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LegacyAdaFortiTranStaticCompat(nn.Module):
    """Compatibility path for the retained pre-refactor local static checkpoint."""

    def __init__(
        self,
        *,
        num_subcarriers: int = 120,
        num_symbols: int = 14,
        pilot_vector_length: int = 80,
        d_enc: int = 32,
        encoder_layers: int = 6,
        num_heads: int = 4,
        ffn_dim: int = 64,
        dropout: float = 0.1,
        patch_subcarriers: int = 3,
        patch_symbols: int = 2,
        shallow_channels: int = 8,
        hidden_channels: int = 32,
        **_: object,
    ) -> None:
        super().__init__()
        patch_dim = int(patch_subcarriers * patch_symbols)
        patch_count = int((num_subcarriers // patch_subcarriers) * (num_symbols // patch_symbols))
        self.num_subcarriers = int(num_subcarriers)
        self.num_symbols = int(num_symbols)
        self.upsampler = LearnedLinearUpsampler(2 * int(pilot_vector_length), 2 * num_subcarriers * num_symbols)
        self.feature_enhancer = nn.Sequential(
            nn.Conv2d(2, shallow_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(shallow_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, shallow_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.shallow_map = nn.Conv2d(shallow_channels, shallow_channels, kernel_size=3, padding=1)
        self.patch_to_scalar = nn.Conv2d(shallow_channels, 1, kernel_size=1)
        self.unfold = nn.Unfold(kernel_size=(patch_subcarriers, patch_symbols), stride=(patch_subcarriers, patch_symbols))
        self.patch_embed = nn.Linear(patch_dim, d_enc)
        self.position = nn.Parameter(torch.zeros(1, patch_count, d_enc))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_enc,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_layers)
        self.patch_unembed = nn.Linear(d_enc, patch_dim)
        self.fold = nn.Fold(
            output_size=(num_subcarriers, num_symbols),
            kernel_size=(patch_subcarriers, patch_symbols),
            stride=(patch_subcarriers, patch_symbols),
        )
        self.patch_to_shallow = nn.Conv2d(1, shallow_channels, kernel_size=1)
        self.reconstructor = LegacyConvRefiner(shallow_channels, hidden_channels, shallow_channels)
        self.output_head = nn.Conv2d(shallow_channels, 2, kernel_size=3, padding=1)

    def forward(
        self,
        inputs: torch.Tensor | None = None,
        *,
        sparse_ls_grid: torch.Tensor | None = None,
        pilot_vector: torch.Tensor | None = None,
        ls_coarse_grid: torch.Tensor | None = None,
        conditioning: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
        active_encoder_layers: int | None = None,
        return_representation: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        del inputs, sparse_ls_grid, ls_coarse_grid, conditioning, active_encoder_layers
        if pilot_vector is None:
            raise ValueError("pilot_vector is required")
        if pilot_vector.ndim != 3 or pilot_vector.shape[1] != 2:
            raise ValueError(f"expected pilot_vector shape [B, 2, L], got {tuple(pilot_vector.shape)}")

        batch_size = int(pilot_vector.shape[0])
        flat_input = pilot_vector.reshape(batch_size, -1)
        dense_grid = self.upsampler(flat_input).view(batch_size, 2, self.num_subcarriers, self.num_symbols)
        shallow_features = self.feature_enhancer(dense_grid)
        shallow_residual = self.shallow_map(shallow_features)
        scalar_grid = self.patch_to_scalar(shallow_residual)
        patch_tokens = self.unfold(scalar_grid).transpose(1, 2).contiguous()
        patch_tokens = self.patch_embed(patch_tokens) + self.position[:, : patch_tokens.shape[1], :]
        encoded = self.encoder(patch_tokens)
        patch_pixels = self.patch_unembed(encoded).transpose(1, 2).contiguous()
        reconstructed_scalar = self.fold(patch_pixels)
        reconstructed_features = self.patch_to_shallow(reconstructed_scalar)
        refined = self.reconstructor(shallow_residual + reconstructed_features)
        output = self.output_head(shallow_residual + refined)
        if not return_representation:
            return output
        return output, encoded.mean(dim=1)
