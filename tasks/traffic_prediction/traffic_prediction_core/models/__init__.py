"""Original-model exports for traffic forecasting."""

from traffic_prediction_core.models.itransformer import ITransformerConfig, ITransformerModel

ITransformerForecaster = ITransformerModel

__all__ = [
    "ITransformerConfig",
    "ITransformerModel",
    "ITransformerForecaster",
]
