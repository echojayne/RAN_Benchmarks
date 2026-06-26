"""Baseline exports for traffic forecasting."""

from traffic_prediction_core.baselines.naive import (
    last_value_forecast,
    naive_last_value,
    seasonal_naive,
    seasonal_naive_forecast,
)
from traffic_prediction_core.baselines.sequence_baselines import LSTMForecaster, TCNForecaster

LSTMForecastModel = LSTMForecaster
TCNForecastModel = TCNForecaster

__all__ = [
    "LSTMForecaster",
    "TCNForecaster",
    "LSTMForecastModel",
    "TCNForecastModel",
    "last_value_forecast",
    "naive_last_value",
    "seasonal_naive",
    "seasonal_naive_forecast",
]
