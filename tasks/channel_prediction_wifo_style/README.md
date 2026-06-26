# Channel Prediction, WiFo Style

Task root for WiFo-style channel prediction benchmarks.

## Dataset

- `datasets/wifo_channel_prediction`: expected location for original WiFo
  train/validation and test data assets.

The dataset contains `D1` through `D18` channel families with MATLAB files such
as `X_train.mat`, `X_val.mat`, and `X_test.mat`.

## Methods

- `methods/wifo`: curated original WiFo entry with method-local
  `wifo_prediction/` source, benchmark config, wrappers, normalized result
  tables, and retained original weights.
