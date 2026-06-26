# Traffic Prediction

Task root for Milan telecom traffic forecasting benchmarks. The benchmark uses
hourly Milan Internet traffic windows to predict future traffic for selected
grid cells.

## Dataset

- `datasets/milan_telecom_traffic`: expected location for raw Telecom Italia
  Milan files and prepared benchmark arrays.

Prepared sample shape:

- input: `[24, 64]`, 24 hourly context steps over 64 selected cells
- target: `[6, 64]`, 6-hour forecast horizon over the same cells

Current split sizes are 860 train, 139 validation, and 139 test windows.

## Shared Implementation

Reusable Milan data loaders, baseline code, iTransformer model code, training
loops, evaluation utilities, and ML-TP paper-aligned helpers live once at
`traffic_prediction_core/`.

## Original Methods

- `methods/itransformer`: iTransformer config, train/eval
  wrappers, retained static checkpoint, and a separate citywide artifact.
- `methods/lstm`: LSTM baseline config, train/eval wrappers, and
  retained checkpoint.
- `methods/tcn`: TCN baseline config, train/eval wrappers, and
  retained checkpoint.
- `methods/naive`: naive baseline config, materialization/eval
  wrappers, and retained eval run.
- `methods/seasonal_naive`: seasonal naive baseline.
- `methods/weekly_seasonal_naive`: weekly seasonal naive baseline.
- `methods/mltp`: paper-aligned ML-TP scripts and retained result artifacts.

Method directories do not contain duplicate `traffic_forecasting/` source
trees. Their wrappers import `traffic_prediction_core` from the task root.

## Protocols

Two traffic protocols are retained and must not be merged in one result table:

- Main benchmark: 64 selected cells, 24-hour context, 6-hour horizon. This is
  the protocol used by `configs/*_benchmark.yaml` and the shared
  `traffic_prediction_core` train/eval wrappers.
- Citywide iTransformer artifact: 9,943 cells after filtering, 3-hour context,
  1-hour horizon. This is retained under
  `methods/itransformer/runs/well_done_citywide_noholiday_r2filtered` with
  copied code/config/docs in `methods/itransformer/citywide_compat/`.

### Main Benchmark Results

Retained checkpoint/test-set evaluation on the 64-cell `24 -> 6` benchmark:

| Method | Protocol | Test MAE | Test RMSE | Test MSE |
|---|---|---:|---:|---:|
| iTransformer static | 64 cells, 24 -> 6 | 545.3032 | 862.6703 | 744200.0 |
| TCN | 64 cells, 24 -> 6 | 928.7461 | 1371.4343 | 1880832.0 |
| LSTM | 64 cells, 24 -> 6 | 1150.5366 | 1809.9117 | 3275780.25 |
| seasonal naive, 24h | 64 cells, 24 -> 6 | 790.8719 | 1616.4517 | 2612916.0 |
| naive last value | 64 cells, 24 -> 6 | 1743.2683 | 2818.1703 | 7942084.0 |

`weekly_seasonal_naive` is not a valid weekly seasonal result under this
protocol: the model asks for a 168-hour seasonal period but the prepared input
contains only 24 context hours, so the implementation falls back to last-value
forecasting. Its retained metrics are therefore identical to naive and should
be labeled unavailable until a 168-hour context dataset is prepared and
evaluated.

### Citywide Artifact

The citywide iTransformer artifact is a separate retained result, not the main
64-cell benchmark:

| Artifact | Cells | Context -> Horizon | Mean MAE | Mean RMSE | Mean R2 | Mean NMAE | Mean NRMSE |
|---|---:|---|---:|---:|---:|---:|---:|
| well_done_citywide_noholiday_r2filtered | 9943 | 3 -> 1 | 25.7390 | 35.8265 | 0.8775 | 0.0342 | 0.0472 |

## Asset Policy

Small retained traffic checkpoints are committed under each method's `weights/`
directory. Raw data, prepared arrays, and full historical runs are external
assets; see `../../docs/DATASETS.md` and `../../docs/ASSETS.md`.
