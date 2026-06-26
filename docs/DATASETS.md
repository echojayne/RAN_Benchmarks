# Dataset Setup

This repository keeps dataset loaders and preparation scripts in Git, but the
datasets themselves are external assets. Place data under the paths below, or
set `RAN_BENCHMARK_ASSET_ROOT` and use symlinks from these task directories.

```bash
export RAN_BENCHMARK_ROOT=/path/to/RAN_Benchmarks
export RAN_BENCHMARK_ASSET_ROOT=/path/to/ran_benchmark_assets
```

## OFDM Channel Estimation

Task path:

```text
tasks/channel_estimation/datasets/ofdm_channel_estimation_common/
```

Source:

- AdaFortiTran-compatible OFDM channel-estimation MATLAB samples generated with
  the public OFDM channel generator used by AdaFortiTran.
- Generator reference recorded in configs:
  `https://github.com/BerkIGuler/OFDMChannelGenerator`.

Expected structure:

```text
tasks/channel_estimation/datasets/ofdm_channel_estimation_common/data/
|-- train_data/
|-- val_data/
`-- test_data/
    |-- SNR_test_set/
    |-- DS_test_set/
    `-- MDS_test_set/
```

Each sample is a MATLAB `.mat` file containing complex `H` shaped
`[120, 14, 2]` and optional scalar `var_hat`. The filename encodes SNR, delay
spread, Doppler, pilot spacing, and TDL profile.

## WiFo Channel Prediction

Task path:

```text
tasks/channel_prediction_wifo_style/datasets/wifo_channel_prediction/
```

Source:

- WiFo channel-prediction benchmark data from the original WiFo release.
- Upstream project: `https://github.com/liuboxun/WiFo`.

Expected structure:

```text
tasks/channel_prediction_wifo_style/datasets/wifo_channel_prediction/data/
|-- train_val_data/
|   |-- D1/
|   |   |-- X_train.mat
|   |   |-- X_val.mat
|   |   `-- X_test.mat
|   `-- ...
`-- test_data/
    |-- D1/
    |   `-- X_test.mat
    `-- ...
```

WiFo test/evaluation commands read
`data/test_data/<dataset>/X_test.mat`, where `<dataset>` is typically `D1`
through `D18`.

## Milan Telecom Traffic

Task path:

```text
tasks/traffic_prediction/datasets/milan_telecom_traffic/
```

Source:

- Telecom Italia Milan open cellular traffic records.
- Download DOIs recorded by the preparation config:
  `https://doi.org/10.7910/DVN/EGZHFV` and
  `https://doi.org/10.7910/DVN/QJWLFU`.

Raw structure:

```text
tasks/traffic_prediction/datasets/milan_telecom_traffic/data/raw_data/
|-- sms-call-internet-mi-YYYY-MM-DD.txt
`-- ...
```

Prepared benchmark structure:

```text
tasks/traffic_prediction/datasets/milan_telecom_traffic/data/prepared_data/
|-- manifest.json
|-- selected_regions.csv
|-- train.npz
|-- val.npz
`-- test.npz
```

The main benchmark uses hourly `internet_traffic` values with input shape
`[24, 64]` and target shape `[6, 64]`.

## BeamFormer Home/Office 28 GHz

Task path:

```text
tasks/beam_management/datasets/homeoffice_28g_beamformer/
```

Source:

- Public BeamFormer ray-tracing channel dataset for the
  `homeoffice-communication` environment at 28 GHz.
- Upstream project: `https://github.com/Shunqiang-Feng/BeamFormer.git`.
- Dataset reference recorded by upstream: IEEE DataPort `10.21227/g1zj-z323`
  and S3 archive `homeoffice-communication-28G-raw-data.zip`.

Expected structure:

```text
tasks/beam_management/datasets/homeoffice_28g_beamformer/data/
|-- raw/
|-- generated_csi/
|-- csi-dataset/
|   `-- homeoffice-communication-28G-csi/
|       |-- t16x16_r2x1_train/
|       |-- t16x16_r2x1_val/
|       |-- t16x16_r2x1_test_small/
|       `-- t16x16_r2x1_test/
`-- mini_demo/
```

The retained BeamFormer README examples use `t16x16_r2x1_test_small` for quick
checks and `t16x16_r2x1_test` for the full 90,856-sample test split.
