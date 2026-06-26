# WiFo Channel Prediction Dataset

This is the task-level dataset entry for the WiFo-style channel-prediction
benchmark. The method directory does not carry a duplicate dataset copy.

## Source

The data comes from the original WiFo channel-prediction benchmark release.
Upstream project: `https://github.com/liuboxun/WiFo`.

## Expected Data

- `data/train_val_data`: train/validation/test MATLAB files for model
  development.
- `data/test_data`: official-style test folders `D1` through `D18` used by the
  retained WiFo entrypoint.

See `../../../../docs/DATASETS.md` for the full setup instructions.

## File Format

`train_val_data` contains one folder per channel family. A representative D1
folder has:

- `X_train.mat`: key `X_train`, shape `[9000, 24, 4, 128]`.
- `X_val.mat`: key `X_test`, shape `[2000, 24, 4, 128]`.
- `X_test.mat`: key `X_val`, shape `[1000, 24, 4, 128]`.

The raw MATLAB v7.3/HDF5 layout stores the same arrays as `[128, 4, 24, N]`;
`hdf5storage` restores them to the MATLAB semantic order above.

`test_data/D*/X_test.mat` is the path used by the original WiFo evaluation
entrypoint. A representative `D17/X_test.mat` contains key `X_val` with shape
`[1000, 16, 32, 32]`.

Physical meaning is inherited from WiFo:

- first dimension after loading: sample/window index.
- temporal dimension: historical channel snapshots.
- remaining dimensions: channel tensor axes used by the WiFo masking and
  reconstruction objective.
- values are complex CSI and are converted to real/imaginary channels by the
  loader.

## Loader Contract

The method loader lives at:

```text
../../methods/wifo/wifo_prediction/DataLoader.py
```

For evaluation, it reads:

```text
data/test_data/<dataset>/X_test.mat
```

by default through the task-level path. Override with `WIFO_DATASET_ROOT` if a
different test-data root is needed.

The loader converts a complex array to a float tensor with real/imaginary
channels and yields `DataLoader` batches for the original WiFo `main.py`.

## How To Use

```bash
cd ${RAN_BENCHMARK_ROOT:-.}/tasks/channel_prediction_wifo_style/methods/wifo
python eval/run_wifo.py \
  --dataset D17 \
  --file_load_path weights/original_weights/wifo_base \
  --batch_size 128
```

The original WiFo weights are method artifacts and are linked only under
`methods/wifo/weights/original_weights`.
