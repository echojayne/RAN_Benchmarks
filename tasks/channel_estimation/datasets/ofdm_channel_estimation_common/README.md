# OFDM Channel Estimation Common Dataset

This is the shared task-level dataset for the channel-estimation methods,
including AdaFortiTran and A-MMSE. The repository keeps the loader/config code;
the `.mat` data itself is an external asset.

## Source

The files are AdaFortiTran-compatible OFDM channel-estimation MATLAB samples.
The generator reference is
`https://github.com/BerkIGuler/OFDMChannelGenerator`. In a local asset mirror,
place the prepared data under
`${RAN_BENCHMARK_ASSET_ROOT}/benchmarks/channel_estimation/common`.

## Expected Data

- `data/train_data`: 100000 `.mat` training samples.
- `data/val_data`: 10000 `.mat` validation samples.
- `data/test_data`: 34000 `.mat` test samples and condition sweeps.

See `../../../../docs/DATASETS.md` for the full setup instructions.

## File Format

Each sample is a MATLAB `.mat` file named with its channel condition, for
example:

```text
5290_SNR-25_DS-200_DOP-650_N-3_TDL-A.mat
```

The fields are:

- `H`: complex channel tensor, shape `[120, 14, 2]`.
- `var_hat`: scalar noise-variance estimate.

Physical meaning:

- `120`: OFDM subcarriers.
- `14`: OFDM symbols in one slot/frame sample.
- `H[:, :, 0]`: dense target channel on the full resource grid.
- `H[:, :, 1]`: sparse pilot/LS observation grid used by pilot-based methods.
- filename `SNR`: signal-to-noise ratio in dB.
- filename `DS`: delay spread in ns.
- filename `DOP`: Doppler in Hz.
- filename `TDL-*`: tapped-delay-line channel profile.

## Loader Contract

The shared loader package is:

```text
../../ofdm_channel_estimation/
```

AdaFortiTran uses `build_split_dataloader`, which returns batches containing:

- `inputs`: `[B, 4, 120, 14]`, real/imag sparse LS grid plus real/imag coarse LS grid.
- `pilot_vector`: `[B, 2, P]`, real/imag pilot vector, where `P` is pilot count.
- `target`: `[B, 2, 120, 14]`, real/imag dense target channel.
- `pilot_mask`: `[120, 14]`.
- `noise_var`, `scale`, and per-sample `metadata`.

A-MMSE uses `build_ammse_dataloader`, which returns:

- `pilot_vector`: `[B, 2, P]`.
- `target_full_grid`: `[B, 2, 120, 14]`.
- `noise_var`, `pilot_mask`, and `metadata`.

## How To Use

Method wrappers read method configs under `methods/<method>/configs/`, and those
configs point to this task-level dataset. Example:

```bash
cd ${RAN_BENCHMARK_ROOT:-.}/tasks/channel_estimation/methods/ammse
python eval/run_ammse.py \
  --train-config configs/ammse_benchmark.yaml \
  --checkpoint weights/paper_strict_current_benchmark_final_best.pt \
  --split test \
  --comparison-split snr \
  --output-dir /tmp/ammse_eval \
  --limit 8 \
  --device cpu
```
