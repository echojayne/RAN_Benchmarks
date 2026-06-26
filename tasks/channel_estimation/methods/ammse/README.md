# A-MMSE for OFDM Channel Estimation

A-MMSE is the retained attention-aided / rank-adaptive MMSE baseline for the
shared OFDM channel-estimation benchmark. The task is to reconstruct a dense
`120 x 14` complex channel grid from 80 sparse pilot observations and an
estimated noise variance.

This directory is curated as a self-contained method entry: method-local code,
one benchmark config, retained weight links, retained run links, normalized
evaluation outputs, and plotting scripts.

## Status

- Method: rank-adaptive A-MMSE on the AdaFortiTran-format OFDM data line.
- Dataset: `${RAN_BENCHMARK_ASSET_ROOT}/benchmarks/channel_estimation/common`.
- Main config: `configs/ammse_benchmark.yaml`.
- Retained checkpoint: `weights/paper_strict_current_benchmark_final_best.pt`.
- Retained evaluation output:
  `performance/raw_data/paper_strict_current_benchmark_final_eval`.

## Results

The retained paper-strict checkpoint was evaluated through the local A-MMSE
evaluator. Metrics are aggregated as mean linear MSE per condition, then
converted to dB.

### MSE by SNR

| SNR (dB) | MSE (dB) |
|---:|---:|
| 0 | -19.7955 |
| 5 | -24.1424 |
| 10 | -27.7547 |
| 15 | -29.8873 |
| 20 | -30.9128 |
| 25 | -31.2517 |

### MSE by Delay Spread

| Delay spread (ns) | MSE (dB) |
|---:|---:|
| 50 | -32.1392 |
| 100 | -30.8679 |
| 150 | -31.1113 |
| 200 | -30.9178 |
| 250 | -32.4823 |
| 300 | -29.5068 |

### MSE by Doppler

| Doppler (Hz) | MSE (dB) |
|---:|---:|
| 200 | -26.5386 |
| 400 | -26.7553 |
| 600 | -26.5603 |
| 800 | -27.2585 |
| 1000 | -26.1879 |

The normalized result table is:

```text
performance/raw_data/paper_strict_current_benchmark_final_eval/paper_strict_current_benchmark_final_metrics.csv
```

Regenerate curve figures with:

```bash
python performance/curves/paper_strict_current_benchmark_final/plot_paper_strict_current_benchmark_final_curves.py
```

## References

- Paper PDF: `papers/2506.00452v4.pdf`
- Upstream repository recorded for provenance:
  `https://github.com/TaeJun1999/Attention-aided-MMSE`
- Retained run:
  `runs/paper_strict_current_benchmark_final`
- Static comparison run:
  `runs/static_baseline`

## Directory Layout

```text
ammse/
|-- README.md
|-- configs/
|   `-- ammse_benchmark.yaml
|-- eval/
|   `-- run_ammse.py
|-- models/
|   `-- ammse_rank_adaptive.py
|-- performance/
|   |-- raw_data/
|   `-- curves/
|-- train/
|   `-- train_ammse.py
|-- weights/
|-- runs/
`-- papers/
```

Shared OFDM data loaders, pilot helpers, and report utilities live once at the
task root: `../../ofdm_channel_estimation/`.

`source/` is intentionally not used. Code needed to run the method lives inside
this method directory.

## Data Format

The config expects:

```text
${RAN_BENCHMARK_ASSET_ROOT}/benchmarks/channel_estimation/common/
|-- train_data/
|-- val_data/
`-- test_data/
    |-- SNR_test_set/
    |-- DS_test_set/
    `-- MDS_test_set/
```

Each `.mat` sample contains:

- `H`: complex array shaped `[120, 14, 2]`
  - `H[:, :, 0]`: dense ground-truth channel
  - `H[:, :, 1]`: sparse LS pilot observation grid
- `var_hat`: scalar noise variance estimate

The A-MMSE loader returns:

- `pilot_vector`: `[B, 2, 80]`, real/imag pilot observations
- `target_full_grid`: `[B, 2, 120, 14]`, real/imag dense target
- `noise_var`: `[B]`, estimated noise variance

Pilot resources are every third subcarrier at OFDM symbols `[2, 11]`
zero-based, giving `40 x 2 = 80` pilots.

## Training

```bash
cd ${RAN_BENCHMARK_ROOT:-.}/tasks/channel_estimation/methods/ammse

python train/train_ammse.py \
  --config configs/ammse_benchmark.yaml
```

CPU smoke test:

```bash
python train/train_ammse.py \
  --config configs/ammse_benchmark.yaml \
  --train-limit 2 \
  --val-limit 2 \
  --batch-size 2 \
  --epochs 1 \
  --num-workers 0 \
  --output-dir /tmp/ammse_train_smoke \
  --device cpu
```

## Evaluation

Evaluate the retained checkpoint on one comparison axis:

```bash
python eval/run_ammse.py \
  --train-config configs/ammse_benchmark.yaml \
  --checkpoint weights/paper_strict_current_benchmark_final_best.pt \
  --split test \
  --comparison-split snr \
  --output-dir /tmp/ammse_eval_snr \
  --batch-size 512 \
  --num-workers 0 \
  --device cpu
```

Use `--comparison-split delay_spread` or `--comparison-split doppler` for the
other retained test axes.

## Retained Asset Notes

The retained checkpoint is committed:

- `weights/paper_strict_current_benchmark_final_best.pt`

The following retained assets are external and can be restored from the asset
bundle described in `../../../../docs/ASSETS.md`:

- `runs/paper_strict_current_benchmark_final`
- `runs/static_baseline`
- `papers/2506.00452v4.pdf`

## Known Caveats

- The original public A-MMSE repository expected a different external 5G NR
  TDL-E data contract. This benchmark maps the method onto the shared
  AdaFortiTran-format TDL-A data line for comparison.
- Matplotlib may warn about a non-writable user cache in restricted
  environments. Set `MPLCONFIGDIR=/tmp/matplotlib` if repeated plotting needs a
  writable cache.
