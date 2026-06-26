# AdaFortiTran for OFDM Channel Estimation

AdaFortiTran is an adaptive transformer baseline for SISO OFDM channel
estimation. This directory packages the method as a self-contained benchmark
entry: local benchmark code, a single benchmark config, retained weights,
official-public wrapper code, evaluation outputs, and plotting scripts.

The benchmark task is to reconstruct the dense OFDM channel grid from sparse
pilot LS observations. Each sample is a MATLAB `.mat` file with a complex
channel tensor `H` shaped `[120, 14, 2]`: 120 subcarriers, 14 OFDM symbols, and
two planes for dense ground truth and sparse pilot observation.

## Status

- Method: AdaFortiTran / FortiTran-style adaptive transformer.
- Dataset: shared OFDM channel-estimation benchmark data under
  `${RAN_BENCHMARK_ASSET_ROOT}/benchmarks/channel_estimation/common`.
- Main config: `configs/adafortitran_benchmark.yaml`.
- Retained public checkpoint:
  `weights/official_public_final_best.pt`.
- Public-code evaluation results are stored in `performance/raw_data/` and
  plotted in `performance/curves/`.

## Results

The retained official-public checkpoint was evaluated through the vendored
public AdaFortiTran path:

```bash
python eval/evaluate_adafortitran_public.py \
  --checkpoint-path weights/official_public_final_best.pt \
  --output-dir /tmp/adafortitran_public_eval \
  --device auto \
  --batch-size 512 \
  --num-workers 0
```

The normalized raw output is:

- `performance/raw_data/official_public_final_eval/test_results.txt`
- `performance/raw_data/official_public_final_eval/official_public_final_metrics.csv`
- `performance/raw_data/official_public_final_eval/official_public_final_metrics.json`

### MSE by SNR

| SNR (dB) | MSE (dB) |
|---:|---:|
| 0 | -21.9348 |
| 5 | -25.8863 |
| 10 | -28.6807 |
| 15 | -30.2801 |
| 20 | -31.0047 |
| 25 | -31.2685 |

### MSE by Delay Spread

| Delay spread (ns) | MSE (dB) |
|---:|---:|
| 50 | -31.5823 |
| 100 | -30.8798 |
| 150 | -29.8710 |
| 200 | -30.9936 |
| 250 | -30.1779 |
| 300 | -28.7014 |

### MSE by Doppler

| Doppler (Hz) | MSE (dB) |
|---:|---:|
| 200 | -30.2800 |
| 400 | -29.9271 |
| 600 | -29.0763 |
| 800 | -28.5583 |
| 1000 | -25.8762 |

Regenerate the benchmark curve PNGs with:

```bash
python performance/curves/official_public_final/plot_official_public_final_curves.py
```

The generated figures are written next to the plotting script.

## References

- Paper PDF:
  `papers/AdaFortiTran_An_Adaptive_Transformer_Model_for_Robust_OFDM_Channel_Estimation.pdf`
- Public implementation snapshot:
  `public_compat/adafortitran_public`
- Upstream project recorded in the vendored snapshot:
  `https://github.com/BerkIGuler/AdaFortiTran`
- Dataset generator recorded in the config:
  `https://github.com/BerkIGuler/OFDMChannelGenerator`

The public snapshot records the vendored commit and local integration notes in
`public_compat/adafortitran_public/LOCAL_INTEGRATION.md`.

## Directory Layout

```text
adafortitran/
|-- README.md
|-- configs/
|   `-- adafortitran_benchmark.yaml
|-- eval/
|   |-- evaluate_adafortitran_public.py
|   `-- run_adafortitran.py
|-- models/
|   `-- adafortitran.py
|-- performance/
|   |-- raw_data/         # copied raw/reference/evaluation data
|   `-- curves/           # plotting scripts and generated curve figures
|-- train/
|   |-- train_adafortitran.py
|   `-- train_adafortitran_public.py
|-- public_compat/
|   `-- adafortitran_public/
|-- weights/
|   `-- official_public_final_best.pt
|-- runs/
`-- papers/
```

Shared OFDM data loaders, LS helpers, pilot-mask utilities, and metric/report
helpers live once at the task root: `../../ofdm_channel_estimation/`.
`source/` is intentionally not used.

## Environment

Use a Python environment with:

- Python 3.10 or newer
- PyTorch
- NumPy
- SciPy
- PyYAML
- tqdm
- matplotlib
- pydantic
- prettytable

Optional:

- tensorboard, for the public wrapper trainer. The vendored wrapper includes a
  no-op fallback when tensorboard is unavailable.

For local runs from this directory:

```bash
cd ${RAN_BENCHMARK_ROOT:-.}/tasks/channel_estimation/methods/adafortitran
```

On restricted environments, prefer `--num-workers 0`; multiprocessing workers
can fail when socket creation is disallowed.

## Configuration

There is one method-level config:

```text
configs/adafortitran_benchmark.yaml
```

It contains:

- dataset root and split paths
- OFDM grid size: `120 x 14`
- pilot layout: every 3rd subcarrier, OFDM symbols `[2, 11]` zero-based
- channel profile metadata: TDL-A, delay spread values, Doppler values
- evaluation axes: SNR, delay spread, Doppler
- local AdaFortiTran model hyperparameters
- local training hyperparameters

The public wrapper also has its original upstream config schema under:

```text
public_compat/adafortitran_public/config/
```

Those files are kept only for the public wrapper code path.

## Data Preparation

The benchmark config expects this data tree:

```text
${RAN_BENCHMARK_ASSET_ROOT}/benchmarks/channel_estimation/common/
|-- train_data/
|-- val_data/
`-- test_data/
    |-- SNR_test_set/
    |-- DS_test_set/
    `-- MDS_test_set/
```

Each `.mat` file must contain:

- `H`: complex array shaped `[120, 14, 2]`
  - `H[:, :, 0]`: dense ground-truth channel
  - `H[:, :, 1]`: sparse LS channel observations at pilot positions
- `var_hat`: optional scalar noise variance estimate

Filenames encode the test condition:

```text
<index>_SNR-<snr>_DS-<delay_spread>_DOP-<doppler>_N-<pilot_spacing>_<profile>.mat
```

Example:

```text
1000_SNR-0_DS-200_DOP-500_N-3_TDL-A.mat
```

## Training

### Local benchmark implementation

Run local AdaFortiTran training with:

```bash
python train/train_adafortitran.py \
  --config configs/adafortitran_benchmark.yaml \
  --output-dir /tmp/adafortitran_train \
  --device auto \
  --num-workers 0
```

Useful smoke-test command:

```bash
python train/train_adafortitran.py \
  --config configs/adafortitran_benchmark.yaml \
  --train-limit 2 \
  --val-limit 2 \
  --batch-size 2 \
  --epochs 1 \
  --num-workers 0 \
  --output-dir /tmp/adafortitran_train_smoke \
  --device cpu
```

The local trainer writes:

- `best.pt`
- `train_history.json`

to the requested output directory.

### Public wrapper implementation

The retained public-code path uses:

```bash
python train/train_adafortitran_public.py \
  --data-config configs/adafortitran_benchmark.yaml \
  --output-root /tmp/official_public_adafortitran \
  --log-root /tmp/official_public_adafortitran_logs \
  --exp-id public_benchmark_run \
  --device auto \
  --batch-size 512 \
  --learning-rate 1e-3 \
  --max-epoch 1000 \
  --patience 15 \
  --num-workers 0
```

For smoke tests, `--train-set` and `--val-set` can override the full benchmark
split directories.

## Evaluation

### Evaluate a local checkpoint

```bash
python eval/run_adafortitran.py \
  --train-config configs/adafortitran_benchmark.yaml \
  --checkpoint /path/to/best.pt \
  --split test \
  --comparison-split snr \
  --output-dir /tmp/adafortitran_eval_snr \
  --batch-size 64 \
  --num-workers 0 \
  --device auto
```

`--comparison-split` accepts:

- `snr`
- `delay_spread`
- `doppler`

The local evaluator writes:

- `per_sample.csv`
- `summary.csv`
- `summary.json`
- `summary.md`

### Evaluate the retained public checkpoint

Use this path for `weights/official_public_final_best.pt`, because that
checkpoint was produced by the vendored public implementation:

```bash
python eval/evaluate_adafortitran_public.py \
  --checkpoint-path weights/official_public_final_best.pt \
  --data-config configs/adafortitran_benchmark.yaml \
  --output-dir /tmp/adafortitran_public_eval \
  --device auto \
  --batch-size 512 \
  --num-workers 0
```

## Retained Assets

- `weights/official_public_final_best.pt`
  - Best retained public-code checkpoint.
- `runs/official_public_final`
  - Retained official-public final run package.
- `runs/static_baseline`
  - Retained local static baseline run package.
- `performance/raw_data/reference_curves`
  - Copied reference/source performance data.
- `performance/raw_data/official_public_final_eval`
  - Normalized public-checkpoint evaluation output.
- `performance/curves/official_public_final`
  - Plotting script and generated curve figures.

The retained checkpoint above is committed. Full retained run directories,
large raw source bundles, and paper PDFs are external assets; see
`../../../../docs/ASSETS.md`.

## Notes

- The reported retained public checkpoint is a benchmark-aligned public-code
  reproduction artifact, not a claim that this local asset exactly reproduces
  every paper figure.
- Keep `public_compat/adafortitran_public` if the public checkpoint or public
  wrapper train/eval path is needed.
- Use `configs/adafortitran_benchmark.yaml` as the single source of truth for
  local benchmark data, model, training, and evaluation settings.
