# RAN Benchmarks

Public benchmark package for several radio-access-network learning tasks used
in the StruJEPA experiments. The repository is organized by task, then by
dataset and method. It keeps source code, configs, public-facing documentation,
small retained weights, and normalized result tables in Git. Large datasets,
large checkpoints, full historical runs, and paper PDFs are distributed outside
Git and are described in `docs/DATASETS.md` and `docs/ASSETS.md`.

## Repository Layout

```text
catalog/
  asset_manifest.yaml        # task/dataset/method asset index
  weights_and_results.yaml   # retained run and result index
docs/
  ASSETS.md                  # weight/result release policy
  DATASETS.md                # data sources and expected folder structure
tasks/
  channel_estimation/
  channel_prediction_wifo_style/
  traffic_prediction/
  beam_management/
```

Task directories follow the same pattern:

```text
datasets/<dataset_id>/
  README.md
  data/          # user-provided data, ignored by Git
  provenance/    # optional local archives, ignored by Git
methods/<method_id>/
  README.md
  configs/
  train/
  eval/
  weights/       # selected small weights in Git; large weights external
  performance/   # normalized summaries and plotting scripts
comparisons/
  elastic_methods.yaml       # optional external elastic-baseline index
```

## Benchmarks

- `tasks/channel_estimation`: OFDM channel estimation with AdaFortiTran and
  A-MMSE on a shared OFDM `.mat` dataset.
- `tasks/channel_prediction_wifo_style`: WiFo-style CSI prediction with
  original WiFo weights and paper-reference normalized results.
- `tasks/traffic_prediction`: Milan cellular traffic forecasting with
  iTransformer, LSTM, TCN, naive baselines, and ML-TP artifacts.
- `tasks/beam_management`: BeamFormer-style beam management on the 28 GHz
  home/office CSI dataset.

## Setup

Use these environment variables when running commands from arbitrary locations:

```bash
export RAN_BENCHMARK_ROOT=/path/to/RAN_Benchmarks
export RAN_BENCHMARK_ASSET_ROOT=/path/to/ran_benchmark_assets
export ELASTIC_INFERENCE_METHODS_ROOT=/path/to/elastic_inference_methods
```

Most examples also work when launched from the method directory because configs
use paths relative to the checkout or `${RAN_BENCHMARK_ASSET_ROOT}`.

## Data And Weights

The repository intentionally does not include full datasets. Download or place
datasets under the exact layouts described in `docs/DATASETS.md`.

Small retained checkpoints are committed for lightweight local evaluation where
reasonable. Larger checkpoints are external assets:

- BeamFormer final weights: roughly 781 MB.
- WiFo base/large weights: roughly 86 MB and 344 MB.
- Full datasets and historical training/evaluation runs.

See `docs/ASSETS.md` for the complete asset policy and expected locations.

## Quick Smoke Checks

After installing each method's dependencies and placing datasets, use method
README files for full commands. A minimal syntax/import sweep can be run with:

```bash
python -m compileall tasks
```

For evaluation examples, start with small limits or CPU mode where supported.
Large BeamFormer and WiFo evaluations require the external data/weight bundles.

## External Elastic Baselines

DynaBERT, MatFormer, and OFA adapters are maintained outside this repository.
When available, task-level `comparisons/elastic_methods.yaml` files point to
`${ELASTIC_INFERENCE_METHODS_ROOT}` so original RAN benchmark methods remain
separate from cross-benchmark elastic inference baselines.
