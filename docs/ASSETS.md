# External Assets

Git stores the runnable benchmark code, configs, documentation, normalized
results, and selected small checkpoints. Full datasets, large checkpoints,
paper PDFs, and historical run directories are intentionally excluded.

## Committed Small Weights

The following checkpoints are committed because they are small enough for normal
Git use and useful for local smoke or retained-result evaluation:

| Task | Method | Path | Size |
|---|---|---|---:|
| channel estimation | AdaFortiTran | `tasks/channel_estimation/methods/adafortitran/weights/official_public_final_best.pt` | 3.6 MB |
| channel estimation | A-MMSE | `tasks/channel_estimation/methods/ammse/weights/paper_strict_current_benchmark_final_best.pt` | 10.1 MB |
| traffic prediction | iTransformer | `tasks/traffic_prediction/methods/itransformer/weights/static_baseline_best.pt` | 2.2 MB |
| traffic prediction | iTransformer citywide artifact | `tasks/traffic_prediction/methods/itransformer/weights/well_done_best.pt` | 3.6 MB |
| traffic prediction | LSTM | `tasks/traffic_prediction/methods/lstm/weights/train_run_best.pt` | 1.1 MB |
| traffic prediction | TCN | `tasks/traffic_prediction/methods/tcn/weights/train_run_best.pt` | 1.7 MB |
| channel prediction | WiFo small/little/tiny | `tasks/channel_prediction_wifo_style/methods/wifo/weights/original_weights/` | 29.5 MB total |

## External Large Weights

Publish these as a Google Drive folder or release asset, then place them at the
paths shown below after cloning the repository:

| Task | Method | External files | Expected path |
|---|---|---|---|
| beam management | BeamFormer | `estimator.pth`, `arn_model.pth`, `generator.pth`, `provenance.json` | `tasks/beam_management/methods/beamformer/weights/original_final/` |
| channel prediction | WiFo | `wifo_base.pkl`, `wifo_large.pkl` | `tasks/channel_prediction_wifo_style/methods/wifo/weights/original_weights/` |
| all tasks | retained runs | training histories, full eval outputs, generated plots | matching `methods/<method>/runs/` directories |

Recommended external asset root:

```text
ran_benchmark_assets/
|-- benchmarks/
|   |-- channel_estimation/
|   |-- channel_prediction/
|   |-- traffic_forecasting/
|   `-- beam_management/
`-- source_snapshots/
```

Set:

```bash
export RAN_BENCHMARK_ASSET_ROOT=/path/to/ran_benchmark_assets
```

For the local migrated workspace on this machine, use:

```bash
export RAN_BENCHMARK_ASSET_ROOT=/mnt/dky/ran_benchmarks/.local_assets/ai_ran_benchmarks
```

Then either copy the required assets into the repository paths listed above, or
create local symlinks from those paths to the asset root. Do not commit local
symlinks that point to machine-specific absolute paths.

## Dataset Assets

Datasets are always external. See `docs/DATASETS.md` for sources and required
directory structures.

## Result Artifacts

Normalized summary CSV/JSON files and plotting scripts are kept in Git when
they are small and directly support README result tables. Large per-sample CSVs,
raw upstream result dumps, and generated experiment runs are ignored by default.
