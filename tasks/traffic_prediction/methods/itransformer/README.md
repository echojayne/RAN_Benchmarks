# iTransformer for Milan Traffic Prediction

Original iTransformer traffic-forecasting method for the Milan benchmark.

## Current Layout

- `traffic_prediction_core/`: task-level shared implementation package, imported from the task root.
- `configs/itransformer_benchmark.yaml`: method-level config.
- `train/train_itransformer.py`: local training wrapper.
- `eval/run_itransformer.py`: local evaluation wrapper.
- `citywide_compat/`: copied code, config, and docs for the retained citywide
  iTransformer artifact.
- `weights/`: committed retained checkpoints for the 64-cell benchmark and the
  separate citywide artifact.
- `runs/`: optional external retained run directories.
- Paper reference: arXiv `2310.06625`.

## Retained Assets

- `weights/static_baseline_best.pt`
- `weights/well_done_best.pt`
- `runs/static_baseline`
- `runs/well_done_citywide_noholiday_r2filtered`

## Protocol Boundary

`configs/itransformer_benchmark.yaml`, `train/train_itransformer.py`, and
`eval/run_itransformer.py` use the main Milan benchmark protocol: 64 selected
cells, 24-hour context, and 6-hour horizon.

`runs/well_done_citywide_noholiday_r2filtered` is a separate citywide artifact:
9,943 filtered cells, 3-hour context, and 1-hour horizon. Its runnable
code/config copy is under `citywide_compat/`; do not compare its metrics in the
same row group as the 64-cell benchmark.

## Commands

```bash
cd ${RAN_BENCHMARK_ROOT:-.}/tasks/traffic_prediction/methods/itransformer

python train/train_itransformer.py --config configs/itransformer_benchmark.yaml

python eval/run_itransformer.py \
  --train-config configs/itransformer_benchmark.yaml \
  --checkpoint weights/static_baseline_best.pt \
  --split test \
  --output-dir /tmp/itransformer_eval \
  --num-workers 0
```
