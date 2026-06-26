# TCN for Milan Traffic Prediction

Static TCN baseline for the Milan benchmark.

## Current Layout

- `traffic_prediction_core/`: task-level shared implementation package, imported from the task root.
- `configs/tcn_benchmark.yaml`: method-level config.
- `train/train_baseline.py`: local training wrapper.
- `eval/run_baseline.py`: local evaluation wrapper.
- `weights/train_run_best.pt`: committed retained checkpoint.
- `runs/train_run`: optional external retained run directory.

## Commands

```bash
cd ${RAN_BENCHMARK_ROOT:-.}/tasks/traffic_prediction/methods/tcn

python train/train_baseline.py --config configs/tcn_benchmark.yaml

python eval/run_baseline.py \
  --train-config configs/tcn_benchmark.yaml \
  --checkpoint weights/train_run_best.pt \
  --split test \
  --output-dir /tmp/tcn_eval \
  --num-workers 0
```
