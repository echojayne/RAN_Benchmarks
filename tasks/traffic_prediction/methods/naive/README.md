# Naive for Milan Traffic Prediction

Parameter-free last-value baseline for the Milan benchmark.

## Current Layout

- `traffic_prediction_core/`: task-level shared implementation package, imported from the task root.
- `configs/naive_benchmark.yaml`: method-level config.
- `train/train_baseline.py`: materializes `checkpoint.json`.
- `eval/run_baseline.py`: local evaluation wrapper.
- `runs/eval_run`: optional external retained evaluation run directory.

## Commands

```bash
cd ${RAN_BENCHMARK_ROOT:-.}/tasks/traffic_prediction/methods/naive

python train/train_baseline.py \
  --config configs/naive_benchmark.yaml \
  --output-dir /tmp/naive_train

python eval/run_baseline.py \
  --train-config configs/naive_benchmark.yaml \
  --checkpoint runs/eval_run/checkpoint.json \
  --split test \
  --output-dir /tmp/naive_eval \
  --num-workers 0
```
