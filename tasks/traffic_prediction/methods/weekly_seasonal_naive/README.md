# Weekly Seasonal Naive for Milan Traffic Prediction

Weekly seasonal naive entry for the Milan benchmark.

## Current Layout

- `traffic_prediction_core/`: task-level shared implementation package, imported from the task root.
- `configs/weekly_seasonal_naive_benchmark.yaml`: method-level config.
- `train/train_baseline.py`: materializes `checkpoint.json`.
- `eval/run_baseline.py`: local evaluation wrapper.
- `runs/eval_run`: optional external retained evaluation run directory.

## Status

This method is not a valid weekly seasonal baseline for the current main
benchmark. The prepared Milan benchmark has only a 24-hour input context, while
this config requests `seasonal_period: 168`. The implementation therefore falls
back to last-value forecasting when the seasonal period is longer than the
available context, so the retained metrics are identical to `methods/naive`.

Treat this entry as unavailable until a 168-hour context dataset is prepared and
re-evaluated. The current retained artifact is a last-value fallback, not a
weekly seasonal result.

## Commands

```bash
cd ${RAN_BENCHMARK_ROOT:-.}/tasks/traffic_prediction/methods/weekly_seasonal_naive

python eval/run_baseline.py \
  --train-config configs/weekly_seasonal_naive_benchmark.yaml \
  --checkpoint runs/eval_run/checkpoint.json \
  --split test \
  --output-dir /tmp/weekly_seasonal_naive_eval \
  --num-workers 0
```

The `runs/eval_run` checkpoint is not committed. Restore it from external
assets, or materialize a new checkpoint with `train/train_baseline.py`.
