# ML-TP for Milan Traffic Prediction

Paper-aligned meta-learning traffic prediction baseline.

## Current Layout

- `traffic_prediction_core/`: task-level shared implementation package,
  including `traffic_prediction_core/paper/`.
- `configs/mltp_benchmark.yaml`: paper-aligned config.
- `eval/run_cell_benchmark.py`: paper-aligned cell benchmark wrapper.
- `eval/run_mltp_week_sweep.py`: ML-TP week-sweep wrapper.
- `runs/paper_aligned_results`: retained result artifacts.
- `papers/A_Meta-Learning_Based_Framework_for_Cell-Level_Mobile_Network_Traffic_Prediction.pdf`

## Commands

```bash
cd ${RAN_BENCHMARK_ROOT:-.}/tasks/traffic_prediction/methods/mltp

python eval/run_mltp_week_sweep.py \
  --baseline-config configs/mltp_benchmark.yaml \
  --output-dir /tmp/mltp_week_sweep \
  --weeks 5
```
