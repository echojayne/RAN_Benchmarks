# WiFo for Channel Prediction

Original WiFo-style channel prediction method.

## Current Layout

- `wifo_prediction/`: method-local WiFo source package.
- `configs/wifo_benchmark.yaml`: benchmark config recording data and weight paths.
- `eval/run_wifo.py`: wrapper for the original CLI entrypoint.
- `train/train_wifo.py`: wrapper for the same upstream train/eval loop.
- `requirements.txt`: WiFo-side dependencies.
- `UPSTREAM_SOURCE.md`: retained upstream source notes.
- `weights/original_weights`: committed tiny/little/small weights; base and
  large weights are external assets.
- Paper reference: arXiv `2412.08908`.
- `performance/normalized/wifo_paper_reference`: normalized paper-reference
  NMSE tables plus the local D17 retained-checkpoint reproduction row.

## Data Contract

The task-level dataset lives at
`../../datasets/wifo_channel_prediction`. The loader reads
`data/test_data/<dataset>/X_test.mat` directly from that task-level dataset and
also supports `WIFO_DATASET_ROOT` for overrides. It converts complex channel
tensors to real/imag channels before feeding the WiFo model.

## Example Command

```bash
cd ${RAN_BENCHMARK_ROOT:-.}/tasks/channel_prediction_wifo_style/methods/wifo

python eval/run_wifo.py \
  --device_id 0 \
  --size small \
  --mask_strategy_random none \
  --mask_strategy temporal \
  --dataset D17 \
  --file_load_path weights/original_weights/wifo_small \
  --few_ratio 0.0 \
  --t_patch_size 4 \
  --patch_size 4 \
  --batch_size 128 \
  --pos_emb SinCos_3D
```

## Caveat

The retained upstream entrypoint is argparse-based, not YAML-config driven. The
YAML config is a benchmark record, not a replacement for the upstream CLI.
For paper-scale base/large runs, download `wifo_base.pkl` and `wifo_large.pkl`
as external assets described in `../../../../docs/ASSETS.md`.
