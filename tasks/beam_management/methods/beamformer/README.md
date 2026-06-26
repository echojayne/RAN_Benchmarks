# BeamFormer for Beam Management

Original BeamFormer beam-management method.

## Current Layout

- `implementation/`: method-local BeamFormer package.
- `configs/`: upstream Python config modules.
- `train/train_beamformer.py`: co-training wrapper.
- `train/train_beamformer_arn.py`: ARN training wrapper.
- `eval/run_cdf_plot.py`: CDF evaluation/plot wrapper.
- `eval/visualize_spectrum.py`: spectrum visualization wrapper.
- `dataset_processing/`: retained MATLAB dataset-processing source.
- `performance/raw_data/upstream_reference_results`: optional external upstream
  reference results.
- `performance/raw_data/upstream_baseline_cdf`: optional external upstream
  baseline CDF JSON.
- `performance/curves/upstream_reference`: retained upstream reference figures.
- `weights/original_final`: external final weights directory.
- `UPSTREAM_README.md`: retained upstream README.
- `upstream_tools/`: Docker and download helper scripts.

## References

- Upstream GitHub: `https://github.com/Shunqiang-Feng/BeamFormer.git`
- Retained commit: `8e9b171d77aeb996e884deb4e29420b2fc472667`

## Data Contract

The benchmark data is 28 GHz home/office CSI. The primary generated CSI family
contains train/validation/test folders for the `t16x16_r2x1` antenna setting.
CSI samples are MATLAB files loaded as complex tensors and evaluated over a
beam-angle spectrum of 1600 query directions.

## Commands

```bash
cd ${RAN_BENCHMARK_ROOT:-.}/tasks/beam_management/methods/beamformer

python eval/visualize_spectrum.py \
  --data_path ../../datasets/homeoffice_28g_beamformer/data/mini_demo/indoor_28g_dataset/t16x16_r2x1_test_small \
  --model_dir ../../datasets/homeoffice_28g_beamformer/data/mini_demo/saved_models \
  --arn_model_dir ../../datasets/homeoffice_28g_beamformer/data/mini_demo/saved_models

python eval/run_cdf_plot.py \
  --data_path ../../datasets/homeoffice_28g_beamformer/data/csi-dataset/homeoffice-communication-28G-csi/t16x16_r2x1_test_small \
  --model_dir weights/original_final \
  --arn_model_dir weights/original_final
```

Generated plots are written under `performance/curves/generated`.

## Retained Asset Notes

`weights/original_final` is the single method-level final-weight entry, but the
large `.pth` files are not committed to Git. The
mini-demo data and demo models live only in the task dataset directory under
`../../datasets/homeoffice_28g_beamformer/data/mini_demo`. Reference result
payloads are kept under `performance/raw_data/upstream_*` so they are not
confused with locally regenerated evaluation outputs.
