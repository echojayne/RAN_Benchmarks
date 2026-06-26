# Beam Management

Task root for BeamFormer-style beam management benchmarks.

## Dataset

- `datasets/homeoffice_28g_beamformer`: expected location for the 28 GHz
  home/office raw data, generated CSI, CSI dataset, and mini-demo assets.

The primary generated CSI family is the `homeoffice-communication-28G-csi`
dataset used by the retained BeamFormer and elastic baseline runs.

## Methods

- `methods/beamformer`: curated original BeamFormer method with method-local
  `implementation/` package, Python configs, wrappers, dataset-processing code,
  retained weights, and retained performance artifacts.

The raw/generated BeamFormer data and final BeamFormer checkpoints are external
assets; see `../../docs/DATASETS.md` and `../../docs/ASSETS.md`.
