# Home/Office 28 GHz BeamFormer Dataset

This is the task-level dataset entry for the BeamFormer beam-management
benchmark. The repository keeps wrappers and documentation; raw/generated CSI
assets are external.

## Source

The raw scenario is the public BeamFormer ray-tracing channel dataset for the
`homeoffice-communication` environment at 28 GHz. The upstream source documents
the IEEE DataPort entry `10.21227/g1zj-z323`, the S3 archive
`homeoffice-communication-28G-raw-data.zip`, and the BeamFormer-Dataset
processing toolbox.

## Expected Data

- `data/raw`: raw ray-tracing SQLite assets.
- `data/generated_csi`: CSI generated from the ray-tracing assets.
- `data/csi-dataset`: train/validation/test CSI splits consumed by BeamFormer.
- `data/mini_demo`: small upstream demo data and demo saved models.

See `../../../../docs/DATASETS.md` for download and placement instructions.

## Ready-To-Use CSI Splits

The primary family is:

```text
data/csi-dataset/homeoffice-communication-28G-csi/
|-- t16x16_r2x1_train       726846 .mat files
|-- t16x16_r2x1_val          90856 .mat files
|-- t16x16_r2x1_test_small    1000 .mat files
`-- t16x16_r2x1_test         90856 .mat files
```

Each `.mat` sample contains a complex variable:

- key: `csi`
- raw shape: `[128, 2, 256]`
- physical meaning:
  - `128`: frequency samples/subcarriers.
  - `2`: flattened 2 x 1 receive array elements.
  - `256`: flattened 16 x 16 transmit array elements.

The BeamFormer loader reshapes this to `[frequency, rx_elements, tx_elements]`
and then computes received signal strength (RSS) under sampled/query beam
weights.

## Task Meaning

Beam management is formulated as beam-spectrum reconstruction. For each CSI
sample, the method observes RSS on 64 reference beams and predicts RSS over
1600 query directions. The 1600-point spectrum corresponds to
`angle_steps_theta x angle_steps_phi = 40 x 40` in the benchmark config.
Evaluation reports RSS loss in dB between the oracle beam and the beam selected
from the predicted spectrum.

## How To Use

BeamFormer wrappers live under:

```text
../../methods/beamformer/
```

Example:

```bash
cd ${RAN_BENCHMARK_ROOT:-.}/tasks/beam_management/methods/beamformer
python eval/run_cdf_plot.py \
  --data_path ../../datasets/homeoffice_28g_beamformer/data/csi-dataset/homeoffice-communication-28G-csi/t16x16_r2x1_test_small \
  --model_dir weights/original_final \
  --arn_model_dir weights/original_final
```

`data/mini_demo` is only a small upstream demo bundle. It is not the canonical
final retained weight location.
