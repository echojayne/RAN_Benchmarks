# Milan Telecom Traffic Dataset

This is the task-level dataset entry for Milan cellular traffic prediction.
The repository keeps loaders and preparation metadata; raw and prepared arrays
are external assets.

## Source

The source is the Telecom Italia Milan open data used by the local traffic
forecasting benchmark. The preparation config records the source DOIs:

- `https://doi.org/10.7910/DVN/EGZHFV`
- `https://doi.org/10.7910/DVN/QJWLFU`

Raw files are 10-minute records with columns:

```text
square_id, time_interval, country_code, sms_in, sms_out,
call_in, call_out, internet_traffic
```

This benchmark uses `internet_traffic` as the target field.

## Expected Data

- `data/raw_data`: raw Milan text files and grid metadata.
- `data/prepared_data`: hourly benchmark arrays produced from the raw files.

See `../../../../docs/DATASETS.md` for download and placement instructions.

## Prepared Format

The prepared arrays are compressed NumPy `.npz` files:

```text
data/prepared_data/
|-- manifest.json
|-- selected_regions.csv
|-- train.npz
|-- val.npz
`-- test.npz
```

Each split contains:

- `inputs`: `[N, 24, 64]`, context traffic windows.
- `targets`: `[N, 6, 64]`, forecast targets.
- `target_times_ns`: `[N, 6]`, timestamps for forecast steps.
- `target_start_time_ns`: `[N]`, first forecast timestamp.
- `sample_index`: `[N]`, sample ids.
- `region_ids`: `[64]`, selected Milan grid-cell ids.

Physical meaning:

- `24`: hourly context length, derived from 10-minute raw records aggregated to
  1-hour intervals.
- `6`: hourly forecast horizon.
- `64`: selected Milan grid cells.
- value: aggregated hourly Internet traffic for each cell.

Current split sizes:

- train: 860 windows.
- validation: 139 windows.
- test: 139 windows.

## Loader Contract

The shared loader package is:

```text
../../traffic_prediction_core/data/
```

`MilanPreparedDataset` returns one sample as:

- `inputs`: `[24, 64]`.
- `targets`: `[6, 64]`.
- `target_times_ns`, `target_start_time_ns`, `sample_index`.
- `metadata` with split name, selected regions, and normalization stats.

The `DataLoader` collates these as `[B, 24, 64]` and `[B, 6, 64]`.

## How To Use

Method wrappers under `../../methods/<method>/` read configs that point to
`data/prepared_data`. Example:

```bash
cd ${RAN_BENCHMARK_ROOT:-.}/tasks/traffic_prediction/methods/itransformer
python eval/run_itransformer.py \
  --train-config configs/itransformer_benchmark.yaml \
  --checkpoint weights/static_baseline_best.pt \
  --split test \
  --output-dir /tmp/itransformer_eval \
  --num-workers 0
```
