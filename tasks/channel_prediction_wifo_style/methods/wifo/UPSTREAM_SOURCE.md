# WiFo Source Snapshot

This directory contains the benchmark-owned WiFo source snapshot. It is the canonical local source for the WiFo backbone and original WiFo entrypoints used by StruJEPA and other methods.

## Contents

- `src/`
  - original WiFo model code and original WiFo inference/evaluation entrypoint
  - main entry:
    - `main.py`: original WiFo inference/evaluation entry
- `reference/`
  - original WiFo reference material
- `requirements.txt`
  - WiFo-side Python dependencies

## Assets

Datasets and original weights are stored under:

```text
${RAN_BENCHMARK_ASSET_ROOT}/benchmarks/channel_prediction/wifo/assets
```

Use the benchmark catalog for stable paths:

```text
catalog/asset_manifest.yaml
```

## Typical Commands

Run original WiFo inference:

```bash
python src/main.py --device_id 0 --size base --mask_strategy_random none --mask_strategy temporal --dataset D17 --file_load_path weights/original_weights/wifo_base --few_ratio 0.0 --t_patch_size 4 --patch_size 4 --batch_size 128 --pos_emb SinCos_3D
```

StruJEPA-specific WiFo training and analysis code lives in the StruJEPA method repository, not in this benchmark source snapshot.
