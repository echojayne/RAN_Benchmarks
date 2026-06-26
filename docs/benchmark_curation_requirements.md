# Benchmark Curation Requirements

Use this checklist when adding or updating a task or method in this public
benchmark repository.

## Principles

1. Keep method code self-contained.
   - Do not rely on code symlinks.
   - Place method-specific packages under the method directory.
   - Place task-shared loaders or metrics once at the task root.

2. Separate code from assets.
   - Commit source, configs, READMEs, normalized summaries, and plotting code.
   - Keep full datasets, large checkpoints, retained runs, PDFs, and raw
     per-sample outputs outside Git.
   - Document every external asset in `docs/DATASETS.md`, `docs/ASSETS.md`, or
     `catalog/asset_manifest.yaml`.

3. Keep configs portable.
   - Use relative paths or environment variables such as
     `${RAN_BENCHMARK_ROOT}`, `${RAN_BENCHMARK_ASSET_ROOT}`, and
     `${ELASTIC_INFERENCE_METHODS_ROOT}`.
   - Do not introduce machine-specific absolute paths in executable defaults.

4. Keep results auditable.
   - Result tables must come from evaluator output, CSV, JSON, or trusted
     retained run artifacts.
   - Do not infer metrics from rendered PNGs.
   - Separate incompatible protocols in different tables.

5. Keep public docs concise and runnable.
   - README commands should reference files that exist in the checkout or are
     clearly marked as external assets.
   - Explain required data layout and checkpoint paths before listing commands.

## Required Questions Per Benchmark

Before committing a benchmark, answer:

- What is the task input, target, and evaluation metric?
- What are the dataset file formats, keys, shapes, and physical dimensions?
- Which train/validation/test splits are used?
- Which implementation is local benchmark code and which code is retained for
  upstream compatibility?
- Which weights/results are committed and which are external?
- How are training, smoke evaluation, full evaluation, and plotting launched?
- Are all result tables aligned to the same protocol and operating point?

## Method Directory Layout

Prefer:

```text
method_name/
|-- README.md
|-- configs/
|   `-- method_benchmark.yaml
|-- <method_package_or_source_dirs>/
|-- train/
|-- eval/
|-- performance/
|   |-- raw_data/
|   |-- normalized/
|   `-- curves/
|-- weights/
`-- runs/
```

Use `public_compat/<upstream_name>/` only when a retained checkpoint requires
the upstream package layout.

## Validation

After structural edits, run the checks that match the change:

```bash
python -m compileall tasks
python train/<train_script>.py --help
python eval/<eval_script>.py --help
```

For small smoke tests, prefer CPU and `--num-workers 0` in restricted
environments. Remove smoke-test outputs before committing.

## Pre-Push Checklist

- no code symlinks
- no committed machine-specific absolute paths
- no accidental dataset, run, PDF, or large checkpoint payloads
- one clear method config unless variants are intentionally documented
- README commands match current paths
- dataset and external-weight setup documented
- result tables identify protocol and retained artifact source
