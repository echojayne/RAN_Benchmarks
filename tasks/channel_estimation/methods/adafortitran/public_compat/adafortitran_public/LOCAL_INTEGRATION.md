# Local Integration Notes

This directory vendors the official public AdaFortiTran repository into the benchmark tree.

- upstream repository: `https://github.com/BerkIGuler/AdaFortiTran`
- vendored commit: `fa49e60f25579f59ba1815a27399e999b1382d5c`
- license: `MIT`

Local integration patches are intentionally thin:

1. `src/main/trainer.py`
   - adds a no-op `SummaryWriter` fallback when `tensorboard` is not installed
2. `src/data/dataset.py`
   - adds `num_workers` to `get_test_dataloaders(...)` so the public `evaluate.py` call path works as written
3. `src/evaluate.py`
   - instantiates the model on the requested runtime device so the public evaluation path does not hit a CPU/CUDA mismatch
   - converts predicted channels to CPU before visualization so the public plotting path works on CUDA runs
4. `src/utils.py`
   - makes `get_ls_mse_per_folder(...)` accept the public 2-slice `H[:,:,0:2]` format used by the generator README

No model-architecture changes were introduced on top of the vendored public code.
