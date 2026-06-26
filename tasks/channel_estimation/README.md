# Channel Estimation

Task root for OFDM channel estimation benchmarks.

## Dataset

- `datasets/ofdm_channel_estimation_common`: shared train/validation/test data
  used by AdaFortiTran, A-MMSE, and elastic baseline adaptations.

## Methods

- `methods/adafortitran`: original AdaFortiTran method, paper PDF, public
  wrapper, official retained public checkpoint, and static baseline run.
- `methods/ammse`: A-MMSE method, paper PDF, retained strict benchmark run, and
  static baseline run.

The shared OFDM data loaders, LS baseline, generated-data helpers, and reporting
utilities live once at `ofdm_channel_estimation/`. Method-specific models live
inside each method directory under `models/`.
