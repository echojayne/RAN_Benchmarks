# WiFo Normalized Result Package

This package records WiFo paper-reference NMSE values in machine-readable CSV
form and keeps the local retained-checkpoint reproduction as a separate row.

- `paper_wifo_base_d1_d16_nmse.csv`: WiFo-Base D1-D16 time-domain and
  frequency-domain NMSE from paper Tables 4 and 5.
- `paper_ablation_nmse.csv`: WiFo-Base ablation and D17/D18 generalization
  values from paper Table 6.
- `paper_scaling_nmse.csv`: model-size and pre-training dataset-scale values
  from paper Table 7.
- `local_reproduction_d17.csv`: local retained `wifo_base` D17 temporal-mask
  evaluation retained for comparison.

The local D17 reproduction row is not mixed into the paper tables because it is
an executed artifact from this repository, while the other rows are reference
values transcribed from arXiv `2412.08908`.
