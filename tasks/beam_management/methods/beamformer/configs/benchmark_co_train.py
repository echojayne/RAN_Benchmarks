import os
from pathlib import Path
from types import SimpleNamespace

from configs.submodules import assumption, dataset, estimator, generator, training


config_name = os.path.splitext(os.path.basename(__file__))[0]
task_root = Path(__file__).resolve().parents[3]
method_root = Path(__file__).resolve().parents[1]
asset_root = os.environ.get("RAN_BENCHMARK_ASSET_ROOT")
default_benchmark_root = (
    os.path.join(asset_root, "benchmarks", "beam_management", "beamformer")
    if asset_root
    else ""
)

root = os.environ.get(
    "BEAMFORMER_BENCHMARK_ROOT",
    default_benchmark_root,
)
csi_root = os.environ.get(
    "BEAMFORMER_CSI_ROOT",
    os.path.join(root, "assets", "csi-dataset", "homeoffice-communication-28G-csi")
    if root
    else str(task_root / "datasets" / "homeoffice_28g_beamformer" / "data" / "csi-dataset" / "homeoffice-communication-28G-csi"),
)
model_root = os.environ.get(
    "BEAMFORMER_MODEL_ROOT",
    os.path.join(root, "assets", "original_weights")
    if root
    else str(method_root / "weights" / "original_final"),
)

ds = dataset.homeoffice_communication_28g()
ds.train_data_path = os.path.join(csi_root, "t16x16_r2x1_train")
ds.test_data_path = os.path.join(csi_root, "t16x16_r2x1_val")

train_cfg = training.co_train("co_train")
train_cfg.model_save_path = os.path.join(model_root, "co_train")
train_cfg.batch_size = int(os.environ.get("BEAMFORMER_BATCH_SIZE", train_cfg.batch_size))
train_cfg.epochs = int(os.environ.get("BEAMFORMER_EPOCHS", train_cfg.epochs))
train_cfg.num_workers = int(os.environ.get("BEAMFORMER_NUM_WORKERS", train_cfg.num_workers))
train_cfg.gpu_num = int(os.environ.get("BEAMFORMER_GPU_NUM", "1"))

config = SimpleNamespace(
    assumption=assumption.beam64(),
    dataset=ds,
    estimator=estimator.PerceiverIO(
        estimator_pretrained_model=os.environ.get("BEAMFORMER_ESTIMATOR_PRETRAINED") or None
    ),
    generator=generator.parametric_generator(
        generator_pretrained_model=os.environ.get("BEAMFORMER_GENERATOR_PRETRAINED") or None
    ),
    training=train_cfg,
)

if os.environ.get("BEAMFORMER_LR_GENERATOR"):
    config.training.lr_generator = float(os.environ["BEAMFORMER_LR_GENERATOR"])
if os.environ.get("BEAMFORMER_LR_ESTIMATOR"):
    config.training.lr_estimator = float(os.environ["BEAMFORMER_LR_ESTIMATOR"])
