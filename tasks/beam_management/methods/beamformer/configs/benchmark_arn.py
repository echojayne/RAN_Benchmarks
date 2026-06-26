import os
from pathlib import Path
from types import SimpleNamespace

from configs.submodules import ARN_model, ARN_training, assumption, dataset, generator


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

config = SimpleNamespace(
    assumption=assumption.beam64(),
    dataset=ds,
    generator=generator.parametric_generator(
        generator_pretrained_model=os.path.join(model_root, "co_train", "generator_epoch_final.pth")
    ),
    arn_model=ARN_model.typical_ARN(
        ARN_model_pretrained_model=os.environ.get("BEAMFORMER_ARN_PRETRAINED") or None
    ),
)

config = ARN_training.add_train_ARN(config, config_name="arn")
config.training.model_save_path = os.path.join(model_root, "arn")
config.training.batch_size = int(os.environ.get("BEAMFORMER_ARN_BATCH_SIZE", config.training.batch_size))
config.training.epochs = int(os.environ.get("BEAMFORMER_ARN_EPOCHS", config.training.epochs))
config.training.num_workers = int(os.environ.get("BEAMFORMER_NUM_WORKERS", config.training.num_workers))
config.training.gpu_num = int(os.environ.get("BEAMFORMER_GPU_NUM", "1"))
if os.environ.get("BEAMFORMER_ARN_LR"):
    config.training.learning_rate = float(os.environ["BEAMFORMER_ARN_LR"])
