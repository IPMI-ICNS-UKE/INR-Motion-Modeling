import copy
import math
import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(os.environ.get("INRMM_OUTPUT_DIR", PROJECT_DIR / "outputs"))
DIRLAB_DIR = Path(os.environ.get("INRMM_DIRLAB_DIR", PROJECT_DIR / "data" / "dirlab"))

default = {
    "reference_phase": 5,
    "respiration": {"method": "lung_volume", "centering": "ref"},
    "sampling": {
        "dense_points_per_batch": 1000,
        "continuous": True,
        "from_mask": True,
    },
    "training": {
        "total_steps": 10000,
        "validation_interval": 5000,
        "lr": 1e-4,
        "scheduler": "cosine",
        "end_lr": 0.0,
        "warmup_steps": 100,
        "track_every": 1000,
    },
    "model": {
        "type": "nonlinear",
        "layers": [5, 256, 256, 256, 3],
        "siren_freq": 30,
        "temporal_freq": math.pi / 2,
    },
    "loss": {
        "laplacian_weight": 0.01,
        "jacobian_weight": 0.01,
        "temporal_weight": 0.1,
        "temporal_order": 1,
        "consistency_weight": 1.0,
    },
    "paths": {
        "aim_repo": OUTPUT_DIR / ".aim",
        "run_folder": OUTPUT_DIR / "motion_models",
        "experiment_name": "motion_model",
        "dirlab_path": DIRLAB_DIR,
        "results_file": OUTPUT_DIR / "results.json",
    },
}

linear_model = copy.deepcopy(default)
linear_model["model"]["type"] = "linear"
linear_model["model"]["layers"] = [3, 256, 256, 256, 6]
linear_model["model"]["temporal_freq"] = None
linear_model["loss"]["temporal_order"] = 1
linear_model["loss"]["temporal_weight"] = 0.1

anisotropic_siren = copy.deepcopy(default)

isotropic_siren = copy.deepcopy(default)
isotropic_siren["model"]["temporal_freq"] = None


configs = {
    "anisotropic_siren": anisotropic_siren,
    "default": anisotropic_siren,
    "isotropic_siren": isotropic_siren,
    "linear_siren": linear_model,
    "linear_model": linear_model,
}
