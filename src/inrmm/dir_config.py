import copy
from pathlib import Path

DEFAULT_DIRINR_CONFIG = {
    "data": {
        "mask_sampling": "lung",
        "sampler": "continuous",
        "dense_points_per_batch": 30000,
    },
    "training": {
        "batch_size": 1,
        "shuffle": True,
        "total_steps": 1500,
        "blur_sigmas": [0.0],
        "fine_start_stage": 0,
    },
    "model": {
        "type": "Siren",
        "layers": [3, 256, 256, 256, 3],
        "siren_freq": 30,
        "coarse_omega0": 10,
        "fine_omega0": 30,
    },
    "optimizer": {
        "lr": 0.001,
        "end_lr": 0.00001,
    },
    "scheduler": {
        "type": "cosine",
        "warmup_steps": 20,
    },
    "loss": {
        "data_loss": "ncc_per_batch",
        "laplace_weight": 0.01,
        "jacobian_weight": 0.01,
        "inverse_consistency_weight": 0.1,
        "fine_reg_scale": 0.1,
    },
    "paths": {
        "run_folder": Path("./reg_runs/"),
        "experiment_name": "dir_inr",
    },
}

SINGLE_DIRINR_CONFIG = copy.deepcopy(DEFAULT_DIRINR_CONFIG)
SINGLE_DIRINR_CONFIG["model"]["type"] = "Siren"
SINGLE_DIRINR_CONFIG["training"]["total_steps"] = 1500
SINGLE_DIRINR_CONFIG["training"]["blur_sigmas"] = [0.0]
SINGLE_DIRINR_CONFIG["training"]["fine_start_stage"] = 0

DUAL_DIRINR_CONFIG = copy.deepcopy(DEFAULT_DIRINR_CONFIG)
DUAL_DIRINR_CONFIG["model"]["type"] = "Dual"
DUAL_DIRINR_CONFIG["model"]["coarse_omega0"] = 10
DUAL_DIRINR_CONFIG["model"]["fine_omega0"] = 30
DUAL_DIRINR_CONFIG["training"]["total_steps"] = 3000
DUAL_DIRINR_CONFIG["training"]["blur_sigmas"] = [4.0, 2.0, 0.0]
DUAL_DIRINR_CONFIG["training"]["fine_start_stage"] = 1

DIRINR_CONFIGS = {
    "single": SINGLE_DIRINR_CONFIG,
    "dual": DUAL_DIRINR_CONFIG,
}
