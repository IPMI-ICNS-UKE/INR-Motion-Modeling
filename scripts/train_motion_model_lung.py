import argparse
import copy
import logging
import random
from pathlib import Path
import json

import numpy as np
import torch
from inrmm.compat import init_fancy_logging
from monai.optimizers.lr_scheduler import WarmupCosineSchedule

from inrmm.model import LinearSiren, Siren
from inrmm.ncc import ncc_per_batch
from inrmm.regularizers import (
    DetJLaplacianLoss,
    DetJLoss,
    TemporalLoss,
    TemporalCurvatureLoss,
)
from inrmm.surrogate import load_lung_volume_surrogate
from inrmm.trainer import MotionModelTrainer, DualMotionModel
from inrmm.utils import (
    load_and_crop_full_dirlab,
    save_config,
    compute_landmark_distance,
    pixel_coords_to_norm,
    upsert_result,
    serialize_config,
)
from inrmm import configs


def train_case(cfg, case, dirlab_path, config_name, device, seed=42):
    case_folder = dirlab_path / f"case_{case:02d}"

    run_folder = Path(cfg["paths"]["run_folder"]) / f"case_{case:02d}"

    phases = list(range(10))

    data = load_and_crop_full_dirlab(case_folder, phases)
    images = data["images"]
    vessel_maps = data["vessel_maps"]
    lung_masks = data["lung_masks"]
    union_lung_mask = data["union_lung_mask"]
    image_spacing = data["image_spacing"]
    landmarks = data["landmarks"]

    # check landmark distances
    for phase, lm in landmarks.items():
        lm1 = pixel_coords_to_norm(
            torch.from_numpy(landmarks[cfg["reference_phase"]]).float(), images[0].shape
        )
        lm2 = pixel_coords_to_norm(torch.from_numpy(lm).float(), images[0].shape)
        dist = compute_landmark_distance(lm2, lm1, images[0].shape, image_spacing)
        print(f"Case {case} Phase {phase} Landmark distance (mm): {dist}")

    lung_vol_csv_path = case_folder / "respiratory" / "lung_volume.csv"
    signal_tensor = torch.from_numpy(
        load_lung_volume_surrogate(
            csv_path=lung_vol_csv_path,
            expected_phases=phases,
            reference_phase=cfg["reference_phase"],
        )
    ).float()

    data_loss_fn = ncc_per_batch
    if cfg["loss"]["jacobian_weight"] > 0 and cfg["loss"]["laplacian_weight"] == 0:
        spatial_reg_function = DetJLoss(detj_weight=cfg["loss"]["jacobian_weight"])
    else:
        spatial_reg_function = DetJLaplacianLoss(
            detj_weight=cfg["loss"]["jacobian_weight"],
            laplace_weight=cfg["loss"]["laplacian_weight"],
        )

    temporal_order = cfg["loss"].get("temporal_order", 1)
    if temporal_order == 1:
        temporal_reg_function = TemporalLoss(weight=cfg["loss"]["temporal_weight"])
    else:
        assert temporal_order == 2
        temporal_reg_function = TemporalCurvatureLoss(
            weight=cfg["loss"]["temporal_weight"]
        )

    def build_model():
        model_type = cfg["model"]["type"].lower()
        if model_type == "linear":
            return LinearSiren(
                layers=cfg["model"]["layers"],
                omega=cfg["model"]["siren_freq"],
            )
        if model_type == "nonlinear":
            return Siren(
                layers=cfg["model"]["layers"],
                omega=cfg["model"]["siren_freq"],
                temporal_omega=cfg["model"]["temporal_freq"],
            )
        raise ValueError(f"Unsupported model type: {model_type}")

    forward_model = build_model()
    backward_model = build_model()
    model = DualMotionModel(forward_model, backward_model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["lr"]))

    scheduler = None
    if cfg["training"]["scheduler"] == "cosine":
        scheduler = WarmupCosineSchedule(
            optimizer,
            warmup_steps=cfg["training"]["warmup_steps"],
            cycles=0.5,
            t_total=cfg["training"]["total_steps"],
            end_lr=cfg["training"]["end_lr"],
        )

    trainer = MotionModelTrainer(
        model=model,
        images=[torch.from_numpy(img).float() for img in images],
        lung_vessel_maps=[torch.from_numpy(vmap).float() for vmap in vessel_maps],
        lung_masks=[torch.from_numpy(mask).int() for mask in lung_masks],
        joint_lung_mask=torch.from_numpy(union_lung_mask).int(),
        reference_phase=cfg["reference_phase"],
        signal=signal_tensor,
        landmarks=landmarks,
        image_spacing=image_spacing,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=[None],
        run_folder=run_folder,
        aim_repo=cfg["paths"]["aim_repo"],
        loss_function=data_loss_fn,
        spatial_reg_function=spatial_reg_function,
        temporal_reg_function=temporal_reg_function,
        val_loader=[None],
        experiment_name=f"{case}_dirlab_{config_name}",
        device=device,
        config=cfg,
        track_every=cfg["training"]["track_every"],
    )

    cfg["case"] = case
    cfg["seed"] = seed
    # cfg["bbox"] = [(box.start, box.stop) for box in bbox]
    save_config(trainer._output_folder / "config.yml", cfg)

    trainer.aim_run["config"] = serialize_config(cfg)

    trainer.run(
        steps=cfg["training"]["total_steps"],
        validation_interval=cfg["training"]["validation_interval"],
    )
    # Final Evaluation
    tre_results = trainer.evaluate_on_landmarks()
    landmark_distances_mean = tre_results["mean"]
    landmark_distances_std = tre_results["std"]
    mean_landmark_displacement = 0.0
    if len(landmark_distances_mean) > 0:
        mean_landmark_displacement = sum(landmark_distances_mean.values()) / len(
            landmark_distances_mean
        )
    for phase, lm_dist in landmark_distances_mean.items():
        print(
            f"Case {case} Phase {phase} Mean Landmark Distance (mm): {lm_dist:.2f} +/- {landmark_distances_std[phase]:.2f}"
        )
    print(
        f"Case {case} Mean Landmark Displacement (mm): {mean_landmark_displacement:.2f}"
    )

    results = trainer.calculate_lung_vessel_dice()
    # save dice scores in model folder
    results_out = []
    for r in results:
        phase = r["phase"]
        results_out.append(
            {
                "case": case,
                "ref_phase": cfg["reference_phase"],
                "phase": phase,
                "tre": landmark_distances_mean.get(phase, None),
                "tre_std": landmark_distances_std.get(phase, None),
                "mean_landmark_displacement": mean_landmark_displacement,
                "dice_lung_vessels": r["dice_score"],
                "map_mae": r["map_mae"],
                "mse": r["mse"],
                #  "folding_percentage": r["folding_percentage"],
            }
        )

    # save case results
    json_path = trainer._output_folder / "results.json"
    with json_path.open("w") as f:
        json.dump(results_out, f, indent=2)

    # update aggregate results
    for r in results_out:
        upsert_result(json_path=cfg["paths"]["results_file"], new_result=r)


if __name__ == "__main__":
    logging.getLogger("vroc").setLevel(logging.DEBUG)
    logging.getLogger("inrmm").setLevel(logging.DEBUG)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    init_fancy_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, default=1)
    parser.add_argument("--config", type=str, default="default")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--total-steps",
        type=int,
        default=None,
        help="Override training.total_steps for smoke tests or scheduled runs",
    )
    parser.add_argument(
        "--temporal-freq",
        type=float,
        default=None,
        help="Override model.temporal_freq for this run",
    )
    parser.add_argument(
        "--run-folder",
        type=str,
        default=None,
        help="Override paths.run_folder for this run",
    )
    parser.add_argument(
        "--consistency-weight",
        type=float,
        default=None,
        help="Override loss.consistency_weight for this run",
    )
    parser.add_argument(
        "--temporal-weight",
        type=float,
        default=None,
        help="Override loss.temporal_weight for this run",
    )
    args = parser.parse_args()
    DEVICE = f"cuda:{args.device}"

    cfg = copy.deepcopy(configs.configs[args.config])
    config_name = args.config
    if args.run_folder is not None:
        cfg["paths"]["run_folder"] = args.run_folder
    if args.total_steps is not None:
        if args.total_steps < 1:
            raise ValueError("--total-steps must be at least one")
        cfg["training"]["total_steps"] = args.total_steps
    if args.temporal_freq is not None:
        cfg["model"]["temporal_freq"] = args.temporal_freq
        config_name = f"{config_name}_tf_{args.temporal_freq}"
    if args.consistency_weight is not None:
        cfg["loss"]["consistency_weight"] = args.consistency_weight
        config_name = f"{config_name}_cw_{args.consistency_weight}"
    if args.temporal_weight is not None:
        cfg["loss"]["temporal_weight"] = args.temporal_weight
        config_name = f"{config_name}_tw_{args.temporal_weight}"
    dirlab_path = Path(cfg["paths"]["dirlab_path"])

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.case == -1:
        for case in range(1, 11):
            train_case(cfg, case, dirlab_path, config_name, DEVICE, seed=args.seed)
    else:
        train_case(cfg, args.case, dirlab_path, config_name, DEVICE, seed=args.seed)
