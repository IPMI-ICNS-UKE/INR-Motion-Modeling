from __future__ import annotations

import argparse
import csv
import gc
import json
import re
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi
import torch
import torch.nn.functional as F
import yaml
from inrmm.deformation import jacobian_determinant

from inrmm.metrics import dice_score
from inrmm.model import LinearSiren, Siren
from inrmm.surrogate import load_lung_volume_surrogate
from inrmm.trainer import DualMotionModel
from inrmm.utils import (
    compute_landmark_distance,
    load_and_crop_full_dirlab,
    make_coordinate_tensor,
    pixel_coords_to_norm,
)


def load_config(run_dir: Path) -> dict:
    config_path = run_dir / "config.yml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.yml in {run_dir}")
    with config_path.open("r") as f:
        return yaml.safe_load(f) or {}


def build_model(cfg: dict) -> DualMotionModel:
    model_cfg = cfg.get("model", {})
    model_type = str(model_cfg.get("type", "nonlinear")).lower()

    if model_type == "linear":
        forward_model = LinearSiren(
            layers=model_cfg["layers"],
            omega=model_cfg["siren_freq"],
        )
        backward_model = LinearSiren(
            layers=model_cfg["layers"],
            omega=model_cfg["siren_freq"],
        )
    elif model_type == "nonlinear":
        forward_model = Siren(
            layers=model_cfg["layers"],
            omega=model_cfg["siren_freq"],
            temporal_omega=model_cfg.get("temporal_freq"),
        )
        backward_model = Siren(
            layers=model_cfg["layers"],
            omega=model_cfg["siren_freq"],
            temporal_omega=model_cfg.get("temporal_freq"),
        )
    else:
        raise ValueError(f"Unsupported model type in config: {model_type}")

    return DualMotionModel(forward_model, backward_model)


def model_variant(cfg: dict) -> str:
    """Return the supported comparison-model name for a run configuration."""
    model_cfg = cfg.get("model", {})
    model_type = str(model_cfg.get("type", "")).lower()
    if model_type == "linear":
        return "linear_siren"
    if model_type == "nonlinear":
        if model_cfg.get("temporal_freq") is None:
            return "isotropic_siren"
        return "anisotropic_siren"
    raise ValueError(f"Unsupported model type in config: {model_type}")


def load_signal(
    cfg: dict,
    case_folder: Path,
    phases: list[int],
    reference_phase: int,
) -> torch.Tensor:
    respiration_method = str(
        cfg.get("respiration", {}).get("method", "lung_volume")
    ).lower()
    if respiration_method != "lung_volume":
        raise ValueError("Only the lung_volume respiratory surrogate is supported.")
    signal_csv = case_folder / "respiratory" / "lung_volume.csv"
    return torch.from_numpy(
        load_lung_volume_surrogate(
            csv_path=signal_csv,
            expected_phases=phases,
            reference_phase=reference_phase,
        )
    ).float()


def infer_case(run_dir: Path, cfg: dict) -> int:
    if "case" in cfg:
        return int(cfg["case"])

    match = re.search(r"case_(\d+)", run_dir.as_posix())
    if match is None:
        raise ValueError(f"Could not infer case id for run: {run_dir}")
    return int(match.group(1))


def _get_base_grid(
    image_shape: tuple[int, int, int],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    d, h, w = image_shape
    z = torch.linspace(-1.0, 1.0, d, device=device, dtype=dtype)
    y = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
    x = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
    return torch.stack([xx, yy, zz], dim=-1)


def _warp_with_grid_sample(
    image: torch.Tensor,
    disp_norm: torch.Tensor,
    base_grid: torch.Tensor,
) -> torch.Tensor:
    disp_grid = torch.empty_like(disp_norm)
    disp_grid[..., 0] = disp_norm[..., 2]
    disp_grid[..., 1] = disp_norm[..., 1]
    disp_grid[..., 2] = disp_norm[..., 0]

    grid = (base_grid + disp_grid).unsqueeze(0)
    return F.grid_sample(
        image,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )


def _compute_hd95_assd(
    pred_map: np.ndarray,
    target_map: np.ndarray,
    mask: np.ndarray,
    image_spacing: np.ndarray,
    threshold: float = 0.5,
) -> tuple[float, float]:
    if pred_map.shape != target_map.shape or target_map.shape != mask.shape:
        raise ValueError(
            "Shape mismatch in HD95/ASSD computation: "
            f"{pred_map.shape=}, {target_map.shape=}, {mask.shape=}"
        )

    mask_bool = mask > 0.5
    pred_bin = (pred_map >= threshold) & mask_bool
    target_bin = (target_map >= threshold) & mask_bool

    pred_count = int(np.sum(pred_bin))
    target_count = int(np.sum(target_bin))
    if pred_count == 0 and target_count == 0:
        return 0.0, 0.0
    if pred_count == 0 or target_count == 0:
        return float("nan"), float("nan")

    structure = np.ones((3, 3, 3), dtype=bool)
    pred_surface = pred_bin ^ ndi.binary_erosion(
        pred_bin, structure=structure, border_value=0
    )
    target_surface = target_bin ^ ndi.binary_erosion(
        target_bin, structure=structure, border_value=0
    )

    if int(np.sum(pred_surface)) == 0 or int(np.sum(target_surface)) == 0:
        return float("nan"), float("nan")

    dt_to_target_surface = ndi.distance_transform_edt(
        ~target_surface, sampling=tuple(image_spacing)
    )
    dt_to_pred_surface = ndi.distance_transform_edt(
        ~pred_surface, sampling=tuple(image_spacing)
    )

    dist_pred_to_target = dt_to_target_surface[pred_surface]
    dist_target_to_pred = dt_to_pred_surface[target_surface]
    if dist_pred_to_target.size == 0 or dist_target_to_pred.size == 0:
        return float("nan"), float("nan")

    all_surface_distances = np.concatenate(
        [dist_pred_to_target, dist_target_to_pred], axis=0
    )
    hd95 = float(np.percentile(all_surface_distances, 95))
    assd = float(np.mean(all_surface_distances))
    return hd95, assd


def evaluate_landmarks(
    model: DualMotionModel,
    landmarks_np: dict[int, np.ndarray],
    signal: torch.Tensor,
    reference_phase: int,
    image_shape: tuple[int, int, int],
    image_spacing: tuple[float, float, float],
    device: torch.device,
    phases_to_evaluate: set[int] | None = None,
) -> dict[str, dict[str, dict[int, float]]]:
    forward_model = model.forward_model
    backward_model = model.backward_model
    forward_model.eval()
    backward_model.eval()

    landmarks = {
        phase: pixel_coords_to_norm(
            torch.from_numpy(lm).float().to(device), image_shape
        )
        for phase, lm in landmarks_np.items()
    }

    def _empty_direction_stats() -> dict[str, dict[int, float]]:
        return {
            "mean": {},
            "std": {},
            "mean_world": {},
            "std_world": {},
        }

    if reference_phase not in landmarks:
        return {
            "fwd": _empty_direction_stats(),
            "bwd": _empty_direction_stats(),
        }

    out = {
        "fwd": _empty_direction_stats(),
        "bwd": _empty_direction_stats(),
    }

    reference_landmarks = landmarks[reference_phase]

    for phase, phase_landmarks in landmarks.items():
        if phases_to_evaluate is not None and phase not in phases_to_evaluate:
            continue
        surrogate = signal[phase].to(device)

        # Forward model: reference grid -> target phase landmark positions
        model_input_fwd = torch.cat(
            [reference_landmarks, surrogate.repeat(reference_landmarks.shape[0], 1)],
            dim=-1,
        )
        with torch.no_grad():
            disp_fwd = forward_model(model_input_fwd)
        pred_landmarks_fwd = reference_landmarks + disp_fwd

        mean_fwd, std_fwd = compute_landmark_distance(
            pred_landmarks_fwd.detach(),
            phase_landmarks.detach(),
            img_shape=image_shape,
            img_spacing=image_spacing,
        )
        mean_world_fwd, std_world_fwd = compute_landmark_distance(
            pred_landmarks_fwd.detach(),
            phase_landmarks.detach(),
            img_shape=image_shape,
            img_spacing=image_spacing,
            snap_to_voxel=False,
        )

        out["fwd"]["mean"][phase] = float(mean_fwd.item())
        out["fwd"]["std"][phase] = float(std_fwd.item())
        out["fwd"]["mean_world"][phase] = float(mean_world_fwd.item())
        out["fwd"]["std_world"][phase] = float(std_world_fwd.item())

        # Backward model: phase grid -> reference landmark positions
        model_input_bwd = torch.cat(
            [phase_landmarks, surrogate.repeat(phase_landmarks.shape[0], 1)],
            dim=-1,
        )
        with torch.no_grad():
            disp_bwd = backward_model(model_input_bwd)
        pred_landmarks_bwd = phase_landmarks + disp_bwd

        mean_bwd, std_bwd = compute_landmark_distance(
            pred_landmarks_bwd.detach(),
            reference_landmarks.detach(),
            img_shape=image_shape,
            img_spacing=image_spacing,
        )
        mean_world_bwd, std_world_bwd = compute_landmark_distance(
            pred_landmarks_bwd.detach(),
            reference_landmarks.detach(),
            img_shape=image_shape,
            img_spacing=image_spacing,
            snap_to_voxel=False,
        )

        out["bwd"]["mean"][phase] = float(mean_bwd.item())
        out["bwd"]["std"][phase] = float(std_bwd.item())
        out["bwd"]["mean_world"][phase] = float(mean_world_bwd.item())
        out["bwd"]["std_world"][phase] = float(std_world_bwd.item())

    return out


def evaluate_vessel_and_image_metrics(
    model: DualMotionModel,
    images: torch.Tensor,
    lung_vessel_maps: torch.Tensor,
    lung_masks: torch.Tensor,
    signal: torch.Tensor,
    reference_phase: int,
    image_shape: tuple[int, int, int],
    image_spacing: tuple[float, float, float] | np.ndarray,
    device: torch.device,
    direction: str,
    phases_to_evaluate: set[int] | None = None,
) -> list[dict[str, float | int]]:
    if direction == "backward":
        metric_model = model.backward_model
    elif direction == "forward":
        metric_model = model.forward_model
    else:
        raise ValueError(f"Unsupported direction: {direction}")

    metric_model.eval()

    base_grid = _get_base_grid(
        image_shape=image_shape, device=device, dtype=images.dtype
    )
    grid = make_coordinate_tensor(dims=image_shape, flatten=True).to(
        device=device, dtype=torch.float32
    )

    output: list[dict[str, float | int]] = []

    for phase in range(images.shape[0]):
        if phases_to_evaluate is not None and phase not in phases_to_evaluate:
            continue
        surrogate = signal[phase].to(device)
        model_input = torch.cat([grid, surrogate.repeat(grid.shape[0], 1)], dim=-1)

        deformation_chunks: list[torch.Tensor] = []
        for chunk in torch.chunk(model_input, chunks=10, dim=0):
            with torch.no_grad():
                deformation_chunks.append(metric_model(chunk))
        deformations = torch.cat(deformation_chunks, dim=0).view(image_shape + (3,))

        if direction == "backward":
            # backward model: phase grid, warp reference (moving) -> phase (fixed)
            moving_map = (
                lung_vessel_maps[reference_phase].float().unsqueeze(0).unsqueeze(0)
            )
            fixed_map = lung_vessel_maps[phase].float().unsqueeze(0).unsqueeze(0)
            moving_image = images[reference_phase].unsqueeze(0).unsqueeze(0)
            fixed_image = images[phase].unsqueeze(0).unsqueeze(0)
            lung_mask = (lung_masks[phase] > 0.5).float().unsqueeze(0).unsqueeze(0)
        else:
            # forward model: reference grid, backwarp phase (moving) -> reference (fixed)
            moving_map = lung_vessel_maps[phase].float().unsqueeze(0).unsqueeze(0)
            fixed_map = (
                lung_vessel_maps[reference_phase].float().unsqueeze(0).unsqueeze(0)
            )
            moving_image = images[phase].unsqueeze(0).unsqueeze(0)
            fixed_image = images[reference_phase].unsqueeze(0).unsqueeze(0)
            lung_mask = (
                (lung_masks[reference_phase] > 0.5).float().unsqueeze(0).unsqueeze(0)
            )

        warped_map = _warp_with_grid_sample(
            image=moving_map,
            disp_norm=deformations,
            base_grid=base_grid,
        )

        warped_map = warped_map * lung_mask
        fixed_map = fixed_map * lung_mask

        map_mae = (
            torch.abs(warped_map - fixed_map) * lung_mask
        ).sum() / lung_mask.sum().clamp_min(1.0)

        warped_mask = (warped_map >= 0.5).float()
        fixed_mask = (fixed_map >= 0.5).float()
        dice_score_val = dice_score(warped_mask, fixed_mask)
        _, assd_vessel_mm = _compute_hd95_assd(
            pred_map=warped_map.squeeze(0).squeeze(0).detach().cpu().numpy(),
            target_map=fixed_map.squeeze(0).squeeze(0).detach().cpu().numpy(),
            mask=lung_mask.squeeze(0).squeeze(0).detach().cpu().numpy(),
            image_spacing=np.asarray(image_spacing, dtype=np.float64),
            threshold=0.5,
        )

        warped_image = _warp_with_grid_sample(
            image=moving_image,
            disp_norm=deformations,
            base_grid=base_grid,
        )
        mse = (
            (warped_image - fixed_image) ** 2 * lung_mask
        ).sum() / lung_mask.sum().clamp_min(1.0)

        deformations_voxel = torch.empty_like(deformations)
        for axis in range(3):
            deformations_voxel[..., axis] = (
                deformations[..., axis] * (image_shape[axis] - 1) / 2
            )
        det_j = jacobian_determinant(
            deformations_voxel.permute(3, 0, 1, 2).unsqueeze(0).detach().cpu().numpy()
        )
        det_j = (
            det_j.detach().cpu().numpy() if isinstance(det_j, torch.Tensor) else det_j
        )

        lung_mask_np = (lung_mask.squeeze(0).squeeze(0) > 0.5).detach().cpu().numpy()
        if min(lung_mask_np.shape) > 4:
            cropped_lung = lung_mask_np[2:-2, 2:-2, 2:-2]
        else:
            cropped_lung = lung_mask_np

        if det_j.shape != cropped_lung.shape:
            min_shape = tuple(
                min(det_dim, mask_dim)
                for det_dim, mask_dim in zip(det_j.shape, cropped_lung.shape)
            )
            det_j = det_j[tuple(slice(0, size) for size in min_shape)]
            cropped_lung = cropped_lung[tuple(slice(0, size) for size in min_shape)]

        det_j_lung = det_j[cropped_lung]
        if det_j_lung.size == 0:
            folding_percentage = 0.0
            detj_std = None
        else:
            folding_percentage = float(np.sum(det_j_lung <= 0) / det_j_lung.size)
            detj_std = float(np.std(det_j_lung))

        output.append(
            {
                "phase": int(phase),
                "dice_score": float(dice_score_val.item()),
                "map_mae": float(map_mae.item()),
                "mse": float(mse.item()),
                "assd_vessel_0p5_mm": assd_vessel_mm,
                "folding_percentage": folding_percentage,
                "detj_std": detj_std,
            }
        )

    return output


def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.array(values, dtype=np.float64)))


def summarize_results(results: list[dict]) -> dict:
    if not results:
        return {}

    ref_phase = results[0].get("ref_phase")

    def _collect(metric_key: str) -> list[float]:
        return [
            float(r[metric_key])
            for r in results
            if r.get("phase") != ref_phase and r.get(metric_key) is not None
        ]

    tre_values = _collect("tre")
    tre_values_fwd = _collect("tre_fwd")
    tre_values_bwd = _collect("tre_bwd")
    tre_world_values = _collect("tre_world")
    tre_world_values_fwd = _collect("tre_world_fwd")
    tre_world_values_bwd = _collect("tre_world_bwd")
    tre_extreme_values = _collect("tre_extreme")
    tre_extreme_values_fwd = _collect("tre_extreme_fwd")
    tre_extreme_values_bwd = _collect("tre_extreme_bwd")

    dice_values_bwd = [
        float(r["dice_lung_vessels_bwd"])
        for r in results
        if r.get("phase") != ref_phase and r.get("dice_lung_vessels_bwd") is not None
    ]
    mae_values_bwd = [
        float(r["map_mae_bwd"])
        for r in results
        if r.get("phase") != ref_phase and r.get("map_mae_bwd") is not None
    ]
    mse_values_bwd = [
        float(r["mse_bwd"])
        for r in results
        if r.get("phase") != ref_phase and r.get("mse_bwd") is not None
    ]
    assd_values_bwd = [
        float(r["assd_vessel_0p5_mm_bwd"])
        for r in results
        if r.get("phase") != ref_phase and r.get("assd_vessel_0p5_mm_bwd") is not None
    ]
    folding_values_bwd = [
        float(r["folding_percentage_bwd"])
        for r in results
        if r.get("phase") != ref_phase and r.get("folding_percentage_bwd") is not None
    ]
    detj_std_values_bwd = [
        float(r["detj_std_bwd"])
        for r in results
        if r.get("phase") != ref_phase and r.get("detj_std_bwd") is not None
    ]

    dice_values_fwd = [
        float(r["dice_lung_vessels_fwd"])
        for r in results
        if r.get("phase") != ref_phase and r.get("dice_lung_vessels_fwd") is not None
    ]
    mae_values_fwd = [
        float(r["map_mae_fwd"])
        for r in results
        if r.get("phase") != ref_phase and r.get("map_mae_fwd") is not None
    ]
    mse_values_fwd = [
        float(r["mse_fwd"])
        for r in results
        if r.get("phase") != ref_phase and r.get("mse_fwd") is not None
    ]
    assd_values_fwd = [
        float(r["assd_vessel_0p5_mm_fwd"])
        for r in results
        if r.get("phase") != ref_phase and r.get("assd_vessel_0p5_mm_fwd") is not None
    ]
    folding_values_fwd = [
        float(r["folding_percentage_fwd"])
        for r in results
        if r.get("phase") != ref_phase and r.get("folding_percentage_fwd") is not None
    ]
    detj_std_values_fwd = [
        float(r["detj_std_fwd"])
        for r in results
        if r.get("phase") != ref_phase and r.get("detj_std_fwd") is not None
    ]

    return {
        "case": results[0].get("case"),
        "ref_phase": ref_phase,
        "mean_tre_over_phases": _mean_or_none(tre_values),
        "mean_tre_over_phases_fwd": _mean_or_none(tre_values_fwd),
        "mean_tre_over_phases_bwd": _mean_or_none(tre_values_bwd),
        "mean_tre_world_over_phases": _mean_or_none(tre_world_values),
        "mean_tre_world_over_phases_fwd": _mean_or_none(tre_world_values_fwd),
        "mean_tre_world_over_phases_bwd": _mean_or_none(tre_world_values_bwd),
        "mean_tre_extreme_over_phases": _mean_or_none(tre_extreme_values),
        "mean_tre_extreme_over_phases_fwd": _mean_or_none(tre_extreme_values_fwd),
        "mean_tre_extreme_over_phases_bwd": _mean_or_none(tre_extreme_values_bwd),
        # Backward metrics (legacy defaults)
        "mean_dice_over_phases": _mean_or_none(dice_values_bwd),
        "mean_map_mae_over_phases": _mean_or_none(mae_values_bwd),
        "mean_mse_over_phases": _mean_or_none(mse_values_bwd),
        "mean_assd_vessel_0p5_mm_over_phases": _mean_or_none(assd_values_bwd),
        "mean_folding_over_phases": _mean_or_none(folding_values_bwd),
        "mean_detj_std_over_phases": _mean_or_none(detj_std_values_bwd),
        # Explicit directional metrics
        "mean_dice_over_phases_bwd": _mean_or_none(dice_values_bwd),
        "mean_map_mae_over_phases_bwd": _mean_or_none(mae_values_bwd),
        "mean_mse_over_phases_bwd": _mean_or_none(mse_values_bwd),
        "mean_assd_vessel_0p5_mm_over_phases_bwd": _mean_or_none(assd_values_bwd),
        "mean_folding_over_phases_bwd": _mean_or_none(folding_values_bwd),
        "mean_detj_std_over_phases_bwd": _mean_or_none(detj_std_values_bwd),
        "mean_dice_over_phases_fwd": _mean_or_none(dice_values_fwd),
        "mean_map_mae_over_phases_fwd": _mean_or_none(mae_values_fwd),
        "mean_mse_over_phases_fwd": _mean_or_none(mse_values_fwd),
        "mean_assd_vessel_0p5_mm_over_phases_fwd": _mean_or_none(assd_values_fwd),
        "mean_folding_over_phases_fwd": _mean_or_none(folding_values_fwd),
        "mean_detj_std_over_phases_fwd": _mean_or_none(detj_std_values_fwd),
        "mean_landmark_displacement": results[0].get("mean_landmark_displacement"),
        "mean_landmark_displacement_world": results[0].get(
            "mean_landmark_displacement_world"
        ),
    }


def summarize_over_all_runs_phase_groups(rows: list[dict]) -> list[dict]:
    metric_keys = [
        "tre",
        "tre_world",
        "tre_extreme",
        "tre_fwd",
        "tre_world_fwd",
        "tre_extreme_fwd",
        "tre_bwd",
        "tre_world_bwd",
        "tre_extreme_bwd",
        "dice_lung_vessels_bwd",
        "map_mae_bwd",
        "mse_bwd",
        "assd_vessel_0p5_mm_bwd",
        "folding_percentage_bwd",
        "detj_std_bwd",
        "dice_lung_vessels_fwd",
        "map_mae_fwd",
        "mse_fwd",
        "assd_vessel_0p5_mm_fwd",
        "folding_percentage_fwd",
        "detj_std_fwd",
    ]

    grouped: dict[tuple[int, int, str, str | None], dict] = {}
    for row in rows:
        case = int(row["case"])
        phase = int(row["phase"])
        variant = str(row["model_variant"])
        signal = row.get("signal")
        key = (case, phase, variant, signal)

        if key not in grouped:
            grouped[key] = {
                "run_dirs": set(),
                **{metric_key: [] for metric_key in metric_keys},
            }

        grouped[key]["run_dirs"].add(str(row.get("run_dir", "")))
        for metric_key in metric_keys:
            value = row.get(metric_key)
            if value is None:
                continue
            value_f = float(value)
            if np.isfinite(value_f):
                grouped[key][metric_key].append(value_f)

    summary_rows: list[dict] = []
    for key in sorted(grouped.keys()):
        case, phase, variant, signal = key
        values = grouped[key]
        out_row: dict[str, int | str | float | None] = {
            "case": case,
            "phase": phase,
            "model_variant": variant,
            "signal": signal,
            "num_runs": len(values["run_dirs"]),
        }
        for metric_key in metric_keys:
            metric_values = np.asarray(values[metric_key], dtype=np.float64)
            out_row[f"{metric_key}_mean"] = (
                float(np.mean(metric_values)) if metric_values.size > 0 else None
            )
            out_row[f"{metric_key}_std"] = (
                float(np.std(metric_values)) if metric_values.size > 0 else None
            )
        summary_rows.append(out_row)

    return summary_rows


def evaluate_single_run(
    run_dir: Path,
    device: torch.device,
    data_root_override: Path | None,
    case_data_cache: dict[tuple[str, int], dict],
    selected_phases: set[int] | None = None,
) -> tuple[dict, list[dict]]:
    model: DualMotionModel | None = None
    images: torch.Tensor | None = None
    vessel_maps: torch.Tensor | None = None
    lung_masks: torch.Tensor | None = None
    signal: torch.Tensor | None = None

    cfg = load_config(run_dir)

    case = infer_case(run_dir=run_dir, cfg=cfg)
    reference_phase = int(cfg.get("reference_phase", 5))
    phases = list(range(10))
    eval_phases = phases if selected_phases is None else sorted(selected_phases)

    if data_root_override is None:
        dirlab_root = Path(cfg["paths"]["dirlab_path"])
    else:
        dirlab_root = data_root_override

    case_folder = dirlab_root / f"case_{case:02d}"
    if not case_folder.exists():
        raise FileNotFoundError(f"Case folder not found: {case_folder}")

    case_key = (str(case_folder.resolve()), case)
    if case_key not in case_data_cache:
        # Keep only one case in cache at a time to avoid unnecessary memory growth.
        case_data_cache.clear()
        case_data_cache[case_key] = load_and_crop_full_dirlab(
            case_folder=case_folder,
            phases=phases,
        )

    data = case_data_cache[case_key]

    try:
        signal = load_signal(
            cfg=cfg,
            case_folder=case_folder,
            phases=phases,
            reference_phase=reference_phase,
        ).to(device)

        model_path = run_dir / "models" / "final.pth"
        if not model_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {model_path}")

        # Load checkpoint on CPU first to avoid temporary duplicated GPU memory.
        model = build_model(cfg)
        state = torch.load(model_path, map_location="cpu")
        model_state = (
            state["model"] if isinstance(state, dict) and "model" in state else state
        )
        model.load_state_dict(model_state)
        model = model.to(device)

        images = torch.stack(
            [torch.from_numpy(img).float() for img in data["images"]], dim=0
        ).to(device)
        vessel_maps = torch.stack(
            [torch.from_numpy(vmap).float() for vmap in data["vessel_maps"]], dim=0
        ).to(device)
        lung_masks = torch.from_numpy(np.asarray(data["lung_masks"])).float().to(device)

        landmark_results = evaluate_landmarks(
            model=model,
            landmarks_np=data["landmarks"],
            signal=signal,
            reference_phase=reference_phase,
            image_shape=images[0].shape,
            image_spacing=data["image_spacing"],
            device=device,
            phases_to_evaluate=set(eval_phases),
        )
        extreme_landmark_results = evaluate_landmarks(
            model=model,
            landmarks_np=data.get("extreme_landmarks", {}),
            signal=signal,
            reference_phase=reference_phase,
            image_shape=images[0].shape,
            image_spacing=data["image_spacing"],
            device=device,
            phases_to_evaluate=set(eval_phases),
        )

        vessel_results_bwd = evaluate_vessel_and_image_metrics(
            model=model,
            images=images,
            lung_vessel_maps=vessel_maps,
            lung_masks=lung_masks,
            signal=signal,
            reference_phase=reference_phase,
            image_shape=images[0].shape,
            image_spacing=data["image_spacing"],
            device=device,
            direction="backward",
            phases_to_evaluate=set(eval_phases),
        )
        vessel_results_fwd = evaluate_vessel_and_image_metrics(
            model=model,
            images=images,
            lung_vessel_maps=vessel_maps,
            lung_masks=lung_masks,
            signal=signal,
            reference_phase=reference_phase,
            image_shape=images[0].shape,
            image_spacing=data["image_spacing"],
            device=device,
            direction="forward",
            phases_to_evaluate=set(eval_phases),
        )

        mean_landmark_displacement = _mean_or_none(
            [
                float(v)
                for phase, v in landmark_results["fwd"]["mean"].items()
                if int(phase) != reference_phase
            ]
        )
        mean_landmark_displacement_world = _mean_or_none(
            [
                float(v)
                for phase, v in landmark_results["fwd"]["mean_world"].items()
                if int(phase) != reference_phase
            ]
        )

        model_cfg = cfg.get("model", {})
        resp_cfg = cfg.get("respiration", {})
        variant = model_variant(cfg)

        run_results: list[dict] = []
        vessel_by_phase_bwd = {int(v["phase"]): v for v in vessel_results_bwd}
        vessel_by_phase_fwd = {int(v["phase"]): v for v in vessel_results_fwd}
        for phase in eval_phases:
            vessel_phase_result_bwd = vessel_by_phase_bwd.get(phase)
            vessel_phase_result_fwd = vessel_by_phase_fwd.get(phase)
            if vessel_phase_result_bwd is None and vessel_phase_result_fwd is None:
                continue
            run_results.append(
                {
                    "case": case,
                    "ref_phase": reference_phase,
                    "phase": phase,
                    # Legacy TRE keys preserve forward-model semantics.
                    "tre": landmark_results["fwd"]["mean"].get(phase),
                    "tre_std": landmark_results["fwd"]["std"].get(phase),
                    "tre_world": landmark_results["fwd"]["mean_world"].get(phase),
                    "tre_std_world": landmark_results["fwd"]["std_world"].get(phase),
                    "tre_extreme": extreme_landmark_results["fwd"]["mean"].get(phase),
                    "tre_extreme_world": extreme_landmark_results["fwd"][
                        "mean_world"
                    ].get(phase),
                    # Explicit directional TRE metrics
                    "tre_fwd": landmark_results["fwd"]["mean"].get(phase),
                    "tre_fwd_std": landmark_results["fwd"]["std"].get(phase),
                    "tre_world_fwd": landmark_results["fwd"]["mean_world"].get(phase),
                    "tre_world_fwd_std": landmark_results["fwd"]["std_world"].get(
                        phase
                    ),
                    "tre_extreme_fwd": extreme_landmark_results["fwd"]["mean"].get(
                        phase
                    ),
                    "tre_extreme_world_fwd": extreme_landmark_results["fwd"][
                        "mean_world"
                    ].get(phase),
                    "tre_bwd": landmark_results["bwd"]["mean"].get(phase),
                    "tre_bwd_std": landmark_results["bwd"]["std"].get(phase),
                    "tre_world_bwd": landmark_results["bwd"]["mean_world"].get(phase),
                    "tre_world_bwd_std": landmark_results["bwd"]["std_world"].get(
                        phase
                    ),
                    "tre_extreme_bwd": extreme_landmark_results["bwd"]["mean"].get(
                        phase
                    ),
                    "tre_extreme_world_bwd": extreme_landmark_results["bwd"][
                        "mean_world"
                    ].get(phase),
                    "mean_landmark_displacement": mean_landmark_displacement,
                    "mean_landmark_displacement_world": mean_landmark_displacement_world,
                    # Legacy keys map to backward metrics for compatibility.
                    "dice_lung_vessels": (
                        vessel_phase_result_bwd["dice_score"]
                        if vessel_phase_result_bwd is not None
                        else None
                    ),
                    "map_mae": (
                        vessel_phase_result_bwd["map_mae"]
                        if vessel_phase_result_bwd is not None
                        else None
                    ),
                    "mse": (
                        vessel_phase_result_bwd["mse"]
                        if vessel_phase_result_bwd is not None
                        else None
                    ),
                    "assd_vessel_0p5_mm": (
                        vessel_phase_result_bwd["assd_vessel_0p5_mm"]
                        if vessel_phase_result_bwd is not None
                        else None
                    ),
                    "folding_percentage": (
                        vessel_phase_result_bwd["folding_percentage"]
                        if vessel_phase_result_bwd is not None
                        else None
                    ),
                    "detj_std": (
                        vessel_phase_result_bwd["detj_std"]
                        if vessel_phase_result_bwd is not None
                        else None
                    ),
                    "dice_lung_vessels_bwd": (
                        vessel_phase_result_bwd["dice_score"]
                        if vessel_phase_result_bwd is not None
                        else None
                    ),
                    "map_mae_bwd": (
                        vessel_phase_result_bwd["map_mae"]
                        if vessel_phase_result_bwd is not None
                        else None
                    ),
                    "mse_bwd": (
                        vessel_phase_result_bwd["mse"]
                        if vessel_phase_result_bwd is not None
                        else None
                    ),
                    "assd_vessel_0p5_mm_bwd": (
                        vessel_phase_result_bwd["assd_vessel_0p5_mm"]
                        if vessel_phase_result_bwd is not None
                        else None
                    ),
                    "folding_percentage_bwd": (
                        vessel_phase_result_bwd["folding_percentage"]
                        if vessel_phase_result_bwd is not None
                        else None
                    ),
                    "detj_std_bwd": (
                        vessel_phase_result_bwd["detj_std"]
                        if vessel_phase_result_bwd is not None
                        else None
                    ),
                    "dice_lung_vessels_fwd": (
                        vessel_phase_result_fwd["dice_score"]
                        if vessel_phase_result_fwd is not None
                        else None
                    ),
                    "map_mae_fwd": (
                        vessel_phase_result_fwd["map_mae"]
                        if vessel_phase_result_fwd is not None
                        else None
                    ),
                    "mse_fwd": (
                        vessel_phase_result_fwd["mse"]
                        if vessel_phase_result_fwd is not None
                        else None
                    ),
                    "assd_vessel_0p5_mm_fwd": (
                        vessel_phase_result_fwd["assd_vessel_0p5_mm"]
                        if vessel_phase_result_fwd is not None
                        else None
                    ),
                    "folding_percentage_fwd": (
                        vessel_phase_result_fwd["folding_percentage"]
                        if vessel_phase_result_fwd is not None
                        else None
                    ),
                    "detj_std_fwd": (
                        vessel_phase_result_fwd["detj_std"]
                        if vessel_phase_result_fwd is not None
                        else None
                    ),
                    "model_type": model_cfg.get("type"),
                    "model_variant": variant,
                    "signal": resp_cfg.get("method"),
                    "run_dir": str(run_dir),
                }
            )

        out_path = run_dir / "retrospective_results.json"
        with out_path.open("w") as f:
            json.dump(run_results, f, indent=2)

        summary = summarize_results(run_results)
        summary["run_dir"] = str(run_dir)

        summary.update(
            {
                "model_type": model_cfg.get("type"),
                "model_variant": variant,
                "signal": resp_cfg.get("method"),
            }
        )

        print(f"Evaluated {run_dir}")
        return summary, run_results
    finally:
        del images, vessel_maps, lung_masks, signal, model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def discover_run_dirs(run_folder: Path) -> list[Path]:
    run_dirs: list[Path] = []
    case_dirs = sorted([path for path in run_folder.glob("case_*") if path.is_dir()])

    if case_dirs:
        for case_dir in case_dirs:
            for run_dir in sorted(case_dir.iterdir()):
                if run_dir.is_dir():
                    run_dirs.append(run_dir)
    else:
        run_dirs = sorted([path for path in run_folder.iterdir() if path.is_dir()])

    return run_dirs


def write_summary_csv(summaries: list[dict], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "run_dir",
                "case",
                "ref_phase",
                "model_type",
                "model_variant",
                "signal",
                "mean_tre_over_phases",
                "mean_tre_over_phases_fwd",
                "mean_tre_over_phases_bwd",
                "mean_tre_world_over_phases",
                "mean_tre_world_over_phases_fwd",
                "mean_tre_world_over_phases_bwd",
                "mean_tre_extreme_over_phases",
                "mean_tre_extreme_over_phases_fwd",
                "mean_tre_extreme_over_phases_bwd",
                "mean_dice_over_phases",
                "mean_map_mae_over_phases",
                "mean_mse_over_phases",
                "mean_assd_vessel_0p5_mm_over_phases",
                "mean_folding_over_phases",
                "mean_detj_std_over_phases",
                "mean_dice_over_phases_bwd",
                "mean_map_mae_over_phases_bwd",
                "mean_mse_over_phases_bwd",
                "mean_assd_vessel_0p5_mm_over_phases_bwd",
                "mean_folding_over_phases_bwd",
                "mean_detj_std_over_phases_bwd",
                "mean_dice_over_phases_fwd",
                "mean_map_mae_over_phases_fwd",
                "mean_mse_over_phases_fwd",
                "mean_assd_vessel_0p5_mm_over_phases_fwd",
                "mean_folding_over_phases_fwd",
                "mean_detj_std_over_phases_fwd",
                "mean_landmark_displacement",
                "mean_landmark_displacement_world",
            ]
        )
        for summary in summaries:
            writer.writerow(
                [
                    summary.get("run_dir"),
                    summary.get("case"),
                    summary.get("ref_phase"),
                    summary.get("model_type"),
                    summary.get("model_variant"),
                    summary.get("signal"),
                    summary.get("mean_tre_over_phases"),
                    summary.get("mean_tre_over_phases_fwd"),
                    summary.get("mean_tre_over_phases_bwd"),
                    summary.get("mean_tre_world_over_phases"),
                    summary.get("mean_tre_world_over_phases_fwd"),
                    summary.get("mean_tre_world_over_phases_bwd"),
                    summary.get("mean_tre_extreme_over_phases"),
                    summary.get("mean_tre_extreme_over_phases_fwd"),
                    summary.get("mean_tre_extreme_over_phases_bwd"),
                    summary.get("mean_dice_over_phases"),
                    summary.get("mean_map_mae_over_phases"),
                    summary.get("mean_mse_over_phases"),
                    summary.get("mean_assd_vessel_0p5_mm_over_phases"),
                    summary.get("mean_folding_over_phases"),
                    summary.get("mean_detj_std_over_phases"),
                    summary.get("mean_dice_over_phases_bwd"),
                    summary.get("mean_map_mae_over_phases_bwd"),
                    summary.get("mean_mse_over_phases_bwd"),
                    summary.get("mean_assd_vessel_0p5_mm_over_phases_bwd"),
                    summary.get("mean_folding_over_phases_bwd"),
                    summary.get("mean_detj_std_over_phases_bwd"),
                    summary.get("mean_dice_over_phases_fwd"),
                    summary.get("mean_map_mae_over_phases_fwd"),
                    summary.get("mean_mse_over_phases_fwd"),
                    summary.get("mean_assd_vessel_0p5_mm_over_phases_fwd"),
                    summary.get("mean_folding_over_phases_fwd"),
                    summary.get("mean_detj_std_over_phases_fwd"),
                    summary.get("mean_landmark_displacement"),
                    summary.get("mean_landmark_displacement_world"),
                ]
            )


def write_all_runs_phase_summary_csv(rows: list[dict], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "case",
                "phase",
                "model_variant",
                "signal",
                "num_runs",
                "tre_mean",
                "tre_std",
                "tre_world_mean",
                "tre_world_std",
                "tre_extreme_mean",
                "tre_extreme_std",
                "tre_fwd_mean",
                "tre_fwd_std",
                "tre_world_fwd_mean",
                "tre_world_fwd_std",
                "tre_extreme_fwd_mean",
                "tre_extreme_fwd_std",
                "tre_bwd_mean",
                "tre_bwd_std",
                "tre_world_bwd_mean",
                "tre_world_bwd_std",
                "tre_extreme_bwd_mean",
                "tre_extreme_bwd_std",
                "dice_lung_vessels_bwd_mean",
                "dice_lung_vessels_bwd_std",
                "map_mae_bwd_mean",
                "map_mae_bwd_std",
                "mse_bwd_mean",
                "mse_bwd_std",
                "assd_vessel_0p5_mm_bwd_mean",
                "assd_vessel_0p5_mm_bwd_std",
                "folding_percentage_bwd_mean",
                "folding_percentage_bwd_std",
                "detj_std_bwd_mean",
                "detj_std_bwd_std",
                "dice_lung_vessels_fwd_mean",
                "dice_lung_vessels_fwd_std",
                "map_mae_fwd_mean",
                "map_mae_fwd_std",
                "mse_fwd_mean",
                "mse_fwd_std",
                "assd_vessel_0p5_mm_fwd_mean",
                "assd_vessel_0p5_mm_fwd_std",
                "folding_percentage_fwd_mean",
                "folding_percentage_fwd_std",
                "detj_std_fwd_mean",
                "detj_std_fwd_std",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.get("case"),
                    row.get("phase"),
                    row.get("model_variant"),
                    row.get("signal"),
                    row.get("num_runs"),
                    row.get("tre_mean"),
                    row.get("tre_std"),
                    row.get("tre_world_mean"),
                    row.get("tre_world_std"),
                    row.get("tre_extreme_mean"),
                    row.get("tre_extreme_std"),
                    row.get("tre_fwd_mean"),
                    row.get("tre_fwd_std"),
                    row.get("tre_world_fwd_mean"),
                    row.get("tre_world_fwd_std"),
                    row.get("tre_extreme_fwd_mean"),
                    row.get("tre_extreme_fwd_std"),
                    row.get("tre_bwd_mean"),
                    row.get("tre_bwd_std"),
                    row.get("tre_world_bwd_mean"),
                    row.get("tre_world_bwd_std"),
                    row.get("tre_extreme_bwd_mean"),
                    row.get("tre_extreme_bwd_std"),
                    row.get("dice_lung_vessels_bwd_mean"),
                    row.get("dice_lung_vessels_bwd_std"),
                    row.get("map_mae_bwd_mean"),
                    row.get("map_mae_bwd_std"),
                    row.get("mse_bwd_mean"),
                    row.get("mse_bwd_std"),
                    row.get("assd_vessel_0p5_mm_bwd_mean"),
                    row.get("assd_vessel_0p5_mm_bwd_std"),
                    row.get("folding_percentage_bwd_mean"),
                    row.get("folding_percentage_bwd_std"),
                    row.get("detj_std_bwd_mean"),
                    row.get("detj_std_bwd_std"),
                    row.get("dice_lung_vessels_fwd_mean"),
                    row.get("dice_lung_vessels_fwd_std"),
                    row.get("map_mae_fwd_mean"),
                    row.get("map_mae_fwd_std"),
                    row.get("mse_fwd_mean"),
                    row.get("mse_fwd_std"),
                    row.get("assd_vessel_0p5_mm_fwd_mean"),
                    row.get("assd_vessel_0p5_mm_fwd_std"),
                    row.get("folding_percentage_fwd_mean"),
                    row.get("folding_percentage_fwd_std"),
                    row.get("detj_std_fwd_mean"),
                    row.get("detj_std_fwd_std"),
                ]
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrospective evaluation of trained motion-model runs."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--run-folder",
        type=str,
        default=None,
        help="Folder with run directories (e.g. case_XX/*).",
    )
    group.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Evaluate one specific run directory.",
    )
    parser.add_argument(
        "--dirlab-path",
        type=str,
        default=None,
        help="Optional override for DIRLAB root path from config.",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--phase",
        type=int,
        action="append",
        dest="phases",
        help=(
            "Phase index to evaluate. Repeat to evaluate multiple phases only "
            "(e.g. --phase 0 --phase 2)."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Aggregate JSON output path (default: <run-folder>/retrospective_summary.json).",
    )
    parser.add_argument(
        "--csv-output",
        type=str,
        default=None,
        help="Aggregate CSV output path (default: <run-folder>/retrospective_summary.csv).",
    )
    parser.add_argument(
        "--all-runs-output",
        type=str,
        default=None,
        help=(
            "All-runs grouped summary JSON path "
            "(default: <run-folder>/retrospective_all_runs_phase_summary.json)."
        ),
    )
    parser.add_argument(
        "--all-runs-csv-output",
        type=str,
        default=None,
        help=(
            "All-runs grouped summary CSV path "
            "(default: <run-folder>/retrospective_all_runs_phase_summary.csv)."
        ),
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first failing run instead of skipping it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"Requested CUDA device '{args.device}', but CUDA is not available."
        )
    device = torch.device(args.device)

    if args.run_dir is not None:
        run_dirs = [Path(args.run_dir)]
        output_root = Path(args.run_dir)
    else:
        run_folder = Path(args.run_folder)
        if not run_folder.exists():
            raise FileNotFoundError(f"Run folder not found: {run_folder}")
        run_dirs = discover_run_dirs(run_folder)
        output_root = run_folder

    if args.output is None:
        output_json = output_root / "retrospective_summary.json"
    else:
        output_json = Path(args.output)

    if args.csv_output is None:
        output_csv = output_root / "retrospective_summary.csv"
    else:
        output_csv = Path(args.csv_output)

    if args.all_runs_output is None:
        all_runs_output_json = output_root / "retrospective_all_runs_phase_summary.json"
    else:
        all_runs_output_json = Path(args.all_runs_output)

    if args.all_runs_csv_output is None:
        all_runs_output_csv = output_root / "retrospective_all_runs_phase_summary.csv"
    else:
        all_runs_output_csv = Path(args.all_runs_csv_output)

    data_root_override = (
        Path(args.dirlab_path) if args.dirlab_path is not None else None
    )
    selected_phases: set[int] | None = None
    if args.phases:
        selected_phases = {int(phase) for phase in args.phases}
        invalid_phases = sorted(
            [phase for phase in selected_phases if phase < 0 or phase > 9]
        )
        if invalid_phases:
            raise ValueError(
                f"Invalid phase indices {invalid_phases}. Expected values in [0, 9]."
            )

    summaries: list[dict] = []
    all_phase_rows: list[dict] = []
    case_data_cache: dict[tuple[str, int], dict] = {}

    for run_dir in run_dirs:
        config_path = run_dir / "config.yml"
        model_path = run_dir / "models" / "final.pth"
        if not config_path.exists() or not model_path.exists():
            print(f"Skipping {run_dir} (missing config.yml or models/final.pth)")
            continue

        try:
            summary, run_rows = evaluate_single_run(
                run_dir=run_dir,
                device=device,
                data_root_override=data_root_override,
                case_data_cache=case_data_cache,
                selected_phases=selected_phases,
            )
            summaries.append(summary)
            all_phase_rows.extend(run_rows)
        except Exception as exc:  # noqa: BLE001
            print(f"Failed {run_dir}: {exc}")
            if args.fail_fast:
                raise

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w") as f:
        json.dump(summaries, f, indent=2)

    write_summary_csv(summaries=summaries, csv_path=output_csv)
    all_runs_phase_summary = summarize_over_all_runs_phase_groups(all_phase_rows)

    all_runs_output_json.parent.mkdir(parents=True, exist_ok=True)
    with all_runs_output_json.open("w") as f:
        json.dump(all_runs_phase_summary, f, indent=2)
    write_all_runs_phase_summary_csv(
        rows=all_runs_phase_summary,
        csv_path=all_runs_output_csv,
    )

    print(f"Wrote {len(summaries)} run summaries to {output_json}")
    print(f"Wrote CSV summary to {output_csv}")
    print(f"Wrote all-runs grouped summary to {all_runs_output_json}")
    print(f"Wrote all-runs grouped CSV summary to {all_runs_output_csv}")


if __name__ == "__main__":
    main()
