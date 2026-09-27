from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi
import SimpleITK as sitk
from scipy.stats import pearsonr, spearmanr

from inrmm import configs
from inrmm.deformation import jacobian_determinant
from inrmm.utils import (
    compute_tre_dense_dvf,
    load_and_crop_full_dirlab,
)


def _parse_phase_from_filename(path: Path) -> int | None:
    match = re.search(r"phase_(\d{2})", path.name)
    if match is None:
        return None
    return int(match.group(1))


def _collect_dvfs_by_phase(dvf_dir: Path, glob_pattern: str) -> dict[int, Path]:
    phase_to_path: dict[int, Path] = {}
    for path in sorted(dvf_dir.glob(glob_pattern)):
        if "backward" in path.name:
            continue
        phase = _parse_phase_from_filename(path)
        if phase is None:
            continue
        if phase in phase_to_path:
            raise ValueError(
                f"Found multiple DVFs for phase {phase:02d}: {phase_to_path[phase]} and {path}"
            )
        phase_to_path[phase] = path
    return phase_to_path


def _load_dvf_xyzc(path: Path) -> np.ndarray:
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(
            f"Expected DVF with shape (z, y, x, 3), got {arr.shape} for {path}"
        )
    # Convert (z, y, x, 3) to (x, y, z, 3) for internal coordinate convention.
    return np.swapaxes(arr, 0, 2)


def _warp_volume_with_dvf(
    moving_volume: np.ndarray, dvf_xyzc: np.ndarray
) -> np.ndarray:
    if moving_volume.shape != dvf_xyzc.shape[:3]:
        raise ValueError(
            "Shape mismatch for warping: moving volume has shape "
            f"{moving_volume.shape}, DVF has spatial shape {dvf_xyzc.shape[:3]}"
        )

    shape = moving_volume.shape
    grid_x, grid_y, grid_z = np.meshgrid(
        np.arange(shape[0], dtype=np.float32),
        np.arange(shape[1], dtype=np.float32),
        np.arange(shape[2], dtype=np.float32),
        indexing="ij",
    )
    sample_coords = np.stack(
        [
            grid_x + dvf_xyzc[..., 0],
            grid_y + dvf_xyzc[..., 1],
            grid_z + dvf_xyzc[..., 2],
        ],
        axis=0,
    )
    return ndi.map_coordinates(
        moving_volume,
        sample_coords,
        order=1,
        mode="constant",
        cval=0.0,
    )


def _compute_masked_mae(
    warped_map: np.ndarray, fixed_map: np.ndarray, lung_mask: np.ndarray
) -> float:
    if warped_map.shape != fixed_map.shape or fixed_map.shape != lung_mask.shape:
        raise ValueError(
            "Shape mismatch in MAE computation: "
            f"{warped_map.shape=}, {fixed_map.shape=}, {lung_mask.shape=}"
        )
    mask = lung_mask > 0.5
    denom = float(np.sum(mask))
    if denom <= 0.0:
        raise ValueError("Lung mask is empty, cannot compute vessel-map MAE.")
    return float(np.sum(np.abs(warped_map - fixed_map) * mask) / denom)


def _compute_masked_mse(
    warped_img: np.ndarray, fixed_img: np.ndarray, lung_mask: np.ndarray
) -> float:
    if warped_img.shape != fixed_img.shape or fixed_img.shape != lung_mask.shape:
        raise ValueError(
            "Shape mismatch in MSE computation: "
            f"{warped_img.shape=}, {fixed_img.shape=}, {lung_mask.shape=}"
        )
    mask = lung_mask > 0.5
    denom = float(np.sum(mask))
    if denom <= 0.0:
        raise ValueError("Lung mask is empty, cannot compute grayscale MSE.")
    sq_error = (warped_img - fixed_img) ** 2
    return float(np.sum(sq_error * mask) / denom)


def _compute_masked_dice(
    pred_map: np.ndarray,
    target_map: np.ndarray,
    mask: np.ndarray,
    threshold: float = 0.5,
) -> float:
    if pred_map.shape != target_map.shape or target_map.shape != mask.shape:
        raise ValueError(
            "Shape mismatch in Dice computation: "
            f"{pred_map.shape=}, {target_map.shape=}, {mask.shape=}"
        )
    mask_bool = mask > 0.5
    pred_bin = (pred_map >= threshold) & mask_bool
    target_bin = (target_map >= threshold) & mask_bool
    pred_sum = int(np.sum(pred_bin))
    target_sum = int(np.sum(target_bin))
    denom = pred_sum + target_sum
    if denom == 0:
        return 1.0
    intersection = int(np.sum(pred_bin & target_bin))
    return float((2.0 * intersection) / denom)


def _compute_probabilistic_soft_dice(
    pred_map: np.ndarray,
    target_map: np.ndarray,
    mask: np.ndarray,
    epsilon: float = 1e-8,
) -> float:
    if pred_map.shape != target_map.shape or target_map.shape != mask.shape:
        raise ValueError(
            "Shape mismatch in probabilistic soft Dice computation: "
            f"{pred_map.shape=}, {target_map.shape=}, {mask.shape=}"
        )
    mask_float = (mask > 0.5).astype(np.float64)
    pred = np.asarray(pred_map, dtype=np.float64) * mask_float
    target = np.asarray(target_map, dtype=np.float64) * mask_float
    numerator = 2.0 * np.sum(pred * target)
    denominator = np.sum(pred) + np.sum(target)
    if denominator <= epsilon:
        return 1.0
    return float((numerator + epsilon) / (denominator + epsilon))


def _morphological_skeleton_3d(binary_mask: np.ndarray) -> np.ndarray:
    if binary_mask.ndim != 3:
        raise ValueError(f"Expected 3D binary mask, got shape {binary_mask.shape}")
    structure = ndi.generate_binary_structure(rank=3, connectivity=1)
    working = binary_mask.astype(bool).copy()
    skeleton = np.zeros_like(working, dtype=bool)
    while np.any(working):
        eroded = ndi.binary_erosion(working, structure=structure, border_value=0)
        opened = ndi.binary_dilation(eroded, structure=structure, border_value=0)
        skeleton |= working & (~opened)
        working = eroded
    return skeleton


def _compute_centerline_dice(
    pred_map: np.ndarray,
    target_map: np.ndarray,
    mask: np.ndarray,
    threshold: float = 0.5,
) -> float:
    if pred_map.shape != target_map.shape or target_map.shape != mask.shape:
        raise ValueError(
            "Shape mismatch in centerline Dice computation: "
            f"{pred_map.shape=}, {target_map.shape=}, {mask.shape=}"
        )
    mask_bool = mask > 0.5
    pred_bin = (pred_map >= threshold) & mask_bool
    target_bin = (target_map >= threshold) & mask_bool

    if int(np.sum(pred_bin)) == 0 and int(np.sum(target_bin)) == 0:
        return 1.0

    pred_cl = _morphological_skeleton_3d(pred_bin)
    target_cl = _morphological_skeleton_3d(target_bin)

    pred_cl_count = int(np.sum(pred_cl))
    target_cl_count = int(np.sum(target_cl))
    if pred_cl_count == 0 or target_cl_count == 0:
        return 0.0

    tprec = float(np.sum(pred_cl & target_bin) / pred_cl_count)
    tsens = float(np.sum(target_cl & pred_bin) / target_cl_count)
    if tprec + tsens == 0.0:
        return 0.0
    return float((2.0 * tprec * tsens) / (tprec + tsens))


def _compute_weighted_mae(
    warped_map: np.ndarray,
    fixed_map: np.ndarray,
    weights: np.ndarray,
    mask: np.ndarray | None = None,
) -> float:
    if warped_map.shape != fixed_map.shape or fixed_map.shape != weights.shape:
        raise ValueError(
            "Shape mismatch in weighted MAE computation: "
            f"{warped_map.shape=}, {fixed_map.shape=}, {weights.shape=}"
        )
    w = np.asarray(weights, dtype=np.float64).copy()
    w = np.clip(w, a_min=0.0, a_max=None)
    if mask is not None:
        if mask.shape != w.shape:
            raise ValueError(
                f"Mask shape mismatch in weighted MAE: {mask.shape=} vs {w.shape=}"
            )
        w *= (mask > 0.5).astype(np.float64)
    denom = float(np.sum(w))
    if denom <= 0.0:
        return float("nan")
    diff = np.abs(fixed_map - warped_map).astype(np.float64)
    return float(np.sum(w * diff) / denom)


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


def _safe_nanmean(values: np.ndarray) -> float:
    valid = np.isfinite(values)
    if not np.any(valid):
        return float("nan")
    return float(np.mean(values[valid]))


def _safe_corr(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.size < 2 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan"), float("nan"), float("nan"), float("nan")
    pearson = pearsonr(x, y)
    spearman = spearmanr(x, y)
    return (
        float(pearson.statistic),
        float(pearson.pvalue),
        float(spearman.statistic),
        float(spearman.pvalue),
    )


def _compute_correlation_rows(rows: list[dict]) -> list[dict]:
    by_folder: dict[str, list[dict]] = {}
    for row in rows:
        by_folder.setdefault(str(row["dvf_folder"]), []).append(row)

    summary_rows: list[dict] = []
    for folder, folder_rows in sorted(by_folder.items()):
        tre = np.array(
            [float(r["tre_extreme_mean_mm"]) for r in folder_rows], dtype=np.float64
        )
        mae_lung = np.array(
            [float(r["mae_lung"]) for r in folder_rows], dtype=np.float64
        )
        mae_vessel_support = np.array(
            [float(r["mae_vessel_support"]) for r in folder_rows], dtype=np.float64
        )
        mae_delta_vs_zero = np.array(
            [float(r["mae_delta_vs_zero_dvf"]) for r in folder_rows], dtype=np.float64
        )
        dice_vessel = np.array(
            [float(r["dice_vessel_0p5"]) for r in folder_rows], dtype=np.float64
        )
        hd95_vessel = np.array(
            [float(r["hd95_vessel_0p5_mm"]) for r in folder_rows], dtype=np.float64
        )
        assd_vessel = np.array(
            [float(r["assd_vessel_0p5_mm"]) for r in folder_rows], dtype=np.float64
        )
        centerline_dice = np.array(
            [float(r["centerline_dice_0p5"]) for r in folder_rows], dtype=np.float64
        )
        weighted_mae_prob = np.array(
            [float(r["weighted_mae_prob"]) for r in folder_rows], dtype=np.float64
        )
        mse_gray = np.array(
            [float(r["mse_gray"]) for r in folder_rows], dtype=np.float64
        )
        soft_dice_prob = np.array(
            [float(r["soft_dice_prob"]) for r in folder_rows], dtype=np.float64
        )
        folding = np.array([float(r["folding_percentage"]) for r in folder_rows])
        detj_std = np.array([float(r["detj_std"]) for r in folder_rows])
        finite_tre = np.isfinite(tre)
        tre = tre[finite_tre]
        mae_lung_for_tre = mae_lung[finite_tre]
        mae_vessel_for_tre = mae_vessel_support[finite_tre]
        mae_delta_for_tre = mae_delta_vs_zero[finite_tre]
        dice_for_tre = dice_vessel[finite_tre]
        hd95_for_tre = hd95_vessel[finite_tre]
        assd_for_tre = assd_vessel[finite_tre]
        centerline_for_tre = centerline_dice[finite_tre]
        weighted_mae_for_tre = weighted_mae_prob[finite_tre]
        mse_for_tre = mse_gray[finite_tre]
        soft_dice_for_tre = soft_dice_prob[finite_tre]
        n = int(len(folder_rows))

        (
            pearson_r_tre_vs_mae_lung,
            pearson_p_tre_vs_mae_lung,
            spearman_r_tre_vs_mae_lung,
            spearman_p_tre_vs_mae_lung,
        ) = _safe_corr(tre, mae_lung_for_tre)
        (
            pearson_r_tre_vs_mae_vessel,
            pearson_p_tre_vs_mae_vessel,
            spearman_r_tre_vs_mae_vessel,
            spearman_p_tre_vs_mae_vessel,
        ) = _safe_corr(tre, mae_vessel_for_tre)
        (
            pearson_r_tre_vs_mae_delta,
            pearson_p_tre_vs_mae_delta,
            spearman_r_tre_vs_mae_delta,
            spearman_p_tre_vs_mae_delta,
        ) = _safe_corr(tre, mae_delta_for_tre)
        (
            pearson_r_tre_vs_dice,
            pearson_p_tre_vs_dice,
            spearman_r_tre_vs_dice,
            spearman_p_tre_vs_dice,
        ) = _safe_corr(tre, dice_for_tre)
        (
            pearson_r_tre_vs_hd95,
            pearson_p_tre_vs_hd95,
            spearman_r_tre_vs_hd95,
            spearman_p_tre_vs_hd95,
        ) = _safe_corr(tre, hd95_for_tre)
        (
            pearson_r_tre_vs_assd,
            pearson_p_tre_vs_assd,
            spearman_r_tre_vs_assd,
            spearman_p_tre_vs_assd,
        ) = _safe_corr(tre, assd_for_tre)
        (
            pearson_r_tre_vs_centerline_dice,
            pearson_p_tre_vs_centerline_dice,
            spearman_r_tre_vs_centerline_dice,
            spearman_p_tre_vs_centerline_dice,
        ) = _safe_corr(tre, centerline_for_tre)
        (
            pearson_r_tre_vs_weighted_mae_prob,
            pearson_p_tre_vs_weighted_mae_prob,
            spearman_r_tre_vs_weighted_mae_prob,
            spearman_p_tre_vs_weighted_mae_prob,
        ) = _safe_corr(tre, weighted_mae_for_tre)
        (
            pearson_r_tre_vs_mse_gray,
            pearson_p_tre_vs_mse_gray,
            spearman_r_tre_vs_mse_gray,
            spearman_p_tre_vs_mse_gray,
        ) = _safe_corr(tre, mse_for_tre)
        (
            pearson_r_tre_vs_soft_dice_prob,
            pearson_p_tre_vs_soft_dice_prob,
            spearman_r_tre_vs_soft_dice_prob,
            spearman_p_tre_vs_soft_dice_prob,
        ) = _safe_corr(tre, soft_dice_for_tre)

        summary_rows.append(
            {
                "dvf_folder": folder,
                "num_samples": n,
                "tre_extreme_mean_mm_over_samples": float(np.mean(tre)),
                "folding_percentage_mean_over_samples": float(np.mean(folding)),
                "detj_std_mean_over_samples": float(np.mean(detj_std)),
                "mae_lung_mean_over_samples": float(np.mean(mae_lung)),
                "mae_vessel_support_mean_over_samples": float(
                    _safe_nanmean(mae_vessel_support)
                ),
                "mae_delta_vs_zero_dvf_mean_over_samples": float(
                    np.mean(mae_delta_vs_zero)
                ),
                "dice_vessel_0p5_mean_over_samples": float(np.mean(dice_vessel)),
                "hd95_vessel_0p5_mm_mean_over_samples": float(
                    _safe_nanmean(hd95_vessel)
                ),
                "assd_vessel_0p5_mm_mean_over_samples": float(
                    _safe_nanmean(assd_vessel)
                ),
                "centerline_dice_0p5_mean_over_samples": float(
                    _safe_nanmean(centerline_dice)
                ),
                "weighted_mae_prob_mean_over_samples": float(
                    _safe_nanmean(weighted_mae_prob)
                ),
                "mse_gray_mean_over_samples": float(_safe_nanmean(mse_gray)),
                "soft_dice_prob_mean_over_samples": float(
                    _safe_nanmean(soft_dice_prob)
                ),
                "pearson_r_tre_vs_mae_lung": pearson_r_tre_vs_mae_lung,
                "pearson_p_tre_vs_mae_lung": pearson_p_tre_vs_mae_lung,
                "spearman_r_tre_vs_mae_lung": spearman_r_tre_vs_mae_lung,
                "spearman_p_tre_vs_mae_lung": spearman_p_tre_vs_mae_lung,
                "pearson_r_tre_vs_mae_vessel_support": pearson_r_tre_vs_mae_vessel,
                "pearson_p_tre_vs_mae_vessel_support": pearson_p_tre_vs_mae_vessel,
                "spearman_r_tre_vs_mae_vessel_support": spearman_r_tre_vs_mae_vessel,
                "spearman_p_tre_vs_mae_vessel_support": spearman_p_tre_vs_mae_vessel,
                "pearson_r_tre_vs_mae_delta_vs_zero_dvf": pearson_r_tre_vs_mae_delta,
                "pearson_p_tre_vs_mae_delta_vs_zero_dvf": pearson_p_tre_vs_mae_delta,
                "spearman_r_tre_vs_mae_delta_vs_zero_dvf": spearman_r_tre_vs_mae_delta,
                "spearman_p_tre_vs_mae_delta_vs_zero_dvf": spearman_p_tre_vs_mae_delta,
                "pearson_r_tre_vs_dice_vessel_0p5": pearson_r_tre_vs_dice,
                "pearson_p_tre_vs_dice_vessel_0p5": pearson_p_tre_vs_dice,
                "spearman_r_tre_vs_dice_vessel_0p5": spearman_r_tre_vs_dice,
                "spearman_p_tre_vs_dice_vessel_0p5": spearman_p_tre_vs_dice,
                "pearson_r_tre_vs_hd95_vessel_0p5_mm": pearson_r_tre_vs_hd95,
                "pearson_p_tre_vs_hd95_vessel_0p5_mm": pearson_p_tre_vs_hd95,
                "spearman_r_tre_vs_hd95_vessel_0p5_mm": spearman_r_tre_vs_hd95,
                "spearman_p_tre_vs_hd95_vessel_0p5_mm": spearman_p_tre_vs_hd95,
                "pearson_r_tre_vs_assd_vessel_0p5_mm": pearson_r_tre_vs_assd,
                "pearson_p_tre_vs_assd_vessel_0p5_mm": pearson_p_tre_vs_assd,
                "spearman_r_tre_vs_assd_vessel_0p5_mm": spearman_r_tre_vs_assd,
                "spearman_p_tre_vs_assd_vessel_0p5_mm": spearman_p_tre_vs_assd,
                "pearson_r_tre_vs_centerline_dice_0p5": pearson_r_tre_vs_centerline_dice,
                "pearson_p_tre_vs_centerline_dice_0p5": pearson_p_tre_vs_centerline_dice,
                "spearman_r_tre_vs_centerline_dice_0p5": spearman_r_tre_vs_centerline_dice,
                "spearman_p_tre_vs_centerline_dice_0p5": spearman_p_tre_vs_centerline_dice,
                "pearson_r_tre_vs_weighted_mae_prob": pearson_r_tre_vs_weighted_mae_prob,
                "pearson_p_tre_vs_weighted_mae_prob": pearson_p_tre_vs_weighted_mae_prob,
                "spearman_r_tre_vs_weighted_mae_prob": spearman_r_tre_vs_weighted_mae_prob,
                "spearman_p_tre_vs_weighted_mae_prob": spearman_p_tre_vs_weighted_mae_prob,
                "pearson_r_tre_vs_mse_gray": pearson_r_tre_vs_mse_gray,
                "pearson_p_tre_vs_mse_gray": pearson_p_tre_vs_mse_gray,
                "spearman_r_tre_vs_mse_gray": spearman_r_tre_vs_mse_gray,
                "spearman_p_tre_vs_mse_gray": spearman_p_tre_vs_mse_gray,
                "pearson_r_tre_vs_soft_dice_prob": pearson_r_tre_vs_soft_dice_prob,
                "pearson_p_tre_vs_soft_dice_prob": pearson_p_tre_vs_soft_dice_prob,
                "spearman_r_tre_vs_soft_dice_prob": spearman_r_tre_vs_soft_dice_prob,
                "spearman_p_tre_vs_soft_dice_prob": spearman_p_tre_vs_soft_dice_prob,
            }
        )
    return summary_rows


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="default")
    parser.add_argument("--cases", type=int, nargs="+", default=list(range(1, 11)))
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["dirlab", "copdgene"],
        default="dirlab",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help=(
            "Dataset root containing case folders. "
            "If omitted, uses config path (dirlab_path / copdgene_path)."
        ),
    )
    parser.add_argument("--dvf-folders", type=str, nargs="+", required=True)
    parser.add_argument("--reference-phase", type=int, default=5)
    parser.add_argument("--glob", type=str, default="*phase_*.nii*")
    parser.add_argument(
        "--vessel-support-threshold",
        type=float,
        default=0.2,
        help="Threshold on reference vessel probability map for vessel-support MAE.",
    )
    parser.add_argument(
        "--vessel-binary-threshold",
        type=float,
        default=0.5,
        help="Threshold for binary vessel-mask metrics (Dice/HD95/ASSD).",
    )
    parser.add_argument(
        "--include-reference-phase",
        action="store_true",
        help="Also evaluate the reference phase if a DVF file exists.",
    )
    parser.add_argument(
        "--crop",
        action="store_true",
        help="Load cropped data (default uses full size, matching padded DVFs).",
    )
    parser.add_argument(
        "--skip-missing-dvf-folder",
        action="store_true",
        help="Skip (instead of fail) when a requested DVF subfolder is missing for a case.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="analysis/dir_extreme_tre_vessel_mae.csv",
    )
    parser.add_argument(
        "--summary-csv",
        type=str,
        default="analysis/dir_extreme_tre_vessel_mae_summary.csv",
    )
    args = parser.parse_args()

    cfg = configs.configs[args.config]
    cfg_paths = cfg["paths"]
    if args.data_root is not None:
        data_root = Path(args.data_root)
    elif args.dataset == "dirlab":
        data_root = Path(cfg_paths["dirlab_path"])
    else:
        copdgene_path = cfg_paths.get("copdgene_path")
        if copdgene_path is None:
            raise ValueError(
                "For --dataset copdgene, provide --data-root or set paths.copdgene_path "
                "in the selected config."
            )
        data_root = Path(copdgene_path)

    phases = list(range(10)) if args.dataset == "dirlab" else [0, 5]

    rows: list[dict] = []
    for case in args.cases:
        case_folder = data_root / f"case_{case:02d}"

        data = load_and_crop_full_dirlab(
            case_folder=case_folder,
            phases=phases,
            crop=args.crop,
        )

        phase_to_index = {phase: idx for idx, phase in enumerate(phases)}
        if args.reference_phase not in phase_to_index:
            raise ValueError(
                f"reference-phase={args.reference_phase} is not part of selected phases {phases}"
            )

        image_spacing = np.asarray(data["image_spacing"], dtype=np.float64).reshape(-1)
        if image_spacing.size != 3:
            raise ValueError(
                f"Expected image spacing with 3 entries, got {image_spacing}."
            )

        extreme_landmarks = data["extreme_landmarks"]
        if args.reference_phase not in extreme_landmarks:
            print(
                f"Case {case:02d}: missing extreme landmarks for reference phase "
                f"{args.reference_phase:02d}, skipping case."
            )
            continue

        vessel_maps = data["vessel_maps"]
        lung_masks = data["lung_masks"]
        images = data["images"]
        ref_idx = phase_to_index[args.reference_phase]
        fixed_map_ref = np.asarray(vessel_maps[ref_idx], dtype=np.float32)
        fixed_lung_mask_ref = np.asarray(lung_masks[ref_idx], dtype=np.float32)
        fixed_image_ref = np.asarray(images[ref_idx], dtype=np.float32)

        print(f"Dataset={args.dataset} | Case {case:02d} | phases={phases}")
        for dvf_folder in args.dvf_folders:
            dvf_dir = case_folder / dvf_folder
            if not dvf_dir.exists():
                if args.skip_missing_dvf_folder:
                    print(f"  {dvf_folder}: missing, skipped.")
                    continue
                raise FileNotFoundError(f"DVF directory does not exist: {dvf_dir}")

            phase_to_dvf = _collect_dvfs_by_phase(
                dvf_dir=dvf_dir, glob_pattern=args.glob
            )
            if not phase_to_dvf:
                print(
                    f"  {dvf_folder}: no matching DVFs found with glob '{args.glob}'."
                )
                continue

            for phase, dvf_path in sorted(phase_to_dvf.items()):
                if phase == args.reference_phase and not args.include_reference_phase:
                    continue
                if phase not in phase_to_index:
                    continue
                dvf_xyzc = _load_dvf_xyzc(dvf_path)
                if dvf_xyzc.shape[:3] != fixed_map_ref.shape:
                    raise ValueError(
                        f"Shape mismatch for {dvf_path}: dvf {dvf_xyzc.shape[:3]} vs "
                        f"reference vessel map {fixed_map_ref.shape}. "
                        "If DVFs are padded to full image size, do not use --crop."
                    )

                vector_field_chw = np.moveaxis(dvf_xyzc, -1, 0)
                if phase in extreme_landmarks:
                    tre_ext, _ = compute_tre_dense_dvf(
                        moving_landmarks=extreme_landmarks[phase],
                        fixed_landmarks=extreme_landmarks[args.reference_phase],
                        vector_field=vector_field_chw,
                        image_spacing=image_spacing,
                        snap_to_voxel=True,
                    )
                else:
                    tre_ext = np.asarray([], dtype=np.float64)

                det_j = jacobian_determinant(vector_field_chw[None, ...])
                jacobian_lung_mask = fixed_lung_mask_ref[2:-2, 2:-2, 2:-2] > 0.5
                det_j_lung = det_j[jacobian_lung_mask]
                folding_percentage = float(100.0 * np.mean(det_j_lung <= 0))
                detj_std = float(np.std(det_j_lung))

                # DVF is fixed(reference) -> moving(phase), so for image comparison in
                # fixed space we must sample the moving phase map on the reference grid.
                moving_idx = phase_to_index[phase]
                moving_vessel_map = np.asarray(
                    vessel_maps[moving_idx], dtype=np.float32
                )
                moving_image = np.asarray(images[moving_idx], dtype=np.float32)
                warped_vessel_map = _warp_volume_with_dvf(
                    moving_volume=moving_vessel_map,
                    dvf_xyzc=dvf_xyzc,
                )
                warped_image = _warp_volume_with_dvf(
                    moving_volume=moving_image,
                    dvf_xyzc=dvf_xyzc,
                )
                mae_lung = _compute_masked_mae(
                    warped_map=warped_vessel_map,
                    fixed_map=fixed_map_ref,
                    lung_mask=fixed_lung_mask_ref,
                )
                mae_lung_zero_dvf = _compute_masked_mae(
                    warped_map=moving_vessel_map,
                    fixed_map=fixed_map_ref,
                    lung_mask=fixed_lung_mask_ref,
                )
                mae_delta_vs_zero_dvf = mae_lung_zero_dvf - mae_lung

                vessel_support_mask = (
                    (fixed_map_ref >= args.vessel_support_threshold)
                    & (fixed_lung_mask_ref > 0.5)
                ).astype(np.float32)
                if np.sum(vessel_support_mask) > 0:
                    mae_vessel_support = _compute_masked_mae(
                        warped_map=warped_vessel_map,
                        fixed_map=fixed_map_ref,
                        lung_mask=vessel_support_mask,
                    )
                else:
                    mae_vessel_support = float("nan")

                dice_vessel_0p5 = _compute_masked_dice(
                    pred_map=warped_vessel_map,
                    target_map=fixed_map_ref,
                    mask=fixed_lung_mask_ref,
                    threshold=args.vessel_binary_threshold,
                )
                hd95_vessel_mm, assd_vessel_mm = _compute_hd95_assd(
                    pred_map=warped_vessel_map,
                    target_map=fixed_map_ref,
                    mask=fixed_lung_mask_ref,
                    image_spacing=image_spacing,
                    threshold=args.vessel_binary_threshold,
                )
                centerline_dice = _compute_centerline_dice(
                    pred_map=warped_vessel_map,
                    target_map=fixed_map_ref,
                    mask=fixed_lung_mask_ref,
                    threshold=args.vessel_binary_threshold,
                )
                weighted_mae_prob = _compute_weighted_mae(
                    warped_map=warped_vessel_map,
                    fixed_map=fixed_map_ref,
                    weights=fixed_map_ref,
                    mask=fixed_lung_mask_ref,
                )
                soft_dice_prob = _compute_probabilistic_soft_dice(
                    pred_map=warped_vessel_map,
                    target_map=fixed_map_ref,
                    mask=fixed_lung_mask_ref,
                )
                mse_gray = _compute_masked_mse(
                    warped_img=warped_image,
                    fixed_img=fixed_image_ref,
                    lung_mask=fixed_lung_mask_ref,
                )

                row = {
                    "case": case,
                    "dvf_folder": dvf_folder,
                    "phase": phase,
                    "reference_phase": args.reference_phase,
                    "dvf_file": str(dvf_path),
                    "num_extreme_landmarks": int(len(tre_ext)),
                    "tre_extreme_mean_mm": float(np.mean(tre_ext))
                    if len(tre_ext)
                    else float("nan"),
                    "tre_extreme_std_mm": float(np.std(tre_ext))
                    if len(tre_ext)
                    else float("nan"),
                    "folding_percentage": folding_percentage,
                    "detj_std": detj_std,
                    "mae_lung": mae_lung,
                    "mae_lung_zero_dvf": mae_lung_zero_dvf,
                    "mae_delta_vs_zero_dvf": mae_delta_vs_zero_dvf,
                    "mae_vessel_support": mae_vessel_support,
                    "dice_vessel_0p5": dice_vessel_0p5,
                    "hd95_vessel_0p5_mm": hd95_vessel_mm,
                    "assd_vessel_0p5_mm": assd_vessel_mm,
                    "centerline_dice_0p5": centerline_dice,
                    "weighted_mae_prob": weighted_mae_prob,
                    "soft_dice_prob": soft_dice_prob,
                    "mse_gray": mse_gray,
                }
                rows.append(row)
                print(
                    f"  {dvf_folder} phase {phase:02d}: "
                    f"TRE(ext)={row['tre_extreme_mean_mm']:.4f} mm, "
                    f"MAE(lung)={row['mae_lung']:.6f}, "
                    f"MAE(vessel)={row['mae_vessel_support']:.6f}, "
                    f"DeltaMAE={row['mae_delta_vs_zero_dvf']:.6f}, "
                    f"Dice={row['dice_vessel_0p5']:.4f}, "
                    f"HD95={row['hd95_vessel_0p5_mm']:.4f} mm, "
                    f"ASSD={row['assd_vessel_0p5_mm']:.4f} mm, "
                    f"CL-Dice={row['centerline_dice_0p5']:.4f}, "
                    f"WMAE={row['weighted_mae_prob']:.6f}, "
                    f"SoftDice={row['soft_dice_prob']:.6f}, "
                    f"MSE(gray)={row['mse_gray']:.6f}"
                )

    if not rows:
        raise RuntimeError(
            "No evaluation rows were produced. Check DVF folders/phases."
        )

    output_csv = Path(args.output_csv)
    summary_csv = Path(args.summary_csv)
    row_fields = [
        "case",
        "dvf_folder",
        "phase",
        "reference_phase",
        "dvf_file",
        "num_extreme_landmarks",
        "tre_extreme_mean_mm",
        "tre_extreme_std_mm",
        "folding_percentage",
        "detj_std",
        "mae_lung",
        "mae_lung_zero_dvf",
        "mae_delta_vs_zero_dvf",
        "mae_vessel_support",
        "dice_vessel_0p5",
        "hd95_vessel_0p5_mm",
        "assd_vessel_0p5_mm",
        "centerline_dice_0p5",
        "weighted_mae_prob",
        "soft_dice_prob",
        "mse_gray",
    ]
    _write_csv(path=output_csv, rows=rows, fieldnames=row_fields)

    summary_rows = _compute_correlation_rows(rows)
    summary_fields = [
        "dvf_folder",
        "num_samples",
        "tre_extreme_mean_mm_over_samples",
        "folding_percentage_mean_over_samples",
        "detj_std_mean_over_samples",
        "mae_lung_mean_over_samples",
        "mae_vessel_support_mean_over_samples",
        "mae_delta_vs_zero_dvf_mean_over_samples",
        "dice_vessel_0p5_mean_over_samples",
        "hd95_vessel_0p5_mm_mean_over_samples",
        "assd_vessel_0p5_mm_mean_over_samples",
        "centerline_dice_0p5_mean_over_samples",
        "weighted_mae_prob_mean_over_samples",
        "soft_dice_prob_mean_over_samples",
        "mse_gray_mean_over_samples",
        "pearson_r_tre_vs_mae_lung",
        "pearson_p_tre_vs_mae_lung",
        "spearman_r_tre_vs_mae_lung",
        "spearman_p_tre_vs_mae_lung",
        "pearson_r_tre_vs_mae_vessel_support",
        "pearson_p_tre_vs_mae_vessel_support",
        "spearman_r_tre_vs_mae_vessel_support",
        "spearman_p_tre_vs_mae_vessel_support",
        "pearson_r_tre_vs_mae_delta_vs_zero_dvf",
        "pearson_p_tre_vs_mae_delta_vs_zero_dvf",
        "spearman_r_tre_vs_mae_delta_vs_zero_dvf",
        "spearman_p_tre_vs_mae_delta_vs_zero_dvf",
        "pearson_r_tre_vs_dice_vessel_0p5",
        "pearson_p_tre_vs_dice_vessel_0p5",
        "spearman_r_tre_vs_dice_vessel_0p5",
        "spearman_p_tre_vs_dice_vessel_0p5",
        "pearson_r_tre_vs_hd95_vessel_0p5_mm",
        "pearson_p_tre_vs_hd95_vessel_0p5_mm",
        "spearman_r_tre_vs_hd95_vessel_0p5_mm",
        "spearman_p_tre_vs_hd95_vessel_0p5_mm",
        "pearson_r_tre_vs_assd_vessel_0p5_mm",
        "pearson_p_tre_vs_assd_vessel_0p5_mm",
        "spearman_r_tre_vs_assd_vessel_0p5_mm",
        "spearman_p_tre_vs_assd_vessel_0p5_mm",
        "pearson_r_tre_vs_centerline_dice_0p5",
        "pearson_p_tre_vs_centerline_dice_0p5",
        "spearman_r_tre_vs_centerline_dice_0p5",
        "spearman_p_tre_vs_centerline_dice_0p5",
        "pearson_r_tre_vs_weighted_mae_prob",
        "pearson_p_tre_vs_weighted_mae_prob",
        "spearman_r_tre_vs_weighted_mae_prob",
        "spearman_p_tre_vs_weighted_mae_prob",
        "pearson_r_tre_vs_mse_gray",
        "pearson_p_tre_vs_mse_gray",
        "spearman_r_tre_vs_mse_gray",
        "spearman_p_tre_vs_mse_gray",
        "pearson_r_tre_vs_soft_dice_prob",
        "pearson_p_tre_vs_soft_dice_prob",
        "spearman_r_tre_vs_soft_dice_prob",
        "spearman_p_tre_vs_soft_dice_prob",
    ]
    _write_csv(path=summary_csv, rows=summary_rows, fieldnames=summary_fields)

    print(f"Wrote detailed results: {output_csv}")
    print(f"Wrote correlation summary: {summary_csv}")
    for row in summary_rows:
        print(
            f"{row['dvf_folder']}: n={row['num_samples']}, "
            f"r(TRE,MAE_lung)={row['pearson_r_tre_vs_mae_lung']:.4f}, "
            f"r(TRE,MAE_vessel)={row['pearson_r_tre_vs_mae_vessel_support']:.4f}, "
            f"r(TRE,DeltaMAE)={row['pearson_r_tre_vs_mae_delta_vs_zero_dvf']:.4f}, "
            f"r(TRE,Dice)={row['pearson_r_tre_vs_dice_vessel_0p5']:.4f}, "
            f"r(TRE,HD95)={row['pearson_r_tre_vs_hd95_vessel_0p5_mm']:.4f}, "
            f"r(TRE,ASSD)={row['pearson_r_tre_vs_assd_vessel_0p5_mm']:.4f}, "
            f"r(TRE,CL-Dice)={row['pearson_r_tre_vs_centerline_dice_0p5']:.4f}, "
            f"r(TRE,WMAE)={row['pearson_r_tre_vs_weighted_mae_prob']:.4f}, "
            f"r(TRE,MSEgray)={row['pearson_r_tre_vs_mse_gray']:.4f}, "
            f"r(TRE,SoftDice)={row['pearson_r_tre_vs_soft_dice_prob']:.4f}"
        )


if __name__ == "__main__":
    main()
