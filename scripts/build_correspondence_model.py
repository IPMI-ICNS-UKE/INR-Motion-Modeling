import argparse
import logging
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import torch
import scipy.ndimage as ndi
from inrmm.compat import init_fancy_logging
from inrmm.deformation import jacobian_determinant

from inrmm import configs
from inrmm.correspondence import CorrespondenceModel
from inrmm.surrogate import load_lung_volume_surrogate
from inrmm.utils import compute_tre_dense_dvf, load_and_crop_full_dirlab


def _save_dvf(
    dvf_vox: np.ndarray,
    image_shape: tuple[int, int, int],
    image_spacing: tuple[float, float, float],
    output_path: Path,
) -> None:
    dvf_sitk = sitk.GetImageFromArray(np.swapaxes(dvf_vox, 0, 2), isVector=True)
    dvf_sitk.SetSpacing(tuple(image_spacing))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(dvf_sitk, str(output_path))


def _pad_dvf_to_full(
    dvf_crop: np.ndarray,
    bbox: tuple[slice, slice, slice],
    full_shape: tuple[int, int, int],
) -> np.ndarray:
    full_dvf = np.zeros(full_shape + (3,), dtype=dvf_crop.dtype)
    full_dvf[bbox] = dvf_crop
    return full_dvf


def _dice_score_numpy(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    eps: float = 1e-8,
) -> float:
    y_pred_f = y_pred.astype(np.float32).reshape(-1)
    y_true_f = y_true.astype(np.float32).reshape(-1)
    intersection = float(np.sum(y_pred_f * y_true_f))
    denom = float(np.sum(y_pred_f) + np.sum(y_true_f))
    return (2.0 * intersection + eps) / (denom + eps)


def _warp_volume_with_voxel_dvf(
    moving_volume: np.ndarray,
    vector_field_chw: np.ndarray,
) -> np.ndarray:
    if vector_field_chw.shape[0] != 3:
        raise ValueError(
            "vector_field_chw must have channel-first shape (3, D, H, W), "
            f"got {vector_field_chw.shape}"
        )
    d, h, w = moving_volume.shape
    grid_d, grid_h, grid_w = np.meshgrid(
        np.arange(d, dtype=np.float32),
        np.arange(h, dtype=np.float32),
        np.arange(w, dtype=np.float32),
        indexing="ij",
    )

    sample_d = grid_d + vector_field_chw[0]
    sample_h = grid_h + vector_field_chw[1]
    sample_w = grid_w + vector_field_chw[2]
    coords = np.stack([sample_d, sample_h, sample_w], axis=0)

    warped = ndi.map_coordinates(
        moving_volume,
        coords,
        order=1,
        mode="constant",
        cval=0.0,
    )
    return warped.astype(np.float32, copy=False)


def _evaluate_forward_tre(
    model: CorrespondenceModel,
    signal: np.ndarray,
    landmarks: dict[int, np.ndarray],
    reference_phase: int,
    image_spacing: tuple[float, float, float],
    logger: logging.Logger,
) -> dict:
    if reference_phase not in landmarks:
        raise ValueError(
            f"Reference phase {reference_phase} not found in available landmarks: "
            f"{sorted(landmarks.keys())}"
        )

    ref_landmarks = landmarks[reference_phase]
    per_phase = []
    all_tre = []
    all_tre_world = []
    for phase in sorted(landmarks.keys()):
        vector_field = model.predict(signal[phase])
        tre, _ = compute_tre_dense_dvf(
            moving_landmarks=landmarks[phase],
            fixed_landmarks=ref_landmarks,
            vector_field=vector_field,
            image_spacing=image_spacing,
            snap_to_voxel=True,
        )
        tre_world, _ = compute_tre_dense_dvf(
            moving_landmarks=landmarks[phase],
            fixed_landmarks=ref_landmarks,
            vector_field=vector_field,
            image_spacing=image_spacing,
            snap_to_voxel=False,
        )
        tre_mean = float(np.mean(tre))
        tre_std = float(np.std(tre))
        tre_world_mean = float(np.mean(tre_world))
        tre_world_std = float(np.std(tre_world))
        if phase != reference_phase:
            all_tre.append(tre)
            all_tre_world.append(tre_world)
        per_phase.append(
            {
                "phase": phase,
                "tre_mean_mm": tre_mean,
                "tre_std_mm": tre_std,
                "tre_world_mean_mm": tre_world_mean,
                "tre_world_std_mm": tre_world_std,
                "n_landmarks": int(tre.shape[0]),
            }
        )
        logger.info(
            "Forward phase %02d: TRE=%.4f±%.4f mm | TRE-world=%.4f±%.4f mm",
            phase,
            tre_mean,
            tre_std,
            tre_world_mean,
            tre_world_std,
        )

    aggregate = {}
    if all_tre and all_tre_world:
        all_tre_flat = np.concatenate(all_tre, axis=0)
        all_tre_world_flat = np.concatenate(all_tre_world, axis=0)
        aggregate = {
            "tre_mean_mm": float(np.mean(all_tre_flat)),
            "tre_std_mm": float(np.std(all_tre_flat)),
            "tre_world_mean_mm": float(np.mean(all_tre_world_flat)),
            "tre_world_std_mm": float(np.std(all_tre_world_flat)),
            "n_landmarks_total": int(all_tre_flat.shape[0]),
            "excluded_phase": int(reference_phase),
        }
        logger.info(
            "Forward aggregate (excluding reference phase %02d): "
            "TRE=%.4f±%.4f mm | TRE-world=%.4f±%.4f mm",
            reference_phase,
            aggregate["tre_mean_mm"],
            aggregate["tre_std_mm"],
            aggregate["tre_world_mean_mm"],
            aggregate["tre_world_std_mm"],
        )

    return {"per_phase": per_phase, "aggregate": aggregate}


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


def _compute_jacobian_metrics(
    vector_field_chw: np.ndarray,
    lung_mask: np.ndarray,
) -> tuple[float, float]:
    if vector_field_chw.ndim != 4 or vector_field_chw.shape[0] != 3:
        raise ValueError(
            "Expected vector_field_chw with shape (3, D, H, W), "
            f"got {vector_field_chw.shape}"
        )
    if lung_mask.shape != tuple(vector_field_chw.shape[1:]):
        raise ValueError(
            f"Lung mask shape {lung_mask.shape} does not match DVF shape "
            f"{vector_field_chw.shape[1:]}"
        )

    det_j = jacobian_determinant(vector_field_chw[np.newaxis, ...])
    det_j = det_j.detach().cpu().numpy() if isinstance(det_j, torch.Tensor) else det_j

    lung_mask_np = (lung_mask > 0.5).astype(bool)
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
        return 0.0, 0.0
    folding_percentage = float(np.sum(det_j_lung <= 0) / det_j_lung.size)
    return folding_percentage, float(np.std(det_j_lung))


def _evaluate_backward_registration_quality(
    model: CorrespondenceModel,
    signal: np.ndarray,
    images: np.ndarray,
    lung_masks: np.ndarray,
    lung_vessel_maps: np.ndarray,
    image_spacing: tuple[float, float, float],
    reference_phase: int,
    phases: list[int],
    logger: logging.Logger,
) -> dict:
    per_phase = []
    dice_values = []
    mse_values = []
    map_mae_values = []
    assd_values = []
    folding_values = []
    detj_std_values = []
    for phase in phases:
        vector_field = model.predict(signal[phase])  # (3, D, H, W), phase -> reference

        moving_map = lung_vessel_maps[reference_phase].astype(np.float32)
        fixed_map = lung_vessel_maps[phase].astype(np.float32)
        moving_image = images[reference_phase].astype(np.float32)
        fixed_image = images[phase].astype(np.float32)
        lung_mask = (lung_masks[phase] > 0.5).astype(np.float32)

        warped_map = _warp_volume_with_voxel_dvf(moving_map, vector_field)
        warped_image = _warp_volume_with_voxel_dvf(moving_image, vector_field)

        warped_map = warped_map * lung_mask
        fixed_map = fixed_map * lung_mask
        warped_image = warped_image * lung_mask
        fixed_image = fixed_image * lung_mask

        mask_den = float(np.sum(lung_mask))
        if mask_den <= 0:
            map_mae = 0.0
            mse = 0.0
        else:
            map_mae = float(
                np.sum(np.abs(warped_map - fixed_map) * lung_mask) / mask_den
            )
            mse = float(
                np.sum((warped_image - fixed_image) ** 2 * lung_mask) / mask_den
            )

        warped_mask = (warped_map >= 0.5).astype(np.float32)
        fixed_mask = (fixed_map >= 0.5).astype(np.float32)
        dice = _dice_score_numpy(warped_mask, fixed_mask)
        _, assd_vessel_mm = _compute_hd95_assd(
            pred_map=warped_map,
            target_map=fixed_map,
            mask=lung_mask,
            image_spacing=np.asarray(image_spacing, dtype=np.float64),
            threshold=0.5,
        )
        folding_percentage, detj_std = _compute_jacobian_metrics(
            vector_field_chw=vector_field,
            lung_mask=lung_mask,
        )

        per_phase.append(
            {
                "phase": phase,
                "dice_lung_vessels": float(dice),
                "mse": mse,
                "map_mae": map_mae,
                "assd_vessel_0p5_mm": assd_vessel_mm,
                "folding_percentage": folding_percentage,
                "detj_std": detj_std,
            }
        )
        if phase != reference_phase:
            dice_values.append(float(dice))
            mse_values.append(mse)
            map_mae_values.append(map_mae)
            if np.isfinite(assd_vessel_mm):
                assd_values.append(float(assd_vessel_mm))
            folding_values.append(float(folding_percentage))
            detj_std_values.append(float(detj_std))
        logger.info(
            "Backward phase %02d: DICE=%.4f | MSE=%.6f | vessel map MAE=%.6f | "
            "ASSD=%.4f mm | folding=%.4f | std(J)=%.4f",
            phase,
            dice,
            mse,
            map_mae,
            assd_vessel_mm,
            folding_percentage,
            detj_std,
        )

    aggregate = {}
    if per_phase:
        aggregate = {
            "dice_lung_vessels_mean": float(np.mean(dice_values)),
            "dice_lung_vessels_std": float(np.std(dice_values)),
            "mse_mean": float(np.mean(mse_values)),
            "mse_std": float(np.std(mse_values)),
            "map_mae_mean": float(np.mean(map_mae_values)),
            "map_mae_std": float(np.std(map_mae_values)),
            "assd_vessel_0p5_mm_mean": float(np.mean(assd_values))
            if assd_values
            else float("nan"),
            "assd_vessel_0p5_mm_std": float(np.std(assd_values))
            if assd_values
            else float("nan"),
            "folding_percentage_mean": float(np.mean(folding_values)),
            "folding_percentage_std": float(np.std(folding_values)),
            "detj_std_mean": float(np.mean(detj_std_values)),
            "detj_std_std": float(np.std(detj_std_values)),
            "excluded_phase": int(reference_phase),
        }
        logger.info(
            "Backward aggregate (excluding reference phase %02d): DICE=%.4f±%.4f | "
            "MSE=%.6f±%.6f | map MAE=%.6f±%.6f | ASSD=%.4f±%.4f mm | "
            "folding=%.4f±%.4f | std(J)=%.4f±%.4f",
            reference_phase,
            aggregate["dice_lung_vessels_mean"],
            aggregate["dice_lung_vessels_std"],
            aggregate["mse_mean"],
            aggregate["mse_std"],
            aggregate["map_mae_mean"],
            aggregate["map_mae_std"],
            aggregate["assd_vessel_0p5_mm_mean"],
            aggregate["assd_vessel_0p5_mm_std"],
            aggregate["folding_percentage_mean"],
            aggregate["folding_percentage_std"],
            aggregate["detj_std_mean"],
            aggregate["detj_std_std"],
        )

    return {"per_phase": per_phase, "aggregate": aggregate}


def _plot_backward_warped_slices(
    model: CorrespondenceModel,
    signal: np.ndarray,
    images: np.ndarray,
    reference_phase: int,
    phases: list[int],
    output_path: Path,
    logger: logging.Logger,
) -> None:
    moving_image = images[reference_phase].astype(np.float32)
    n_phases = len(phases)
    n_cols = 5
    n_rows = int(np.ceil(n_phases / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.0 * n_cols, 3.0 * n_rows))
    axes = np.atleast_1d(axes).reshape(-1)

    # Keep contrast consistent across all plotted slices.
    vmin = float(np.percentile(images, 1))
    vmax = float(np.percentile(images, 99))

    for i_phase, phase in enumerate(phases):
        vector_field = model.predict(signal[phase])  # (3, D, H, W), phase -> reference
        warped_image = _warp_volume_with_voxel_dvf(moving_image, vector_field)
        idx = warped_image.shape[2] // 2

        axes[i_phase].imshow(
            warped_image[:, idx, :],
            cmap="gray",
            origin="lower",
            vmin=vmin,
            vmax=vmax,
        )
        axes[i_phase].set_title(f"{reference_phase:02d} -> {phase:02d}")
        axes[i_phase].axis("off")

    for i in range(n_phases, len(axes)):
        axes[i].axis("off")

    fig.suptitle("Backward Sanity Check: Warped Reference Slices", y=0.995)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved backward sanity-check slices to %s", output_path)


def main() -> None:
    logging.getLogger("vroc").setLevel(logging.DEBUG)
    logging.getLogger("inrmm").setLevel(logging.DEBUG)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    init_fancy_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, default=1)
    parser.add_argument("--config", type=str, default="default")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument(
        "--method-name",
        type=str,
        required=True,
        help="Stable method label used to keep baseline outputs separate",
    )
    parser.add_argument(
        "--dvf-root",
        "--DVF-input",
        dest="dvf_root",
        type=str,
        required=True,
        help="Case-relative canonical DVF root containing forward/ and backward/.",
    )
    parser.add_argument(
        "--resp-method",
        type=str,
        choices=["lung_volume"],
        default="lung_volume",
    )
    parser.add_argument(
        "--resp-centering", type=str, choices=["ref", "mean"], default="ref"
    )
    parser.add_argument("--reference-phase", type=int, default=5)
    parser.add_argument(
        "--leave-out-phase",
        type=int,
        default=None,
        help="Optional phase index to exclude while fitting the correspondence model.",
    )
    parser.add_argument(
        "--direction", type=str, choices=["forward", "backward", "both"], default="both"
    )
    parser.add_argument(
        "--no-masked-registration",
        action="store_true",
        help="Do not use lung masks during registration",
    )
    args = parser.parse_args()

    cfg = configs.configs[args.config]
    if args.reference_phase is not None:
        cfg["reference_phase"] = args.reference_phase

    case_folder = Path(cfg["paths"]["dirlab_path"]) / f"case_{args.case:02d}"

    phases = list(range(10))
    data = load_and_crop_full_dirlab(case_folder, phases)
    bbox = data["bbox"]

    images = np.stack(data["images"], axis=0)
    lung_masks = data["lung_masks"]
    lung_vessel_maps = np.stack(data["vessel_maps"], axis=0)
    landmarks = data["extreme_landmarks"]
    image_spacing = data["image_spacing"]

    lung_vol_csv_path = case_folder / "respiratory" / "lung_volume.csv"
    signal_tensor = torch.from_numpy(
        load_lung_volume_surrogate(
            csv_path=lung_vol_csv_path,
            expected_phases=phases,
            reference_phase=args.reference_phase,
        )
    ).float()

    signal = signal_tensor.cpu().numpy()

    if args.no_masked_registration:
        masked_registration = False
    else:
        masked_registration = True

    # load DVFs
    vector_fields = None
    if args.dvf_root is not None:
        vector_fields = []
        for phase in phases:
            dvf_path = (
                case_folder / args.dvf_root / "forward" / f"phase_{phase:02d}.nii.gz"
            )
            dvf_sitk = sitk.ReadImage(str(dvf_path))
            dvf_array = sitk.GetArrayFromImage(dvf_sitk)
            dvf_array = np.swapaxes(dvf_array, 0, 2)
            dvf_array = dvf_array[bbox]
            vector_fields.append(dvf_array)

        vector_fields = np.stack(vector_fields, axis=0)
        vector_fields = np.moveaxis(vector_fields, -1, 1)  # (phases, 3, D, H, W)

        bwd_vector_fields = []
        for phase in phases:
            dvf_path = (
                case_folder / args.dvf_root / "backward" / f"phase_{phase:02d}.nii.gz"
            )
            dvf_sitk = sitk.ReadImage(str(dvf_path))
            dvf_array = sitk.GetArrayFromImage(dvf_sitk)
            dvf_array = np.swapaxes(dvf_array, 0, 2)
            dvf_array = dvf_array[bbox]
            bwd_vector_fields.append(dvf_array)

        bwd_vector_fields = np.stack(bwd_vector_fields, axis=0)
        bwd_vector_fields = np.moveaxis(
            bwd_vector_fields, -1, 1
        )  # (phases, 3, D, H, W)

    forward_model = None
    backward_model = None

    if args.direction in ["forward", "both"]:
        forward_model = CorrespondenceModel.build_default(
            images=images,
            signals=signal,
            vector_fields=vector_fields,
            masks=lung_masks,
            centering=args.resp_centering,
            device=args.device,
            reference_phase=cfg["reference_phase"],
            phase_to_leave_out=args.leave_out_phase,
            masked_registration=masked_registration,
            direction="forward",
        )
        forward_eval = _evaluate_forward_tre(
            model=forward_model,
            signal=signal,
            landmarks=landmarks,
            reference_phase=cfg["reference_phase"],
            image_spacing=image_spacing,
            logger=logger,
        )
        output_root = Path(args.output) if args.output is not None else case_folder
        output_folder = output_root / f"correspondence_{args.method_name}_forward"
        output_folder.mkdir(parents=True, exist_ok=True)
        output_path = output_folder / f"ref{args.reference_phase}_fwd.pkl"
        forward_model.save(output_path)
        logger.info("Saved correspondence model to %s", output_path)
        with (output_folder / f"ref{args.reference_phase}_fwd_eval.json").open(
            "w"
        ) as f:
            json.dump(forward_eval, f, indent=2)
        logger.info(
            "Saved forward evaluation to %s",
            output_folder / f"ref{args.reference_phase}_fwd_eval.json",
        )

    if args.direction in ["backward", "both"]:
        backward_model = CorrespondenceModel.build_default(
            images=images,
            signals=signal,
            vector_fields=bwd_vector_fields,
            masks=lung_masks,
            centering=args.resp_centering,
            device=args.device,
            reference_phase=cfg["reference_phase"],
            phase_to_leave_out=args.leave_out_phase,
            masked_registration=masked_registration,
            direction="backward",
        )
        backward_eval = _evaluate_backward_registration_quality(
            model=backward_model,
            signal=signal,
            images=images,
            lung_masks=lung_masks,
            lung_vessel_maps=lung_vessel_maps,
            image_spacing=image_spacing,
            reference_phase=cfg["reference_phase"],
            phases=phases,
            logger=logger,
        )
        output_root = Path(args.output) if args.output is not None else case_folder
        output_folder = output_root / f"correspondence_{args.method_name}_backward"
        output_folder.mkdir(parents=True, exist_ok=True)
        output_path = output_folder / f"ref{args.reference_phase}_bwd.pkl"
        backward_model.save(output_path)
        logger.info("Saved correspondence model to %s", output_path)
        with (output_folder / f"ref{args.reference_phase}_bwd_eval.json").open(
            "w"
        ) as f:
            json.dump(backward_eval, f, indent=2)
        logger.info(
            "Saved backward evaluation to %s",
            output_folder / f"ref{args.reference_phase}_bwd_eval.json",
        )
        _plot_backward_warped_slices(
            model=backward_model,
            signal=signal,
            images=images,
            reference_phase=cfg["reference_phase"],
            phases=phases,
            output_path=output_folder
            / f"ref{args.reference_phase}_bwd_warped_slices.png",
            logger=logger,
        )


if __name__ == "__main__":
    main()
