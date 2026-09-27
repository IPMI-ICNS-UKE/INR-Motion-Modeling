from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk
from inrmm.compat import init_fancy_logging

try:
    from vroc.registration import VrocRegistration
except ImportError:
    VrocRegistration = None

from inrmm import configs
from inrmm.utils import compute_tre_dense_dvf, load_and_crop_full_dirlab


def _to_numpy(array: Any) -> np.ndarray:
    if isinstance(array, np.ndarray):
        return array
    if hasattr(array, "detach"):
        array = array.detach()
    if hasattr(array, "cpu"):
        array = array.cpu()
    return np.asarray(array)


def _to_xyzc(vector_field: np.ndarray) -> np.ndarray:
    if vector_field.ndim != 4:
        raise ValueError(
            f"Expected 4D vector field, but got shape {vector_field.shape}"
        )
    if vector_field.shape[-1] == 3:
        return vector_field
    if vector_field.shape[0] == 3:
        return np.moveaxis(vector_field, 0, -1)
    raise ValueError(
        f"Expected vector dimension to be first or last axis with size 3. "
        f"Got shape {vector_field.shape}"
    )


def _pad_dvf_to_full(
    dvf_crop: np.ndarray,
    bbox: tuple[slice, slice, slice] | None,
    full_shape: tuple[int, int, int],
) -> np.ndarray:
    if bbox is None:
        return dvf_crop
    full_dvf = np.zeros(full_shape + (3,), dtype=dvf_crop.dtype)
    full_dvf[bbox] = dvf_crop
    return full_dvf


def _save_dvf(
    dvf_vox: np.ndarray,
    image_spacing: tuple[float, float, float],
    output_path: Path,
) -> None:
    dvf_sitk = sitk.GetImageFromArray(np.swapaxes(dvf_vox, 0, 2), isVector=True)
    dvf_sitk.SetSpacing(tuple(image_spacing))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(dvf_sitk, str(output_path))


def _registrations_from_reference(
    images: np.ndarray, masks: np.ndarray | None, reference_phase: int
):
    for phase in range(len(images)):
        yield {
            "fixed_image": images[phase],
            "moving_image": images[reference_phase],
            "fixed_mask": masks[phase] if masks is not None else None,
            "moving_mask": masks[reference_phase] if masks is not None else None,
        }


def _compute_vroc_vector_fields(
    images: np.ndarray,
    masks: np.ndarray | None,
    device: str,
    reference_phase: int,
    image_spacing: tuple[float, float, float],
    masked_registration: bool,
    direction: str,
) -> np.ndarray:
    if direction not in ("forward", "backward"):
        raise ValueError("direction must be 'forward' or 'backward'")

    if VrocRegistration is None:
        raise RuntimeError(
            "The optional VROC backend is not installed. Install an authorized "
            "VROC distribution before running this exporter."
        )
    registration = VrocRegistration(device=device)
    vector_fields = []

    for data in _registrations_from_reference(images, masks, reference_phase):
        moving_image = data["moving_image"]
        fixed_image = data["fixed_image"]
        moving_mask = data["moving_mask"] if masked_registration else None
        fixed_mask = data["fixed_mask"] if masked_registration else None

        if direction == "forward":
            moving_image, fixed_image = fixed_image, moving_image
            moving_mask, fixed_mask = fixed_mask, moving_mask

        registration_result = registration.register(
            moving_image=moving_image,
            fixed_image=fixed_image,
            moving_mask=moving_mask,
            fixed_mask=fixed_mask,
            register_affine=False,
            image_spacing=image_spacing,
            default_parameters={
                "iterations": 800,
                "tau": 2.25,
                "tau_level_decay": 0.0,
                "tau_iteration_decay": 0.0,
                "sigma_x": 1.25,
                "sigma_y": 1.25,
                "sigma_z": 1.25,
                "sigma_level_decay": 0.0,
                "sigma_iteration_decay": 0.0,
                "n_levels": 3,
                "largest_scale_factor": 1.0,
            },
        )
        vector_fields.append(_to_numpy(registration_result.composed_vector_field))

    return np.stack(vector_fields, axis=0)


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
    parser.add_argument("--output-folder", type=str, default="dvfs/vroc")
    parser.add_argument("--reference-phase", type=int, default=None)
    parser.add_argument(
        "--direction",
        type=str,
        choices=["forward", "backward"],
        default="forward",
    )
    parser.add_argument(
        "--masked-registration",
        action="store_true",
        help="Use lung masks during registration",
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

    images = np.stack(data["images"], axis=0)
    lung_masks = data["lung_masks"]

    if args.no_masked_registration:
        masked_registration = False
    elif args.masked_registration:
        masked_registration = True
    else:
        masked_registration = True

    vector_fields = _compute_vroc_vector_fields(
        images=images,
        masks=lung_masks,
        device=args.device,
        reference_phase=cfg["reference_phase"],
        image_spacing=data["image_spacing"],
        masked_registration=masked_registration,
        direction=args.direction,
    )

    output_root = case_folder / args.output_folder
    output_dir = output_root / args.direction
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Saving %d vector fields to %s", len(vector_fields), output_dir)

    bbox = data["bbox"]
    original_shape = data["original_shape"]
    image_spacing = data["image_spacing"]
    landmarks = data.get("landmarks", {})

    tre_log_path = output_dir / "tre_scores.txt"
    with tre_log_path.open("w") as tre_log:
        for phase, dvf in enumerate(vector_fields):
            dvf_xyzc = _to_xyzc(dvf)
            dvf_full = _pad_dvf_to_full(
                dvf_crop=dvf_xyzc,
                bbox=bbox,
                full_shape=original_shape,
            )
            dvf_path = output_dir / f"phase_{phase:02d}.nii.gz"
            _save_dvf(
                dvf_vox=dvf_full,
                image_spacing=image_spacing,
                output_path=dvf_path,
            )
            logger.info("Saved %s", dvf_path)

            vector_field_chw = np.moveaxis(dvf_xyzc, -1, 0)
            if landmarks and cfg["reference_phase"] in landmarks and phase in landmarks:
                if args.direction == "forward":
                    # For forward DVF, we want to warp the moving landmarks to the fixed space
                    moving_landmarks = landmarks[phase]
                    fixed_landmarks = landmarks[cfg["reference_phase"]]
                else:
                    # For backward DVF, we want to warp the fixed landmarks to the moving space
                    moving_landmarks = landmarks[cfg["reference_phase"]]
                    fixed_landmarks = landmarks[phase]
                tre, _ = compute_tre_dense_dvf(
                    moving_landmarks=moving_landmarks,
                    fixed_landmarks=fixed_landmarks,
                    vector_field=vector_field_chw,
                    image_spacing=image_spacing,
                    snap_to_voxel=True,
                )
                tre_mean = float(np.mean(tre))
                tre_std = float(np.std(tre))
                logger.info(
                    "Phase %02d: TRE mean=%.4f mm std=%.4f mm",
                    phase,
                    tre_mean,
                    tre_std,
                )
                tre_log.write(
                    f"Phase {phase:02d}: TRE mean={tre_mean:.4f} mm std={tre_std:.4f} mm\n"
                )

    manifest = {
        "method": "vroc",
        "reference_phase": cfg["reference_phase"],
        "directions": {
            "forward": "reference_to_phase",
            "backward": "phase_to_reference",
        },
        "units": "voxel",
        "file_pattern": "{direction}/phase_{phase:02d}.nii.gz",
    }
    with (output_root / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()
