import argparse
import json
import copy
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from inrmm import configs
from inrmm.dir_config import DIRINR_CONFIGS
from inrmm.dir_inr import DirINR
from inrmm.utils import compute_tre_dense_dvf, load_and_crop_full_dirlab


def _pad_dvf_to_full(
    dvf_crop: np.ndarray,
    bbox: tuple[slice, slice, slice],
    full_shape: tuple[int, int, int],
) -> np.ndarray:
    full_dvf = np.zeros(full_shape + (3,), dtype=dvf_crop.dtype)
    full_dvf[bbox] = dvf_crop
    return full_dvf


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


def _log_tre(
    tre_log,
    phase: int,
    label: str,
    moving_landmarks: np.ndarray,
    fixed_landmarks: np.ndarray,
    image_spacing: tuple[float, float, float],
    vector_field_chw: np.ndarray | None = None,
) -> None:
    tre, _ = compute_tre_dense_dvf(
        moving_landmarks=moving_landmarks,
        fixed_landmarks=fixed_landmarks,
        vector_field=vector_field_chw,
        image_spacing=image_spacing,
        snap_to_voxel=True,
    )
    tre_mean = float(np.mean(tre))
    tre_std = float(np.std(tre))
    print(f"Phase {phase:02d}: {label} mean={tre_mean:.4f} mm std={tre_std:.4f} mm")
    tre_log.write(
        f"Phase {phase:02d}: {label} mean={tre_mean:.4f} mm std={tre_std:.4f} mm\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, required=True)
    parser.add_argument("--config", type=str, default="default")
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
    parser.add_argument("--reference-phase", type=int, default=5)
    parser.add_argument("--output-folder", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--dirinr-model",
        type=str,
        choices=["single", "dual"],
        default="single",
        help="Select the predefined DirINR python config preset.",
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

    case_folder = data_root / f"case_{args.case:02d}"
    phases = list(range(10)) if args.dataset == "dirlab" else [0, 5]
    data = load_and_crop_full_dirlab(case_folder, phases, crop=True)

    images = data["images"]
    lung_masks = data["lung_masks"]
    image_spacing = data["image_spacing"]
    bbox = data["bbox"]
    original_shape = data["original_shape"]
    landmarks = data.get("landmarks", {})
    extreme_landmarks = data.get("extreme_landmarks", {})
    if args.dataset == "copdgene":
        # COPDGene contains only extreme landmark annotations (phases 00/05).
        landmarks = {}

    reference_phase = args.reference_phase
    if reference_phase not in phases:
        raise ValueError(
            f"reference-phase={reference_phase} is not part of selected phases: {phases}"
        )
    phase_to_index = {phase: idx for idx, phase in enumerate(phases)}
    ref_idx = phase_to_index[reference_phase]
    reference_image = images[ref_idx]
    reference_lung_mask = lung_masks[ref_idx]
    print(
        f"Dataset={args.dataset} | Case {args.case:02d} | "
        f"reference_phase={reference_phase} | phases={phases}"
    )
    print(
        f"Cropped shape={reference_image.shape} | original_shape={original_shape} | bbox={bbox}"
    )

    if args.output_folder is None:
        output_dir = case_folder / "dvfs" / f"inr_{args.dirinr_model}"
    else:
        output_dir = case_folder / args.output_folder
    forward_dir = output_dir / "forward"
    backward_dir = output_dir / "backward"
    forward_dir.mkdir(parents=True, exist_ok=True)
    backward_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    tre_log_path = output_dir / "tre_scores.txt"
    tre_log = tre_log_path.open("w")

    dir_inr_cfg = copy.deepcopy(DIRINR_CONFIGS[args.dirinr_model])
    print(f"Using DirINR preset: {args.dirinr_model}")
    dir_inr = DirINR(config=dir_inr_cfg)

    zero_dvf = np.zeros(original_shape + (3,), dtype=np.float32)
    _save_dvf(
        zero_dvf,
        original_shape,
        image_spacing,
        forward_dir / f"phase_{reference_phase:02d}.nii.gz",
    )
    _save_dvf(
        zero_dvf,
        original_shape,
        image_spacing,
        backward_dir / f"phase_{reference_phase:02d}.nii.gz",
    )

    for phase, phase_image, phase_lung_mask in zip(
        phases, images, lung_masks, strict=True
    ):
        if phase == reference_phase:
            print(f"Skipping phase {phase:02d} (reference phase).")
            continue

        # We always save the reference->phase DVF.
        fixed_phase = reference_phase
        moving_phase = phase
        fixed_image = reference_image
        fixed_lung_mask = reference_lung_mask
        moving_image = phase_image
        moving_lung_mask = phase_lung_mask

        print(f"Registering phase {moving_phase:02d} -> {fixed_phase:02d}")

        if (
            extreme_landmarks
            and reference_phase in extreme_landmarks
            and phase in extreme_landmarks
        ):
            _log_tre(
                tre_log=tre_log,
                phase=phase,
                label="START TRE(extreme)",
                moving_landmarks=extreme_landmarks[phase],
                fixed_landmarks=extreme_landmarks[reference_phase],
                image_spacing=image_spacing,
                vector_field_chw=None,
            )

        registration_result = dir_inr.register(
            fixed_image=fixed_image,
            moving_image=moving_image,
            fixed_lung_mask=fixed_lung_mask,
            moving_lung_mask=moving_lung_mask,
            image_spacing=image_spacing,
            device=args.device,
        )

        dvf_crop_fwd = registration_result["forward_dvf"]
        dvf_crop_bwd = registration_result["backward_dvf"]

        dvf_crop_fwd_vox = dvf_crop_fwd.copy()
        dvf_crop_bwd_vox = dvf_crop_bwd.copy()
        for i in range(3):
            dvf_crop_fwd_vox[..., i] = (
                dvf_crop_fwd_vox[..., i] * (dvf_crop_fwd.shape[i] - 1) / 2
            )
            dvf_crop_bwd_vox[..., i] = (
                dvf_crop_bwd_vox[..., i] * (dvf_crop_bwd.shape[i] - 1) / 2
            )

        dvf_full_fwd = _pad_dvf_to_full(
            dvf_crop=dvf_crop_fwd_vox,
            bbox=bbox,
            full_shape=original_shape,
        )
        dvf_full_bwd = _pad_dvf_to_full(
            dvf_crop=dvf_crop_bwd_vox,
            bbox=bbox,
            full_shape=original_shape,
        )

        out_path_fwd = forward_dir / f"phase_{phase:02d}.nii.gz"
        out_path_bwd = backward_dir / f"phase_{phase:02d}.nii.gz"
        _save_dvf(
            dvf_vox=dvf_full_fwd,
            image_shape=original_shape,
            image_spacing=image_spacing,
            output_path=out_path_fwd,
        )
        _save_dvf(
            dvf_vox=dvf_full_bwd,
            image_shape=original_shape,
            image_spacing=image_spacing,
            output_path=out_path_bwd,
        )
        print(f"Saved DVF (forward): {out_path_fwd}")
        print(f"Saved DVF (backward): {out_path_bwd}")

        vector_field_fwd_chw = np.moveaxis(dvf_crop_fwd_vox, -1, 0)
        vector_field_bwd_chw = np.moveaxis(dvf_crop_bwd_vox, -1, 0)

        if landmarks and reference_phase in landmarks and phase in landmarks:
            _log_tre(
                tre_log=tre_log,
                phase=phase,
                label="TRE",
                moving_landmarks=landmarks[phase],
                fixed_landmarks=landmarks[reference_phase],
                image_spacing=image_spacing,
                vector_field_chw=vector_field_fwd_chw,
            )
            _log_tre(
                tre_log=tre_log,
                phase=phase,
                label="TRE(backward)",
                moving_landmarks=landmarks[reference_phase],
                fixed_landmarks=landmarks[phase],
                image_spacing=image_spacing,
                vector_field_chw=vector_field_bwd_chw,
            )

        if (
            extreme_landmarks
            and reference_phase in extreme_landmarks
            and phase in extreme_landmarks
        ):
            _log_tre(
                tre_log=tre_log,
                phase=phase,
                label="TRE(extreme)",
                moving_landmarks=extreme_landmarks[phase],
                fixed_landmarks=extreme_landmarks[reference_phase],
                image_spacing=image_spacing,
                vector_field_chw=vector_field_fwd_chw,
            )
            _log_tre(
                tre_log=tre_log,
                phase=phase,
                label="TRE(extreme,backward)",
                moving_landmarks=extreme_landmarks[reference_phase],
                fixed_landmarks=extreme_landmarks[phase],
                image_spacing=image_spacing,
                vector_field_chw=vector_field_bwd_chw,
            )

    tre_log.close()
    manifest = {
        "method": f"inr_{args.dirinr_model}",
        "reference_phase": reference_phase,
        "directions": {
            "forward": "reference_to_phase",
            "backward": "phase_to_reference",
        },
        "units": "voxel",
        "file_pattern": "{direction}/phase_{phase:02d}.nii.gz",
    }
    with (output_dir / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()
