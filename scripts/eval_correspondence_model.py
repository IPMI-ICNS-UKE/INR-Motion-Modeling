import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from inrmm.deformation import jacobian_determinant

from inrmm import configs
from inrmm.correspondence import CorrespondenceModel
from inrmm.surrogate import load_lung_volume_surrogate
from inrmm.utils import load_and_crop_full_dirlab, load_config


def _load_cfg(config_path: Path | None, config_name: str) -> dict:
    if config_path is not None:
        return load_config(config_path)
    return configs.configs[config_name]


def load_surrogate_signal(
    cfg: dict,
    case_folder: Path,
    phases: list[int],
    reference_phase: int,
) -> np.ndarray:
    respiration_method = str(
        cfg.get("respiration", {}).get("method", "lung_volume")
    ).lower()
    if respiration_method != "lung_volume":
        raise ValueError("Only the lung_volume respiratory surrogate is supported.")
    signal_path = case_folder / "respiratory" / "lung_volume.csv"
    return load_lung_volume_surrogate(
        csv_path=signal_path,
        expected_phases=phases,
        reference_phase=reference_phase,
    )


def compute_displacement_grid(
    model: CorrespondenceModel,
    reference_lung_mask: np.ndarray,
    image_spacing: tuple[float, float, float],
    steps: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    amps = np.arange(0.0 - steps[0], 1.0 + steps[0], steps[0], dtype=np.float32)
    vels = np.arange(-1.0 - steps[1], 1.0 + steps[1], steps[1], dtype=np.float32)
    grid_mag = np.zeros((len(amps), len(vels)), dtype=np.float32)
    grid_negative_detj = np.zeros((len(amps), len(vels)), dtype=np.float32)
    grid_detj_std = np.full((len(amps), len(vels)), np.nan, dtype=np.float32)

    spacing = np.asarray(image_spacing, dtype=np.float32).reshape(1, 1, 1, 3)

    for i, amp in enumerate(amps):
        for j, vel in enumerate(vels):
            signal = np.array([amp, vel], dtype=np.float32)
            vector_field = model.predict(signal)
            displacement_xyz = np.moveaxis(vector_field, 0, -1)
            disp_mm = displacement_xyz * spacing
            grid_mag[i, j] = float(np.linalg.norm(disp_mm, axis=-1).mean())

            det_j = jacobian_determinant(vector_field[None, ...])
            det_j = (
                det_j.detach().cpu().numpy()
                if hasattr(det_j, "detach")
                else np.asarray(det_j)
            )

            if min(reference_lung_mask.shape) > 4:
                cropped_lung = reference_lung_mask[2:-2, 2:-2, 2:-2]
            else:
                cropped_lung = reference_lung_mask
            if det_j.shape != cropped_lung.shape:
                min_shape = tuple(
                    min(det_dim, mask_dim)
                    for det_dim, mask_dim in zip(det_j.shape, cropped_lung.shape)
                )
                det_j = det_j[tuple(slice(0, size) for size in min_shape)]
                cropped_lung = cropped_lung[tuple(slice(0, size) for size in min_shape)]

            det_j_lung = det_j[cropped_lung]
            if det_j_lung.size == 0:
                negative_percentage = 0.0
                detj_std = float("nan")
            else:
                negative_percentage = float(
                    np.sum(det_j_lung <= 0) / det_j_lung.size * 100.0
                )
                detj_std = float(np.std(det_j_lung))
            grid_negative_detj[i, j] = negative_percentage
            grid_detj_std[i, j] = detj_std

    return amps, vels, grid_mag, grid_negative_detj, grid_detj_std


def plot_grid(
    amps: np.ndarray,
    vels: np.ndarray,
    grid_mag: np.ndarray,
    negative_detj_percentage: np.ndarray,
    detj_std: np.ndarray,
    surrogate_signal: np.ndarray | None,
    output_path: Path,
) -> None:
    fig, (ax_mag, ax_jac, ax_detj_std) = plt.subplots(
        1,
        3,
        figsize=(20, 6),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    extent = [vels.min(), vels.max(), amps.min(), amps.max()]

    im_mag = ax_mag.imshow(
        grid_mag,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="viridis",
    )
    im_jac = ax_jac.imshow(
        negative_detj_percentage,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="magma",
    )
    im_detj_std = ax_detj_std.imshow(
        detj_std,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="cividis",
    )

    for ax in (ax_mag, ax_jac, ax_detj_std):
        if surrogate_signal is not None:
            ax.scatter(
                surrogate_signal[:, 1],
                surrogate_signal[:, 0],
                marker="x",
                c="red",
                s=40,
                linewidths=1.5,
            )
        ax.set_xlabel("Velocity")

    ax_mag.set_title("Mean Disp Mag")
    ax_jac.set_title("det(J) <= 0 (%)")
    ax_detj_std.set_title("std(det(J))")
    ax_mag.set_ylabel("Amplitude")

    cbar_mag = fig.colorbar(im_mag, ax=ax_mag, pad=0.02, fraction=0.046)
    cbar_mag.set_label("Mean disp. (mm)")
    cbar_mag.ax.tick_params(labelsize=8)
    cbar_jac = fig.colorbar(im_jac, ax=ax_jac, pad=0.02, fraction=0.046)
    cbar_jac.set_label("Negative det(J) (%)")
    cbar_jac.ax.tick_params(labelsize=8)
    cbar_detj_std = fig.colorbar(im_detj_std, ax=ax_detj_std, pad=0.02, fraction=0.046)
    cbar_detj_std.set_label("std(det(J))")
    cbar_detj_std.ax.tick_params(labelsize=8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def save_grid_data_txt(
    amps: np.ndarray,
    vels: np.ndarray,
    grid_mag: np.ndarray,
    negative_detj_percentage: np.ndarray,
    detj_std: np.ndarray,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        f.write("# amp vel mean_disp_mm negative_detj_percent detj_std\n")
        for i, amp in enumerate(amps):
            for j, vel in enumerate(vels):
                f.write(
                    f"{amp:.6f} {vel:.6f} {grid_mag[i, j]:.6f}"
                    f" {negative_detj_percentage[i, j]:.6f}"
                    f" {detj_std[i, j]:.6f}\n"
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--config-path", type=str, default=None)
    parser.add_argument("--config", type=str, default="default")
    parser.add_argument("--case", type=int, default=None)
    parser.add_argument("--reference-phase", type=int, default=None)
    parser.add_argument(
        "--direction",
        type=str,
        default="forward",
        choices=["forward", "backward"],
        help="forward: fixed=reference, moving=phase; backward: fixed=phase, moving=reference",
    )
    parser.add_argument(
        "--steps",
        type=float,
        nargs="+",
        default=[0.05, 0.1],
        help="Grid step for amplitude/velocity in [-1,1]",
    )
    parser.add_argument(
        "--ood-output",
        type=str,
        default=None,
        help="Output path for OOD displacement heatmaps",
    )
    parser.add_argument(
        "--ood-data-output",
        type=str,
        default=None,
        help=(
            "Output txt path for OOD grid data (default: <ood_output_stem>_data.txt)."
        ),
    )
    parser.add_argument(
        "--resp-signal", type=str, default="lung_volume", choices=["lung_volume"]
    )
    args = parser.parse_args()

    model = CorrespondenceModel.load(args.model_path)

    cfg = _load_cfg(Path(args.config_path) if args.config_path else None, args.config)

    if args.case is not None:
        case = args.case
    else:
        case = cfg.get("case")
        if case is None:
            raise ValueError("case must be provided via --case or config file")

    reference_phase = (
        args.reference_phase
        if args.reference_phase is not None
        else cfg.get("reference_phase", model.reference_phase)
    )

    case_folder = Path(cfg["paths"]["dirlab_path"]) / f"case_{case:02d}"
    phases = list(range(10))
    data = load_and_crop_full_dirlab(case_folder, phases)

    image_spacing = data["image_spacing"]
    lung_masks = data["lung_masks"]

    # set signal in cfg
    cfg["respiration"]["method"] = args.resp_signal

    signals = load_surrogate_signal(
        cfg=cfg,
        case_folder=case_folder,
        phases=phases,
        reference_phase=reference_phase,
    )

    print(
        f"Case {case:02d} | reference_phase={reference_phase} | direction={args.direction}"
    )

    reference_lung_mask = (lung_masks[reference_phase] > 0.5).astype(np.bool_)
    amps, vels, grid_mag, negative_detj_percentage, detj_std = (
        compute_displacement_grid(
            model=model,
            reference_lung_mask=reference_lung_mask,
            image_spacing=image_spacing,
            steps=args.steps,
        )
    )
    output_path = (
        Path(args.ood_output)
        if args.ood_output is not None
        else Path(args.model_path).parent / "ood_displacement_heatmaps.png"
    )
    if args.ood_data_output is None:
        data_output_path = output_path.with_name(f"{output_path.stem}_data.txt")
    else:
        data_output_path = Path(args.ood_data_output)

    plot_grid(
        amps=amps,
        vels=vels,
        grid_mag=grid_mag,
        negative_detj_percentage=negative_detj_percentage,
        detj_std=detj_std,
        surrogate_signal=signals,
        output_path=output_path,
    )
    save_grid_data_txt(
        amps=amps,
        vels=vels,
        grid_mag=grid_mag,
        negative_detj_percentage=negative_detj_percentage,
        detj_std=detj_std,
        output_path=data_output_path,
    )
    print(f"Wrote OOD plot to {output_path}")
    print(f"Wrote OOD grid data to {data_output_path}")


if __name__ == "__main__":
    main()
