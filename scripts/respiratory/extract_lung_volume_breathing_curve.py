from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

from inrmm.utils import load_full_dirlab


def extract_lung_volume_curve(
    case_folder: Path,
    phases: list[int],
    lung_classes: list[int],
) -> list[dict[str, int]]:
    data = load_full_dirlab(case_folder, phases)
    masks = data["masks"]

    curve: list[dict[str, int]] = []
    for idx, phase in enumerate(phases):
        lung_mask = np.isin(masks[idx], lung_classes)
        lung_volume_voxels = int(np.sum(lung_mask))
        curve.append({"phase": phase, "lung_volume_voxels": lung_volume_voxels})
    return curve


def _rescale_to_unit_interval(values: list[int]) -> list[float]:
    values_np = np.asarray(values, dtype=np.float32)
    # center around 0 and rescale to [-1, 1]
    value_range = float(np.max(values_np) - np.min(values_np))
    if value_range <= 0:
        return [0.0 for _ in values]
    normalized = 2 * (values_np - np.min(values_np)) / value_range - 1

    return normalized.tolist()


def _compute_smoothed_gradient(
    phases: list[int],
    amplitudes: list[float],
    interp_points_per_interval: int,
    smoothing_sigma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(phases) != len(amplitudes):
        raise ValueError(
            "phases and amplitudes must have same length, "
            f"got {len(phases)} and {len(amplitudes)}"
        )
    if len(phases) < 2:
        raise ValueError("Need at least two phase points to compute a gradient.")
    if interp_points_per_interval < 1:
        raise ValueError(
            f"interp_points_per_interval must be >= 1, got {interp_points_per_interval}"
        )
    if smoothing_sigma < 0:
        raise ValueError(f"smoothing_sigma must be >= 0, got {smoothing_sigma}")

    x_phase = np.asarray(phases, dtype=np.float32)
    y_phase = np.asarray(amplitudes, dtype=np.float32)

    # Step 1: interpolate to a denser curve.
    n_dense = (len(x_phase) - 1) * interp_points_per_interval + 1
    x_dense = np.linspace(
        float(x_phase[0]), float(x_phase[-1]), num=n_dense, dtype=np.float32
    )
    y_dense = np.interp(x_dense, x_phase, y_phase)

    # Step 2: smooth interpolated curve.
    y_smooth = gaussian_filter1d(y_dense, sigma=smoothing_sigma, mode="nearest")

    # Step 3: gradient on smoothed curve.
    grad_dense = np.gradient(y_smooth, x_dense)

    # Map dense gradient back to original phases so each phase has one value.
    grad_phase = np.interp(x_phase, x_dense, grad_dense)
    return x_dense, y_smooth, grad_dense, grad_phase


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract lung-volume breathing curves (voxel sum over binary lung masks) "
            "for all 10 DIRLAB cases and plot them together."
        )
    )
    parser.add_argument(
        "--dirlab-dir",
        type=str,
        required=True,
        help="Path to DIRLAB converted test folder containing case_01 ... case_10.",
    )
    parser.add_argument(
        "--lung-classes",
        type=int,
        nargs="+",
        default=[10, 11, 12, 13, 14],
        help="Segmentation class ids considered lung.",
    )
    parser.add_argument(
        "--plot-output",
        type=str,
        default="./lung_volume_breathing_curve.png",
        help="Output image path for the combined plot.",
    )
    parser.add_argument(
        "--interp-points-per-interval",
        type=int,
        default=25,
        help=(
            "Number of interpolation samples between consecutive phases before smoothing "
            "and gradient computation."
        ),
    )
    parser.add_argument(
        "--smoothing-sigma",
        type=float,
        default=2.0,
        help="Gaussian smoothing sigma (in interpolated sample units).",
    )
    args = parser.parse_args()

    phases = list(range(10))
    dirlab_dir = Path(args.dirlab_dir)

    plt.figure(figsize=(10, 5))

    for case in range(1, 11):
        case_folder = dirlab_dir / f"case_{case:02d}"
        if not case_folder.exists():
            raise FileNotFoundError(f"Case folder not found: {case_folder}")

        curve = extract_lung_volume_curve(
            case_folder=case_folder,
            phases=phases,
            lung_classes=args.lung_classes,
        )

        print(f"Case {case:02d}")
        print("phase,lung_volume_voxels,amplitude,gradient")
        phases_list = [row["phase"] for row in curve]
        volumes = [row["lung_volume_voxels"] for row in curve]
        volumes_norm = _rescale_to_unit_interval(volumes)

        _, _, _, gradients = _compute_smoothed_gradient(
            phases=phases_list,
            amplitudes=volumes_norm,
            interp_points_per_interval=args.interp_points_per_interval,
            smoothing_sigma=args.smoothing_sigma,
        )

        for row in curve:
            phase = int(row["phase"])
            idx = phases_list.index(phase)
            print(
                f"{phase},"
                f"{row['lung_volume_voxels']},"
                f"{volumes_norm[idx]:.6f},"
                f"{float(gradients[idx]):.6f}"
            )
        ordered_values = ",".join(str(row["lung_volume_voxels"]) for row in curve)
        print("lung_volume_values")
        print(ordered_values)
        gradient_values = ",".join(f"{float(g):.6f}" for g in gradients)
        print("gradient_values")
        print(gradient_values)

        # write CSV for this case
        respiratory_dir = case_folder / "respiratory"
        respiratory_dir.mkdir(exist_ok=True)
        csv_output_path = respiratory_dir / "lung_volume.csv"
        with open(csv_output_path, "w") as f:
            f.write("phase,amplitude,gradient\n")
            for row, amplitude, gradient in zip(
                curve, volumes_norm, gradients, strict=True
            ):
                f.write(
                    f"phase_{row['phase']:02d},{amplitude:.6f},{float(gradient):.6f}\n"
                )
        print(f"Saved lung volume curve CSV for case {case:02d} to {csv_output_path}")

        plt.plot(phases_list, volumes_norm, marker="o", label=f"Case {case:02d}")
        plt.plot(
            phases_list,
            gradients,
            marker="x",
            linestyle="--",
            label=f"Gradient Case {case:02d}",
        )

    plt.xlabel("Phase")
    plt.ylabel("Normalized Lung Volume")
    plt.title("Lung Volume Breathing Curves")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(args.plot_output, dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
