from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from inrmm import configs


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing evaluation file: {path}")
    with path.open("r") as f:
        return json.load(f)


def _require_metric(container: dict, key: str, source_path: Path) -> float:
    if key not in container:
        raise KeyError(f"Missing key '{key}' in {source_path}")
    return float(container[key])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="default")
    parser.add_argument("--method-name", type=str, required=True)
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="DIRLAB root override; otherwise use the selected config.",
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
    parser.add_argument("--cases", type=int, nargs="+", default=list(range(1, 11)))
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--output-csv", type=str, default=None)
    args = parser.parse_args()

    cfg = configs.configs[args.config]
    dirlab_root = (
        Path(args.data_root)
        if args.data_root is not None
        else Path(cfg["paths"]["dirlab_path"])
    )

    per_case_rows: list[dict[str, float | int]] = []

    for case in sorted(args.cases):
        case_folder = dirlab_root / f"case_{case:02d}"
        fwd_eval_path = (
            case_folder
            / f"correspondence_{args.method_name}_forward"
            / f"ref{args.reference_phase}_fwd_eval.json"
        )
        bwd_eval_path = (
            case_folder
            / f"correspondence_{args.method_name}_backward"
            / f"ref{args.reference_phase}_bwd_eval.json"
        )

        fwd_eval = _load_json(fwd_eval_path)
        bwd_eval = _load_json(bwd_eval_path)

        fwd_agg = fwd_eval.get("aggregate", {})
        bwd_agg = bwd_eval.get("aggregate", {})

        row = {
            "case": case,
            "tre_mean_mm": _require_metric(fwd_agg, "tre_mean_mm", fwd_eval_path),
            "mse_mean": _require_metric(bwd_agg, "mse_mean", bwd_eval_path),
            "dice_mean": _require_metric(
                bwd_agg, "dice_lung_vessels_mean", bwd_eval_path
            ),
            "map_mae_mean": _require_metric(bwd_agg, "map_mae_mean", bwd_eval_path),
            "folding_percentage_mean": _require_metric(
                bwd_agg, "folding_percentage_mean", bwd_eval_path
            ),
            "detj_std_mean": _require_metric(bwd_agg, "detj_std_mean", bwd_eval_path),
        }
        per_case_rows.append(row)

    if not per_case_rows:
        raise RuntimeError("No cases available for aggregation.")

    tre_values = np.array(
        [row["tre_mean_mm"] for row in per_case_rows], dtype=np.float64
    )
    mse_values = np.array([row["mse_mean"] for row in per_case_rows], dtype=np.float64)
    dice_values = np.array(
        [row["dice_mean"] for row in per_case_rows], dtype=np.float64
    )
    map_mae_values = np.array(
        [row["map_mae_mean"] for row in per_case_rows], dtype=np.float64
    )

    folding_values = np.array(
        [row["folding_percentage_mean"] for row in per_case_rows], dtype=np.float64
    )
    detj_std_values = np.array(
        [row["detj_std_mean"] for row in per_case_rows], dtype=np.float64
    )

    aggregate = {
        "n_cases": int(len(per_case_rows)),
        "tre_mean_mm": float(np.mean(tre_values)),
        "tre_std_mm": float(np.std(tre_values)),
        "mse_mean": float(np.mean(mse_values)),
        "mse_std": float(np.std(mse_values)),
        "dice_mean": float(np.mean(dice_values)),
        "dice_std": float(np.std(dice_values)),
        "map_mae_mean": float(np.mean(map_mae_values)),
        "map_mae_std": float(np.std(map_mae_values)),
        "folding_percentage_mean": float(np.mean(folding_values)),
        "folding_percentage_std": float(np.std(folding_values)),
        "detj_std_mean": float(np.mean(detj_std_values)),
        "detj_std_std": float(np.std(detj_std_values)),
    }

    output = {
        "config": args.config,
        "method_name": args.method_name,
        "resp_method": args.resp_method,
        "resp_centering": args.resp_centering,
        "reference_phase": args.reference_phase,
        "cases": sorted(args.cases),
        "per_case": per_case_rows,
        "aggregate_over_cases": aggregate,
    }

    default_stem = (
        f"correspondence_{args.method_name}_aggregate_resp_"
        f"{args.resp_method}_{args.resp_centering}_ref{args.reference_phase}"
    )
    output_json = (
        Path(args.output_json)
        if args.output_json is not None
        else Path("results") / f"{default_stem}.json"
    )
    output_csv = (
        Path(args.output_csv)
        if args.output_csv is not None
        else Path("results") / f"{default_stem}_per_case.csv"
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_json.open("w") as f:
        json.dump(output, f, indent=2)

    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(per_case_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(per_case_rows)

    print(f"Saved aggregate JSON: {output_json}")
    print(f"Saved per-case CSV: {output_csv}")
    print("Aggregate over cases:")
    print(
        "TRE={:.4f}±{:.4f} mm | DICE={:.4f}±{:.4f} | "
        "folding={:.6f}±{:.6f} | std(J)={:.6f}±{:.6f}".format(
            aggregate["tre_mean_mm"],
            aggregate["tre_std_mm"],
            aggregate["dice_mean"],
            aggregate["dice_std"],
            aggregate["folding_percentage_mean"],
            aggregate["folding_percentage_std"],
            aggregate["detj_std_mean"],
            aggregate["detj_std_std"],
        )
    )


if __name__ == "__main__":
    main()
