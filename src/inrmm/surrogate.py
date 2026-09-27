from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np


def _parse_phase(value: str) -> int | None:
    """Parse a numeric phase or a value formatted as phase_XX."""
    value = value.strip()
    if value.isdigit():
        return int(value)
    match = re.fullmatch(r"phase_(\d+)", value)
    return int(match.group(1)) if match is not None else None


def load_lung_volume_surrogate(
    csv_path: Path,
    expected_phases: list[int],
    reference_phase: int,
) -> np.ndarray:
    """Load the centered, normalized lung-volume respiratory surrogate.

    Args:
        csv_path: CSV containing phase, amplitude, and gradient columns.
        expected_phases: Phase identifiers in the desired output order.
        reference_phase: Phase whose surrogate values define the zero point.

    Returns:
        Array with shape (len(expected_phases), 2) containing amplitude and gradient.
    """
    phase_to_signal: dict[int, tuple[float, float]] = {}
    with csv_path.open("r", newline="") as file:
        reader = csv.DictReader(file)
        required_columns = {"phase", "amplitude", "gradient"}
        actual_columns = set(reader.fieldnames or [])
        missing_columns = required_columns - actual_columns
        if missing_columns:
            raise ValueError(
                f"Missing columns in lung-volume surrogate CSV {csv_path}: "
                f"{sorted(missing_columns)}"
            )

        for row in reader:
            phase_value = row["phase"]
            phase = _parse_phase(phase_value)
            if phase is None:
                raise ValueError(
                    f"Invalid phase value {phase_value!r} in surrogate CSV {csv_path}"
                )
            if phase in phase_to_signal:
                raise ValueError(f"Duplicate phase {phase} in surrogate CSV {csv_path}")
            phase_to_signal[phase] = (
                float(row["amplitude"]),
                float(row["gradient"]),
            )

    missing_phases = [
        phase for phase in expected_phases if phase not in phase_to_signal
    ]
    if missing_phases:
        raise ValueError(
            f"Missing phases in lung-volume surrogate CSV {csv_path}: {missing_phases}"
        )
    if reference_phase not in phase_to_signal:
        raise ValueError(
            f"Reference phase {reference_phase} is missing from surrogate CSV {csv_path}"
        )

    signal = np.asarray(
        [phase_to_signal[phase] for phase in expected_phases], dtype=np.float32
    )
    reference_signal = np.asarray(phase_to_signal[reference_phase], dtype=np.float32)
    signal -= reference_signal

    scales = np.max(np.abs(signal), axis=0)
    nonzero = scales > 0
    signal[:, nonzero] /= scales[nonzero]
    return signal
