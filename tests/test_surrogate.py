from pathlib import Path

import numpy as np
import pytest

from inrmm.surrogate import load_lung_volume_surrogate


def _write_csv(path: Path, contents: str) -> None:
    path.write_text(contents)


def test_load_lung_volume_surrogate_centers_and_normalizes(tmp_path: Path) -> None:
    csv_path = tmp_path / "lung_volume.csv"
    _write_csv(
        csv_path,
        "phase,amplitude,gradient\n"
        "phase_00,-1.0,-0.5\n"
        "phase_01,0.0,0.5\n"
        "phase_02,1.0,0.0\n",
    )

    signal = load_lung_volume_surrogate(csv_path, [0, 1, 2], reference_phase=1)

    np.testing.assert_allclose(
        signal,
        np.asarray([[-1.0, -1.0], [0.0, 0.0], [1.0, -0.5]], dtype=np.float32),
    )
