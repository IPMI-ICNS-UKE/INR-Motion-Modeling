import numpy as np
import pytest

from inrmm.correspondence import CorrespondenceModel


def test_correspondence_requires_precomputed_dvfs() -> None:
    with pytest.raises(ValueError, match="Precomputed vector_fields are required"):
        CorrespondenceModel.build_default(
            images=np.zeros((3, 4, 4, 4), dtype=np.float32),
            signals=np.zeros((3, 2), dtype=np.float32),
            vector_fields=None,
        )


def test_correspondence_recovers_linear_motion() -> None:
    signals = np.asarray(
        [[-1.0, 0.0], [0.0, 0.0], [0.0, 1.0], [1.0, -1.0]], dtype=np.float32
    )
    coefficients = np.asarray([2.0, -3.0], dtype=np.float32)
    scalar_motion = signals @ coefficients
    vector_fields = np.zeros((4, 3, 2, 2, 2), dtype=np.float32)
    vector_fields[:, 0] = scalar_motion[:, None, None, None]

    model = CorrespondenceModel.build_default(
        images=np.zeros((4, 2, 2, 2), dtype=np.float32),
        signals=signals,
        vector_fields=vector_fields,
        reference_phase=1,
    )
    predicted = model.predict(np.asarray([0.5, 0.25], dtype=np.float32))
    np.testing.assert_allclose(predicted[0], 0.25, atol=1e-4)
