import pytest
import torch

from inrmm.ncc import ncc_per_batch


def test_ncc_per_batch_identical_signals() -> None:
    signal = torch.tensor([[1.0, 2.0, 3.0], [3.0, 1.0, 2.0]])

    loss = ncc_per_batch(signal, signal)

    assert loss.item() == pytest.approx(0.0, abs=1e-5)


def test_ncc_per_batch_respects_mask() -> None:
    fixed = torch.tensor([[1.0, 2.0, 100.0]])
    warped = torch.tensor([[1.0, 2.0, -100.0]])
    mask = torch.tensor([[1, 1, 0]], dtype=torch.bool)

    loss = ncc_per_batch(fixed, warped, mask)

    assert loss.item() == pytest.approx(0.0, abs=1e-5)


def test_ncc_per_batch_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="equal shapes"):
        ncc_per_batch(torch.zeros(1, 2), torch.zeros(1, 3))
