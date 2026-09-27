import numpy as np
import torch

from inrmm.deformation import SpatialTransformer, dice_coefficient, jacobian_determinant
from inrmm.utils import get_bounding_box


def test_jacobian_identity_displacement() -> None:
    displacement = np.zeros((1, 3, 9, 10, 11), dtype=np.float32)
    determinant = jacobian_determinant(displacement)
    assert determinant.shape == (5, 6, 7)
    np.testing.assert_allclose(determinant, 1.0)


def test_jacobian_linear_expansion() -> None:
    displacement = np.zeros((1, 3, 9, 9, 9), dtype=np.float32)
    coordinate = np.arange(9, dtype=np.float32)
    displacement[0, 0] = 0.1 * coordinate[:, None, None]
    determinant = jacobian_determinant(displacement)
    np.testing.assert_allclose(determinant, 1.1, rtol=1e-5)


def test_spatial_transformer_identity() -> None:
    image = torch.arange(5 * 6 * 7, dtype=torch.float32).reshape(1, 1, 5, 6, 7)
    displacement = torch.zeros((1, 3, 5, 6, 7), dtype=torch.float32)
    warped = SpatialTransformer()(image, displacement)
    torch.testing.assert_close(warped, image)


def test_dice_and_bounding_box() -> None:
    mask = torch.zeros((8, 9, 10), dtype=torch.bool)
    mask[2:5, 3:7, 4:8] = True
    assert get_bounding_box(mask, padding=1) == (
        slice(1, 6),
        slice(2, 8),
        slice(3, 9),
    )
    assert dice_coefficient(mask.numpy(), mask.numpy()) == 1.0
