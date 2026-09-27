"""Deformation-field utilities shared by registration and evaluation."""

from __future__ import annotations

import numpy as np
import scipy.ndimage as ndi
import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_coefficient(prediction: np.ndarray, target: np.ndarray) -> float:
    """Compute binary Dice, returning one when both masks are empty."""
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    if prediction.shape != target.shape:
        raise ValueError(f"Mask shape mismatch: {prediction.shape} != {target.shape}")
    denominator = int(prediction.sum() + target.sum())
    if denominator == 0:
        return 1.0
    return float(2 * np.logical_and(prediction, target).sum() / denominator)


def jacobian_determinant(displacement: np.ndarray | torch.Tensor) -> np.ndarray:
    """Return det(I + grad(u)) for voxel-space fields shaped ``(B, 3, X, Y, Z)``."""
    if isinstance(displacement, torch.Tensor):
        displacement = displacement.detach().cpu().numpy()
    displacement = np.asarray(displacement)
    if displacement.ndim != 5 or displacement.shape[1] != 3:
        raise ValueError(
            f"Expected displacement shape (B, 3, X, Y, Z), got {displacement.shape}"
        )

    gradients = np.empty((3, 3, *displacement.shape[2:]), dtype=displacement.dtype)
    for component in range(3):
        for axis in range(3):
            gradients[component, axis] = ndi.correlate1d(
                displacement[0, component],
                weights=np.asarray([-0.5, 0.0, 0.5]),
                axis=axis,
                mode="constant",
                cval=0.0,
            )
    jacobian = gradients + np.eye(3, dtype=displacement.dtype)[..., None, None, None]
    jacobian = jacobian[:, :, 2:-2, 2:-2, 2:-2]
    return np.linalg.det(np.moveaxis(jacobian, (0, 1), (-2, -1)))


class SpatialTransformer(nn.Module):
    """Warp 3D tensors with channel-first voxel displacement fields."""

    def forward(
        self,
        image: torch.Tensor,
        transformation: torch.Tensor,
        default_value: float = 0.0,
        mode: str = "bilinear",
        padding_mode: str = "zeros",
    ) -> torch.Tensor:
        if image.ndim != 5 or transformation.ndim != 5:
            raise ValueError("Expected image and transformation to be five-dimensional")
        if transformation.shape[1] != 3 or image.shape[2:] != transformation.shape[2:]:
            raise ValueError(
                f"Incompatible image/DVF shapes: {image.shape}, {transformation.shape}"
            )

        spatial_shape = image.shape[2:]
        axes = [
            torch.arange(size, device=image.device, dtype=transformation.dtype)
            for size in spatial_shape
        ]
        identity = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=0).unsqueeze(0)
        grid = identity + transformation
        scale = torch.as_tensor(spatial_shape, device=image.device, dtype=grid.dtype)
        grid = 2.0 * (grid / (scale[None, :, None, None, None] - 1.0) - 0.5)
        grid = grid.permute(0, 2, 3, 4, 1)[..., [2, 1, 0]]

        image_dtype = image.dtype
        warped = F.grid_sample(
            image.float(),
            grid,
            mode=mode,
            padding_mode=padding_mode,
            align_corners=True,
        )
        if default_value != 0.0:
            inside = ((grid >= -1.0) & (grid <= 1.0)).all(dim=-1).unsqueeze(1)
            warped = torch.where(
                inside, warped, torch.as_tensor(default_value, device=image.device)
            )
        return warped.to(image_dtype)
