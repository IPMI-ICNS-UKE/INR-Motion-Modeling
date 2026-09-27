import torch


def ncc_per_batch(
    x1: torch.Tensor,
    x2: torch.Tensor,
    mask: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Args:
        x1: (B, N) intensities from image 1
        x2: (B, N) intensities from image 2
        mask: (B, N) binary mask where 1=valid, 0=invalid, or None
        eps: numerical stability constant
    """
    if x1.shape != x2.shape:
        raise ValueError(
            f"NCC inputs must have equal shapes, got {x1.shape} and {x2.shape}"
        )
    if x1.ndim != 2:
        raise ValueError(f"NCC inputs must have shape (B, N), got {x1.shape}")

    if mask is not None:
        # Ensure mask is binary
        mask = mask.float()

        # Compute masked means
        mask_sum = mask.sum(dim=1, keepdim=True).clamp(min=1)  # Avoid division by zero
        x1_mean = (x1 * mask).sum(dim=1, keepdim=True) / mask_sum
        x2_mean = (x2 * mask).sum(dim=1, keepdim=True) / mask_sum

        # Center the data
        x1_centered = (x1 - x1_mean) * mask
        x2_centered = (x2 - x2_mean) * mask

        # Compute masked cross-correlation and standard deviations
        valid_count = mask_sum.squeeze(1)
        cc = (x1_centered * x2_centered).sum(dim=1) / valid_count
        std1 = torch.sqrt(((x1_centered**2).sum(dim=1) / valid_count) + eps)
        std2 = torch.sqrt(((x2_centered**2).sum(dim=1) / valid_count) + eps)
    else:
        # Original unmasked version
        x1_mean = x1.mean(dim=1, keepdim=True)
        x2_mean = x2.mean(dim=1, keepdim=True)
        x1_centered = x1 - x1_mean
        x2_centered = x2 - x2_mean

        cc = (x1_centered * x2_centered).mean(dim=1)
        std1 = torch.sqrt((x1_centered**2).mean(dim=1) + eps)
        std2 = torch.sqrt((x2_centered**2).mean(dim=1) + eps)

    # Normalized cross-correlation
    ncc = cc / (std1 * std2)

    return 1.0 - ncc.mean()
