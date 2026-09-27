import torch


def dice_score(
    y_pred: torch.Tensor, y_true: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    # Both tensors expected as binary with shape (1, 1, D, H, W).
    y_pred_f = y_pred.float().reshape(-1)
    y_true_f = y_true.float().reshape(-1)
    intersection = (y_pred_f * y_true_f).sum()
    denom = y_pred_f.sum() + y_true_f.sum()
    return (2.0 * intersection + eps) / (denom + eps)


def soft_dice_score(
    y_pred: torch.Tensor, y_true: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    # Both tensors expected as probabilities with shape (1, 1, D, H, W).
    y_pred_f = y_pred.float().reshape(-1)
    y_true_f = y_true.float().reshape(-1)
    intersection = (y_pred_f * y_true_f).sum()
    denom = y_pred_f.sum() + y_true_f.sum()
    return (2.0 * intersection + eps) / (denom + eps)


def hausdorff_distance(
    y_pred: torch.Tensor, y_true: torch.Tensor, spacing: tuple[float, ...]
) -> torch.Tensor:
    # Both tensors expected as binary with shape (1, 1, D, H, W).
    pred_idx = torch.nonzero(y_pred > 0.5, as_tuple=False)
    true_idx = torch.nonzero(y_true > 0.5, as_tuple=False)

    if pred_idx.numel() == 0 and true_idx.numel() == 0:
        return torch.tensor(0.0)
    if pred_idx.numel() == 0 or true_idx.numel() == 0:
        return torch.tensor(0.0)

    # Drop batch/channel dims, keep spatial (D, H, W).
    pred_pts = pred_idx.squeeze(0).squeeze(0).float()
    true_pts = true_idx.squeeze(0).squeeze(0).float()

    spacing_t = torch.tensor(spacing, dtype=pred_pts.dtype)
    pred_pts = pred_pts * spacing_t
    true_pts = true_pts * spacing_t

    dists = torch.cdist(pred_pts, true_pts, p=2)
    hd_pred = dists.min(dim=1).values.max()
    hd_true = dists.min(dim=0).values.max()
    return torch.max(hd_pred, hd_true)
