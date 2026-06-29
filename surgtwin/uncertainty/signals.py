from typing import Optional, Tuple

import torch


def compute_photo_residual(
    rgb_pred: torch.Tensor,
    rgb_gt: torch.Tensor,
    detach_pred: bool = True,
) -> torch.Tensor:
    if rgb_pred.shape != rgb_gt.shape:
        raise ValueError(f"Shape mismatch: pred {rgb_pred.shape} vs gt {rgb_gt.shape}")
    pred = rgb_pred.detach() if detach_pred else rgb_pred
    residual = (pred[..., :3] - rgb_gt[..., :3]).abs().mean(dim=-1)
    return residual


def compute_p95_scale(
    rgb_residual: torch.Tensor,
    eps: float = 1e-4,
) -> torch.Tensor:
    flat = rgb_residual.flatten()
    q = torch.quantile(flat, 0.95).detach()
    scale = q.clamp(min=eps)
    return scale


def compute_u_photo(
    rgb_residual: torch.Tensor,
    scale: torch.Tensor,
    clamp_max: float = 1.0,
) -> torch.Tensor:
    return (rgb_residual / scale).clamp(0.0, clamp_max)


def compute_w_photo(
    u_photo: torch.Tensor,
    alpha: float = 2.0,
    w_min: float = 0.15,
) -> torch.Tensor:
    return torch.exp(-alpha * u_photo).clamp(w_min, 1.0)


def compute_w_photo_with_mask(
    u_photo: torch.Tensor,
    mask: Optional[torch.Tensor],
    alpha: float = 2.0,
    w_min: float = 0.15,
    mask_boost: float = 0.5,
) -> torch.Tensor:
    if mask is not None:
        if mask.shape != u_photo.shape:
            raise ValueError(f"Mask shape {mask.shape} != u_photo shape {u_photo.shape}")
        u_photo = (u_photo + mask_boost * mask.float()).clamp(0.0, 1.0)
    return torch.exp(-alpha * u_photo).clamp(w_min, 1.0)


def w_photo_distribution_stats(w_photo: torch.Tensor, w_min: float = 0.15) -> dict:
    flat = w_photo.flatten()
    with torch.no_grad():
        p10 = torch.quantile(flat, 0.10).item()
        p50 = torch.quantile(flat, 0.50).item()
        p90 = torch.quantile(flat, 0.90).item()
        min_val = flat.min().item()
        max_val = flat.max().item()
        mean_val = flat.mean().item()
        frac_at_min = (flat <= w_min + 1e-6).float().mean().item()
        frac_at_one = (flat >= 1.0 - 1e-6).float().mean().item()
        p90_minus_p10 = p90 - p10
    return {
        "w_photo_mean": mean_val,
        "w_photo_min": min_val,
        "w_photo_max": max_val,
        "w_photo_p10": p10,
        "w_photo_p50": p50,
        "w_photo_p90": p90,
        "fraction_w_photo_at_min": frac_at_min,
        "fraction_w_photo_at_one": frac_at_one,
        "w_photo_p90_minus_p10": p90_minus_p10,
    }
