from typing import Dict, Optional, Tuple

import torch

from surgtwin.uncertainty.signals import (
    compute_photo_residual,
    compute_p95_scale,
    compute_u_photo,
    compute_w_photo,
    compute_w_photo_with_mask,
    w_photo_distribution_stats,
)


def uncertainty_weighted_photometric_l1(
    rgb_pred: torch.Tensor,
    rgb_gt: torch.Tensor,
    alpha: float = 2.0,
    w_min: float = 0.15,
    mask: Optional[torch.Tensor] = None,
    mask_boost: float = 0.5,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    if rgb_pred.shape != rgb_gt.shape:
        raise ValueError(f"Shape mismatch: pred {rgb_pred.shape} vs gt {rgb_gt.shape}")

    residual = compute_photo_residual(rgb_pred, rgb_gt, detach_pred=True)
    scale = compute_p95_scale(residual)
    u_photo = compute_u_photo(residual, scale)

    if mask is not None:
        w_photo = compute_w_photo_with_mask(
            u_photo, mask, alpha=alpha, w_min=w_min, mask_boost=mask_boost
        )
    else:
        w_photo = compute_w_photo(u_photo, alpha=alpha, w_min=w_min)

    diff = (rgb_pred[..., :3] - rgb_gt[..., :3]).abs().mean(dim=-1)
    eps = 1e-6
    loss_weighted = (w_photo * diff).sum() / w_photo.sum().clamp(min=eps)

    with torch.no_grad():
        diagnostics = w_photo_distribution_stats(w_photo, w_min)
        diagnostics["u_photo_mean"] = u_photo.mean().item()
        diagnostics["u_photo_median"] = u_photo.median().item()
        diagnostics["p95_scale"] = scale.item()
        diagnostics["normalization_mode"] = "p95_detached"
        diagnostics["mask_used"] = mask is not None
        if mask is not None:
            mask_f = mask.float()
            if mask.any():
                diagnostics["w_photo_in_mask_mean"] = w_photo[mask].mean().item()
            else:
                diagnostics["w_photo_in_mask_mean"] = None
            diagnostics["w_photo_out_mask_mean"] = w_photo[~mask].mean().item()
            diagnostics["mask_coverage"] = mask_f.mean().item()
        loss_clone = loss_weighted.detach()

    return loss_weighted, diagnostics
