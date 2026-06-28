from typing import Dict, Tuple

import torch


_DEPTH_SEMANTICS_ERROR = (
    "Depth loss requires depth_semantics='metric_meters', "
    "but got '{semantics}'. "
    "Run M2-A depth semantics verification first."
)


def depth_l1(
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    depth_semantics: str,
    near_m: float = 0.02,
    far_m: float = 0.30,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    if depth_semantics != "metric_meters":
        raise ValueError(_DEPTH_SEMANTICS_ERROR.format(semantics=depth_semantics))
    if pred_depth.shape != gt_depth.shape:
        raise ValueError(f"Shape mismatch: pred {pred_depth.shape} vs gt {gt_depth.shape}")

    valid = (
        torch.isfinite(gt_depth)
        & torch.isfinite(pred_depth)
        & (gt_depth >= near_m)
        & (gt_depth <= far_m)
    )

    pred_clamped = pred_depth.clamp(near_m, far_m)
    diff_abs = (pred_clamped - gt_depth).abs()
    diff_raw_abs = (pred_depth - gt_depth).abs()

    if valid.any():
        depth_loss = diff_abs[valid].mean()
        rmse_raw = ((pred_depth - gt_depth) ** 2)[valid].mean().sqrt()
        rmse_clipped = ((pred_clamped - gt_depth) ** 2)[valid].mean().sqrt()
        mae_raw = diff_raw_abs[valid].mean()
        mae_clipped = diff_abs[valid].mean()
        abs_rel = (diff_raw_abs / gt_depth.abs().clamp(min=1e-8))[valid].mean()
    else:
        device = pred_depth.device
        depth_loss = torch.tensor(0.0, device=device)
        rmse_raw = torch.tensor(0.0, device=device)
        rmse_clipped = torch.tensor(0.0, device=device)
        mae_raw = torch.tensor(0.0, device=device)
        mae_clipped = torch.tensor(0.0, device=device)
        abs_rel = torch.tensor(0.0, device=device)

    valid_ratio = valid.float().mean()

    diagnostics = {
        "depth_loss_raw_m": diff_raw_abs[valid].mean() if valid.any() else torch.tensor(0.0, device=pred_depth.device),
        "depth_loss_weighted": depth_loss.detach(),
        "depth_rmse_m_raw": rmse_raw.detach(),
        "depth_rmse_m_clipped": rmse_clipped.detach(),
        "depth_mae_m_raw": mae_raw.detach(),
        "depth_mae_m_clipped": mae_clipped.detach(),
        "abs_rel": abs_rel.detach(),
        "depth_valid_ratio": valid_ratio.detach(),
    }

    return depth_loss, diagnostics
