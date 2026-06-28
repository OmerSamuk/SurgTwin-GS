from typing import Dict, Optional, Tuple

import torch

from surgtwin.evaluation.depth_diagnostics import median_align_depth

_METRIC_GUARD = (
    "depth_semantics='{semantics}' but metric depth required. "
    "All geometry metrics reject non-metric depth to prevent silent scale errors."
)


def _validate_metric_depth(semantics: str) -> None:
    if semantics != "metric_meters":
        raise ValueError(_METRIC_GUARD.format(semantics=semantics))


def depth_rmse(
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    valid_mask: torch.Tensor,
    depth_semantics: str,
) -> torch.Tensor:
    _validate_metric_depth(depth_semantics)
    if not valid_mask.any():
        return torch.tensor(0.0, device=pred_depth.device)
    return ((pred_depth[valid_mask] - gt_depth[valid_mask]) ** 2).mean().sqrt()


def depth_rmse_clipped(
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    valid_mask: torch.Tensor,
    depth_semantics: str,
    near_m: float = 0.02,
    far_m: float = 0.30,
) -> torch.Tensor:
    _validate_metric_depth(depth_semantics)
    if not valid_mask.any():
        return torch.tensor(0.0, device=pred_depth.device)
    clipped = pred_depth.clamp(near_m, far_m)
    return ((clipped[valid_mask] - gt_depth[valid_mask]) ** 2).mean().sqrt()


def depth_mae(
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    valid_mask: torch.Tensor,
    depth_semantics: str,
) -> torch.Tensor:
    _validate_metric_depth(depth_semantics)
    if not valid_mask.any():
        return torch.tensor(0.0, device=pred_depth.device)
    return (pred_depth[valid_mask] - gt_depth[valid_mask]).abs().mean()


def depth_mae_clipped(
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    valid_mask: torch.Tensor,
    depth_semantics: str,
    near_m: float = 0.02,
    far_m: float = 0.30,
) -> torch.Tensor:
    _validate_metric_depth(depth_semantics)
    if not valid_mask.any():
        return torch.tensor(0.0, device=pred_depth.device)
    clipped = pred_depth.clamp(near_m, far_m)
    return (clipped[valid_mask] - gt_depth[valid_mask]).abs().mean()


def abs_rel(
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    valid_mask: torch.Tensor,
    depth_semantics: str,
    eps: float = 1e-8,
) -> torch.Tensor:
    _validate_metric_depth(depth_semantics)
    if not valid_mask.any():
        return torch.tensor(0.0, device=pred_depth.device)
    return (
        (pred_depth[valid_mask] - gt_depth[valid_mask]).abs()
        / gt_depth[valid_mask].abs().clamp(min=eps)
    ).mean()


def valid_depth_ratio(valid_mask: torch.Tensor) -> torch.Tensor:
    return valid_mask.float().mean()


def delta_thresholds(
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    valid_mask: torch.Tensor,
    depth_semantics: str,
    thresholds: Tuple[float, ...] = (1.01, 1.02, 1.03, 1.05, 1.10),
) -> Dict[str, float]:
    _validate_metric_depth(depth_semantics)
    result = {}
    if not valid_mask.any():
        for t in thresholds:
            result[f"delta_{t:.2f}"] = 0.0
        return result
    ratio = pred_depth[valid_mask] / gt_depth[valid_mask].clamp(min=1e-8)
    for t in thresholds:
        result[f"delta_{t:.2f}"] = float((ratio < t).float().mean().item())
    return result


def median_aligned_rmse(
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    aligned, scale, shift = median_align_depth(pred_depth, gt_depth, valid_mask, eps)
    if valid_mask is None:
        valid_mask = (gt_depth > 0) & torch.isfinite(gt_depth)
    if not valid_mask.any():
        return torch.tensor(0.0, device=pred_depth.device)
    return ((aligned[valid_mask] - gt_depth[valid_mask]) ** 2).mean().sqrt()


def geometry_metrics_report(
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    depth_semantics: str,
    near_m: float = 0.02,
    far_m: float = 0.30,
) -> Dict[str, float]:
    valid_mask = (
        torch.isfinite(gt_depth)
        & torch.isfinite(pred_depth)
        & (gt_depth >= near_m)
        & (gt_depth <= far_m)
    )

    report = {
        "depth_rmse_m_raw": float(depth_rmse(pred_depth, gt_depth, valid_mask, depth_semantics).item()),
        "depth_rmse_m_clipped": float(depth_rmse_clipped(pred_depth, gt_depth, valid_mask, depth_semantics, near_m, far_m).item()),
        "depth_mae_m_raw": float(depth_mae(pred_depth, gt_depth, valid_mask, depth_semantics).item()),
        "depth_mae_m_clipped": float(depth_mae_clipped(pred_depth, gt_depth, valid_mask, depth_semantics, near_m, far_m).item()),
        "abs_rel": float(abs_rel(pred_depth, gt_depth, valid_mask, depth_semantics).item()),
        "depth_valid_ratio": float(valid_depth_ratio(valid_mask).item()),
        "depth_semantics": depth_semantics,
    }

    deltas = delta_thresholds(pred_depth, gt_depth, valid_mask, depth_semantics)
    report.update(deltas)

    aligned_rmse_val = median_aligned_rmse(pred_depth, gt_depth, valid_mask)
    report["median_aligned_rmse_m"] = float(aligned_rmse_val.item())

    return report
