from typing import Dict, Optional, Tuple

import torch


def compare_depth_distributions(
    rendered: torch.Tensor,
    gt: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> Dict:
    if rendered.shape != gt.shape:
        raise ValueError(f"Shape mismatch: rendered {rendered.shape} vs gt {gt.shape}")
    if valid_mask is None:
        valid_mask = (gt > 0) & torch.isfinite(gt)

    r_valid = rendered[valid_mask]
    g_valid = gt[valid_mask]

    def stats(t: torch.Tensor) -> dict:
        if t.numel() == 0:
            return {}
        return {
            "min": round(float(t.min()), 6),
            "max": round(float(t.max()), 6),
            "median": round(float(t.median()), 6),
            "mean": round(float(t.mean()), 6),
            "std": round(float(t.std()), 6),
        }

    return {
        "shape": list(rendered.shape),
        "dtype": str(rendered.dtype),
        "rendered": stats(r_valid),
        "gt": stats(g_valid),
        "valid_ratio": round(float(valid_mask.float().mean()), 6),
        "finite_ratio": round(float(torch.isfinite(rendered).float().mean()), 6),
        "rendered_has_invalid": bool((~torch.isfinite(rendered)).any().item()),
        "gt_has_invalid": bool((~torch.isfinite(gt)).any().item()),
    }


def median_align_depth(
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, float, float]:
    if valid_mask is None:
        valid_mask = (gt_depth > 0) & torch.isfinite(gt_depth)
    p_valid = pred_depth[valid_mask]
    g_valid = gt_depth[valid_mask]
    if p_valid.numel() < 10 or g_valid.numel() < 10:
        return pred_depth, 1.0, 0.0
    med_pred = float(p_valid.median())
    med_gt = float(g_valid.median())
    if med_pred < eps:
        return pred_depth, 1.0, 0.0
    scale = med_gt / med_pred
    aligned = pred_depth * scale
    shift = med_gt - aligned[valid_mask].median().item()
    return aligned, round(scale, 6), round(shift, 6)


def classify_scale(scale: Optional[float]) -> str:
    if scale is None:
        return "fail"
    rel = abs(scale - 1.0)
    if rel <= 0.05:
        return "green"
    elif rel <= 0.10:
        return "acceptable"
    elif rel <= 0.20:
        return "diagnostic"
    else:
        return "fail"


def depth_in_servct_range(
    stats: Dict,
    lo_m: float = 0.02,
    hi_m: float = 0.30,
) -> bool:
    rendered = stats.get("rendered", {})
    median = rendered.get("median")
    if median is None:
        return False
    return lo_m <= median <= hi_m


def depth_scale_ok(
    rendered: torch.Tensor,
    gt: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    tolerance: float = 0.10,
) -> Tuple[bool, str, float]:
    if valid_mask is None:
        valid_mask = (gt > 0) & torch.isfinite(gt)
    _, scale, _ = median_align_depth(rendered, gt, valid_mask)
    tier = classify_scale(scale)
    ok = tier in ("green", "acceptable")
    return ok, tier, scale
