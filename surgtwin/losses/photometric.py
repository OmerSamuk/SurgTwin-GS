import torch


def photometric_l1(pred_rgb: torch.Tensor, gt_rgb: torch.Tensor) -> torch.Tensor:
    if pred_rgb.shape != gt_rgb.shape:
        raise ValueError(f"Shape mismatch: pred {pred_rgb.shape} vs gt {gt_rgb.shape}")
    return (pred_rgb[..., :3] - gt_rgb[..., :3]).abs().mean()
