import torch


def initialize_gaussians_from_rgbd(
    rgb: torch.Tensor,
    depth_m: torch.Tensor,
    K: torch.Tensor,
    c2w: torch.Tensor,
    num_points: int = 20000,
) -> dict:
    H, W = depth_m.shape
    valid = (depth_m > 0) & torch.isfinite(depth_m)
    if valid.sum() == 0:
        raise ValueError("No valid depth pixels found for Gaussian initialization.")

    idxs = torch.where(valid)
    n_valid = valid.sum().item()
    n_sample = min(num_points, n_valid)
    perm = torch.randperm(n_valid, device=depth_m.device)[:n_sample]
    ys = idxs[0][perm]
    xs = idxs[1][perm]
    sampled_depths = depth_m[ys, xs]
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    z_cam = sampled_depths
    x_cam = (xs.float() - cx) * z_cam / fx
    y_cam = (ys.float() - cy) * z_cam / fy
    ones = torch.ones_like(z_cam)
    points_cam = torch.stack([x_cam, y_cam, z_cam, ones], dim=-1)
    means = (c2w @ points_cam.T).T[:, :3]

    colors = rgb[ys, xs]

    scales = torch.clamp(sampled_depths.unsqueeze(-1) * 0.002, min=1e-5, max=3e-3)
    scales = scales.expand(-1, 3).contiguous()

    quats = torch.zeros(n_sample, 4, device=depth_m.device)
    quats[:, 0] = 1.0

    opacity_logit = torch.logit(torch.tensor(0.1, device=depth_m.device))
    opacities = opacity_logit.expand(n_sample)

    return {
        "means": means,
        "scales": scales,
        "quats": quats,
        "opacities": opacities,
        "colors": colors,
        "reliability_logits": torch.zeros(n_sample, device=depth_m.device),
    }
