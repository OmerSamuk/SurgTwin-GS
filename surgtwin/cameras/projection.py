import torch


def unproject_depth_to_points(
    depth_m: torch.Tensor, K: torch.Tensor, c2w: torch.Tensor, valid_mask: torch.Tensor = None
) -> torch.Tensor:
    H, W = depth_m.shape
    if valid_mask is None:
        valid_mask = (depth_m > 0) & torch.isfinite(depth_m)
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    ys, xs = torch.meshgrid(
        torch.arange(H, device=depth_m.device, dtype=torch.float32),
        torch.arange(W, device=depth_m.device, dtype=torch.float32),
        indexing="ij",
    )
    z_cam = depth_m[valid_mask]
    x_cam = (xs[valid_mask] - cx) * z_cam / fx
    y_cam = (ys[valid_mask] - cy) * z_cam / fy
    ones = torch.ones_like(z_cam)
    points_cam = torch.stack([x_cam, y_cam, z_cam, ones], dim=-1)
    points_world = (c2w @ points_cam.T).T[:, :3]
    return points_world


def project_points_to_image(
    points_world: torch.Tensor, K: torch.Tensor, w2c: torch.Tensor, height: int, width: int
) -> tuple:
    ones = torch.ones(points_world.shape[0], 1, device=points_world.device, dtype=points_world.dtype)
    points_h = torch.cat([points_world, ones], dim=-1)
    points_cam = (w2c @ points_h.T).T
    z_cam = points_cam[:, 2]
    x_img = points_cam[:, 0] * K[0, 0] / z_cam + K[0, 2]
    y_img = points_cam[:, 1] * K[1, 1] / z_cam + K[1, 2]
    uv = torch.stack([x_img, y_img], dim=-1)
    return uv, z_cam


def compute_reprojection_error(
    depth_m: torch.Tensor, K: torch.Tensor, c2w: torch.Tensor, w2c: torch.Tensor
) -> dict:
    valid = (depth_m > 0) & torch.isfinite(depth_m)
    if valid.sum() < 10:
        return {"mean_error_px": float("inf"), "valid_points": int(valid.sum().item())}
    points_world = unproject_depth_to_points(depth_m, K, c2w, valid)
    uv_reproj, _ = project_points_to_image(points_world, K, w2c, depth_m.shape[0], depth_m.shape[1])
    ys, xs = torch.meshgrid(
        torch.arange(depth_m.shape[0], device=depth_m.device, dtype=torch.float32),
        torch.arange(depth_m.shape[1], device=depth_m.device, dtype=torch.float32),
        indexing="ij",
    )
    uv_orig = torch.stack([xs[valid], ys[valid]], dim=-1)
    err = torch.norm(uv_reproj - uv_orig, dim=-1)
    return {
        "mean_error_px": err.mean().item(),
        "max_error_px": err.max().item(),
        "valid_points": int(valid.sum().item()),
    }
