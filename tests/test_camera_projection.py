import torch
import numpy as np
from surgtwin.cameras.projection import unproject_depth_to_points, project_points_to_image, compute_reprojection_error


def test_unproject_project_roundtrip_synthetic():
    H, W = 100, 200
    fx, fy, cx, cy = 200.0, 200.0, W / 2, H / 2
    K = torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=torch.float32)
    c2w = torch.eye(4, dtype=torch.float32)
    w2c = torch.eye(4, dtype=torch.float32)

    ys, xs = torch.meshgrid(
        torch.arange(H, dtype=torch.float32),
        torch.arange(W, dtype=torch.float32),
        indexing="ij",
    )
    depth = 0.5 + 0.1 * torch.sin(xs / W * 2 * np.pi) + 0.05 * torch.cos(ys / H * 2 * np.pi)

    reproj = compute_reprojection_error(depth, K, c2w, w2c)
    assert reproj["mean_error_px"] < 1e-3, f"Reprojection error too high: {reproj['mean_error_px']}"
    assert reproj["valid_points"] > 0


def test_unproject_project_roundtrip_single_pixel():
    H, W = 50, 50
    fx, fy, cx, cy = 100.0, 100.0, 25.0, 25.0
    K = torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=torch.float32)
    c2w = torch.eye(4, dtype=torch.float32)
    w2c = torch.eye(4, dtype=torch.float32)

    depth = torch.zeros(H, W, dtype=torch.float32)
    depth[10, 10] = 0.5

    points = unproject_depth_to_points(depth, K, c2w)
    assert points.shape[0] == 1, f"Expected 1 point, got {points.shape[0]}"
    uv, z = project_points_to_image(points, K, w2c, H, W)
    assert torch.allclose(uv, torch.tensor([[10.0, 10.0]]), atol=1e-4), f"UV mismatch: {uv}"
    assert torch.allclose(z, torch.tensor([0.5]), atol=1e-4), f"Depth mismatch: {z}"
