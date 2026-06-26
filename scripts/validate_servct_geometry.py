import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from surgtwin.cameras.projection import unproject_depth_to_points, project_points_to_image, compute_reprojection_error
from surgtwin.data.depth_io import load_servct_depth
from surgtwin.data.manifest import load_manifest, get_sample_by_index, validate_manifest_entry


def main():
    parser = argparse.ArgumentParser(description="Validate SERV-CT geometry.")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="outputs/debug/sprint0_geometry")
    parser.add_argument("--num_samples", type=int, default=500)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    entries = load_manifest(manifest_path)
    entry = get_sample_by_index(entries, args.sample_index)
    validate_manifest_entry(entry)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    left_rgb = cv2.cvtColor(cv2.imread(entry["left_rgb_path"]), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    left_depth = load_servct_depth(Path(entry["left_depth_path"]))

    K_left = torch.tensor(entry["K_left"], dtype=torch.float32)
    c2w_left = torch.tensor(entry["c2w_left"], dtype=torch.float32)
    w2c_left = torch.tensor(entry["w2c_left"], dtype=torch.float32)

    height, width = entry["height"], entry["width"]
    assert left_rgb.shape[0] == height and left_rgb.shape[1] == width, (
        f"RGB shape {left_rgb.shape[:2]} != manifest ({height}, {width})"
    )
    assert left_depth.shape == (height, width), (
        f"Depth shape {left_depth.shape} != manifest ({height}, {width})"
    )

    valid_mask = (left_depth > 0) & torch.isfinite(left_depth)
    valid_ratio = valid_mask.float().mean().item()
    depth_min = left_depth[valid_mask].min().item() if valid_mask.any() else 0.0
    depth_max = left_depth[valid_mask].max().item() if valid_mask.any() else 0.0
    depth_median = left_depth[valid_mask].median().item() if valid_mask.any() else 0.0

    E = w2c_left @ c2w_left - torch.eye(4, dtype=torch.float32)
    pose_rot_err = E[:3, :3].abs().max().item()
    pose_trans_err = E[:3, 3].abs().max().item()

    reproj = compute_reprojection_error(left_depth, K_left, c2w_left, w2c_left)

    geometry_report = {
        "sample_id": entry["sample_id"],
        "valid_depth_ratio": valid_ratio,
        "depth_min_m": depth_min,
        "depth_max_m": depth_max,
        "depth_median_m": depth_median,
        "pose_identity_rotation_error": pose_rot_err,
        "pose_identity_translation_error_m": pose_trans_err,
        "pose_rotation_error_pass": pose_rot_err < 1e-4,
        "pose_translation_error_pass": pose_trans_err < 1e-3,
        "reprojection": reproj,
    }

    cv2.imwrite(str(out_dir / "left_rgb.png"), cv2.cvtColor((left_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

    depth_color = (left_depth / (depth_max + 1e-8) * 255).byte().numpy()
    depth_color = cv2.applyColorMap(depth_color, cv2.COLORMAP_JET)
    cv2.imwrite(str(out_dir / "left_depth_color.png"), depth_color)

    overlay = (left_rgb.copy() * 255).astype(np.uint8)
    valid_idxs = torch.where(valid_mask)
    n_valid = valid_mask.sum().item()
    n_sample = min(args.num_samples, n_valid)
    if n_sample > 0:
        perm = torch.randperm(n_valid)[:n_sample]
        ys_sampled = valid_idxs[0][perm].numpy()
        xs_sampled = valid_idxs[1][perm].numpy()
        sampled_depths = left_depth[valid_idxs[0][perm], valid_idxs[1][perm]]
        valid_pts = torch.ones_like(sampled_depths, dtype=torch.bool)

        fx = K_left[0, 0].item()
        fy = K_left[1, 1].item()
        cx = K_left[0, 2].item()
        cy = K_left[1, 2].item()
        z_cam = sampled_depths.numpy()
        x_cam = (xs_sampled.astype(np.float32) - cx) * z_cam / fx
        y_cam = (ys_sampled.astype(np.float32) - cy) * z_cam / fy
        ones = np.ones_like(z_cam)
        pts_cam = np.stack([x_cam, y_cam, z_cam, ones], axis=-1)
        c2w_np = c2w_left.numpy()
        pts_world = (c2w_np @ pts_cam.T).T[:, :3]

        w2c_np = w2c_left.numpy()
        pts_h = np.concatenate([pts_world, np.ones((pts_world.shape[0], 1))], axis=-1)
        pts_cam_reproj = (w2c_np @ pts_h.T).T
        z_reproj = pts_cam_reproj[:, 2]
        x_reproj = pts_cam_reproj[:, 0] * fx / z_reproj + cx
        y_reproj = pts_cam_reproj[:, 1] * fy / z_reproj + cy
        uv_reproj = np.stack([x_reproj, y_reproj], axis=-1)

        for i in range(n_sample):
            ox, oy = int(round(xs_sampled[i])), int(round(ys_sampled[i]))
            cv2.circle(overlay, (ox, oy), 2, (0, 0, 255), -1)

            rx, ry = int(round(uv_reproj[i, 0])), int(round(uv_reproj[i, 1]))
            if 0 <= rx < width and 0 <= ry < height:
                cv2.circle(overlay, (rx, ry), 2, (0, 255, 0), -1)
                cv2.line(overlay, (ox, oy), (rx, ry), (255, 255, 0), 1)

    cv2.imwrite(str(out_dir / "reprojection_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    with open(out_dir / "geometry_report.json", "w") as f:
        json.dump(geometry_report, f, indent=2)

    print(f"Geometry report for {entry['sample_id']}:")
    print(f"  Valid depth ratio: {valid_ratio:.4f}")
    print(f"  Depth range: {depth_min:.4f} - {depth_max:.4f} m, median: {depth_median:.4f}")
    print(f"  Pose rotation error: {pose_rot_err:.6e} (pass={pose_rot_err < 1e-4})")
    print(f"  Pose translation error: {pose_trans_err:.6e} m (pass={pose_trans_err < 1e-3})")
    print(f"  Reprojection mean error: {reproj['mean_error_px']:.4f} px")
    print(f"Output saved to {out_dir}")


if __name__ == "__main__":
    main()
