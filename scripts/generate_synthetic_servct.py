import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from surgtwin.data.servct_calibration import rectified_to_camera_data


def create_synthetic_plane_scene(height=576, width=720):
    xx, yy = np.meshgrid(np.arange(width), np.arange(height))
    depth_mm = 50.0 + np.clip(0.5 * yy, 0, 100)

    depth_mm = depth_mm.astype(np.float32)

    r = np.clip(0.5 + 0.3 * np.cos(yy / height * 2 * np.pi), 0, 1).astype(np.float32)
    g = np.clip(0.5 + 0.3 * np.sin(xx / width * 2 * np.pi), 0, 1).astype(np.float32)
    b = np.clip(0.5 + 0.3 * np.cos((xx + yy) / (width + height) * 2 * np.pi), 0, 1).astype(np.float32)

    rgb = (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)

    baseline_mm = 5.0
    K = np.array([[500.0, 0.0, width / 2], [0.0, 500.0, height / 2], [0.0, 0.0, 1.0]], dtype=np.float64)
    P1 = np.hstack([K, np.zeros((3, 1), dtype=np.float64)])
    P2 = np.hstack([K, np.array([[-K[0, 0] * baseline_mm], [0.0], [0.0]], dtype=np.float64)])
    Q = np.array(
        [
            [1.0, 0.0, 0.0, -K[0, 2]],
            [0.0, 1.0, 0.0, -K[1, 2]],
            [0.0, 0.0, 0.0, K[0, 0]],
            [0.0, 0.0, 1.0 / baseline_mm, 0.0],
        ],
        dtype=np.float64,
    )

    return rgb, depth_mm, P1, P2, Q, (height, width)


def depth_to_servct_raw(depth_mm: np.ndarray) -> np.ndarray:
    raw = np.round(depth_mm * 256.0).astype(np.uint16)
    return raw


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic SERV-CT scene.")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--experiment_name", type=str, default="Experiment_1")
    parser.add_argument("--num_frames", type=int, default=4)
    args = parser.parse_args()

    out_root = Path(args.output_dir)
    exp_dir = out_root / args.experiment_name
    subdirs = ["Left_rectified", "Right_rectified", "Ground_truth_CT/DepthL", "Ground_truth_CT/DepthR",
               "Ground_truth_CT/OcclusionL", "Ground_truth_CT/OcclusionR", "Rectified_calibration"]
    for sd in subdirs:
        (exp_dir / sd).mkdir(parents=True, exist_ok=True)

    rgb_ref, depth_mm_ref, P1, P2, Q, (H, W) = create_synthetic_plane_scene()

    right_shift_px = 1
    rgb_right = np.zeros_like(rgb_ref)

    calibration = {
        "P1": P1.tolist(),
        "P2": P2.tolist(),
        "Q": Q.tolist(),
    }

    calib_data = rectified_to_camera_data(P1, P2, Q, H, W)

    for i in range(args.num_frames):
        frame_name = f"{i + 1:03d}"
        z_offset = i * 5.0
        depth_mm = depth_mm_ref + z_offset

        cv2.imwrite(str(exp_dir / f"Left_rectified/{frame_name}.png"),
                    cv2.cvtColor(rgb_ref, cv2.COLOR_RGB2BGR))

        if i % 2 == 0:
            rgb_right[:, :-right_shift_px] = rgb_ref[:, right_shift_px:]
            rgb_right[:, -right_shift_px:] = 0
        else:
            rgb_right[:, right_shift_px:] = rgb_ref[:, :-right_shift_px]
            rgb_right[:, :right_shift_px] = 0

        cv2.imwrite(str(exp_dir / f"Right_rectified/{frame_name}.png"),
                    cv2.cvtColor(rgb_right, cv2.COLOR_RGB2BGR))

        depth_raw = depth_to_servct_raw(depth_mm)
        cv2.imwrite(str(exp_dir / f"Ground_truth_CT/DepthL/{frame_name}.png"), depth_raw)

        right_depth_mm = depth_mm
        right_depth_raw = depth_to_servct_raw(right_depth_mm)
        cv2.imwrite(str(exp_dir / f"Ground_truth_CT/DepthR/{frame_name}.png"), right_depth_raw)

        occlusion = np.ones((H, W, 3), dtype=np.uint8) * 255
        cv2.imwrite(str(exp_dir / f"Ground_truth_CT/OcclusionL/{frame_name}.png"),
                    cv2.cvtColor(occlusion, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(exp_dir / f"Ground_truth_CT/OcclusionR/{frame_name}.png"),
                    cv2.cvtColor(occlusion, cv2.COLOR_RGB2BGR))

    calib_path = exp_dir / "Rectified_calibration/Calibration.json"
    with open(calib_path, "w") as f:
        json.dump(calibration, f, indent=2)

    print(f"Synthetic SERV-CT data written to {exp_dir}")
    print(f"  Frames: {args.num_frames} ({args.experiment_name})")
    print(f"  Resolution: {W}x{H}")
    print(f"  Depth range: {depth_mm_ref.min():.1f} - {depth_mm_ref.max():.1f} mm")
    print(f"  Calibration: {calib_path}")
    print(f"  Baseline: {-P2[0, 3] / P2[0, 0]:.2f} mm")
    print(f"  Left K: fx={calib_data['K_left'][0,0].item():.1f}, fy={calib_data['K_left'][1,1].item():.1f}")


if __name__ == "__main__":
    main()
