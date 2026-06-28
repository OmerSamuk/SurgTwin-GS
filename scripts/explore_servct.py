import argparse
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from surgtwin.data.manifest import assign_split
from surgtwin.data.servct_calibration import parse_rectified_calibration, rectified_to_camera_data


def main():
    parser = argparse.ArgumentParser(description="Build SERV-CT manifest.")
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--output_manifest", type=str, required=True)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    if not dataset_root.exists():
        raise FileNotFoundError(f"SERV-CT dataset root not found: {dataset_root}")

    entries = []
    experiment_dirs = sorted(d for d in dataset_root.iterdir() if d.is_dir() and d.name.startswith("Experiment_"))

    if not experiment_dirs:
        raise FileNotFoundError(
            f"No Experiment_* directories found in {dataset_root}. "
            "SERV-CT directory layout: Experiment_1/ (frames 001-008), Experiment_2/ (frames 009-016)"
        )

    for exp_dir in experiment_dirs:
        left_dir = exp_dir / "Left_rectified"
        right_dir = exp_dir / "Right_rectified"
        gt_dir = exp_dir / "Ground_truth_CT"
        calib_dir = exp_dir / "Rectified_calibration"

        if not all(d.exists() for d in [left_dir, right_dir, gt_dir, calib_dir]):
            raise FileNotFoundError(
                f"Missing subdirectory in {exp_dir}. "
                "Expected: Left_rectified/, Right_rectified/, Ground_truth_CT/, Rectified_calibration/"
            )

        frames = sorted(left_dir.glob("*.png"))
        if not frames:
            raise FileNotFoundError(f"No PNG files found in {left_dir}")

        for frame_path in frames:
            frame_name = frame_path.stem
            sample_id = f"{exp_dir.name}_{frame_name}"

            left_rgb = left_dir / f"{frame_name}.png"
            right_rgb = right_dir / f"{frame_name}.png"
            left_depth = gt_dir / "DepthL" / f"{frame_name}.png"
            right_depth = gt_dir / "DepthR" / f"{frame_name}.png"

            img = cv2.imread(str(left_rgb))
            if img is None:
                raise FileNotFoundError(f"Failed to read left RGB image: {left_rgb}")
            height, width = img.shape[:2]

            calib_json = calib_dir / "Calibration.json"
            calib_alt = calib_dir / f"{frame_name}.json"
            if calib_json.exists():
                calib_path = calib_json
            elif calib_alt.exists():
                calib_path = calib_alt
            else:
                calib_path = list(calib_dir.glob("*.json"))
                if not calib_path:
                    raise FileNotFoundError(f"No JSON calibration file found in {calib_dir}")
                calib_path = calib_path[0]

            P1, P2, Q = parse_rectified_calibration(calib_path)
            cam_data = rectified_to_camera_data(P1, P2, Q, height, width)

            entry = {
                "sample_id": sample_id,
                "sequence_id": exp_dir.name,
                "frame_index": int(frame_name),
                "left_rgb_path": str(left_rgb.resolve()),
                "right_rgb_path": str(right_rgb.resolve()),
                "left_depth_path": str(left_depth.resolve()),
                "right_depth_path": str(right_depth.resolve()) if right_depth.exists() else None,
                "K_left": cam_data["K_left"].tolist(),
                "K_right": cam_data["K_right"].tolist(),
                "c2w_left": cam_data["c2w_left"].tolist(),
                "c2w_right": cam_data["c2w_right"].tolist(),
                "w2c_left": cam_data["w2c_left"].tolist(),
                "w2c_right": cam_data["w2c_right"].tolist(),
                "height": height,
                "width": width,
                "depth_unit": "meter",
                "depth_scale_applied": 0.000256,
            }
            entries.append(entry)

    assign_split(entries)

    out_path = Path(args.output_manifest)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    print(f"Manifest written to {out_path} with {len(entries)} entries.")
    print(f"Detected experiments: {[e.name for e in experiment_dirs]}")
    print(f"Frames per experiment: {[len(list(e.glob('Left_rectified/*.png'))) for e in experiment_dirs]}")


if __name__ == "__main__":
    main()
