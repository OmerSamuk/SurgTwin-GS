import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def parse_poses_from_file(pose_path: Path) -> Optional[np.ndarray]:
    ext = pose_path.suffix.lower()
    if ext == ".npy":
        arr = np.load(str(pose_path))
        if arr.ndim == 3 and arr.shape[1:] == (4, 4):
            return arr.astype(np.float64)
        return None
    elif ext == ".csv":
        return _parse_poses_csv(pose_path)
    elif ext == ".json":
        return _parse_poses_json(pose_path)
    elif ext == ".txt":
        return _parse_poses_txt(pose_path)
    return None


def _parse_poses_csv(path: Path) -> Optional[np.ndarray]:
    try:
        data = np.loadtxt(str(path), delimiter=",")
    except Exception:
        return None
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.ndim == 2 and data.shape[1] == 16:
        n = data.shape[0]
        poses = data.reshape(n, 4, 4)
        return poses.astype(np.float64)
    if data.ndim == 2 and data.shape[1] == 7:
        return _poses_from_quat_trans(data)
    if data.ndim == 2 and data.shape[1] == 12:
        n = data.shape[0]
        poses = np.zeros((n, 4, 4), dtype=np.float64)
        poses[:, :3, :] = data.reshape(n, 3, 4)
        poses[:, 3, 3] = 1.0
        return poses
    return None


def _parse_poses_json(path: Path) -> Optional[np.ndarray]:
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None
    if isinstance(data, list) and len(data) > 0:
        matrices = []
        for item in data:
            mat = None
            if isinstance(item, dict):
                for key in ("c2w", "pose", "matrix", "transform"):
                    val = item.get(key)
                    if val is not None:
                        mat = np.array(val, dtype=np.float64).reshape(4, 4)
                        break
            elif isinstance(item, list) and len(item) == 16:
                mat = np.array(item, dtype=np.float64).reshape(4, 4)
            if mat is not None:
                matrices.append(mat)
        if matrices:
            return np.stack(matrices, axis=0)
    if isinstance(data, dict):
        for key in ("c2w", "poses", "matrices", "camera_poses"):
            val = data.get(key)
            if val is not None:
                arr = np.array(val, dtype=np.float64)
                if arr.ndim == 3 and arr.shape[1:] == (4, 4):
                    return arr
    return None


def _parse_poses_txt(path: Path) -> Optional[np.ndarray]:
    try:
        lines = [l.strip() for l in path.read_text().splitlines() if l.strip()]
    except Exception:
        return None
    matrices = []
    current = []
    for line in lines:
        if line.startswith("#") or line.startswith("//"):
            continue
        try:
            row = [float(x) for x in line.replace(",", " ").split()]
        except ValueError:
            continue
        if len(row) == 4:
            current.append(row)
            if len(current) == 4:
                matrices.append(np.array(current, dtype=np.float64))
                current = []
    if len(current) == 0 and matrices:
        return np.stack(matrices, axis=0)
    return None


def _poses_from_quat_trans(data: np.ndarray) -> np.ndarray:
    n = data.shape[0]
    poses = np.zeros((n, 4, 4), dtype=np.float64)
    for i in range(n):
        qw, qx, qy, qz = data[i, 0], data[i, 1], data[i, 2], data[i, 3]
        tx, ty, tz = data[i, 4], data[i, 5], data[i, 6]
        qx2 = qx * qx; qy2 = qy * qy; qz2 = qz * qz
        poses[i, 0, 0] = 1 - 2*(qy2 + qz2)
        poses[i, 0, 1] = 2*(qx*qy - qz*qw)
        poses[i, 0, 2] = 2*(qx*qz + qy*qw)
        poses[i, 0, 3] = tx
        poses[i, 1, 0] = 2*(qx*qy + qz*qw)
        poses[i, 1, 1] = 1 - 2*(qx2 + qz2)
        poses[i, 1, 2] = 2*(qy*qz - qx*qw)
        poses[i, 1, 3] = ty
        poses[i, 2, 0] = 2*(qx*qz - qy*qw)
        poses[i, 2, 1] = 2*(qy*qz + qx*qw)
        poses[i, 2, 2] = 1 - 2*(qx2 + qy2)
        poses[i, 2, 3] = tz
        poses[i, 3, 3] = 1.0
    return poses


def parse_intrinsics(intrinsics_path: Path) -> Optional[np.ndarray]:
    ext = intrinsics_path.suffix.lower()
    if ext == ".npy":
        arr = np.load(str(intrinsics_path))
        if arr.shape == (3, 3) or arr.shape == (4, 4):
            return arr[:3, :3].astype(np.float64)
        return None
    if ext == ".json":
        try:
            data = json.loads(intrinsics_path.read_text())
        except Exception:
            return None
        if isinstance(data, dict):
            for key in ("K", "intrinsics", "camera_matrix", "calibration"):
                val = data.get(key)
                if val is not None:
                    return np.array(val, dtype=np.float64).reshape(3, 3)
        return None
    if ext == ".txt":
        try:
            K = np.loadtxt(str(intrinsics_path))
            if K.shape == (3, 3):
                return K.astype(np.float64)
            if K.size == 4:
                fx, fy, cx, cy = K.flatten()
                return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        except Exception:
            return None
    if ext == ".csv":
        try:
            K = np.loadtxt(str(intrinsics_path), delimiter=",")
            if K.shape == (3, 3):
                return K.astype(np.float64)
            if K.size == 4:
                fx, fy, cx, cy = K.flatten()
                return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        except Exception:
            return None
    return None


def determine_convention(c2w_candidate: np.ndarray) -> str:
    if c2w_candidate.ndim != 3 or c2w_candidate.shape[1:] != (4, 4):
        return "unknown"
    identity_ratio = 0
    n = c2w_candidate.shape[0]
    for i in range(min(n, 10)):
        R = c2w_candidate[i, :3, :3]
        trace = np.trace(R)
        angle = np.degrees(np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0)))
        if angle < 1.0:
            identity_ratio += 1
    if any(np.linalg.det(c2w_candidate[i, :3, :3]) > 0 for i in range(min(n, 5))):
        return "c2w"
    return "unknown"


def assign_split(sequence_id: str, frame_id: int) -> str:
    seq = sequence_id.upper()
    if seq.startswith("P1"):
        if frame_id % 5 == 0:
            return "val"
        return "train"
    if seq.startswith("P2") or seq.startswith("P3"):
        return "test"
    return "train"


def build_manifest_entry(
    sequence_id: str,
    frame_id: int,
    left_path: Path,
    right_path: Path,
    mask_path: Optional[Path],
    K: np.ndarray,
    c2w: np.ndarray,
    w2c: np.ndarray,
    width: int,
    height: int,
) -> Dict:
    return {
        "sample_id": f"{sequence_id}_{frame_id:06d}",
        "dataset_name": "stereomis",
        "sequence_id": sequence_id,
        "frame_id": frame_id,
        "split": assign_split(sequence_id, frame_id),
        "left_rgb_path": str(left_path),
        "right_rgb_path": str(right_path),
        "mask_path": str(mask_path) if mask_path else None,
        "depth_path": None,
        "intrinsics": K.tolist(),
        "c2w": c2w.tolist(),
        "w2c": w2c.tolist(),
        "width": width,
        "height": height,
        "depth_semantics": "none",
        "pose_source": "forward_kinematics",
        "license_note": "CC BY-NC-SA 4.0",
    }


def discover_sequence_dirs(sequences_root: Path) -> List[Path]:
    return sorted([d for d in sequences_root.iterdir() if d.is_dir()])


def find_fk_file(seq_dir: Path) -> Optional[Path]:
    candidates = [
        "fk", "FK", "pose", "poses", "camera_pose", "camera_poses",
        "forward_kinematics",
    ]
    for cand in candidates:
        for ext in (".npy", ".csv", ".json", ".txt"):
            p = seq_dir / f"{cand}{ext}"
            if p.exists():
                return p
        p_dir = seq_dir / cand
        if p_dir.is_dir():
            for f in sorted(p_dir.iterdir()):
                if f.suffix.lower() in (".npy", ".csv", ".json", ".txt"):
                    return f
    for f in sorted(seq_dir.iterdir()):
        if f.suffix.lower() in (".npy", ".csv", ".json", ".txt"):
            if "intrinsic" not in f.name.lower() and "calib" not in f.name.lower():
                return f
    return None


def find_intrinsics_file(seq_dir: Path) -> Optional[Path]:
    candidates = [
        "intrinsics", "Intrinsics", "K", "calibration", "calib",
        "camera_intrinsics", "camera_parameters",
    ]
    for cand in candidates:
        for ext in (".npy", ".json", ".txt", ".csv"):
            p = seq_dir / f"{cand}{ext}"
            if p.exists():
                return p
    for f in sorted(seq_dir.iterdir()):
        if f.suffix.lower() in (".npy", ".json", ".txt", ".csv"):
            if "intrinsic" in f.name.lower() or "calib" in f.name.lower() or f.stem.lower() == "k":
                return f
    return None


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate StereoMIS manifest from FK poses, intrinsics, and extracted frames."
    )
    parser.add_argument(
        "--sequences-dir", type=str, default="data/external/stereomis/raw",
        help="Directory containing StereoMIS sequence subdirectories",
    )
    parser.add_argument(
        "--frames-root", type=str, default="data/processed/stereomis/frames",
        help="Root directory of extracted frames",
    )
    parser.add_argument(
        "--masks-root", type=str, default="data/processed/stereomis/masks",
        help="Root directory of extracted masks (optional)",
    )
    parser.add_argument(
        "--output", "-o", type=str,
        default="data/processed/manifests/stereomis_manifest.jsonl",
        help="Output manifest path",
    )
    parser.add_argument(
        "--intrinsics", type=str, default=None,
        help="Global intrinsics file (if all sequences share the same K)",
    )
    parser.add_argument(
        "--default-width", type=int, default=640,
        help="Default image width (used if frame files not found)",
    )
    parser.add_argument(
        "--default-height", type=int, default=512,
        help="Default image height (used if frame files not found)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    sequences_root = Path(args.sequences_dir)
    if not sequences_root.exists():
        print(f"ERROR: sequences dir not found: {sequences_root}", file=sys.stderr)
        return 1

    frames_root = Path(args.frames_root)
    masks_root = Path(args.masks_root) if args.masks_root else None

    global_K = None
    if args.intrinsics:
        global_K = parse_intrinsics(Path(args.intrinsics))
        if global_K is None:
            print(f"ERROR: could not parse intrinsics from {args.intrinsics}", file=sys.stderr)
            return 1

    seq_dirs = discover_sequence_dirs(sequences_root)
    if not seq_dirs:
        print(f"ERROR: no sequence directories found in {sequences_root}", file=sys.stderr)
        return 1

    all_entries = []
    errors = []
    for seq_dir in seq_dirs:
        seq_id = seq_dir.name
        fk_file = find_fk_file(seq_dir)
        if fk_file is None:
            errors.append(f"{seq_id}: no FK/pose file found")
            continue
        poses = parse_poses_from_file(fk_file)
        if poses is None:
            errors.append(f"{seq_id}: could not parse FK poses from {fk_file.name}")
            continue

        K = global_K
        if K is None:
            intr_file = find_intrinsics_file(seq_dir)
            if intr_file:
                K = parse_intrinsics(intr_file)
        if K is None:
            errors.append(f"{seq_id}: no intrinsics found, using default K")
            K = np.eye(3, dtype=np.float64)

        c2w = poses
        w2c = np.array([np.linalg.inv(m) for m in c2w])

        n_poses = poses.shape[0]
        frame_dir = frames_root / seq_id
        left_dir = frame_dir / "left"
        right_dir = frame_dir / "right"
        mask_dir = (masks_root / seq_id) if masks_root else None
        width = args.default_width
        height = args.default_height

        left_files = sorted(left_dir.glob("*.png")) if left_dir.exists() else []
        right_files = sorted(right_dir.glob("*.png")) if right_dir.exists() else []

        n_frames = min(n_poses, len(left_files), len(right_files)) if left_files and right_files else n_poses

        for fi in range(n_frames):
            left_path = left_files[fi] if fi < len(left_files) else left_dir / f"{fi:06d}.png"
            right_path = right_files[fi] if fi < len(right_files) else right_dir / f"{fi:06d}.png"
            mp = None
            if mask_dir and mask_dir.exists():
                mask_path = mask_dir / f"{fi:06d}.png"
                if mask_path.exists():
                    mp = mask_path
            entry = build_manifest_entry(
                sequence_id=seq_id,
                frame_id=fi,
                left_path=left_path,
                right_path=right_path,
                mask_path=mp,
                K=K,
                c2w=c2w[fi],
                w2c=w2c[fi],
                width=width,
                height=height,
            )
            all_entries.append(entry)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for entry in all_entries:
            f.write(json.dumps(entry) + "\n")

    split_counts = {}
    for e in all_entries:
        split_counts[e["split"]] = split_counts.get(e["split"], 0) + 1
    print(f"Manifest written: {out_path}")
    print(f"  Total entries: {len(all_entries)}")
    for split_name, cnt in sorted(split_counts.items()):
        print(f"    {split_name}: {cnt}")
    if errors:
        print(f"  Errors: {len(errors)}")
        for e in errors[:10]:
            print(f"    - {e}")
    return 0 if not any("no FK" in e or "could not parse" in e for e in errors[:5]) else 1


if __name__ == "__main__":
    sys.exit(main())
