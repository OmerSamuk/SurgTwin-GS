"""
Audit camera geometry diversity from a manifest or pose metadata file.

Usage:
    python scripts/audit_camera_geometry.py --manifest data/manifest.jsonl
    python scripts/audit_camera_geometry.py --manifest data/manifest.jsonl --output report.json
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def _rotation_matrix_from_4x4(M: np.ndarray) -> np.ndarray:
    return M[:3, :3]


def _translation_from_4x4(M: np.ndarray) -> np.ndarray:
    return M[:3, 3]


def _is_identity(R: np.ndarray, rot_thresh_deg: float = 1.0) -> bool:
    angle = rotation_angle(R)
    return angle < rot_thresh_deg


def rotation_angle(R: np.ndarray) -> float:
    trace = np.trace(R)
    trace = np.clip(trace, -1.0, 3.0)
    angle = math.degrees(math.acos((trace - 1.0) / 2.0))
    return angle


def quaternion_from_matrix(R: np.ndarray) -> np.ndarray:
    q = np.zeros(4)
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        q[0] = 0.25 / s
        q[1] = (R[2, 1] - R[1, 2]) * s
        q[2] = (R[0, 2] - R[2, 0]) * s
        q[3] = (R[1, 0] - R[0, 1]) * s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            q[0] = (R[1, 2] - R[2, 1]) / s
            q[1] = 0.25 * s
            q[2] = (R[1, 0] + R[0, 1]) / s
            q[3] = (R[2, 0] + R[0, 2]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            q[0] = (R[2, 0] - R[0, 2]) / s
            q[1] = (R[1, 0] + R[0, 1]) / s
            q[2] = 0.25 * s
            q[3] = (R[2, 1] + R[1, 2]) / s
        else:
            s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            q[0] = (R[0, 1] - R[1, 0]) / s
            q[1] = (R[2, 0] + R[0, 2]) / s
            q[2] = (R[1, 2] + R[2, 1]) / s
            q[3] = 0.25 * s
    return q


def quaternion_angle(q1: np.ndarray, q2: np.ndarray) -> float:
    dot = np.clip(np.abs(np.dot(q1, q2)), 0.0, 1.0)
    return math.degrees(2.0 * math.acos(dot))


def mean_quaternion(quaternions: np.ndarray) -> np.ndarray:
    Q = quaternions.T
    M = Q @ Q.T
    eigenvalues, eigenvectors = np.linalg.eigh(M)
    return eigenvectors[:, -1]


class CameraGeometryAuditor:
    def __init__(self, c2w_matrices: np.ndarray):
        if c2w_matrices.ndim != 3 or c2w_matrices.shape[1:] != (4, 4):
            raise ValueError(f"Expected (N, 4, 4) array, got {c2w_matrices.shape}")
        self.c2w = c2w_matrices
        self.N = c2w_matrices.shape[0]
        self._rotations = np.array(
            [_rotation_matrix_from_4x4(M) for M in c2w_matrices]
        )
        self._translations = np.array(
            [_translation_from_4x4(M) for M in c2w_matrices]
        )

    def unique_pose_count(self, rot_thresh_deg: float = 1.0,
                          trans_thresh: float = 0.01) -> int:
        unique = 0
        seen = []
        for i in range(self.N):
            is_unique = True
            for j in seen:
                R_diff = self._rotations[i].T @ self._rotations[j]
                rot_dist = rotation_angle(R_diff)
                trans_dist = np.linalg.norm(self._translations[i] - self._translations[j])
                if rot_dist < rot_thresh_deg and trans_dist < trans_thresh:
                    is_unique = False
                    break
            if is_unique:
                unique += 1
                seen.append(i)
        return unique

    def rotation_variation(self) -> Dict[str, float]:
        if self.N < 2:
            return {"min_deg": 0.0, "max_deg": 0.0, "mean_deg": 0.0}
        quats = np.array([quaternion_from_matrix(R) for R in self._rotations])
        q_mean = mean_quaternion(quats)
        angles = np.array([quaternion_angle(q, q_mean) for q in quats])
        return {
            "min_deg": float(angles.min()),
            "max_deg": float(angles.max()),
            "mean_deg": float(angles.mean()),
        }

    def translation_variation(self) -> Dict[str, float]:
        if self.N < 2:
            return {"min_m": 0.0, "max_m": 0.0, "mean_m": 0.0}
        t_mean = self._translations.mean(axis=0)
        dists = np.linalg.norm(self._translations - t_mean, axis=1)
        return {
            "min_m": float(dists.min()),
            "max_m": float(dists.max()),
            "mean_m": float(dists.mean()),
        }

    def identity_camera_ratio(self, rot_thresh_deg: float = 1.0,
                              trans_thresh: float = 0.01) -> float:
        if self.N == 0:
            return 1.0
        id_count = 0
        identity = np.eye(3)
        for i in range(self.N):
            rot_dist = rotation_angle(self._rotations[i].T @ identity)
            trans_dist = np.linalg.norm(self._translations[i])
            if rot_dist < rot_thresh_deg and trans_dist < trans_thresh:
                id_count += 1
        return id_count / self.N

    def zero_angular_diversity_flag(self, threshold_deg: float = 5.0) -> bool:
        rv = self.rotation_variation()
        return rv["mean_deg"] < threshold_deg

    def approximate_parallax(self) -> Optional[float]:
        if self.N < 2:
            return None
        num_pairs = 0
        total_baseline = 0.0
        for i in range(self.N):
            for j in range(i + 1, self.N):
                baseline = np.linalg.norm(self._translations[i] - self._translations[j])
                total_baseline += baseline
                num_pairs += 1
        mean_baseline = total_baseline / num_pairs if num_pairs > 0 else 0.0
        return float(mean_baseline)

    def full_report(self, rot_thresh_deg: float = 1.0,
                    trans_thresh: float = 0.01,
                    angular_diversity_threshold: float = 5.0) -> Dict:
        rv = self.rotation_variation()
        tv = self.translation_variation()
        unique_count = self.unique_pose_count(rot_thresh_deg, trans_thresh)
        id_ratio = self.identity_camera_ratio(rot_thresh_deg, trans_thresh)
        zero_flag = self.zero_angular_diversity_flag(angular_diversity_threshold)
        parallax = self.approximate_parallax()
        return {
            "num_frames": self.N,
            "unique_camera_pose_count": unique_count,
            "rotation_variation_deg": rv,
            "translation_variation_m": tv,
            "identity_camera_ratio": id_ratio,
            "zero_angular_diversity_flag": zero_flag,
            "approximate_mean_baseline_m": parallax,
            "threshold_checks": {
                "valid_validation_frames_ge_4": self.N >= 4,
                "valid_validation_frames_ge_6": self.N >= 6,
                "unique_poses_ge_3": unique_count >= 3,
                "rotation_variation_ge_5_deg": not zero_flag,
                "identity_camera_ratio_low": id_ratio < 0.5,
                "non_trivial_baseline": parallax is not None and parallax > 0.01,
            },
        }

    def train_val_comparison(self, train_mask: np.ndarray,
                              val_mask: np.ndarray,
                              **kwargs) -> Dict:
        train_indices = np.where(train_mask)[0]
        val_indices = np.where(val_mask)[0]
        if len(train_indices) == 0 or len(val_indices) == 0:
            return {"error": "train or val set empty"}
        train_auditor = CameraGeometryAuditor(self.c2w[train_indices])
        val_auditor = CameraGeometryAuditor(self.c2w[val_indices])
        train_report = train_auditor.full_report(**kwargs)
        val_report = val_auditor.full_report(**kwargs)
        return {
            "train": train_report,
            "val": val_report,
            "val_to_train_diversity_ratio": (
                val_report["rotation_variation_deg"]["mean_deg"]
                / train_report["rotation_variation_deg"]["mean_deg"]
                if train_report["rotation_variation_deg"]["mean_deg"] > 0
                else None
            ),
        }


def load_c2w_from_manifest(manifest_path: str) -> Optional[np.ndarray]:
    matrices = []
    path = Path(manifest_path)
    if not path.exists():
        return None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            c2w = entry.get("c2w")
            if c2w is None:
                w2c = entry.get("w2c")
                if w2c is not None:
                    w2c_arr = np.array(w2c, dtype=np.float32).reshape(4, 4)
                    c2w_arr = np.linalg.inv(w2c_arr)
                else:
                    continue
            else:
                c2w_arr = np.array(c2w, dtype=np.float32).reshape(4, 4)
            matrices.append(c2w_arr)
    if len(matrices) == 0:
        return None
    return np.stack(matrices, axis=0)


def load_c2w_from_npy(path: str) -> Optional[np.ndarray]:
    return np.load(path)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit camera geometry diversity from a manifest or pose file."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--manifest", type=str,
                             help="Path to manifest JSONL file with c2w entries")
    input_group.add_argument("--npy", type=str,
                             help="Path to .npy file with (N, 4, 4) c2w array")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Path to write JSON report (optional)")
    parser.add_argument("--rot-thresh", type=float, default=1.0,
                        help="Rotation threshold for identity detection (deg)")
    parser.add_argument("--trans-thresh", type=float, default=0.01,
                        help="Translation threshold for identity detection (m)")
    parser.add_argument("--angular-thresh", type=float, default=5.0,
                        help="Angular diversity threshold (deg)")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if args.manifest:
        c2w = load_c2w_from_manifest(args.manifest)
        if c2w is None:
            print(f"ERROR: could not parse poses from {args.manifest}", file=sys.stderr)
            return 1
    elif args.npy:
        c2w = load_c2w_from_npy(args.npy)
        if c2w is None:
            print(f"ERROR: could not load {args.npy}", file=sys.stderr)
            return 1
    else:
        print("ERROR: --manifest or --npy required", file=sys.stderr)
        return 1

    auditor = CameraGeometryAuditor(c2w)
    report = auditor.full_report(
        rot_thresh_deg=args.rot_thresh,
        trans_thresh=args.trans_thresh,
        angular_diversity_threshold=args.angular_thresh,
    )

    output = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).write_text(output)
        print(f"Report written to {args.output}")
    else:
        print(output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
