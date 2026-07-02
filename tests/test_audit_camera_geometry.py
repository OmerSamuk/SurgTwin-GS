import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from scripts.audit_camera_geometry import (
    CameraGeometryAuditor,
    load_c2w_from_manifest,
    quaternion_angle,
    quaternion_from_matrix,
    rotation_angle,
)


def _identity_c2w():
    M = np.eye(4, dtype=np.float32)
    return M


def _translated_c2w(x=0.0, y=0.0, z=1.0):
    M = np.eye(4, dtype=np.float32)
    M[:3, 3] = [x, y, z]
    return M


def _rotated_c2w(deg=30.0, axis="x"):
    rad = math.radians(deg)
    c, s = math.cos(rad), math.sin(rad)
    M = np.eye(4, dtype=np.float32)
    if axis == "x":
        M[:3, :3] = [[1, 0, 0], [0, c, -s], [0, s, c]]
    elif axis == "y":
        M[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    elif axis == "z":
        M[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    return M


class TestRotationAngle:
    def test_identity(self):
        assert rotation_angle(np.eye(3)) == pytest.approx(0.0, abs=1e-6)

    def test_30_deg_x(self):
        rad = math.radians(30)
        c, s = math.cos(rad), math.sin(rad)
        R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        assert rotation_angle(R) == pytest.approx(30.0, abs=1e-4)

    def test_90_deg_y(self):
        rad = math.radians(90)
        c, s = math.cos(rad), math.sin(rad)
        R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        assert rotation_angle(R) == pytest.approx(90.0, abs=1e-4)


class TestQuaternion:
    def test_identity_to_quaternion(self):
        q = quaternion_from_matrix(np.eye(3))
        assert np.abs(q[0]) == pytest.approx(1.0, abs=1e-6)
        assert np.linalg.norm(q) == pytest.approx(1.0, abs=1e-6)

    def test_quaternion_angle_identity(self):
        q1 = quaternion_from_matrix(np.eye(3))
        angle = quaternion_angle(q1, q1)
        assert angle == pytest.approx(0.0, abs=1e-6)

    def test_quaternion_angle_90_deg(self):
        R1 = np.eye(3)
        rad = math.radians(90)
        c, s = math.cos(rad), math.sin(rad)
        R2 = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        q1 = quaternion_from_matrix(R1)
        q2 = quaternion_from_matrix(R2)
        angle = quaternion_angle(q1, q2)
        assert angle == pytest.approx(90.0, abs=1.0)


class TestCameraGeometryAuditor:
    def test_single_camera(self):
        c2w = np.stack([_identity_c2w()], axis=0)
        auditor = CameraGeometryAuditor(c2w)
        report = auditor.full_report()
        assert report["num_frames"] == 1
        assert report["unique_camera_pose_count"] == 1
        assert report["identity_camera_ratio"] == 1.0
        assert report["rotation_variation_deg"]["mean_deg"] == 0.0
        assert report["threshold_checks"]["identity_camera_ratio_low"] is False

    def test_two_identical(self):
        c2w = np.stack([_identity_c2w(), _identity_c2w()], axis=0)
        auditor = CameraGeometryAuditor(c2w)
        assert auditor.unique_pose_count() == 1
        assert auditor.identity_camera_ratio() == 1.0

    def test_two_different_translation(self):
        c2w = np.stack([_translated_c2w(z=0.0), _translated_c2w(z=0.5)], axis=0)
        auditor = CameraGeometryAuditor(c2w)
        assert auditor.unique_pose_count() == 2
        tv = auditor.translation_variation()
        assert tv["mean_m"] > 0.0
        assert tv["min_m"] >= 0.0

    def test_identity_camera_ratio_mixed(self):
        c2w_list = [
            _identity_c2w(),
            _identity_c2w(),
            _translated_c2w(x=0.5, z=1.0),
            _rotated_c2w(deg=45, axis="y"),
        ]
        c2w = np.stack(c2w_list, axis=0)
        auditor = CameraGeometryAuditor(c2w)
        ratio = auditor.identity_camera_ratio()
        assert ratio == pytest.approx(0.5, abs=0.01)

    def test_rotation_variation(self):
        c2w_list = [
            _identity_c2w(),
            _rotated_c2w(deg=45, axis="y"),
            _rotated_c2w(deg=90, axis="y"),
        ]
        c2w = np.stack(c2w_list, axis=0)
        auditor = CameraGeometryAuditor(c2w)
        rv = auditor.rotation_variation()
        assert rv["min_deg"] >= 0.0
        assert rv["max_deg"] > rv["min_deg"]

    def test_zero_angular_diversity_flag(self):
        c2w = np.stack([_identity_c2w(), _identity_c2w()], axis=0)
        auditor = CameraGeometryAuditor(c2w)
        assert auditor.zero_angular_diversity_flag() is True

    def test_zero_angular_diversity_flag_false(self):
        c2w = np.stack([_identity_c2w(), _rotated_c2w(deg=90, axis="z")], axis=0)
        auditor = CameraGeometryAuditor(c2w)
        assert auditor.zero_angular_diversity_flag() is False

    def test_approximate_parallax(self):
        c2w = np.stack([_translated_c2w(z=0.0), _translated_c2w(z=0.5)], axis=0)
        auditor = CameraGeometryAuditor(c2w)
        parallax = auditor.approximate_parallax()
        assert parallax is not None
        assert parallax > 0.0

    def test_single_no_parallax(self):
        c2w = np.stack([_identity_c2w()], axis=0)
        auditor = CameraGeometryAuditor(c2w)
        assert auditor.approximate_parallax() is None

    def test_rotation_variation_single(self):
        c2w = np.stack([_identity_c2w()], axis=0)
        auditor = CameraGeometryAuditor(c2w)
        rv = auditor.rotation_variation()
        assert rv["min_deg"] == 0.0
        assert rv["max_deg"] == 0.0
        assert rv["mean_deg"] == 0.0

    def test_full_report_structure(self):
        c2w_list = [
            _identity_c2w(),
            _translated_c2w(x=0.1, z=1.0),
            _translated_c2w(x=0.2, z=1.5),
            _rotated_c2w(deg=30, axis="x"),
        ]
        c2w = np.stack(c2w_list, axis=0)
        auditor = CameraGeometryAuditor(c2w)
        report = auditor.full_report()
        assert "num_frames" in report
        assert "unique_camera_pose_count" in report
        assert "rotation_variation_deg" in report
        assert "translation_variation_m" in report
        assert "identity_camera_ratio" in report
        assert "zero_angular_diversity_flag" in report
        assert "approximate_mean_baseline_m" in report
        assert "threshold_checks" in report

    def test_train_val_comparison(self):
        train = np.stack([
            _identity_c2w(),
            _rotated_c2w(deg=10, axis="y"),
        ], axis=0)
        val = np.stack([
            _rotated_c2w(deg=30, axis="y"),
            _rotated_c2w(deg=45, axis="y"),
        ], axis=0)
        all_c2w = np.concatenate([train, val], axis=0)
        train_mask = np.array([True, True, False, False])
        val_mask = np.array([False, False, True, True])
        auditor = CameraGeometryAuditor(all_c2w)
        comp = auditor.train_val_comparison(train_mask, val_mask)
        assert "train" in comp
        assert "val" in comp
        assert comp["val_to_train_diversity_ratio"] is not None
        assert comp["val_to_train_diversity_ratio"] > 0.0

    def test_invalid_shape(self):
        with pytest.raises(ValueError):
            CameraGeometryAuditor(np.zeros((5, 3, 3)))


class TestLoadManifest:
    def test_load_valid_c2w(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for x in [0.0, 0.5, 1.0]:
                c2w = np.eye(4).tolist()
                c2w[0][3] = x
                f.write(json.dumps({"sample_id": f"frame_{x}", "c2w": c2w}) + "\n")
            f_path = f.name
        c2w = load_c2w_from_manifest(f_path)
        Path(f_path).unlink()
        assert c2w is not None
        assert c2w.shape == (3, 4, 4)

    def test_load_with_w2c(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            w2c = np.eye(4).tolist()
            f.write(json.dumps({"sample_id": "frame_0", "w2c": w2c}) + "\n")
            f_path = f.name
        c2w = load_c2w_from_manifest(f_path)
        Path(f_path).unlink()
        assert c2w is not None
        assert c2w.shape == (1, 4, 4)

    def test_load_empty(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write("")
            f_path = f.name
        c2w = load_c2w_from_manifest(f_path)
        Path(f_path).unlink()
        assert c2w is None

    def test_load_nonexistent(self):
        c2w = load_c2w_from_manifest("/nonexistent/path.jsonl")
        assert c2w is None
