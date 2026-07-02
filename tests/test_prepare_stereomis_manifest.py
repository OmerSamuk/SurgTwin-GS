import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from scripts.prepare_stereomis_manifest import (
    _parse_poses_csv,
    _parse_poses_json,
    _parse_poses_txt,
    _poses_from_quat_trans,
    assign_split,
    build_manifest_entry,
    determine_convention,
    discover_sequence_dirs,
    find_fk_file,
    find_intrinsics_file,
    parse_intrinsics,
    parse_poses_from_file,
)


@pytest.fixture
def identity_4x4():
    return np.eye(4, dtype=np.float64)


@pytest.fixture
def sample_poses_4x4():
    poses = np.zeros((4, 4, 4), dtype=np.float64)
    for i in range(4):
        poses[i] = np.eye(4, dtype=np.float64)
        poses[i, 0, 3] = i * 0.1
    return poses


def test_parse_poses_csv_16cols(tmp_path: Path, sample_poses_4x4):
    f = tmp_path / "poses.csv"
    flat = sample_poses_4x4.reshape(4, 16)
    np.savetxt(str(f), flat, delimiter=",")
    result = _parse_poses_csv(f)
    assert result is not None
    assert result.shape == (4, 4, 4)
    assert np.allclose(result, sample_poses_4x4)


def test_parse_poses_csv_7cols(tmp_path: Path):
    data = np.array([[0.707, 0.0, 0.0, 0.707, 1.0, 0.0, 0.0]], dtype=np.float64)
    f = tmp_path / "quat.csv"
    np.savetxt(str(f), data, delimiter=",")
    result = _parse_poses_csv(f)
    assert result is not None
    assert result.shape == (1, 4, 4)


def test_parse_poses_csv_12cols(tmp_path: Path):
    data = np.eye(3, dtype=np.float64).reshape(1, 9)
    data = np.hstack([data, np.array([[0.0, 0.0, 0.0]])])
    f = tmp_path / "rt.csv"
    np.savetxt(str(f), data, delimiter=",")
    result = _parse_poses_csv(f)
    assert result is not None
    assert result.shape == (1, 4, 4)


def test_parse_poses_csv_bad_shape(tmp_path: Path):
    data = np.array([[1.0, 2.0, 3.0]])
    f = tmp_path / "bad.csv"
    np.savetxt(str(f), data, delimiter=",")
    assert _parse_poses_csv(f) is None


def test_parse_poses_json_list_of_dicts(tmp_path: Path, sample_poses_4x4):
    entries = [{"c2w": sample_poses_4x4[i].tolist()} for i in range(4)]
    f = tmp_path / "poses.json"
    f.write_text(json.dumps(entries))
    result = _parse_poses_json(f)
    assert result is not None
    assert result.shape == (4, 4, 4)


def test_parse_poses_json_list_of_lists(tmp_path: Path, sample_poses_4x4):
    entries = [sample_poses_4x4[i].flatten().tolist() for i in range(4)]
    f = tmp_path / "poses.json"
    f.write_text(json.dumps(entries))
    result = _parse_poses_json(f)
    assert result is not None
    assert result.shape == (4, 4, 4)


def test_parse_poses_json_dict_key(tmp_path: Path, sample_poses_4x4):
    data = {"poses": sample_poses_4x4.tolist()}
    f = tmp_path / "poses.json"
    f.write_text(json.dumps(data))
    result = _parse_poses_json(f)
    assert result is not None
    assert result.shape == (4, 4, 4)


def test_parse_poses_json_empty_list(tmp_path: Path):
    f = tmp_path / "empty.json"
    f.write_text("[]")
    assert _parse_poses_json(f) is None


def test_parse_poses_txt_4x4_blocks(tmp_path: Path, identity_4x4):
    lines = []
    for i in range(2):
        for row in identity_4x4:
            lines.append(" ".join(f"{v:.6f}" for v in row))
        lines.append("")
    f = tmp_path / "poses.txt"
    f.write_text("\n".join(lines))
    result = _parse_poses_txt(f)
    assert result is not None
    assert result.shape == (2, 4, 4)


def test_parse_poses_txt_comments(tmp_path: Path, identity_4x4):
    lines = ["# comment line"]
    for row in identity_4x4:
        lines.append(" ".join(f"{v:.6f}" for v in row))
    f = tmp_path / "poses.txt"
    f.write_text("\n".join(lines))
    result = _parse_poses_txt(f)
    assert result is not None
    assert result.shape == (1, 4, 4)


def test_parse_poses_txt_no_valid(tmp_path: Path):
    f = tmp_path / "poses.txt"
    f.write_text("not a number line")
    assert _parse_poses_txt(f) is None


def test_poses_from_quat_trans():
    data = np.array([[1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0]], dtype=np.float64)
    result = _poses_from_quat_trans(data)
    assert result.shape == (1, 4, 4)
    assert np.allclose(result[0, :3, 3], [1.0, 2.0, 3.0])


def test_parse_poses_from_file_npy(tmp_path: Path, sample_poses_4x4):
    f = tmp_path / "poses.npy"
    np.save(str(f), sample_poses_4x4)
    result = parse_poses_from_file(f)
    assert result is not None
    assert result.shape == (4, 4, 4)


def test_parse_poses_from_file_unknown_ext(tmp_path: Path):
    f = tmp_path / "poses.dat"
    f.write_text("data")
    assert parse_poses_from_file(f) is None


def test_parse_intrinsics_npy_3x3(tmp_path: Path):
    K = np.eye(3, dtype=np.float64)
    K[0, 0] = 500.0
    f = tmp_path / "K.npy"
    np.save(str(f), K)
    result = parse_intrinsics(f)
    assert result is not None
    assert result[0, 0] == 500.0


def test_parse_intrinsics_json(tmp_path: Path):
    K = [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]]
    f = tmp_path / "intrinsics.json"
    f.write_text(json.dumps({"K": K}))
    result = parse_intrinsics(f)
    assert result is not None
    assert np.allclose(result[0, 0], 500.0)


def test_parse_intrinsics_txt_3x3(tmp_path: Path):
    K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
    f = tmp_path / "K.txt"
    np.savetxt(str(f), K)
    result = parse_intrinsics(f)
    assert result is not None


def test_parse_intrinsics_txt_4param(tmp_path: Path):
    f = tmp_path / "K.txt"
    f.write_text("500.0 500.0 320.0 240.0")
    result = parse_intrinsics(f)
    assert result is not None
    assert result.shape == (3, 3)
    assert result[0, 0] == 500.0


def test_parse_intrinsics_csv_4param(tmp_path: Path):
    f = tmp_path / "K.csv"
    np.savetxt(str(f), np.array([[500.0, 500.0, 320.0, 240.0]]), delimiter=",")
    result = parse_intrinsics(f)
    assert result is not None
    assert result[0, 0] == 500.0


def test_parse_intrinsics_unsupported_ext(tmp_path: Path):
    f = tmp_path / "K.yaml"
    f.write_text("fx: 500.0")
    assert parse_intrinsics(f) is None


def test_determine_convention_identity():
    poses = np.tile(np.eye(4)[np.newaxis, :, :], (5, 1, 1))
    assert determine_convention(poses) in ("c2w", "unknown")


def test_determine_convention_bad_shape():
    assert determine_convention(np.zeros((5, 3, 3))) == "unknown"


def test_assign_split_p1_train():
    assert assign_split("P1_01", 1) == "train"


def test_assign_split_p1_val():
    assert assign_split("P1_01", 5) == "val"


def test_assign_split_p2_test():
    assert assign_split("P2_01", 0) == "test"


def test_assign_split_p3_test():
    assert assign_split("P3_01", 0) == "test"


def test_build_manifest_entry(tmp_path: Path, identity_4x4):
    left = tmp_path / "left.png"
    left.write_text("dummy")
    right = tmp_path / "right.png"
    right.write_text("dummy")
    K = np.eye(3, dtype=np.float64)
    entry = build_manifest_entry(
        sequence_id="P1_01",
        frame_id=0,
        left_path=left,
        right_path=right,
        mask_path=None,
        K=K,
        c2w=identity_4x4,
        w2c=identity_4x4,
        width=640,
        height=480,
    )
    assert entry["sample_id"] == "P1_01_000000"
    assert entry["dataset_name"] == "stereomis"
    assert entry["depth_semantics"] == "none"
    assert entry["pose_source"] == "forward_kinematics"
    assert entry["license_note"] == "CC BY-NC-SA 4.0"
    assert entry["depth_path"] is None
    assert entry["mask_path"] is None
    assert entry["width"] == 640


def test_discover_sequence_dirs(tmp_path: Path):
    (tmp_path / "P1_01").mkdir()
    (tmp_path / "P1_02").mkdir()
    (tmp_path / "not_a_seq.txt").write_text("")
    dirs = discover_sequence_dirs(tmp_path)
    assert len(dirs) == 2


def test_discover_sequence_dirs_empty(tmp_path: Path):
    assert discover_sequence_dirs(tmp_path) == []


def test_find_fk_file_root(tmp_path: Path):
    (tmp_path / "fk.npy").write_text("dummy")
    assert find_fk_file(tmp_path) == tmp_path / "fk.npy"


def test_find_fk_file_subdir(tmp_path: Path):
    sub = tmp_path / "FK"
    sub.mkdir()
    (sub / "pose.csv").write_text("dummy")
    assert find_fk_file(tmp_path) == sub / "pose.csv"


def test_find_fk_file_fallback(tmp_path: Path):
    (tmp_path / "poses.npy").write_text("dummy")
    assert find_fk_file(tmp_path) == tmp_path / "poses.npy"


def test_find_fk_file_none(tmp_path: Path):
    assert find_fk_file(tmp_path) is None


def test_find_intrinsics_file(tmp_path: Path):
    (tmp_path / "intrinsics.json").write_text("{}")
    assert find_intrinsics_file(tmp_path) == tmp_path / "intrinsics.json"


def test_find_intrinsics_file_fallback(tmp_path: Path):
    (tmp_path / "K.txt").write_text("500 500 320 240")
    assert find_intrinsics_file(tmp_path) == tmp_path / "K.txt"


def test_find_intrinsics_file_none(tmp_path: Path):
    (tmp_path / "random.txt").write_text("data")
    assert find_intrinsics_file(tmp_path) is None


def test_find_intrinsics_file_skips_fk(tmp_path: Path):
    (tmp_path / "pose.npy").write_text("dummy")
    (tmp_path / "calib.json").write_text("{}")
    found = find_intrinsics_file(tmp_path)
    assert found is not None
    assert "calib" in found.name.lower()
