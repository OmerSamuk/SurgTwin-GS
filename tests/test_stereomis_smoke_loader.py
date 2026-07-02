import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from scripts.smoke_stereomis_render import (
    build_smoke_report,
    check_image_paths,
    count_split_entries,
    load_manifest,
    validate_camera_matrices,
    validate_manifest_entry,
)


@pytest.fixture
def sample_entry():
    return {
        "sample_id": "P1_01_000000",
        "dataset_name": "stereomis",
        "sequence_id": "P1_01",
        "frame_id": 0,
        "split": "train",
        "left_rgb_path": "left/000000.png",
        "right_rgb_path": "right/000000.png",
        "mask_path": None,
        "depth_path": None,
        "intrinsics": [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]],
        "c2w": [[1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]],
        "w2c": [[1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]],
        "width": 640,
        "height": 480,
        "depth_semantics": "none",
        "pose_source": "forward_kinematics",
        "license_note": "CC BY-NC-SA 4.0",
    }


@pytest.fixture
def valid_manifest_file(tmp_path: Path, sample_entry):
    f = tmp_path / "manifest.jsonl"
    with open(f, "w") as fh:
        fh.write(json.dumps(sample_entry) + "\n")
        entry2 = dict(sample_entry, sample_id="P1_01_000001", frame_id=1)
        fh.write(json.dumps(entry2) + "\n")
    return f


def test_load_manifest(valid_manifest_file: Path):
    entries = load_manifest(valid_manifest_file)
    assert len(entries) == 2
    assert entries[0]["sample_id"] == "P1_01_000000"
    assert entries[1]["sample_id"] == "P1_01_000001"


def test_load_manifest_empty(tmp_path: Path):
    f = tmp_path / "empty.jsonl"
    f.write_text("")
    assert load_manifest(f) == []


def test_validate_manifest_entry_valid(sample_entry):
    result = validate_manifest_entry(sample_entry)
    assert result["valid"] is True
    assert result["missing_fields"] == []


def test_validate_manifest_entry_missing_field(sample_entry):
    del sample_entry["c2w"]
    result = validate_manifest_entry(sample_entry)
    assert result["valid"] is False
    assert "c2w" in result["missing_fields"]


def test_validate_manifest_entry_missing_multiple(sample_entry):
    del sample_entry["intrinsics"]
    del sample_entry["w2c"]
    result = validate_manifest_entry(sample_entry)
    assert result["valid"] is False
    assert len(result["missing_fields"]) == 2


def test_validate_camera_matrices_valid(sample_entry):
    result = validate_camera_matrices(sample_entry)
    assert result["valid"] is True
    assert result["K_positive_fx"] is True
    assert result["c2w_w2c_consistent"] is True
    assert result["c2w_rotation_valid"] is True


def test_validate_camera_matrices_bad_K(sample_entry):
    sample_entry["intrinsics"] = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    result = validate_camera_matrices(sample_entry)
    assert result["valid"] is False
    assert result["K_positive_fx"] is False


def test_validate_camera_matrices_inconsistent_c2w_w2c(sample_entry):
    sample_entry["c2w"] = [[1.0, 0.0, 0.0, 0.1],
                            [0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0]]
    sample_entry["w2c"] = [[1.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0]]
    result = validate_camera_matrices(sample_entry)
    assert result["valid"] is True
    assert result["c2w_w2c_consistent"] is False


def test_check_image_paths_existing(tmp_path: Path, sample_entry):
    (tmp_path / "left").mkdir(parents=True)
    (tmp_path / "right").mkdir()
    left_png = tmp_path / "left" / "000000.png"
    left_png.write_text("dummy")
    right_png = tmp_path / "right" / "000000.png"
    right_png.write_text("dummy")
    sample_entry["left_rgb_path"] = str(left_png)
    sample_entry["right_rgb_path"] = str(right_png)
    result = check_image_paths(sample_entry)
    assert result["valid"] is True
    assert result["left_exists"] is True


def test_check_image_paths_missing(tmp_path: Path, sample_entry):
    (tmp_path / "left").mkdir(parents=True)
    sample_entry["left_rgb_path"] = str(tmp_path / "left" / "missing.png")
    sample_entry["right_rgb_path"] = str(tmp_path / "nonexistent.png")
    result = check_image_paths(sample_entry)
    assert result["valid"] is False


def test_check_image_paths_mask(tmp_path: Path, sample_entry):
    (tmp_path / "left").mkdir(parents=True)
    (tmp_path / "right").mkdir()
    (tmp_path / "left" / "000000.png").write_text("dummy")
    (tmp_path / "right" / "000000.png").write_text("dummy")
    (tmp_path / "mask").mkdir()
    (tmp_path / "mask" / "000000.png").write_text("dummy")
    sample_entry["left_rgb_path"] = str(tmp_path / "left" / "000000.png")
    sample_entry["right_rgb_path"] = str(tmp_path / "right" / "000000.png")
    sample_entry["mask_path"] = str(tmp_path / "mask" / "000000.png")
    result = check_image_paths(sample_entry)
    assert result["mask_exists"] is True


def test_count_split_entries():
    entries = [
        {"split": "train"}, {"split": "train"}, {"split": "val"}, {"split": "test"},
    ]
    counts = count_split_entries(entries)
    assert counts["train"] == 2
    assert counts["val"] == 1
    assert counts["test"] == 1


def test_count_split_entries_empty():
    assert count_split_entries([]) == {}


def test_count_split_entries_unknown():
    entries = [{"no_split": "x"}]
    counts = count_split_entries(entries)
    assert counts.get("unknown", 0) == 1


def test_build_smoke_report_valid():
    entry = {"sample_id": "test"}
    mc = {"valid": True, "missing_fields": [], "sample_id": "test"}
    cc = {"valid": True, "K_finite": True, "K_positive_fx": True,
          "K_positive_fy": True, "c2w_finite": True, "w2c_finite": True,
          "c2w_w2c_consistent": True, "c2w_rotation_valid": True}
    pc = {"valid": True, "left_exists": True, "right_exists": True, "mask_exists": None}
    report = build_smoke_report([entry], [mc], [cc], [pc], {"train": 1})
    assert report["smoke_passed"] is True
    assert report["manifest_valid"] is True
    assert report["camera_matrices_valid"] is True


def test_build_smoke_report_invalid():
    entry = {"sample_id": "test"}
    mc = {"valid": False, "missing_fields": ["c2w"], "sample_id": "test"}
    cc = {"valid": False, "K_finite": False}
    pc = {"valid": False, "left_exists": False, "right_exists": False, "mask_exists": None}
    report = build_smoke_report([entry], [mc], [cc], [pc], {"train": 1})
    assert report["smoke_passed"] is False


def test_build_smoke_report_with_render():
    entry = {"sample_id": "test"}
    mc = {"valid": True, "missing_fields": []}
    cc = {"valid": True}
    pc = {"valid": True, "left_exists": True, "right_exists": True, "mask_exists": None}
    render_results = [{"cuda_available": False, "skipped": True}]
    report = build_smoke_report([entry], [mc], [cc], [pc], {"train": 1}, render_results)
    assert report["render_smoke"] is not None
    assert report["render_smoke_passed"] is None


def test_build_smoke_report_render_fail():
    entry = {"sample_id": "test"}
    mc = {"valid": True, "missing_fields": []}
    cc = {"valid": True}
    pc = {"valid": True, "left_exists": True, "right_exists": True}
    render_results = [{"cuda_available": True, "render_completed": False, "error": "render failed"}]
    report = build_smoke_report([entry], [mc], [cc], [pc], {"train": 1}, render_results)
    assert report["render_smoke_passed"] is False
