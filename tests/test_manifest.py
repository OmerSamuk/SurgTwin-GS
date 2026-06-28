import json
from pathlib import Path
from surgtwin.data.manifest import assign_split, load_manifest, get_sample_by_index, validate_manifest_entry, filter_by_split


def _make_valid_entry(sample_id="test_001"):
    return {
        "sample_id": sample_id,
        "sequence_id": "Experiment_1",
        "frame_index": 1,
        "left_rgb_path": "/data/left.png",
        "right_rgb_path": "/data/right.png",
        "left_depth_path": "/data/depth.png",
        "right_depth_path": None,
        "left_tool_mask_path": None,
        "right_tool_mask_path": None,
        "left_specular_mask_path": None,
        "right_specular_mask_path": None,
        "K_left": [[500.0, 0.0, 360.0], [0.0, 500.0, 288.0], [0.0, 0.0, 1.0]],
        "K_right": [[500.0, 0.0, 360.0], [0.0, 500.0, 288.0], [0.0, 0.0, 1.0]],
        "c2w_left": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        "c2w_right": [[1.0, 0.0, 0.0, 0.005], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        "w2c_left": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        "w2c_right": [[1.0, 0.0, 0.0, -0.005], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        "height": 576,
        "width": 720,
        "depth_unit": "meter",
        "depth_scale_applied": 0.000256,
        "split": "train",
    }


def test_load_manifest(tmp_path):
    entries = [_make_valid_entry("frame_001"), _make_valid_entry("frame_002")]
    p = tmp_path / "manifest.jsonl"
    with open(p, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")

    loaded = load_manifest(p)
    assert len(loaded) == 2
    assert loaded[0]["sample_id"] == "frame_001"


def test_get_sample_by_index():
    entries = [_make_valid_entry("a"), _make_valid_entry("b")]
    assert get_sample_by_index(entries, 0)["sample_id"] == "a"
    assert get_sample_by_index(entries, 1)["sample_id"] == "b"


def test_get_sample_by_index_out_of_range():
    import pytest
    with pytest.raises(IndexError):
        get_sample_by_index([_make_valid_entry()], 5)


def test_validate_valid_entry():
    validate_manifest_entry(_make_valid_entry())


def test_validate_missing_field():
    import pytest
    entry = _make_valid_entry()
    del entry["depth_unit"]
    with pytest.raises(ValueError, match="depth_unit"):
        validate_manifest_entry(entry)


def test_validate_wrong_depth_unit():
    import pytest
    entry = _make_valid_entry()
    entry["depth_unit"] = "millimeter"
    with pytest.raises(ValueError, match="meter"):
        validate_manifest_entry(entry)


def test_validate_bad_k_matrix():
    import pytest
    entry = _make_valid_entry()
    entry["K_left"] = [[1.0, 2.0]]
    with pytest.raises(ValueError, match="K_left"):
        validate_manifest_entry(entry)


def _make_entries_for_split():
    entries = []
    for exp, start, end in [("Experiment_1", 1, 8), ("Experiment_2", 1, 8)]:
        for fi in range(start, end + 1):
            entries.append({
                "sample_id": f"{exp}_{fi:03d}",
                "sequence_id": exp,
                "frame_index": fi,
            })
    return entries


def test_assign_split_experiment1_train():
    entries = _make_entries_for_split()
    assign_split(entries)
    for e in entries:
        if e["sequence_id"] == "Experiment_1" and 1 <= e["frame_index"] <= 6:
            assert e["split"] == "train", f"{e['sample_id']} should be train"
        elif e["sequence_id"] == "Experiment_1" and e["frame_index"] >= 7:
            assert e["split"] == "val", f"{e['sample_id']} should be val"
        elif e["sequence_id"] == "Experiment_2":
            assert e["split"] == "test", f"{e['sample_id']} should be test"


def test_assign_split_counts():
    entries = _make_entries_for_split()
    assign_split(entries)
    train = [e for e in entries if e["split"] == "train"]
    val = [e for e in entries if e["split"] == "val"]
    test = [e for e in entries if e["split"] == "test"]
    assert len(train) == 6, f"Expected 6 train, got {len(train)}"
    assert len(val) == 2, f"Expected 2 val, got {len(val)}"
    assert len(test) == 8, f"Expected 8 test, got {len(test)}"


def test_filter_by_split():
    entries = _make_entries_for_split()
    assign_split(entries)
    train = filter_by_split(entries, "train")
    val = filter_by_split(entries, "val")
    test = filter_by_split(entries, "test")
    assert len(train) == 6
    assert len(val) == 2
    assert len(test) == 8


def test_assign_split_unknown_sequence():
    import pytest
    entries = [{"sequence_id": "Unknown", "frame_index": 1}]
    with pytest.raises(ValueError, match="Unknown"):
        assign_split(entries)
