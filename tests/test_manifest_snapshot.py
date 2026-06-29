import json
import tempfile
from pathlib import Path
from surgtwin.training.manifest_snapshot import write_manifest_snapshot


def _make_entry(sample_id, split="train"):
    return {
        "sample_id": sample_id,
        "sequence_id": "Experiment_1",
        "frame_index": 1,
        "left_rgb_path": f"/data/{sample_id}_left.png",
        "right_rgb_path": f"/data/{sample_id}_right.png",
        "left_depth_path": f"/data/{sample_id}_depth.png",
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
        "split": split,
    }


def test_manifest_snapshot_schema():
    entries = [_make_entry(f"frame_{i:02d}", "train" if i <= 6 else "val") for i in range(1, 9)]
    train = [e for e in entries if e["split"] == "train"]
    val = [e for e in entries if e["split"] == "val"]

    with tempfile.TemporaryDirectory() as tmp:
        manifest_path = Path(tmp) / "manifest.jsonl"
        with open(manifest_path, "w") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")

        output_dir = Path(tmp) / "run_output"
        output_dir.mkdir()
        write_manifest_snapshot(
            output_dir=output_dir,
            manifest_path=manifest_path,
            entries=entries,
            train_entries=train,
            val_entries=val,
        )

        snapshot_path = output_dir / "manifest_snapshot.json"
        assert snapshot_path.exists(), "manifest_snapshot.json not created"
        data = json.loads(snapshot_path.read_text())
        assert data["schema_version"] == "v1"
        assert data["n_entries_total"] == 8
        assert data["n_train"] == 6
        assert data["n_val"] == 2
        assert len(data["sample_ids"]) == 8


def test_manifest_snapshot_sha256():
    entries = [_make_entry("test_001", "train")]
    with tempfile.TemporaryDirectory() as tmp:
        manifest_path = Path(tmp) / "manifest.jsonl"
        with open(manifest_path, "w") as f:
            f.write(json.dumps(entries[0]) + "\n")

        output_dir = Path(tmp) / "run_output"
        output_dir.mkdir()
        write_manifest_snapshot(
            output_dir=output_dir,
            manifest_path=manifest_path,
            entries=entries,
            train_entries=entries,
            val_entries=[],
        )

        data = json.loads((output_dir / "manifest_snapshot.json").read_text())
        assert data["manifest_sha256"]
        assert data["manifest_mtime_iso"]
        assert data["sample_ids_sha256"]


def test_manifest_snapshot_extra():
    entries = [_make_entry("test_001", "train")]
    with tempfile.TemporaryDirectory() as tmp:
        manifest_path = Path(tmp) / "manifest.jsonl"
        with open(manifest_path, "w") as f:
            f.write(json.dumps(entries[0]) + "\n")

        output_dir = Path(tmp) / "run_output"
        output_dir.mkdir()
        write_manifest_snapshot(
            output_dir=output_dir,
            manifest_path=manifest_path,
            entries=entries,
            train_entries=entries,
            val_entries=[],
            extra={"variant": "h1", "alpha": 2.0},
        )

        data = json.loads((output_dir / "manifest_snapshot.json").read_text())
        assert data["trainer_extra"]["variant"] == "h1"
        assert data["trainer_extra"]["alpha"] == 2.0
