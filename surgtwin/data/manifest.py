import json
from pathlib import Path
from typing import Dict, List, Optional


SPLIT_SEED = 42


def _sequence_and_index_seed(seq: str, fi: int, seed: int) -> int:
    """Hash sequence_id and frame_index into a signed 32-bit int for deterministic seeding."""
    import hashlib

    s = f"{seq}_{fi}_{seed}"
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)


def assign_split(entries: List[Dict]) -> None:
    """Deterministic frame-index split.

    Experiment_1 frame 0-5  → train
    Experiment_1 frame 6-7  → val
    Experiment_2 frame 0-7  → test

    Mutates entries in place.
    """
    for entry in entries:
        seq = entry.get("sequence_id", "")
        fi = entry.get("frame_index", -1)
        if seq == "Experiment_1" and 1 <= fi <= 6:
            entry["split"] = "train"
        elif seq == "Experiment_1" and fi >= 7:
            entry["split"] = "val"
        elif seq == "Experiment_2":
            entry["split"] = "test"
        else:
            raise ValueError(
                f"Cannot determine split for sequence_id={seq!r} frame_index={fi}. "
                f"Expected Experiment_1 (frames 1-8) or Experiment_2 (frames 1-8)."
            )


def filter_by_split(entries: List[Dict], split_name: str) -> List[Dict]:
    """Return entries whose split field matches split_name."""
    return [e for e in entries if e.get("split") == split_name]


def load_manifest(path: Path) -> List[Dict]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found at {path}. Run scripts/explore_servct.py first.")
    entries = []
    with open(path, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_num} of {path}: {e}")
    if not entries:
        raise ValueError(f"Manifest at {path} contains no valid entries.")
    return entries


def get_sample_by_index(entries: List[Dict], index: int) -> Dict:
    if index < 0 or index >= len(entries):
        raise IndexError(
            f"Sample index {index} out of range (manifest has {len(entries)} samples)"
        )
    return entries[index]


def validate_manifest_entry(entry: Dict) -> None:
    required_fields = [
        "sample_id", "sequence_id", "frame_index",
        "left_rgb_path", "right_rgb_path", "left_depth_path",
        "K_left", "K_right", "c2w_left", "c2w_right",
        "w2c_left", "w2c_right", "height", "width",
        "depth_unit", "depth_scale_applied",
    ]
    missing = [f for f in required_fields if f not in entry]
    if missing:
        raise ValueError(f"Manifest entry {entry.get('sample_id', 'unknown')} missing fields: {missing}")
    if entry.get("depth_unit") != "meter":
        raise ValueError(f"Expected depth_unit 'meter', got '{entry.get('depth_unit')}'")
    for key in ["K_left", "K_right"]:
        val = entry[key]
        if len(val) != 3 or any(len(row) != 3 for row in val):
            raise ValueError(f"{key} must be 3x3, got {val}")
    for key in ["c2w_left", "c2w_right", "w2c_left", "w2c_right"]:
        val = entry[key]
        if len(val) != 4 or any(len(row) != 4 for row in val):
            raise ValueError(f"{key} must be 4x4, got {val}")
