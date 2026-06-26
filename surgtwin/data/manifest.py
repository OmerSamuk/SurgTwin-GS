import json
from pathlib import Path
from typing import Dict, List, Optional


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
