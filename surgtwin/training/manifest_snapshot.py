import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _iso_mtime(path: Path) -> str:
    mtime = path.stat().st_mtime
    return datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()


def write_manifest_snapshot(
    output_dir: Path,
    manifest_path: Path,
    entries: List[Dict],
    train_entries: List[Dict],
    val_entries: List[Dict],
    extra: Optional[Dict] = None,
) -> None:
    manifest_path = manifest_path.resolve()
    sample_ids = [e.get("sample_id") for e in entries]
    snapshot = {
        "schema_version": "v1",
        "captured_by": "trainer.fit()",
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "manifest_mtime_iso": _iso_mtime(manifest_path),
        "n_entries_total": len(entries),
        "n_train": len(train_entries),
        "n_val": len(val_entries),
        "n_test": sum(1 for e in entries if e.get("split") == "test"),
        "sample_ids": sample_ids,
        "sample_ids_sha256": _sha256_str("\n".join(sample_ids)),
        "train_sample_ids": [e.get("sample_id") for e in train_entries],
        "val_sample_ids": [e.get("sample_id") for e in val_entries],
    }
    if extra:
        snapshot["trainer_extra"] = extra
    from surgtwin.training.logging_utils import write_json as _write_json
    _write_json(output_dir / "manifest_snapshot.json", snapshot)
