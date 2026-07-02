import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from scripts.prepare_stereomis_frames import (
    build_report,
    detect_vertical_stack,
    discover_videos,
)
from scripts.prepare_stereomis_frames import _read_png_dimensions


def _create_dummy_png(path: Path, width: int, height: int):
    import struct
    import zlib
    path.parent.mkdir(parents=True, exist_ok=True)
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    ihdr_crc = zlib.crc32(b"IHDR" + ihdr_data)
    ihdr = struct.pack(">I", 13) + b"IHDR" + ihdr_data + struct.pack(">I", ihdr_crc & 0xFFFFFFFF)
    raw_data = b""
    for y in range(height):
        raw_data += b"\x00" + b"\x00\xff\x00" * width
    compressed = zlib.compress(raw_data)
    idat_crc = zlib.crc32(b"IDAT" + compressed)
    idat = struct.pack(">I", len(compressed)) + b"IDAT" + compressed + struct.pack(">I", idat_crc & 0xFFFFFFFF)
    iend_crc = zlib.crc32(b"IEND")
    iend = struct.pack(">I", 0) + b"IEND" + struct.pack(">I", iend_crc & 0xFFFFFFFF)
    path.write_bytes(sig + ihdr + idat + iend)


def test_discover_videos_empty(tmp_path: Path):
    assert discover_videos(tmp_path) == []


def test_discover_videos_finds_mp4(tmp_path: Path):
    vid = tmp_path / "seq" / "P1_01.mp4"
    vid.parent.mkdir(parents=True)
    vid.write_text("fake video")
    found = discover_videos(tmp_path)
    assert len(found) == 1
    assert found[0].name == "P1_01.mp4"


def test_discover_videos_multiple_extensions(tmp_path: Path):
    (tmp_path / "a.mp4").write_text("")
    (tmp_path / "b.avi").write_text("")
    (tmp_path / "c.mov").write_text("")
    found = discover_videos(tmp_path)
    assert len(found) == 3


def test_detect_vertical_stack_returns_zero_for_unknown(tmp_path: Path):
    vid = tmp_path / "test.mp4"
    vid.write_text("fake")
    result = detect_vertical_stack(vid, method="auto")
    assert result == 0


def test_read_png_dimensions():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "test.png"
        _create_dummy_png(p, 640, 480)
        dims = _read_png_dimensions(p)
        assert dims == (640, 480)


def test_read_png_dimensions_not_png(tmp_path: Path):
    p = tmp_path / "not.png"
    p.write_text("not a png")
    assert _read_png_dimensions(p) is None


def test_build_report_empty():
    report = build_report([])
    assert report["sequences_processed"] == 0
    assert report["total_frame_count"] == 0
    assert report["error_count"] == 0


def test_build_report_single():
    results = [{
        "sequence_id": "P1_01",
        "input_video": "/videos/P1_01.mp4",
        "frame_count": 100,
        "extracted_left_count": 100,
        "extracted_right_count": 100,
        "mask_count": 100,
        "width": 640,
        "height": 240,
        "dropped_frames": [],
        "extraction_errors": [],
    }]
    report = build_report(results)
    assert report["sequences_processed"] == 1
    assert report["total_frame_count"] == 100
    assert report["error_count"] == 0


def test_build_report_with_errors():
    results = [{
        "sequence_id": "P1_01",
        "input_video": "/v/P1_01.mp4",
        "frame_count": 0,
        "extracted_left_count": 0,
        "extracted_right_count": 0,
        "mask_count": 0,
        "width": 0,
        "height": 0,
        "dropped_frames": [],
        "extraction_errors": ["ffmpeg failed"],
        "error": "ffmpeg failed for left_half: error msg",
    }]
    report = build_report(results)
    assert report["error_count"] == 2


def test_build_report_multiple():
    results = [
        {"sequence_id": "S1", "frame_count": 50, "extracted_left_count": 50,
         "extracted_right_count": 50, "mask_count": 0, "dropped_frames": [],
         "extraction_errors": [], "input_video": "/v/S1.mp4", "width": 640, "height": 240},
        {"sequence_id": "S2", "frame_count": 75, "extracted_left_count": 75,
         "extracted_right_count": 75, "mask_count": 0, "dropped_frames": [],
         "extraction_errors": [], "input_video": "/v/S2.mp4", "width": 640, "height": 240},
    ]
    report = build_report(results)
    assert report["total_frame_count"] == 125
    assert report["total_left_count"] == 125
    assert report["total_right_count"] == 125


def test_discover_videos_skips_non_video(tmp_path: Path):
    (tmp_path / "readme.txt").write_text("not a video")
    assert discover_videos(tmp_path) == []


def test_discover_videos_subdir(tmp_path: Path):
    sub = tmp_path / "sub" / "nested"
    sub.mkdir(parents=True)
    vid = sub / "vid.mp4"
    vid.write_text("")
    found = discover_videos(tmp_path)
    assert len(found) == 1
    assert found[0] == vid
