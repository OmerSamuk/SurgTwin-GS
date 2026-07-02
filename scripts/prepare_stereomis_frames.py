import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional


STEREO_VALIDATION_METHODS = ("auto", "aspect", "ffprobe", "cv2")


def discover_videos(
    data_root: Path, extensions: tuple = (".mp4", ".avi", ".mov", ".mkv")
) -> List[Path]:
    videos = []
    for ext in extensions:
        videos.extend(data_root.rglob(f"*{ext}"))
    return sorted(videos)


def detect_vertical_stack(video_path: Path, method: str = "auto") -> int:
    if method == "auto":
        width_est, height_est = _probe_resolution(video_path)
        if width_est is None or height_est is None:
            return 0
        if height_est > width_est * 1.5:
            return height_est // 2
        return 0
    return 0


def _probe_resolution(video_path: Path):
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height",
                "-of", "csv=p=0",
                str(video_path),
            ],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            return None, None
        parts = result.stdout.strip().split(",")
        if len(parts) != 2:
            return None, None
        return int(parts[0]), int(parts[1])
    except (subprocess.SubprocessError, ValueError, FileNotFoundError):
        return None, None


def extract_frames(
    video_path: Path,
    output_dir: Path,
    half_height: int,
) -> Dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    left_dir = output_dir / "left"
    right_dir = output_dir / "right"
    left_dir.mkdir(exist_ok=True)
    right_dir.mkdir(exist_ok=True)
    template = str(output_dir / "%06d.png")
    left_template = str(left_dir / "%06d.png")
    right_template = str(right_dir / "%06d.png")
    left_filter = f"crop=iw:ih/2:0:0"
    right_filter = f"crop=iw:ih/2:0:ih/2"

    for label, filt, tpl in [
        ("left_full", None, template),
        ("left_half", left_filter, left_template),
        ("right_half", right_filter, right_template),
    ]:
        cmd = ["ffmpeg", "-y", "-i", str(video_path)]
        if filt:
            cmd.extend(["-vf", filt])
        cmd.extend(["-qscale:v", "1", "-pix_fmt", "yuv420p", tpl])
        try:
            subprocess.run(
                cmd, capture_output=True, text=True, timeout=300,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            return {
                "error": f"ffmpeg failed for {label}: {e.stderr[-500:] if e.stderr else 'no stderr'}",
            }

    left_files = sorted(left_dir.glob("*.png"))
    right_files = sorted(right_dir.glob("*.png"))

    if half_height > 0:
        if left_files and right_files:
            first_left = _read_png_dimensions(left_files[0])
            first_right = _read_png_dimensions(right_files[0])
            expected_left_h = half_height
            expected_right_h = half_height
            expected_w = first_left[0] if first_left else 0
            if first_left and first_left[1] != expected_left_h:
                pass
            if first_right and first_right[1] != expected_right_h:
                pass

    return {
        "sequence_id": video_path.stem,
        "input_video": str(video_path),
        "frame_count": len(left_files),
        "extracted_left_count": len(left_files),
        "extracted_right_count": len(right_files),
        "width": first_left[0] if left_files and (first_left := _read_png_dimensions(left_files[0])) else 0,
        "height": first_left[1] if left_files and (first_left := _read_png_dimensions(left_files[0])) else 0,
        "dropped_frames": [],
        "extraction_errors": [],
    }


def _read_png_dimensions(png_path: Path):
    try:
        import struct
        with open(png_path, "rb") as f:
            f.read(8)
            f.read(4)
            chunk_type = f.read(4)
            if chunk_type != b"IHDR":
                return None
            data = f.read(8)
            width, height = struct.unpack(">II", data)
            return width, height
    except Exception:
        return None


def pair_masks(
    mask_source_dir: Path,
    frame_left_dir: Path,
    mask_output_dir: Path,
) -> int:
    import shutil
    mask_output_dir.mkdir(parents=True, exist_ok=True)
    paired = 0
    for frame_path in sorted(frame_left_dir.glob("*.png")):
        mask_candidates = [
            mask_source_dir / frame_path.name,
            mask_source_dir / f"{frame_path.stem}.png",
        ]
        for candidate in mask_candidates:
            if candidate.exists():
                shutil.copy2(candidate, mask_output_dir / frame_path.name)
                paired += 1
                break
    return paired


def build_report(all_results: List[Dict]) -> Dict:
    total_frames = sum(r.get("frame_count", 0) for r in all_results)
    total_left = sum(r.get("extracted_left_count", 0) for r in all_results)
    total_right = sum(r.get("extracted_right_count", 0) for r in all_results)
    errors = []
    for r in all_results:
        if r.get("error"):
            errors.append(r["error"])
        errors.extend(r.get("extraction_errors", []))
    return {
        "sequences_processed": len(all_results),
        "total_frame_count": total_frames,
        "total_left_count": total_left,
        "total_right_count": total_right,
        "error_count": len(errors),
        "errors": errors[:20],
        "sequence_details": all_results,
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract and split StereoMIS video frames (vertical stack stereo)."
    )
    parser.add_argument(
        "--data-root", type=str, default="data/external/stereomis/raw",
        help="Root directory containing StereoMIS video files",
    )
    parser.add_argument(
        "--output-root", type=str, default="data/processed/stereomis",
        help="Root directory for extracted frames and masks",
    )
    parser.add_argument(
        "--skip-extract", action="store_true",
        help="Skip ffmpeg extraction (reuse existing frames)",
    )
    parser.add_argument(
        "--mask-source", type=str, default=None,
        help="Optional path to mask directory (mapped by filename)",
    )
    parser.add_argument(
        "--output-report", type=str,
        default="outputs/reports/stereomis_frame_extraction_report.json",
        help="Path to write extraction report JSON",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    data_root = Path(args.data_root)
    if not data_root.exists():
        print(f"ERROR: data root not found: {data_root}", file=sys.stderr)
        return 1

    videos = discover_videos(data_root)
    if not videos:
        print(f"ERROR: no video files found in {data_root}", file=sys.stderr)
        return 1

    print(f"Found {len(videos)} video(s) in {data_root}")
    all_results = []
    for vpath in videos:
        print(f"  Processing: {vpath.name}...")
        seq_id = vpath.stem
        frames_out = Path(args.output_root) / "frames" / seq_id
        if not args.skip_extract:
            half_h = detect_vertical_stack(vpath)
            result = extract_frames(vpath, frames_out, half_h)
        else:
            left_files = sorted((frames_out / "left").glob("*.png"))
            right_files = sorted((frames_out / "right").glob("*.png"))
            result = {
                "sequence_id": seq_id,
                "input_video": str(vpath),
                "frame_count": len(left_files),
                "extracted_left_count": len(left_files),
                "extracted_right_count": len(right_files),
                "width": 0,
                "height": 0,
                "dropped_frames": [],
                "extraction_errors": [],
            }

        if args.mask_source and "error" not in result:
            mask_src = Path(args.mask_source) / seq_id
            mask_out = Path(args.output_root) / "masks" / seq_id
            if mask_src.exists():
                paired = pair_masks(mask_src, frames_out / "left", mask_out)
                result["mask_count"] = paired
            else:
                result["mask_count"] = 0
        else:
            result["mask_count"] = 0

        all_results.append(result)
        status = "OK" if "error" not in result else f"FAIL ({result['error'][:80]})"
        print(f"    {status}: {result.get('frame_count', 0)} frames, {result.get('mask_count', 0)} masks")

    report = build_report(all_results)
    report_path = Path(args.output_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\nExtraction report written to {report_path}")

    if report["error_count"] > 0:
        print(f"WARNING: {report['error_count']} error(s) occurred during extraction.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
