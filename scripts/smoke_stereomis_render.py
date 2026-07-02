import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def load_manifest(manifest_path: Path) -> List[Dict]:
    entries = []
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def validate_manifest_entry(entry: Dict) -> Dict[str, Any]:
    checks = {}
    required_fields = [
        "sample_id", "dataset_name", "sequence_id", "frame_id", "split",
        "left_rgb_path", "right_rgb_path", "intrinsics", "c2w", "w2c",
        "width", "height", "depth_semantics", "pose_source",
    ]
    missing = [f for f in required_fields if f not in entry]
    checks["missing_fields"] = missing
    checks["sample_id"] = entry.get("sample_id", "N/A")
    checks["dataset_name"] = entry.get("dataset_name")
    checks["depth_semantics"] = entry.get("depth_semantics")
    checks["pose_source"] = entry.get("pose_source")
    checks["valid"] = len(missing) == 0
    return checks


def validate_camera_matrices(entry: Dict) -> Dict[str, Any]:
    checks = {}
    intrinsics = np.array(entry.get("intrinsics", []), dtype=np.float64).reshape(3, 3)
    c2w = np.array(entry.get("c2w", []), dtype=np.float64).reshape(4, 4)
    w2c = np.array(entry.get("w2c", []), dtype=np.float64).reshape(4, 4)
    checks["K_finite"] = bool(np.all(np.isfinite(intrinsics)))
    checks["K_positive_fx"] = bool(intrinsics[0, 0] > 0)
    checks["K_positive_fy"] = bool(intrinsics[1, 1] > 0)
    checks["c2w_finite"] = bool(np.all(np.isfinite(c2w)))
    checks["w2c_finite"] = bool(np.all(np.isfinite(w2c)))
    checks["c2w_w2c_consistent"] = bool(
        np.allclose(c2w @ w2c, np.eye(4), atol=1e-4)
    )
    R = c2w[:3, :3]
    det_ok = np.abs(np.linalg.det(R) - 1.0) < 0.1
    checks["c2w_rotation_valid"] = bool(det_ok)
    checks["valid"] = all([
        checks["K_finite"], checks["K_positive_fx"], checks["K_positive_fy"],
        checks["c2w_finite"], checks["w2c_finite"],
    ])
    return checks


def check_image_paths(entry: Dict, frames_root: Optional[Path] = None) -> Dict[str, Any]:
    checks = {}
    left_path = entry.get("left_rgb_path", "")
    right_path = entry.get("right_rgb_path", "")
    if frames_root:
        left_full = frames_root / left_path if not Path(left_path).is_absolute() else Path(left_path)
        right_full = frames_root / right_path if not Path(right_path).is_absolute() else Path(right_path)
    else:
        left_full = Path(left_path)
        right_full = Path(right_path)
    checks["left_exists"] = left_full.exists()
    checks["right_exists"] = right_full.exists()
    mask_path = entry.get("mask_path")
    if mask_path:
        if frames_root:
            mask_full = frames_root / mask_path if not Path(mask_path).is_absolute() else Path(mask_path)
        else:
            mask_full = Path(mask_path)
        checks["mask_exists"] = mask_full.exists()
    else:
        checks["mask_exists"] = None
    checks["valid"] = checks["left_exists"] and checks["right_exists"]
    return checks


def count_split_entries(entries: List[Dict]) -> Dict[str, int]:
    counts = {}
    for e in entries:
        s = e.get("split", "unknown")
        counts[s] = counts.get(s, 0) + 1
    return counts


def load_sample_image(path: Path):
    import cv2
    img = cv2.imread(str(path))
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def render_smoke(entry: Dict, num_gaussians: int = 100) -> Dict[str, Any]:
    import torch
    if not torch.cuda.is_available():
        return {"cuda_available": False, "skipped": True,
                "reason": "CUDA not available, render smoke skipped"}

    from surgtwin.cameras.camera_types import CameraData
    from surgtwin.gaussian.backend_gsplat import GsplatBackend
    from surgtwin.gaussian.initialization import initialize_gaussians_from_rgbd

    width = entry.get("width", 640)
    height = entry.get("height", 480)
    K = torch.tensor(np.array(entry["intrinsics"], dtype=np.float32), dtype=torch.float32)
    c2w_t = torch.tensor(np.array(entry["c2w"], dtype=np.float32), dtype=torch.float32)
    w2c_t = torch.tensor(np.array(entry["w2c"], dtype=np.float32), dtype=torch.float32)
    camera = CameraData(K=K, c2w=c2w_t, w2c=w2c_t, height=height, width=width)

    left_path = entry.get("left_rgb_path", "")
    img = load_sample_image(Path(left_path))
    if img is None:
        return {"error": f"could not load image: {left_path}"}

    rgb_t = torch.from_numpy(img.astype(np.float32) / 255.0)
    dummy_depth = torch.ones((height, width), dtype=torch.float32) * 0.15

    gaussians = initialize_gaussians_from_rgbd(
        rgb=rgb_t,
        depth_m=dummy_depth,
        K=K,
        c2w=c2w_t,
        num_points=num_gaussians,
    )
    gaussians_cuda = gaussians.to(torch.device("cuda"))

    backend = GsplatBackend()
    output = backend.render(
        gaussians=gaussians_cuda,
        camera=camera,
        image_height=height,
        image_width=width,
        render_depth=True,
    )

    result = {
        "cuda_available": True,
        "render_completed": True,
        "render_rgb_shape": list(output.rgb.shape),
        "render_rgb_finite": bool(output.rgb.isfinite().all()),
        "render_depth_shape": list(output.depth.shape) if output.depth is not None else None,
        "render_depth_finite": bool(output.depth.isfinite().all()) if output.depth is not None else None,
        "depth_semantics": output.aux.get("depth_semantics", "unknown"),
        "metric_depth_verified": output.aux.get("metric_depth_verified", False),
        "supports_contrib": output.aux.get("supports_contrib", False),
        "supports_alpha": output.aux.get("supports_alpha", False),
    }
    return result


def build_smoke_report(
    entries: List[Dict],
    manifest_checks: List[Dict],
    camera_checks: List[Dict],
    path_checks: List[Dict],
    split_counts: Dict[str, int],
    render_results: Optional[List[Dict]] = None,
) -> Dict:
    manifest_valid = all(c["valid"] for c in manifest_checks)
    camera_valid = all(c["valid"] for c in camera_checks)
    path_valid = all(c["valid"] for c in path_checks)
    overall_valid = manifest_valid and camera_valid and path_valid
    report: Dict[str, Any] = {
        "smoke_passed": overall_valid,
        "total_entries": len(entries),
        "manifest_valid": manifest_valid,
        "camera_matrices_valid": camera_valid,
        "image_paths_valid": path_valid,
        "split_distribution": split_counts,
        "details": {
            "manifest_checks": manifest_checks[:5],
            "camera_checks": camera_checks[:5],
            "path_checks": path_checks[:5],
        },
    }
    if render_results:
        report["render_smoke"] = render_results
        attempted = [r for r in render_results if not r.get("skipped")]
        if not attempted:
            report["render_smoke_passed"] = None
        elif all(r.get("render_completed", False) for r in attempted):
            report["render_smoke_passed"] = True
        else:
            report["render_smoke_passed"] = False
    return report


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke test the StereoMIS data pipeline: manifest, loading, camera matrices, and optional gsplat render."
    )
    parser.add_argument(
        "--manifest", type=str, required=True,
        help="Path to StereoMIS manifest JSONL",
    )
    parser.add_argument(
        "--frames-root", type=str, default=None,
        help="Root directory for frame paths (used to resolve relative paths)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs/runs/stereomis_smoke",
        help="Output directory for smoke report",
    )
    parser.add_argument(
        "--num-samples", type=int, default=2,
        help="Number of manifest samples to validate (default: 2)",
    )
    parser.add_argument(
        "--num-gaussians", type=int, default=100,
        help="Number of Gaussians for render smoke (default: 100)",
    )
    parser.add_argument(
        "--skip-render", action="store_true",
        help="Skip the gsplat render smoke (useful without CUDA)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        return 1

    entries = load_manifest(manifest_path)
    print(f"Loaded manifest: {len(entries)} entries")

    frames_root = Path(args.frames_root) if args.frames_root else None
    n = min(args.num_samples, len(entries))

    manifest_checks = []
    camera_checks = []
    path_checks = []
    for i in range(n):
        mc = validate_manifest_entry(entries[i])
        cc = validate_camera_matrices(entries[i])
        pc = check_image_paths(entries[i], frames_root)
        manifest_checks.append(mc)
        camera_checks.append(cc)
        path_checks.append(pc)
        print(f"  Sample {i} ({entries[i]['sample_id']}): "
              f"manifest={'OK' if mc['valid'] else 'FAIL'} "
              f"camera={'OK' if cc['valid'] else 'FAIL'} "
              f"paths={'OK' if pc['valid'] else 'FAIL'}")

    split_counts = count_split_entries(entries)
    print(f"  Splits: {split_counts}")

    render_results = None
    if not args.skip_render:
        try:
            render_results = []
            for i in range(n):
                rr = render_smoke(entries[i], num_gaussians=args.num_gaussians)
                render_results.append(rr)
                if rr.get("skipped"):
                    print(f"  Render sample {i}: SKIPPED ({rr.get('reason', '')})")
                elif rr.get("error"):
                    print(f"  Render sample {i}: FAIL ({rr['error']})")
                else:
                    print(f"  Render sample {i}: OK (rgb={rr.get('render_rgb_shape')}, "
                          f"depth={rr.get('render_depth_shape')})")
        except Exception as e:
            print(f"  Render smoke ERROR: {e}", file=sys.stderr)
            render_results = [{"error": str(e)}]

    report = build_smoke_report(entries, manifest_checks, camera_checks,
                                 path_checks, split_counts, render_results)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "smoke_report.json"
    report_path.write_text(json.dumps(report, indent=2))

    summary_path = out_dir / "stereomis_render_smoke_report.md"
    lines = [
        "# StereoMIS Render Smoke Report",
        "",
        f"**Date:** 2026-07-02",
        f"**Manifest:** {manifest_path}",
        f"**Total entries:** {len(entries)}",
        "",
        "## Smoke Results",
        "",
        f"- Manifest valid: {report['manifest_valid']}",
        f"- Camera matrices valid: {report['camera_matrices_valid']}",
        f"- Image paths valid: {report['image_paths_valid']}",
        f"- Overall smoke: **{'PASS' if report['smoke_passed'] else 'FAIL'}**",
    ]
    if render_results:
        if report["render_smoke_passed"]:
            lines.append("- Render smoke: **PASS**")
        elif report["render_smoke_passed"] is False:
            lines.append("- Render smoke: **FAIL**")
        else:
            lines.append("- Render smoke: SKIPPED")
    lines.extend([
        "",
        "## Split Distribution",
        "",
    ])
    for split_name, cnt in sorted(split_counts.items()):
        lines.append(f"- {split_name}: {cnt}")
    lines.extend([
        "",
        "## Details",
        "",
        "### Validation frames",
        f"- Valid validation frames: {split_counts.get('val', 0)}",
        f"- Meets >= 4 threshold: {split_counts.get('val', 0) >= 4}",
        f"- Meets >= 6 threshold: {split_counts.get('val', 0) >= 6}",
    ])
    if render_results:
        lines.extend([
            "",
            "### Render Smoke Details",
        ])
        for i, rr in enumerate(render_results or []):
            if rr.get("skipped"):
                lines.append(f"- Sample {i}: SKIPPED ({rr.get('reason', '')})")
            elif rr.get("error"):
                lines.append(f"- Sample {i}: ERROR ({rr['error']})")
            else:
                lines.append(f"- Sample {i}: RGB shape {rr.get('render_rgb_shape')}, depth shape {rr.get('render_depth_shape')}")
    lines.append("")
    summary_path.write_text("\n".join(lines) + "\n")

    print(f"\nSmoke report written to {summary_path}")
    print(f"Smoke report JSON: {report_path}")
    print(f"\nOverall smoke: {'PASS' if report['smoke_passed'] else 'FAIL'}")
    return 0 if report['smoke_passed'] else 1


if __name__ == "__main__":
    sys.exit(main())
