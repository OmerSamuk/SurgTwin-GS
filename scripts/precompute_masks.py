import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from surgtwin.data.manifest import load_manifest
from surgtwin.masks.specular import detect_specular_hsv


def _load_rgb_cv(path: str) -> np.ndarray:
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return img


def main():
    parser = argparse.ArgumentParser(
        description="Precompute specular occlusion masks for SERV-CT"
    )
    parser.add_argument("--manifest", type=str, required=True,
                        help="Path to manifest JSONL")
    parser.add_argument("--output_dir", type=str, default="data/processed/masks",
                        help="Output directory for masks")
    parser.add_argument("--mask_types", type=str, nargs="+", default=["specular"],
                        choices=["specular"],
                        help="Mask types to precompute")
    parser.add_argument("--save_overlays", action="store_true", default=True,
                        help="Save visual overlay PNGs")
    parser.add_argument("--v_threshold", type=float, default=220.0,
                        help="HSV V threshold for specular detection")
    parser.add_argument("--s_threshold", type=float, default=40.0,
                        help="HSV S threshold for specular detection")
    args = parser.parse_args()

    entries = load_manifest(Path(args.manifest))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mask_quality = {}

    for entry in entries:
        sid = entry["sample_id"]
        left_rgb_path = entry["left_rgb_path"]

        if not Path(left_rgb_path).exists():
            print(f"  WARNING: RGB not found for {sid}: {left_rgb_path}")
            continue

        rgb = _load_rgb_cv(left_rgb_path)

        for mask_type in args.mask_types:
            if mask_type == "specular":
                mask = detect_specular_hsv(
                    rgb,
                    v_threshold=args.v_threshold,
                    s_threshold=args.s_threshold,
                )
            coverage = float(mask.mean())
            mask_path = out_dir / f"{sid}_specular.npy"
            np.save(str(mask_path), mask)
            mask_quality[sid] = {
                "mask_path": str(mask_path),
                "coverage": coverage,
                "height": int(mask.shape[0]),
                "width": int(mask.shape[1]),
                "dtype": str(mask.dtype),
            }
            print(f"  {sid}: specular coverage={coverage:.4f} -> {mask_path}")

            if args.save_overlays:
                overlay = np.zeros((*mask.shape, 3), dtype=np.uint8)
                overlay[mask] = [255, 0, 0]
                alpha = 0.3
                rgb_255 = (rgb * 255).astype(np.uint8)
                blended = (alpha * overlay + (1 - alpha) * rgb_255).astype(np.uint8)
                overlay_path = out_dir / f"{sid}_specular_overlay.png"
                cv2.imwrite(str(overlay_path), cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
                mask_quality[sid]["overlay_path"] = str(overlay_path)

    report = {
        "mask_type": "specular",
        "num_samples_total": len(entries),
        "num_samples_processed": len(mask_quality),
        "parameters": {
            "v_threshold": args.v_threshold,
            "s_threshold": args.s_threshold,
        },
        "per_frame": mask_quality,
    }

    if mask_quality:
        coverages = [v["coverage"] for v in mask_quality.values()]
        report["summary"] = {
            "mean_coverage": float(np.mean(coverages)),
            "min_coverage": float(np.min(coverages)),
            "max_coverage": float(np.max(coverages)),
            "frames_with_no_specular": sum(1 for c in coverages if c < 1e-6),
        }

    report_path = out_dir / "mask_quality_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\nMask quality report: {report_path.resolve()}")
    print("Done.")


if __name__ == "__main__":
    main()
