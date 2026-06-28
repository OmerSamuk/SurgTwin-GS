import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from surgtwin.cameras.camera_types import CameraData
from surgtwin.data.depth_io import load_servct_depth
from surgtwin.data.manifest import load_manifest, filter_by_split, validate_manifest_entry
from surgtwin.evaluation.depth_diagnostics import (
    compare_depth_distributions,
    depth_in_servct_range,
    depth_scale_ok,
    classify_scale,
)
from surgtwin.gaussian.backend_gsplat import GsplatBackend
from surgtwin.gaussian.initialization import initialize_gaussians_from_rgbd


def load_rgb(path: str) -> torch.Tensor:
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(img)


def save_depth_viz(tensor, path):
    arr = tensor.cpu().numpy()
    valid = arr > 0
    if valid.any():
        lo, hi = arr[valid].min(), arr[valid].max()
        viz = np.clip((arr - lo) / (hi - lo + 1e-8), 0, 1)
    else:
        viz = np.zeros_like(arr)
    viz_u8 = (viz * 255).astype(np.uint8)
    viz_u8 = cv2.applyColorMap(viz_u8, cv2.COLORMAP_JET)
    cv2.imwrite(str(path), viz_u8)


def main():
    parser = argparse.ArgumentParser(description="Verify gsplat render depth semantics on real SERV-CT data.")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--sequence_id", type=str, default="Experiment_1", help="Sequence to verify (Experiment_2 held-out)")
    parser.add_argument("--output_dir", type=str, default="outputs/runs/depth_semantics_m2a")
    parser.add_argument("--num_points", type=int, default=20000)
    parser.add_argument("--scale_tolerance", type=float, default=0.10)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for real depth verification (gsplat rasterization).")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    entries = load_manifest(Path(args.manifest))
    for e in entries:
        validate_manifest_entry(e)

    seq_entries = [e for e in entries if e["sequence_id"] == args.sequence_id]
    if not seq_entries:
        raise ValueError(f"No entries found for sequence '{args.sequence_id}'")

    seq_entries.sort(key=lambda e: e["frame_index"])
    print(f"Found {len(seq_entries)} samples for {args.sequence_id}")

    backend = GsplatBackend()
    device = torch.device("cuda")

    per_sample = []

    for idx, entry in enumerate(seq_entries):
        sid = entry["sample_id"]
        split = entry.get("split", "unknown")
        print(f"\n[{idx+1}/{len(seq_entries)}] Processing {sid} (split={split})...")

        rgb = load_rgb(entry["left_rgb_path"]).to(device)
        depth_m = load_servct_depth(Path(entry["left_depth_path"])).to(device)
        K = torch.tensor(entry["K_left"], dtype=torch.float32, device=device)
        c2w = torch.tensor(entry["c2w_left"], dtype=torch.float32, device=device)
        w2c = torch.tensor(entry["w2c_left"], dtype=torch.float32, device=device)
        H, W = entry["height"], entry["width"]

        camera = CameraData(K=K, c2w=c2w, w2c=w2c, height=H, width=W)

        gaussians = initialize_gaussians_from_rgbd(
            rgb=rgb, depth_m=depth_m, K=K, c2w=c2w, num_points=args.num_points,
        ).to(device)

        output = backend.render(
            gaussians=gaussians,
            camera=camera,
            image_height=H,
            image_width=W,
            render_depth=True,
        )

        rendered_depth = output.depth

        comparison = compare_depth_distributions(rendered_depth, depth_m)
        in_range = depth_in_servct_range(comparison)
        shape_ok = (rendered_depth.shape[0] == H and rendered_depth.shape[1] == W)
        finite_ok = rendered_depth.isfinite().float().mean().item() >= 0.8

        scale_ok, scale_tier, scale_val = depth_scale_ok(
            rendered_depth, depth_m, tolerance=args.scale_tolerance
        )

        sample_result = {
            "sample_id": sid,
            "split": split,
            "frame_index": entry["frame_index"],
            "depth_semantics": output.aux["depth_semantics"],
            "metric_depth_verified": output.aux["metric_depth_verified"],
            "distribution": comparison,
            "range_ok": in_range,
            "shape_ok": shape_ok,
            "finite_ok": finite_ok,
            "scale_ok": scale_ok,
            "scale_tier": scale_tier,
            "scale_ratio": scale_val,
        }
        per_sample.append(sample_result)

        print(f"  depth_semantics={output.aux['depth_semantics']}  "
              f"shape_ok={shape_ok}  range_ok={in_range}  "
              f"scale={scale_val:.4f} ({scale_tier})  finite_ok={finite_ok}")

        save_depth_viz(rendered_depth, out_dir / f"render_depth_{sid}.png")
        save_depth_viz(depth_m, out_dir / f"gt_depth_{sid}.png")

        if idx < 3:
            rgb_np = (rgb.cpu().numpy() * 255).astype(np.uint8)
            rgb_out = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(out_dir / f"input_rgb_{sid}.png"), rgb_out)

    scale_tiers = [s["scale_tier"] for s in per_sample]
    total = len(per_sample)
    metric_count = sum(1 for s in per_sample if s["scale_ok"])
    shape_pass = all(s["shape_ok"] for s in per_sample)
    range_pass = all(s["range_ok"] for s in per_sample)
    finite_pass = all(s["finite_ok"] for s in per_sample)

    real_metric_ok = metric_count >= (total * 0.75)

    result = {
        "sequence_id": args.sequence_id,
        "num_samples": total,
        "samples_processed": total,
        "scale_tolerance": args.scale_tolerance,
        "backend_self_check_metric": per_sample[0]["metric_depth_verified"] if per_sample else False,
        "shape_pass": shape_pass,
        "range_pass": range_pass,
        "finite_pass": finite_pass,
        "real_metric_ok": real_metric_ok,
        "metric_samples": metric_count,
        "metric_ratio": round(metric_count / total, 4) if total else 0,
        "scale_tiers": {t: scale_tiers.count(t) for t in set(scale_tiers)},
        "per_sample": per_sample,
    }

    out_path = out_dir / "real_verification.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nReal verification report saved to {out_path}")

    failed = []
    if not shape_pass:
        failed.append("shape")
    if not range_pass:
        failed.append("range")
    if not finite_pass:
        failed.append("finite")
    if not real_metric_ok:
        failed.append("metric (need >=75% samples within ±10% tolerance)")

    if failed:
        print(f"\nWARNING: Real verification FAILED on: {', '.join(failed)}")
        sys.exit(1)

    print("\nReal verification PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
