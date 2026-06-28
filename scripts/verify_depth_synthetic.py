import argparse
import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from surgtwin.cameras.camera_types import CameraData
from surgtwin.data.depth_io import load_servct_depth
from surgtwin.evaluation.depth_diagnostics import (
    compare_depth_distributions,
    depth_in_servct_range,
    depth_scale_ok,
)
from surgtwin.gaussian.backend_gsplat import GsplatBackend
from surgtwin.gaussian.initialization import initialize_gaussians_from_rgbd
from scripts.generate_synthetic_servct import create_synthetic_plane_scene, depth_to_servct_raw


def main():
    parser = argparse.ArgumentParser(description="Verify gsplat render depth semantics on synthetic SERV-CT plane.")
    parser.add_argument("--output_dir", type=str, default="outputs/runs/depth_semantics_m2a")
    parser.add_argument("--num_points", type=int, default=20000)
    parser.add_argument("--scale_tolerance", type=float, default=0.10)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for synthetic depth verification (gsplat rasterization).")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Generating synthetic SERV-CT plane scene (50-150mm)...")
    rgb_np, depth_mm, P1, P2, Q, (height, width) = create_synthetic_plane_scene()
    K = P1[:3, :3].astype(np.float32)

    print(f"  Shape: {height}x{width}, Depth range: {depth_mm.min():.0f}-{depth_mm.max():.0f} mm")
    print(f"  K fx={K[0,0]:.1f} fy={K[1,1]:.1f} cx={K[0,2]:.1f} cy={K[1,2]:.1f}")

    raw_depth = depth_to_servct_raw(depth_mm)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        from PIL import Image
        Image.fromarray(raw_depth, mode="I;16").save(tmp.name)
        depth_m = load_servct_depth(Path(tmp.name))
        Path(tmp.name).unlink()

    rgb_t = torch.from_numpy(rgb_np.astype(np.float32) / 255.0)
    K_t = torch.tensor(K, dtype=torch.float32)
    c2w = torch.eye(4, dtype=torch.float32)
    w2c = torch.eye(4, dtype=torch.float32)

    camera = CameraData(K=K_t, c2w=c2w, w2c=w2c, height=height, width=width)

    print(f"Initializing {args.num_points} Gaussians from synthetic RGB-D...")
    gaussians = initialize_gaussians_from_rgbd(
        rgb=rgb_t,
        depth_m=depth_m,
        K=K_t,
        c2w=c2w,
        num_points=args.num_points,
    )
    gaussians_cuda = gaussians.to(torch.device("cuda"))
    print(f"  Gaussians initialized: {gaussians_cuda.num_gaussians()}")

    print("Rendering with gsplat (mode=RGB+ED)...")
    backend = GsplatBackend()
    output = backend.render(
        gaussians=gaussians_cuda,
        camera=camera,
        image_height=height,
        image_width=width,
        render_depth=True,
    )

    rendered_depth = output.depth
    rendered_rgb = output.rgb

    print(f"  Depth shape: {rendered_depth.shape}, dtype: {rendered_depth.dtype}")
    print(f"  Depth semantics: {output.aux['depth_semantics']}")
    print(f"  Metric depth verified: {output.aux['metric_depth_verified']}")

    result = {
        "backend": "gsplat",
        "synthetic_scene": True,
        "image_size": {"H": height, "W": width},
        "num_points_init": args.num_points,
        "scale_tolerance": args.scale_tolerance,
        "depth_semantics": output.aux["depth_semantics"],
        "metric_depth_verified": output.aux["metric_depth_verified"],
        "backend_self_check": output.aux["supports_metric_depth"],
    }

    comparison = compare_depth_distributions(rendered_depth, depth_m)
    result["distribution"] = comparison

    in_range = depth_in_servct_range(comparison)
    result["range_ok"] = in_range
    result["range_info"] = "SERV-CT reference range: 0.02-0.30m" if in_range else "OUT OF RANGE"

    scale_ok, scale_tier, scale_val = depth_scale_ok(
        rendered_depth, depth_m, tolerance=args.scale_tolerance
    )
    result["scale_ok"] = scale_ok
    result["scale_tier"] = scale_tier
    result["scale_ratio"] = scale_val

    shape_ok = (rendered_depth.shape[0] == height and rendered_depth.shape[1] == width)
    finite_ok = rendered_depth.isfinite().float().mean().item() >= 0.8
    result["shape_ok"] = shape_ok
    result["finite_ok"] = finite_ok

    synthetic_ok = (
        result["depth_semantics"] == "metric_meters"
        and shape_ok
        and in_range
        and scale_ok
        and finite_ok
    )
    result["synthetic_ok"] = synthetic_ok

    print(f"\n  shape_ok={shape_ok}  range_ok={in_range}  scale_ok={scale_ok} ({scale_tier}, {scale_val:.4f})  finite_ok={finite_ok}")
    print(f"  synthetic_ok={synthetic_ok}")

    out_path = out_dir / "synthetic_verification.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nVerification report saved to {out_path}")

    # Save visualization
    def save_depth_viz(tensor, path, cmap=True):
        arr = tensor.cpu().numpy()
        valid = arr > 0
        if valid.any():
            lo, hi = arr[valid].min(), arr[valid].max()
            viz = np.clip((arr - lo) / (hi - lo + 1e-8), 0, 1)
        else:
            viz = np.zeros_like(arr)
        viz_u8 = (viz * 255).astype(np.uint8)
        if cmap:
            viz_u8 = cv2.applyColorMap(viz_u8, cv2.COLORMAP_JET)
        cv2.imwrite(str(path), viz_u8)

    save_depth_viz(rendered_depth, out_dir / "synthetic_render_depth_color.png")
    save_depth_viz(depth_m, out_dir / "synthetic_gt_depth_color.png")

    rgb_bgr = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out_dir / "synthetic_input_rgb.png"), rgb_bgr)

    print(f"Visualizations saved to {out_dir}")

    if not synthetic_ok:
        print("\nWARNING: Synthetic verification FAILED. Check the report for details.")
        sys.exit(1)

    print("\nSynthetic verification PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
