import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from surgtwin.cameras.camera_types import CameraData
from surgtwin.data.depth_io import load_servct_depth
from surgtwin.data.manifest import load_manifest, get_sample_by_index, validate_manifest_entry
from surgtwin.gaussian.backend_gsplat import GsplatBackend
from surgtwin.gaussian.initialization import initialize_gaussians_from_rgbd


def main():
    parser = argparse.ArgumentParser(description="Sprint 0 SERV-CT render with gsplat.")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--num_points", type=int, default=20000)
    parser.add_argument("--output_dir", type=str, default="outputs/debug/sprint0_render")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for Sprint 0 gsplat rendering. "
            "Check NVIDIA driver, CUDA runtime, PyTorch CUDA build, and GPU availability."
        )

    manifest_path = Path(args.manifest)
    entries = load_manifest(manifest_path)
    entry = get_sample_by_index(entries, args.sample_index)
    validate_manifest_entry(entry)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    left_rgb = cv2.cvtColor(cv2.imread(entry["left_rgb_path"]), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    left_depth = load_servct_depth(Path(entry["left_depth_path"]))

    K_left = torch.tensor(entry["K_left"], dtype=torch.float32)
    c2w_left = torch.tensor(entry["c2w_left"], dtype=torch.float32)
    w2c_left = torch.tensor(entry["w2c_left"], dtype=torch.float32)
    height, width = entry["height"], entry["width"]

    camera = CameraData(
        K=K_left,
        c2w=c2w_left,
        w2c=w2c_left,
        height=height,
        width=width,
    )

    rgb_t = torch.from_numpy(left_rgb)
    depth_t = left_depth

    print(f"Initializing {args.num_points} Gaussians from RGB-D...")
    gaussians = initialize_gaussians_from_rgbd(
        rgb=rgb_t,
        depth_m=depth_t,
        K=K_left,
        c2w=c2w_left,
        num_points=args.num_points,
    )

    backend = GsplatBackend()
    gpu_name = torch.cuda.get_device_name(0)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    print("Rendering with gsplat...")
    output = backend.render(
        gaussians={k: v.cuda() for k, v in gaussians.items()},
        camera=camera,
        image_height=height,
        image_width=width,
        render_depth=True,
    )

    render_time = time.time() - t0
    vram_allocated = torch.cuda.max_memory_allocated() / 1024**3

    rgb_out = (output.rgb.cpu().numpy() * 255).astype(np.uint8)
    rgb_bgr = cv2.cvtColor(rgb_out, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out_dir / "render_rgb.png"), rgb_bgr)

    if output.depth is not None:
        depth_np = output.depth.cpu().numpy().squeeze()
        if depth_np[depth_np > 0].size > 0:
            d_min, d_max = depth_np[depth_np > 0].min(), depth_np.max()
            depth_viz = np.clip((depth_np - d_min) / (d_max - d_min + 1e-8), 0, 1)
        else:
            depth_viz = np.zeros_like(depth_np)
        depth_viz = (depth_viz * 255).astype(np.uint8)
        depth_viz = cv2.applyColorMap(depth_viz, cv2.COLORMAP_JET)
        cv2.imwrite(str(out_dir / "render_depth_color.png"), depth_viz)

    alpha_np = output.alpha.cpu().numpy()
    alpha_viz = (alpha_np * 255).astype(np.uint8)
    cv2.imwrite(str(out_dir / "render_alpha.png"), alpha_viz)

    n_gaussians = gaussians["means"].shape[0]
    render_report = {
        "backend": "gsplat",
        "num_gaussians": n_gaussians,
        "image_size": [height, width],
        "gpu_name": gpu_name,
        "vram_allocated_gb": round(vram_allocated, 3),
        "render_time_s": round(render_time, 4),
        "depth_semantics": output.aux["depth_semantics"],
        "metric_depth_available": output.aux["supports_metric_depth"],
        "metric_depth_verified": output.aux.get("metric_depth_verified", False),
    }

    with open(out_dir / "render_report.json", "w") as f:
        json.dump(render_report, f, indent=2)

    print(f"Render complete:")
    print(f"  Backend: {backend.name}")
    print(f"  Gaussians: {n_gaussians}")
    print(f"  Render time: {render_time:.4f}s")
    print(f"  VRAM allocated: {vram_allocated:.3f} GB")
    print(f"  Depth semantics: {output.aux['depth_semantics']}")
    print(f"Output saved to {out_dir}")


if __name__ == "__main__":
    main()
