import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.optim import Adam

from surgtwin.cameras.camera_types import CameraData
from surgtwin.data.manifest import filter_by_split
from surgtwin.evaluation.image_metrics import lpips_score, psnr, ssim
from surgtwin.gaussian.backend_gsplat import GsplatBackend
from surgtwin.gaussian.gaussian_model import GaussianModel
from surgtwin.gaussian.initialization import initialize_gaussians_from_rgbd
from surgtwin.gaussian.renderer_interface import RendererBackend
from surgtwin.training.checkpointing import save_checkpoint
from surgtwin.training.config import BaselineConfig
from surgtwin.training.logging_utils import JsonlLogger, collect_environment, write_json
from surgtwin.training.seed import set_seed

try:
    import cv2
    import numpy as np
except ImportError:
    pass


def _load_rgb(path: str) -> torch.Tensor:
    import cv2
    import numpy as np

    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(img)


def _load_depth(path: str) -> torch.Tensor:
    from surgtwin.data.depth_io import load_servct_depth
    from pathlib import Path

    return load_servct_depth(Path(path))


class BaselineTrainer:
    def __init__(
        self,
        train_entries: List[Dict],
        val_entries: List[Dict],
        backend: RendererBackend,
        config: BaselineConfig,
        output_dir: Path,
    ):
        self.train_entries = train_entries
        self.val_entries = val_entries
        self.backend = backend
        self.config = config
        self.output_dir = output_dir
        set_seed(config.seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is required for baseline training. "
                "Check NVIDIA driver, CUDA runtime, and PyTorch CUDA build."
            )

        self.gaussians: Optional[GaussianModel] = None
        self.optimizer: Optional[Adam] = None
        self.metrics_logger = JsonlLogger(output_dir / "metrics.jsonl")
        self.split_strategy = "Experiment_1: 6 train / 2 val; Experiment_2 test (held-out)"

    def _init_from_first_sample(self) -> None:
        entry = self.train_entries[0]
        rgb = _load_rgb(entry["left_rgb_path"]).to(self.device)
        depth = _load_depth(entry["left_depth_path"]).to(self.device)
        K = torch.tensor(entry["K_left"], dtype=torch.float32, device=self.device)
        c2w = torch.tensor(entry["c2w_left"], dtype=torch.float32, device=self.device)

        self.gaussians = initialize_gaussians_from_rgbd(
            rgb=rgb,
            depth_m=depth,
            K=K,
            c2w=c2w,
            num_points=self.config.init_num_points,
        )
        self.gaussians = self.gaussians.to(self.device)
        for p in self.gaussians.state_dict().values():
            p.requires_grad_(True)

    def _build_optimizer(self) -> None:
        if self.gaussians is None:
            raise RuntimeError("Gaussians not initialized before building optimizer.")
        gd = {k: v for k, v in self.gaussians.__dict__.items() if isinstance(v, torch.Tensor)}
        self.optimizer = Adam([
            {"params": [gd["means"]], "lr": self.config.lr_means},
            {"params": [gd["scales"]], "lr": self.config.lr_scales},
            {"params": [gd["quats"]], "lr": self.config.lr_quats},
            {"params": [gd["opacities"]], "lr": self.config.lr_opacities},
            {"params": [gd["colors"]], "lr": self.config.lr_colors},
        ])

    def _sample_train_entry(self) -> Dict:
        idx = torch.randint(len(self.train_entries), (1,)).item()
        return self.train_entries[idx]

    def _entry_to_camera(self, entry: Dict) -> CameraData:
        return CameraData(
            K=torch.tensor(entry["K_left"], dtype=torch.float32, device=self.device),
            c2w=torch.tensor(entry["c2w_left"], dtype=torch.float32, device=self.device),
            w2c=torch.tensor(entry["w2c_left"], dtype=torch.float32, device=self.device),
            height=entry["height"],
            width=entry["width"],
        )

    def _run_val(self, iter_idx: int) -> Dict[str, float]:
        self.gaussians.means.requires_grad_(False)
        self.gaussians.scales.requires_grad_(False)
        self.gaussians.quats.requires_grad_(False)
        self.gaussians.opacities.requires_grad_(False)
        self.gaussians.colors.requires_grad_(False)

        psnr_list, ssim_list = [], []
        lpips_list = []
        lpips_unavailable = None

        for entry in self.val_entries:
            gt_rgb = _load_rgb(entry["left_rgb_path"]).to(self.device)
            camera = self._entry_to_camera(entry)
            out = self.backend.render(
                gaussians=self.gaussians,
                camera=camera,
                image_height=entry["height"],
                image_width=entry["width"],
                render_depth=False,
            )
            pred_rgb = out.rgb[..., :3]
            psnr_list.append(psnr(pred_rgb, gt_rgb))
            ssim_list.append(ssim(pred_rgb, gt_rgb))
            val_score = lpips_score(pred_rgb, gt_rgb, self.device)
            if val_score is not None:
                lpips_list.append(val_score)
            elif lpips_unavailable is None:
                lpips_unavailable = "LPIPS call returned None"

            snapshot_dir = self.output_dir / "renders"
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            self._save_snapshot(pred_rgb, snapshot_dir / f"iter_{iter_idx:06d}_val_{entry['sample_id']}_rgb.png")

        self.gaussians.means.requires_grad_(True)
        self.gaussians.scales.requires_grad_(True)
        self.gaussians.quats.requires_grad_(True)
        self.gaussians.opacities.requires_grad_(True)
        self.gaussians.colors.requires_grad_(True)

        result = {
            "val_psnr": float(torch.tensor(psnr_list).mean()) if psnr_list else 0.0,
            "val_ssim": float(torch.tensor(ssim_list).mean()) if ssim_list else 0.0,
        }
        if lpips_list:
            result["val_lpips"] = float(torch.tensor(lpips_list).mean())
            result["val_lpips_unavailable_reason"] = None
        else:
            result["val_lpips"] = None
            result["val_lpips_unavailable_reason"] = lpips_unavailable or "LPIPS not computed"
        return result

    def _save_snapshot(self, rgb_tensor: torch.Tensor, path: Path) -> None:
        import cv2
        import numpy as np

        arr = (rgb_tensor.detach().cpu().numpy() * 255).astype(np.uint8)
        if arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[..., :3]
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(path), bgr)

    def setup(self) -> None:
        write_json(self.output_dir / "config.json", asdict(self.config))
        write_json(self.output_dir / "environment.json", collect_environment(self.config.backend))
        self._init_from_first_sample()
        self._build_optimizer()

    def train_step(self, iter_idx: int) -> Dict[str, float]:
        if self.gaussians is None or self.optimizer is None:
            raise RuntimeError("Trainer not set up. Call setup() first.")

        entry = self._sample_train_entry()
        gt_rgb = _load_rgb(entry["left_rgb_path"]).to(self.device)
        camera = self._entry_to_camera(entry)

        self.optimizer.zero_grad()
        out = self.backend.render(
            gaussians=self.gaussians,
            camera=camera,
            image_height=entry["height"],
            image_width=entry["width"],
            render_depth=False,
        )
        l1 = torch.abs(out.rgb[..., :3] - gt_rgb).mean()
        l1.backward()
        self.optimizer.step()

        with torch.no_grad():
            out.rgb[..., :3].clamp_(0.0, 1.0)
            self.gaussians.scales.data.clamp_(min=1e-5)
            self.gaussians.opacities.data.clamp_(min=-10.0, max=10.0)
            p = psnr(out.rgb[..., :3], gt_rgb)

        return {
            "loss": l1.item(),
            "psnr": p,
            "n_gaussians": self.gaussians.num_gaussians(),
        }

    def save(self, iter_idx: int) -> None:
        ckpt_dir = self.output_dir / "checkpoints"
        ckpt_path = ckpt_dir / f"ckpt_{iter_idx:06d}.pt"
        save_checkpoint(
            path=ckpt_path,
            gaussians=self.gaussians,
            optimizer_state_dict=self.optimizer.state_dict(),
            iteration=iter_idx,
            config=asdict(self.config),
            backend_name=self.config.backend,
            seed=self.config.seed,
        )

    def fit(self) -> Dict:
        self.setup()
        config = self.config
        final_metrics = {}

        for i in range(1, config.iterations + 1):
            t0 = time.time()
            step = self.train_step(i)
            step_time = time.time() - t0

            vram = 0.0
            if torch.cuda.is_available():
                vram = torch.cuda.max_memory_allocated() / 1024**3
                torch.cuda.reset_peak_memory_stats()

            metrics = {
                "iter": i,
                "loss": round(step["loss"], 6),
                "psnr": round(step["psnr"], 4),
                "n_gaussians": step["n_gaussians"],
                "iter_time_s": round(step_time, 4),
                "vram_gb": round(vram, 3),
            }
            self.metrics_logger.log(metrics)

            if i == 1:
                final_metrics["initial_loss"] = step["loss"]

            if i % config.log_every == 0:
                print(f"iter {i:5d}/{config.iterations}  loss={metrics['loss']:.6f}  psnr={metrics['psnr']:.2f}  "
                      f"gaussians={metrics['n_gaussians']}  time={metrics['iter_time_s']:.3f}s  "
                      f"vram={metrics['vram_gb']:.3f}GB")

            if i % config.val_every == 0:
                val_metrics = self._run_val(i)
                val_metrics["iter"] = i
                self.metrics_logger.log(val_metrics)
                print(f"VAL iter {i}: psnr={val_metrics['val_psnr']:.2f}  ssim={val_metrics['val_ssim']:.4f}  "
                      f"lpips={val_metrics.get('val_lpips', 'N/A')}")

            if i % config.ckpt_every == 0:
                self.save(i)

        final_metrics["final_loss"] = step["loss"]
        final_metrics["n_gaussians"] = step["n_gaussians"]
        val_metrics = self._run_val(config.iterations)
        final_metrics.update(val_metrics)
        final_metrics["loss_decreased"] = final_metrics["final_loss"] < final_metrics["initial_loss"]
        final_metrics["iterations"] = config.iterations
        final_metrics["split_strategy"] = self.split_strategy
        final_metrics["render_depth_semantics"] = "not_used_for_baseline"
        final_metrics["depth_semantics_verified"] = False
        final_metrics["enable_densification"] = config.enable_densification

        write_json(self.output_dir / "final_metrics.json", final_metrics)
        self.save(config.iterations)
        self._write_report(final_metrics)

        return final_metrics

    def _write_report(self, fm: Dict) -> None:
        lines = [
            "# Baseline Run Report",
            "",
            "## 1. Dataset Split (deterministic, seed={})".format(self.config.seed),
            "- train: Experiment_1 frame 1-6 (n={})".format(len(self.train_entries)),
            "- val: Experiment_1 frame 7-8 (n={})".format(len(self.val_entries)),
            "- split_strategy: " + self.split_strategy,
            "",
            "## 2. Environment",
            "See `environment.json` for full details.",
            "",
            "## 3. Run Summary",
            "- iterations: {}".format(self.config.iterations),
            "- initial_loss (iter 1): {:.6f}".format(fm.get("initial_loss", 0)),
            "- final_loss (iter {}): {:.6f}".format(self.config.iterations, fm.get("final_loss", 0)),
            "- loss_decreased: {}".format(fm.get("loss_decreased", False)),
            "- n_gaussians (final): {}".format(fm.get("n_gaussians", 0)),
            "- val_psnr: {:.4f}".format(fm.get("val_psnr", 0)),
            "- val_ssim: {:.4f}".format(fm.get("val_ssim", 0)),
            "- val_lpips: {}".format(fm.get("val_lpips", "N/A")),
            "",
            "## 4. Render Depth Semantics",
            "- status: NOT_USED",
            "- reason: Baseline is RGB-only L1; render depth semantics re-verified at Milestone 2.",
            "",
            "## 5. Outputs",
            "- config.json: yes",
            "- environment.json: yes",
            "- metrics.jsonl: yes",
            "- checkpoints/: yes",
            "- renders/: yes",
            "- final_metrics.json: yes",
            "",
            "## 6. Known Limitations",
            "- Baseline no-densification: PSNR plateau expected",
            "- LPIPS may be unavailable if weights not cached",
            "- Render depth not used; Milestone 2 prerequisite",
            "",
            "## 7. Next Steps",
            "- Milestone 2: depth-guided GS with lambda_depth=0.2",
            "- Render depth semantics verification required before Milestone 2",
        ]
        report_path = self.output_dir / "report.md"
        report_path.write_text("\n".join(lines) + "\n")
