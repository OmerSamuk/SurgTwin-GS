import copy
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch

from surgtwin.data.depth_io import load_servct_depth
from surgtwin.evaluation.geometry_metrics import geometry_metrics_report
from surgtwin.gaussian.gaussian_model import GaussianModel
from surgtwin.gaussian.renderer_interface import RendererBackend
from surgtwin.losses.depth import depth_l1
from surgtwin.losses.photometric import photometric_l1
from surgtwin.losses.regularizers import REGISTRY as REG_REGISTRY
from surgtwin.training.config import BaselineConfig
from surgtwin.training.depth_guided_config import DepthGuidedConfig
from surgtwin.training.trainer import BaselineTrainer, _load_rgb


def _verify_m2a_artifact(path: Optional[Path]) -> None:
    if path is None or not path.exists():
        raise RuntimeError(
            f"M2-A depth semantics artifact not found at '{path}'. "
            f"Run scripts/verify_render_depth_semantics.py first "
            f"and pass --depth_semantics_artifact."
        )
    data = json.loads(path.read_text())
    if not data.get("depth_semantics_verified", False):
        raise RuntimeError(
            f"M2-A gate not PASS: {path}. "
            f"depth_semantics_verified=false. "
            f"Resolve depth semantics verification before M2-B."
        )


def _save_depth_color(depth_tensor: torch.Tensor, path: Path, near_m: float = 0.02, far_m: float = 0.30) -> None:
    arr = depth_tensor.detach().cpu().numpy()
    valid = (arr > near_m) & (arr < far_m) & np.isfinite(arr)
    if valid.any():
        vmin, vmax = float(arr[valid].min()), float(arr[valid].max())
    else:
        vmin, vmax = near_m, far_m
    if vmax - vmin < 1e-6:
        vmax = vmin + 1e-6
    normalized = np.clip((arr - vmin) / (vmax - vmin), 0, 1)
    colored = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    cv2.imwrite(str(path), colored)


class DepthGuidedTrainer(BaselineTrainer):
    def __init__(
        self,
        train_entries: List[Dict],
        val_entries: List[Dict],
        backend: RendererBackend,
        config: DepthGuidedConfig,
        output_dir: Path,
    ):
        baseline_cfg = BaselineConfig(
            iterations=config.iterations,
            log_every=config.log_every,
            val_every=config.val_every,
            ckpt_every=config.ckpt_every,
            lr_means=config.lr_means,
            lr_scales=config.lr_scales,
            lr_quats=config.lr_quats,
            lr_opacities=config.lr_opacities,
            lr_colors=config.lr_colors,
            init_num_points=config.init_num_points,
            enable_densification=config.enable_densification,
            seed=config.seed,
            backend=config.backend,
        )
        super().__init__(train_entries, val_entries, backend, baseline_cfg, output_dir)
        self.dg_config = config
        self.init_scales: Optional[torch.Tensor] = None

    def _check_m2a_gate(self) -> None:
        if self.dg_config.depth_semantics_artifact_path:
            _verify_m2a_artifact(Path(self.dg_config.depth_semantics_artifact_path))

    def setup(self) -> None:
        self._check_m2a_gate()
        super().setup()
        self.init_scales = self.gaussians.scales.data.clone().detach()

    def train_step(self, iter_idx: int) -> Dict[str, float]:
        if self.gaussians is None or self.optimizer is None:
            raise RuntimeError("Trainer not set up. Call setup() first.")

        entry = self._sample_train_entry()
        gt_rgb = _load_rgb(entry["left_rgb_path"]).to(self.device)
        gt_depth = load_servct_depth(Path(entry["left_depth_path"])).to(self.device)
        camera = self._entry_to_camera(entry)

        self.optimizer.zero_grad()

        out = self.backend.render(
            gaussians=self.gaussians,
            camera=camera,
            image_height=entry["height"],
            image_width=entry["width"],
            render_depth=True,
        )

        if out.aux.get("depth_semantics") != "metric_meters":
            raise RuntimeError(
                f"Depth-guided training requires metric depth, "
                f"but got depth_semantics='{out.aux.get('depth_semantics')}'"
            )

        loss_photo = photometric_l1(out.rgb[..., :3], gt_rgb)
        loss_depth, depth_diag = depth_l1(
            pred_depth=out.depth,
            gt_depth=gt_depth,
            depth_semantics=out.aux["depth_semantics"],
            near_m=self.dg_config.depth_near_m,
            far_m=self.dg_config.depth_far_m,
        )

        reg_fn = REG_REGISTRY.get(self.dg_config.reg_type)
        if reg_fn is not None and self.init_scales is not None and self.dg_config.lambda_reg > 0:
            loss_reg = reg_fn(self.gaussians.scales, self.init_scales)
        else:
            loss_reg = torch.tensor(0.0, device=self.device)

        loss_total = loss_photo + self.dg_config.lambda_depth * loss_depth + self.dg_config.lambda_reg * loss_reg
        loss_total.backward()

        grad_norm_before = 0.0
        grad_norm_after = 0.0
        if self.dg_config.clip_grad_norm:
            params = [p for p in self.gaussians.state_dict().values() if p.grad is not None]
            if params:
                grad_norm_before = torch.nn.utils.clip_grad_norm_(
                    params, max_norm=self.dg_config.max_grad_norm
                ).item()
            else:
                grad_norm_before = 0.0
            grad_norm_after = sum(p.grad.norm().item() ** 2 for p in params if p.grad is not None) ** 0.5
            if not np.isfinite(grad_norm_after):
                grad_norm_after = grad_norm_before

        self.optimizer.step()

        with torch.no_grad():
            out.rgb[..., :3].clamp_(0.0, 1.0)
            self.gaussians.scales.data.clamp_(min=1e-5)
            self.gaussians.opacities.data.clamp_(min=-10.0, max=10.0)
            psnr_val = self._compute_psnr(out.rgb[..., :3], gt_rgb)

        return {
            "loss_total": loss_total.item(),
            "photo_loss": loss_photo.item(),
            "depth_loss_raw_m": depth_diag["depth_loss_raw_m"].item(),
            "depth_loss_weighted": (self.dg_config.lambda_depth * loss_depth.detach()).item(),
            "reg_loss_raw": loss_reg.item(),
            "reg_loss_weighted": (self.dg_config.lambda_reg * loss_reg.detach()).item(),
            "psnr": psnr_val,
            "depth_rmse_m_raw": depth_diag["depth_rmse_m_raw"].item(),
            "depth_mae_m_raw": depth_diag["depth_mae_m_raw"].item(),
            "abs_rel": depth_diag["abs_rel"].item(),
            "depth_valid_ratio": depth_diag["depth_valid_ratio"].item(),
            "grad_norm_before_clip": grad_norm_before,
            "grad_norm_after_clip": grad_norm_after,
            "n_gaussians": self.gaussians.num_gaussians(),
        }

    def _compute_psnr(self, pred: torch.Tensor, gt: torch.Tensor) -> float:
        from surgtwin.evaluation.image_metrics import psnr
        return psnr(pred, gt)

    def _run_val(self, iter_idx: int) -> Dict[str, float]:
        self.gaussians.means.requires_grad_(False)
        self.gaussians.scales.requires_grad_(False)
        self.gaussians.quats.requires_grad_(False)
        self.gaussians.opacities.requires_grad_(False)
        self.gaussians.colors.requires_grad_(False)

        psnr_list, ssim_list = [], []
        lpips_list = []
        all_depth_reports = []

        snapshot_dir = self.output_dir / "renders"
        depth_dir = self.output_dir / "depth"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        depth_dir.mkdir(parents=True, exist_ok=True)

        for entry in self.val_entries:
            gt_rgb = _load_rgb(entry["left_rgb_path"]).to(self.device)
            gt_depth = load_servct_depth(Path(entry["left_depth_path"])).to(self.device)
            camera = self._entry_to_camera(entry)

            out = self.backend.render(
                gaussians=self.gaussians,
                camera=camera,
                image_height=entry["height"],
                image_width=entry["width"],
                render_depth=True,
            )

            pred_rgb = out.rgb[..., :3]
            psnr_list.append(self._compute_psnr(pred_rgb, gt_rgb))
            ssim_list.append(self._compute_ssim(pred_rgb, gt_rgb))

            val_score = self._lpips_score(pred_rgb, gt_rgb, self.device)
            if val_score is not None:
                lpips_list.append(val_score)

            sid = entry["sample_id"]
            self._save_snapshot(pred_rgb, snapshot_dir / f"iter_{iter_idx:06d}_val_{sid}_rgb.png")

            if out.depth is not None and out.aux.get("depth_semantics") == "metric_meters":
                self._save_snapshot_depth(out.depth, gt_depth, depth_dir, iter_idx, sid)
                depth_report = geometry_metrics_report(
                    pred_depth=out.depth,
                    gt_depth=gt_depth,
                    depth_semantics=out.aux["depth_semantics"],
                    near_m=self.dg_config.depth_near_m,
                    far_m=self.dg_config.depth_far_m,
                )
                all_depth_reports.append(depth_report)

        self._restore_grad()

        result: Dict[str, float] = {
            "val_psnr": float(np.mean(psnr_list)) if psnr_list else 0.0,
            "val_ssim": float(np.mean(ssim_list)) if ssim_list else 0.0,
        }

        if lpips_list:
            result["val_lpips"] = float(np.mean(lpips_list))
            result["val_lpips_unavailable_reason"] = None
        else:
            result["val_lpips"] = None
            result["val_lpips_unavailable_reason"] = "LPIPS not computed"

        if all_depth_reports:
            for key in ("depth_rmse_m_raw", "depth_mae_m_raw", "abs_rel", "depth_valid_ratio",
                        "depth_rmse_m_clipped", "depth_mae_m_clipped", "median_aligned_rmse_m"):
                vals = [r[key] for r in all_depth_reports if key in r]
                if vals:
                    result[f"val_{key}"] = float(np.mean(vals))
            result["val_depth_semantics"] = "metric_meters"
            result["val_num_depth_samples"] = len(all_depth_reports)

        return result

    def _save_snapshot_depth(self, pred_depth: torch.Tensor, gt_depth: torch.Tensor,
                              depth_dir: Path, iter_idx: int, sample_id: str) -> None:
        np.save(str(depth_dir / f"iter_{iter_idx:06d}_val_{sample_id}_depth.npy"),
                pred_depth.detach().cpu().numpy())
        _save_depth_color(pred_depth, depth_dir / f"iter_{iter_idx:06d}_val_{sample_id}_depth_color.png")
        _save_depth_color(gt_depth, depth_dir / f"iter_{iter_idx:06d}_val_{sample_id}_gt_depth_color.png")
        error_map = (pred_depth - gt_depth).abs()
        _save_depth_color(error_map, depth_dir / f"iter_{iter_idx:06d}_val_{sample_id}_depth_error_color.png")

    def _compute_ssim(self, pred: torch.Tensor, gt: torch.Tensor) -> float:
        from surgtwin.evaluation.image_metrics import ssim
        return ssim(pred, gt)

    def _lpips_score(self, pred: torch.Tensor, gt: torch.Tensor, device: torch.device):
        from surgtwin.evaluation.image_metrics import lpips_score
        return lpips_score(pred, gt, device)

    def _restore_grad(self) -> None:
        self.gaussians.means.requires_grad_(True)
        self.gaussians.scales.requires_grad_(True)
        self.gaussians.quats.requires_grad_(True)
        self.gaussians.opacities.requires_grad_(True)
        self.gaussians.colors.requires_grad_(True)

    def fit(self) -> Dict:
        self.setup()
        config = self.dg_config
        final_metrics = {}

        for i in range(1, config.iterations + 1):
            t0 = time.time()
            step = self.train_step(i)
            step_time = time.time() - t0

            vram = 0.0
            if torch.cuda.is_available():
                vram = torch.cuda.max_memory_allocated() / 1024 ** 3
                torch.cuda.reset_peak_memory_stats()

            metrics = {
                "iter": i,
                "loss_total": round(step["loss_total"], 6),
                "photo_loss": round(step["photo_loss"], 6),
                "depth_loss_raw_m": round(step["depth_loss_raw_m"], 8),
                "depth_loss_weighted": round(step["depth_loss_weighted"], 8),
                "reg_loss_raw": round(step["reg_loss_raw"], 8),
                "reg_loss_weighted": round(step["reg_loss_weighted"], 8),
                "psnr": round(step["psnr"], 4),
                "depth_rmse_m_raw": round(step["depth_rmse_m_raw"], 6),
                "depth_mae_m_raw": round(step["depth_mae_m_raw"], 6),
                "abs_rel": round(step["abs_rel"], 6),
                "depth_valid_ratio": round(step["depth_valid_ratio"], 4),
                "n_gaussians": step["n_gaussians"],
                "iter_time_s": round(step_time, 4),
                "vram_gb": round(vram, 3),
            }

            if config.log_grad_norm:
                metrics["grad_norm_before_clip"] = round(step["grad_norm_before_clip"], 6)
                metrics["grad_norm_after_clip"] = round(step["grad_norm_after_clip"], 6)

            self.metrics_logger.log(metrics)

            if i == 1:
                final_metrics["initial_loss_total"] = step["loss_total"]

            if i % config.log_every == 0:
                msg = (f"iter {i:5d}/{config.iterations}  "
                       f"loss={metrics['loss_total']:.6f}  "
                       f"photo={metrics['photo_loss']:.6f}  "
                       f"depth={metrics['depth_loss_raw_m']:.6f}  "
                       f"reg={metrics['reg_loss_raw']:.6f}  "
                       f"psnr={metrics['psnr']:.2f}  "
                       f"gaussians={metrics['n_gaussians']}  "
                       f"time={metrics['iter_time_s']:.3f}s")
                if self.dg_config.log_grad_norm:
                    msg += f"  grad_before={metrics.get('grad_norm_before_clip', 0):.4f}"
                print(msg)

            if i % config.val_every == 0:
                val_metrics = self._run_val(i)
                val_metrics["iter"] = i
                self.metrics_logger.log(val_metrics)
                lpips_str = f"{val_metrics.get('val_lpips', 'N/A')}" if val_metrics.get('val_lpips') is not None else "N/A"
                depth_str = ""
                if "val_depth_rmse_m_raw" in val_metrics:
                    depth_str = (f"  depth_rmse={val_metrics['val_depth_rmse_m_raw']:.4f}  "
                                 f"depth_mae={val_metrics['val_depth_mae_m_raw']:.4f}")
                print(f"VAL iter {i}: psnr={val_metrics['val_psnr']:.2f}  "
                      f"ssim={val_metrics['val_ssim']:.4f}  "
                      f"lpips={lpips_str}{depth_str}")

            if i % config.ckpt_every == 0:
                self.save(i)

        final_metrics["final_loss_total"] = step["loss_total"]
        final_metrics["final_photo_loss"] = step["photo_loss"]
        final_metrics["n_gaussians"] = step["n_gaussians"]
        val_metrics = self._run_val(config.iterations)
        final_metrics.update(val_metrics)
        final_metrics["loss_decreased"] = final_metrics["final_loss_total"] < final_metrics["initial_loss_total"]
        final_metrics["iterations"] = config.iterations
        final_metrics["split_strategy"] = self.split_strategy
        final_metrics["lambda_depth"] = config.lambda_depth
        final_metrics["lambda_reg"] = config.lambda_reg
        final_metrics["reg_type"] = config.reg_type
        final_metrics["clip_grad_norm"] = config.clip_grad_norm
        final_metrics["max_grad_norm"] = config.max_grad_norm
        final_metrics["enable_densification"] = config.enable_densification
        final_metrics["depth_semantics"] = "metric_meters"

        write_json = __import__("surgtwin.training.logging_utils", fromlist=["write_json"]).write_json
        write_json(self.output_dir / "final_metrics.json", final_metrics)
        self.save(config.iterations)
        self._write_report(final_metrics)

        return final_metrics

    def save(self, iter_idx: int) -> None:
        ckpt_dir = self.output_dir / "checkpoints"
        ckpt_path = ckpt_dir / f"ckpt_{iter_idx:06d}.pt"
        from surgtwin.training.checkpointing import save_checkpoint
        save_checkpoint(
            path=ckpt_path,
            gaussians=self.gaussians,
            optimizer_state_dict=self.optimizer.state_dict(),
            iteration=iter_idx,
            config=asdict(self.dg_config),
            backend_name=self.dg_config.backend,
            seed=self.dg_config.seed,
        )

    def _write_report(self, fm: Dict) -> None:
        lines = [
            "# Depth-Guided Run Report (M2-B)",
            "",
            "## 1. Dataset Split (deterministic, seed={})".format(self.dg_config.seed),
            "- train: Experiment_1 frame 1-6 (n={})".format(len(self.train_entries)),
            "- val: Experiment_1 frame 7-8 (n={})".format(len(self.val_entries)),
            "- split_strategy: " + self.split_strategy,
            "",
            "## 2. Environment",
            "See `environment.json` for full details.",
            "",
            "## 3. Run Summary",
            "- iterations: {}".format(self.dg_config.iterations),
            "- initial_loss_total (iter 1): {:.6f}".format(fm.get("initial_loss_total", 0)),
            "- final_loss_total (iter {}): {:.6f}".format(self.dg_config.iterations, fm.get("final_loss_total", 0)),
            "- loss_decreased: {}".format(fm.get("loss_decreased", False)),
            "- n_gaussians (final): {}".format(fm.get("n_gaussians", 0)),
            "- val_psnr: {:.4f}".format(fm.get("val_psnr", 0)),
            "- val_ssim: {:.4f}".format(fm.get("val_ssim", 0)),
            "- val_lpips: {}".format(fm.get("val_lpips", "N/A")),
            "",
            "## 4. Depth Configuration",
            "- lambda_depth: {}".format(self.dg_config.lambda_depth),
            "- lambda_reg: {}".format(self.dg_config.lambda_reg),
            "- reg_type: {}".format(self.dg_config.reg_type),
            "- depth_near_m: {}".format(self.dg_config.depth_near_m),
            "- depth_far_m: {}".format(self.dg_config.depth_far_m),
            "- clip_grad_norm: {}".format(self.dg_config.clip_grad_norm),
            "- max_grad_norm: {}".format(self.dg_config.max_grad_norm),
            "- depth_semantics: metric_meters",
            "- M2-A artifact: {}".format(self.dg_config.depth_semantics_artifact_path),
            "",
            "## 4.5 Depth Loss Diagnostics",
        ]
        for key in ("val_depth_rmse_m_raw", "val_depth_rmse_m_clipped",
                     "val_depth_mae_m_raw", "val_depth_mae_m_clipped",
                     "val_abs_rel", "val_depth_valid_ratio",
                     "val_median_aligned_rmse_m"):
            if key in fm:
                lines.append("- {}: {}".format(key, fm[key]))
        lines.extend([
            "- num_depth_val_samples: {}".format(fm.get("val_num_depth_samples", 0)),
            "",
            "## 5. Outputs",
            "- config.json: yes",
            "- environment.json: yes",
            "- metrics.jsonl: yes",
            "- checkpoints/: yes",
            "- renders/: yes",
            "- depth/: yes",
            "- final_metrics.json: yes",
            "",
            "## 6. Known Limitations",
            "- No uncertainty weighting (M3)",
            "- No density control (M4)",
            f"- Fixed {fm.get('n_gaussians', 20000)} Gaussians, densification off",
            "",
            "## 7. Next Steps",
            "- Milestone 3: uncertainty-weighted loss",
            "- Milestone 4: multi-criteria density control",
        ])
        report_path = self.output_dir / "report.md"
        report_path.write_text("\n".join(lines) + "\n")

    def save_side_by_side_panels(self) -> None:
        panel_dir = self.output_dir / "panels"
        panel_dir.mkdir(parents=True, exist_ok=True)

        for entry in self.val_entries:
            gt_rgb = _load_rgb(entry["left_rgb_path"]).to(self.device)
            gt_depth = load_servct_depth(Path(entry["left_depth_path"])).to(self.device)
            camera = self._entry_to_camera(entry)

            out = self.backend.render(
                gaussians=self.gaussians,
                camera=camera,
                image_height=entry["height"],
                image_width=entry["width"],
                render_depth=True,
            )

            pred_rgb_np = (out.rgb[..., :3].detach().cpu().numpy() * 255).astype(np.uint8)
            gt_rgb_np = (gt_rgb.cpu().numpy() * 255).astype(np.uint8)
            pred_d_np = out.depth.detach().cpu().numpy() if out.depth is not None else np.zeros((entry["height"], entry["width"]), dtype=np.float32)
            gt_d_np = gt_depth.cpu().numpy()

            def _norm_depth(d):
                valid = (d > 0.02) & (d < 0.30) & np.isfinite(d)
                if not valid.any():
                    return np.zeros_like(d, dtype=np.uint8)
                vmin, vmax = d[valid].min(), d[valid].max()
                if vmax - vmin < 1e-6:
                    vmax = vmin + 1e-6
                norm = np.clip((d - vmin) / (vmax - vmin), 0, 1)
                return (norm * 255).astype(np.uint8)

            error_map = np.abs(pred_d_np - gt_d_np)
            panels = [
                cv2.cvtColor(gt_rgb_np, cv2.COLOR_RGB2BGR),
                cv2.cvtColor(pred_rgb_np, cv2.COLOR_RGB2BGR),
                cv2.applyColorMap(_norm_depth(gt_d_np), cv2.COLORMAP_INFERNO),
                cv2.applyColorMap(_norm_depth(pred_d_np), cv2.COLORMAP_INFERNO),
                cv2.applyColorMap(_norm_depth(error_map), cv2.COLORMAP_VIRIDIS),
            ]
            h, w = gt_rgb_np.shape[:2]
            top = np.hstack([panels[0], panels[1]])
            mid = np.hstack([panels[2], panels[3]])
            bottom = np.hstack([panels[4], np.zeros((h, w, 3), dtype=np.uint8)])
            full = np.vstack([top, mid, bottom])
            panel_path = panel_dir / f"val_{entry['sample_id']}_panel.png"
            cv2.imwrite(str(panel_path), full)
