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
from surgtwin.losses.uncertainty_weighted import uncertainty_weighted_photometric_l1
from surgtwin.masks.io import load_specular_mask, mask_coverage
from surgtwin.training.config import BaselineConfig
from surgtwin.training.densification import select_densification_candidates, DensificationSelection
from surgtwin.training.depth_guided_config import DepthGuidedConfig
from surgtwin.training.depth_guided_trainer import DepthGuidedTrainer, _verify_m2a_artifact, _save_depth_color
from surgtwin.training.uncertainty_config import UncertaintyConfig
from surgtwin.training.checkpointing import save_checkpoint
from surgtwin.training.logging_utils import write_json
from surgtwin.uncertainty.signals import compute_photo_residual, compute_p95_scale, compute_u_photo, compute_w_photo, compute_w_photo_with_mask


def _resolve_mask_path(entry: Dict, mask_dir: Optional[str]) -> Optional[Path]:
    raw = entry.get("left_specular_mask_path")
    if raw:
        p = Path(raw)
        if p.exists():
            return p
    if mask_dir:
        sid = entry.get("sample_id", "unknown")
        candidate = Path(mask_dir) / f"{sid}_specular.npy"
        if candidate.exists():
            return candidate
    return None


def _save_u_photo_color(u_tensor: torch.Tensor, path: Path) -> None:
    arr = u_tensor.detach().cpu().numpy()
    normalized = np.clip(arr, 0, 1)
    colored = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    cv2.imwrite(str(path), colored)


def _save_w_photo_color(w_tensor: torch.Tensor, path: Path, w_min: float = 0.15) -> None:
    arr = w_tensor.detach().cpu().numpy()
    normalized = np.clip((arr - w_min) / (1.0 - w_min), 0, 1)
    colored = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    cv2.imwrite(str(path), colored)


def _load_rgb(path: str) -> torch.Tensor:
    import cv2, numpy as np
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(img)


class UncertaintyTrainer(DepthGuidedTrainer):
    def __init__(
        self,
        train_entries: List[Dict],
        val_entries: List[Dict],
        backend: RendererBackend,
        config: UncertaintyConfig,
        output_dir: Path,
    ):
        dg_cfg = DepthGuidedConfig(
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
            lambda_depth=config.lambda_depth,
            lambda_reg=config.lambda_reg,
            depth_near_m=config.depth_near_m,
            depth_far_m=config.depth_far_m,
            clip_grad_norm=config.clip_grad_norm,
            max_grad_norm=config.max_grad_norm,
            log_grad_norm=config.log_grad_norm,
            depth_semantics_artifact_path=config.depth_semantics_artifact_path,
        )
        super().__init__(train_entries, val_entries, backend, dg_cfg, output_dir)
        self.unc_config = config
        self.mask_cache: Dict[str, Optional[torch.Tensor]] = {}
        self._warmup_base_lrs: Optional[List[float]] = None
        self._best_val: Optional[Dict] = None

    def _init_warmup(self) -> None:
        if self.unc_config.warmup_iters > 0 and self.optimizer is not None:
            self._warmup_base_lrs = [pg["lr"] for pg in self.optimizer.param_groups]
            factor = 1.0 / self.unc_config.warmup_iters
            for pg in self.optimizer.param_groups:
                pg["lr"] = pg["lr"] * factor

    def _apply_warmup(self, iter_idx: int) -> None:
        if self._warmup_base_lrs is not None and self.optimizer is not None:
            factor = min(1.0, iter_idx / self.unc_config.warmup_iters)
            for pg, base_lr in zip(self.optimizer.param_groups, self._warmup_base_lrs):
                pg["lr"] = base_lr * factor

    def _get_mask_for_entry(self, entry: Dict) -> Optional[torch.Tensor]:
        if entry.get("sample_id") in self.mask_cache:
            return self.mask_cache[entry["sample_id"]]
        path = _resolve_mask_path(entry, self.unc_config.mask_dir)
        if path is not None:
            mask = load_specular_mask(path, device=self.device)
            desired_h, desired_w = entry["height"], entry["width"]
            if mask.shape != (desired_h, desired_w):
                mask = mask[:desired_h, :desired_w]
        else:
            mask = None
        self.mask_cache[entry["sample_id"]] = mask
        return mask

    def setup(self) -> None:
        self._check_m2a_gate()
        super(DepthGuidedTrainer, self).setup()
        self.init_scales = self.gaussians.scales.data.clone().detach()
        self._init_warmup()

    def train_step(self, iter_idx: int, return_context: bool = False):
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
                f"Uncertainty training requires metric depth, "
                f"but got depth_semantics='{out.aux.get('depth_semantics')}'"
            )

        mask = self._get_mask_for_entry(entry) if self.unc_config.variant in ("h2", "h3") else None

        loss_uw, uw_diag = uncertainty_weighted_photometric_l1(
            rgb_pred=out.rgb[..., :3],
            rgb_gt=gt_rgb,
            alpha=self.unc_config.alpha,
            w_min=self.unc_config.w_photo_min,
            mask=mask,
            mask_boost=self.unc_config.mask_boost,
        )

        loss_depth, depth_diag = depth_l1(
            pred_depth=out.depth,
            gt_depth=gt_depth,
            depth_semantics=out.aux["depth_semantics"],
            near_m=self.unc_config.depth_near_m,
            far_m=self.unc_config.depth_far_m,
        )

        loss_total = loss_uw + self.unc_config.lambda_depth * loss_depth
        loss_total.backward()

        grad_norm_before = 0.0
        grad_norm_after = 0.0
        if self.unc_config.clip_grad_norm:
            params = [p for p in self.gaussians.state_dict().values() if p.grad is not None]
            if params:
                grad_norm_before = torch.nn.utils.clip_grad_norm_(
                    params, max_norm=self.unc_config.max_grad_norm
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
            diff = (out.rgb[..., :3] - gt_rgb).abs().mean()
            coeff = loss_uw.detach() / (diff + 1e-8)
            psnr_val = self._compute_psnr(out.rgb[..., :3], gt_rgb)

        result = {
            "loss_total": loss_total.item(),
            "photo_loss_weighted": loss_uw.item(),
            "depth_loss_raw_m": depth_diag["depth_loss_raw_m"].item(),
            "depth_loss_weighted": (self.unc_config.lambda_depth * loss_depth.detach()).item(),
            "psnr": psnr_val,
            "depth_rmse_m_raw": depth_diag["depth_rmse_m_raw"].item(),
            "depth_mae_m_raw": depth_diag["depth_mae_m_raw"].item(),
            "abs_rel": depth_diag["abs_rel"].item(),
            "depth_valid_ratio": depth_diag["depth_valid_ratio"].item(),
            "grad_norm_before_clip": grad_norm_before,
            "grad_norm_after_clip": grad_norm_after,
            "n_gaussians": self.gaussians.num_gaussians(),
        }

        for key in ("w_photo_mean", "w_photo_min", "w_photo_max",
                     "w_photo_p10", "w_photo_p50", "w_photo_p90",
                     "fraction_w_photo_at_min", "fraction_w_photo_at_one",
                     "w_photo_p90_minus_p10", "u_photo_mean", "p95_scale"):
            val = uw_diag.get(key)
            if val is not None:
                result[key] = val

        if return_context:
            context = {
                "entry": entry,
                "camera": camera,
                "sample_id": entry.get("sample_id", "unknown"),
            }
            return result, context
        return result

    def _migrate_optimizer_state(self, old_opt_state, clone_parent_map, keep_mask, remapped_clone, pre_prune_clone_indices=None) -> None:
        try:
            param_keys = ["means", "scales", "quats", "opacities", "colors"]
            gd = {k: v for k, v in self.gaussians.__dict__.items() if isinstance(v, torch.Tensor)}
            if old_opt_state is None or not old_opt_state.get("state"):
                self._build_optimizer()
                return
            old_state = old_opt_state["state"]
            self._build_optimizer()
            for pg_idx, field in enumerate(param_keys):
                if pg_idx >= len(self.optimizer.param_groups):
                    break
                pg = self.optimizer.param_groups[pg_idx]
                if not pg["params"]:
                    continue
                p = pg["params"][0]
                old_pid = sorted(old_state.keys())[pg_idx] if pg_idx < len(old_state) else None
                if old_pid is None:
                    continue
                old_s = old_state[old_pid]
                old_exp_avg = old_s.get("exp_avg")
                old_exp_avg_sq = old_s.get("exp_avg_sq")
                if old_exp_avg is None:
                    continue
                new_exp_avg = torch.zeros_like(p)
                new_exp_avg_sq = torch.zeros_like(p)
                if keep_mask is not None:
                    old_kept_avg = old_exp_avg[keep_mask]
                    old_kept_avg_sq = old_exp_avg_sq[keep_mask]
                else:
                    old_kept_avg = old_exp_avg
                    old_kept_avg_sq = old_exp_avg_sq
                n_kept = old_kept_avg.shape[0]
                new_exp_avg[:n_kept] = old_kept_avg
                new_exp_avg_sq[:n_kept] = old_kept_avg_sq
                if clone_parent_map is not None and clone_parent_map.numel() > 0:
                    n_clone = clone_parent_map.shape[0]
                    if n_clone > 0:
                        parent_idx = (pre_prune_clone_indices if pre_prune_clone_indices is not None else remapped_clone)[:n_clone]
                        parent_avg = old_exp_avg[parent_idx]
                        parent_avg_sq = old_exp_avg_sq[parent_idx]
                        clone_start = n_kept
                        clone_end = n_kept + n_clone
                        if field == "means":
                            new_exp_avg[clone_start:clone_end] = parent_avg * self.unc_config.clone_means_exp_avg_scale
                        else:
                            new_exp_avg[clone_start:clone_end] = parent_avg
                        new_exp_avg_sq[clone_start:clone_end] = parent_avg_sq
                self.optimizer.state[p] = {
                    "step": old_s.get("step", 0),
                    "exp_avg": new_exp_avg,
                    "exp_avg_sq": new_exp_avg_sq,
                }
        except Exception as e:
            import warnings
            warnings.warn(f"Optimizer state migration failed ({e}), falling back to rebuild from scratch.")
            self._build_optimizer()

    def _densification_step(self, iter_idx: int, context: dict,
                            clip_active_ratio: float = 0.0) -> DensificationSelection:
        config = self.unc_config
        entry = context["entry"]
        camera = context["camera"]
        sample_id = context["sample_id"]

        gt_rgb = _load_rgb(entry["left_rgb_path"]).to(self.device)
        gt_depth = load_servct_depth(Path(entry["left_depth_path"])).to(self.device)

        with torch.no_grad():
            out = self.backend.render(
                gaussians=self.gaussians,
                camera=camera,
                image_height=entry["height"],
                image_width=entry["width"],
                render_depth=True,
            )

            mask = self._get_mask_for_entry(entry) if config.variant in ("h2", "h3") else None

            residual = compute_photo_residual(out.rgb[..., :3], gt_rgb, detach_pred=True)
            scale = compute_p95_scale(residual)
            u_photo = compute_u_photo(residual, scale)
            if mask is not None:
                w_photo = compute_w_photo_with_mask(
                    u_photo, mask, alpha=config.alpha,
                    w_min=config.w_photo_min, mask_boost=config.mask_boost,
                )
            else:
                w_photo = compute_w_photo(u_photo, alpha=config.alpha, w_min=config.w_photo_min)

            selection = select_densification_candidates(
                gaussians=self.gaussians,
                rendered_depth=out.depth,
                gt_depth=gt_depth,
                rendered_rgb=out.rgb[..., :3],
                gt_rgb=gt_rgb,
                w_photo=w_photo,
                camera=camera,
                config=config,
                sample_id=sample_id,
                frame_id=entry.get("frame_id", str(iter_idx)),
                split="train",
                iter_idx=iter_idx,
            )

        n_before = self.gaussians.num_gaussians()
        keep_mask = None
        old_opt_state = self.optimizer.state_dict() if self.optimizer is not None else None

        if selection.n_pruned > 0:
            keep_mask = selection.prune_mask
            self.gaussians.remove_gaussians(keep_mask)
            if hasattr(self, "init_scales") and self.init_scales is not None:
                self.init_scales = self.init_scales[keep_mask].detach().clone().requires_grad_(True)
            old_n = n_before
            new_n = self.gaussians.num_gaussians()

            offset = torch.zeros(self.gaussians.num_gaussians(), device=self.device, dtype=torch.long)
            new_indices = torch.arange(new_n, device=self.device)
            old_to_new = torch.zeros(old_n, dtype=torch.long, device=self.device)
            old_to_new[keep_mask] = new_indices
            remapped_clone = old_to_new[selection.clone_indices]
        else:
            remapped_clone = selection.clone_indices

        clone_parent_map = None
        if selection.n_cloned > 0:
            clone_parent_map = self.gaussians.clone_gaussians(
                remapped_clone, selection.clone_offsets, return_parent_mapping=True
            )
            if hasattr(self, "init_scales") and self.init_scales is not None:
                cloned_init = self.init_scales[remapped_clone].detach().clone().requires_grad_(True)
                self.init_scales = torch.cat([self.init_scales, cloned_init], dim=0)

        if selection.n_cloned > 0 or selection.n_pruned > 0:
            self._migrate_optimizer_state(old_opt_state, clone_parent_map, keep_mask, remapped_clone, pre_prune_clone_indices=selection.clone_indices)

        n_after = self.gaussians.num_gaussians()
        log_entry = {
            "iter": iter_idx,
            "n_gaussians_before": n_before,
            "n_gaussians_after": n_after,
            "n_cloned": selection.n_cloned,
            "n_pruned": selection.n_pruned,
            "selected_candidate_count": selection.selected_candidate_count,
            "selected_candidate_ratio": round(selection.selected_candidate_ratio, 6),
            "selected_mean_depth_residual": round(selection.selected_mean_depth_residual, 6),
            "selected_p50_depth_residual": round(selection.selected_p50_depth_residual, 6),
            "selected_p90_depth_residual": round(selection.selected_p90_depth_residual, 6),
            "selected_mean_w_photo": round(selection.selected_mean_w_photo, 6),
            "selected_p10_w_photo": round(selection.selected_p10_w_photo, 6),
            "selected_min_w_photo": round(selection.selected_min_w_photo, 6),
            "selected_p01_w_photo": round(selection.selected_p01_w_photo, 6),
            "selected_p05_w_photo": round(selection.selected_p05_w_photo, 6),
            "w_photo_threshold": round(selection.w_photo_threshold, 6),
            "w_photo_leak_count": selection.w_photo_leak_count,
            "w_photo_near_threshold_count": selection.w_photo_near_threshold_count,
            "w_photo_threshold_margin_min": round(selection.w_photo_threshold_margin_min, 6),
            "opacity_mean": round(selection.opacity_mean, 6),
            "opacity_std": round(selection.opacity_std, 6),
            "opacity_min": round(selection.opacity_min, 6),
            "opacity_max": round(selection.opacity_max, 6),
            "clip_active_ratio": round(clip_active_ratio, 4),
            "max_gaussians_hit": selection.max_gaussians_hit,
            "clone_offset_mode": selection.clone_offset_mode,
            "clone_offset_scale": selection.clone_offset_scale,
            "mapping_mode": selection.mapping_mode,
            "trigger_reason": selection.trigger_reason,
            "sample_id": selection.sample_id,
            "frame_id": selection.frame_id,
            "split": selection.split,
            "densification_entry_mode": selection.densification_entry_mode,
        }
        dens_log_path = self.output_dir / "densification_log.jsonl"
        dens_log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dens_log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        return selection

    def _run_val(self, iter_idx: int) -> Dict[str, float]:
        self.gaussians.means.requires_grad_(False)
        self.gaussians.scales.requires_grad_(False)
        self.gaussians.quats.requires_grad_(False)
        self.gaussians.opacities.requires_grad_(False)
        self.gaussians.colors.requires_grad_(False)

        psnr_list, ssim_list = [], []
        lpips_list = []
        all_depth_reports = []
        w_photo_stats_list = []

        snapshot_dir = self.output_dir / "renders"
        depth_dir = self.output_dir / "depth"
        unc_dir = self.output_dir / "uncertainty"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        depth_dir.mkdir(parents=True, exist_ok=True)
        unc_dir.mkdir(parents=True, exist_ok=True)

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

            mask = self._get_mask_for_entry(entry) if self.unc_config.variant in ("h2", "h3") else None

            residual = compute_photo_residual(pred_rgb, gt_rgb, detach_pred=True)
            scale = compute_p95_scale(residual)
            u_photo = compute_u_photo(residual, scale)
            if mask is not None:
                w_photo = compute_w_photo_with_mask(
                    u_photo, mask, alpha=self.unc_config.alpha,
                    w_min=self.unc_config.w_photo_min, mask_boost=self.unc_config.mask_boost,
                )
            else:
                w_photo = compute_w_photo(u_photo, alpha=self.unc_config.alpha, w_min=self.unc_config.w_photo_min)

            _save_u_photo_color(u_photo, unc_dir / f"iter_{iter_idx:06d}_val_{sid}_u_photo_color.png")
            _save_w_photo_color(w_photo, unc_dir / f"iter_{iter_idx:06d}_val_{sid}_w_photo_color.png",
                                w_min=self.unc_config.w_photo_min)

            _, uw_diag = uncertainty_weighted_photometric_l1(
                rgb_pred=pred_rgb,
                rgb_gt=gt_rgb,
                alpha=self.unc_config.alpha,
                w_min=self.unc_config.w_photo_min,
                mask=mask,
                mask_boost=self.unc_config.mask_boost,
            )

            w_photo_stats_list.append(uw_diag)

            if out.depth is not None and out.aux.get("depth_semantics") == "metric_meters":
                self._save_snapshot_depth(out.depth, gt_depth, depth_dir, iter_idx, sid)
                depth_report = geometry_metrics_report(
                    pred_depth=out.depth,
                    gt_depth=gt_depth,
                    depth_semantics=out.aux["depth_semantics"],
                    near_m=self.unc_config.depth_near_m,
                    far_m=self.unc_config.depth_far_m,
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

        if w_photo_stats_list:
            for key in ("w_photo_mean", "w_photo_min", "w_photo_max",
                         "w_photo_p10", "w_photo_p50", "w_photo_p90",
                         "fraction_w_photo_at_min", "fraction_w_photo_at_one",
                         "w_photo_p90_minus_p10", "normalization_mode",
                         "mask_used", "mask_coverage",
                         "w_photo_in_mask_mean", "w_photo_out_mask_mean"):
                vals = [d.get(key) for d in w_photo_stats_list if d.get(key) is not None]
                if vals:
                    if isinstance(vals[0], str):
                        result[f"val_{key}"] = vals[0]
                    else:
                        result[f"val_{key}"] = float(np.mean(vals))

            avg_coverage = result.get("val_mask_coverage")
            if avg_coverage is not None:
                result["val_mask_effective"] = avg_coverage >= 0.01

        return result

    def _resolve_val_metric(self, metric_name: str) -> str:
        mapping = {
            "depth_rmse": "val_depth_rmse_m_raw",
            "psnr": "val_psnr",
        }
        return mapping.get(metric_name, metric_name)

    def _maybe_save_best_val(self, iter_idx: int, val_metrics: Dict) -> None:
        if not self.unc_config.best_val_enabled:
            return
        metric_key = self._resolve_val_metric(self.unc_config.best_val_metric)
        tie_key = self._resolve_val_metric(self.unc_config.best_val_tiebreaker)
        metric_value = val_metrics.get(metric_key)
        tie_value = val_metrics.get(tie_key)
        if metric_value is None:
            return
        if self._best_val is None:
            improved = True
        else:
            if self.unc_config.best_val_metric_mode == "min":
                improved = metric_value < self._best_val["metric_value"]
            else:
                improved = metric_value > self._best_val["metric_value"]
            if not improved and metric_value == self._best_val["metric_value"]:
                if tie_value is not None:
                    if self.unc_config.best_val_tiebreaker_mode == "max":
                        improved = tie_value > self._best_val.get("tie_value", float("-inf"))
                    else:
                        improved = tie_value < self._best_val.get("tie_value", float("inf"))
        if improved:
            self._best_val = {
                "iter": iter_idx,
                "metric_value": metric_value,
                "tie_value": tie_value,
                "val_metrics": val_metrics,
            }
            ckpt_dir = self.output_dir / "checkpoints"
            ckpt_path = ckpt_dir / "best_val.pt"
            save_checkpoint(
                path=ckpt_path,
                gaussians=self.gaussians,
                optimizer_state_dict=self.optimizer.state_dict(),
                iteration=iter_idx,
                config=asdict(self.unc_config),
                backend_name=self.unc_config.backend,
                seed=self.unc_config.seed,
                extra={
                    "best_val_metric": self.unc_config.best_val_metric,
                    "best_val_metric_mode": self.unc_config.best_val_metric_mode,
                    "best_val_tiebreaker": self.unc_config.best_val_tiebreaker,
                    "best_val_score": metric_value,
                    "best_val_tiebreaker_value": tie_value,
                    "val_metrics": val_metrics,
                },
            )
            best_meta = {
                "best_iter": iter_idx,
                "best_val_metric": self.unc_config.best_val_metric,
                "best_val_metric_mode": self.unc_config.best_val_metric_mode,
                "best_val_metric_value": metric_value,
                "best_val_tiebreaker": self.unc_config.best_val_tiebreaker,
                "best_val_tiebreaker_value": tie_value,
                "val_psnr": val_metrics.get("val_psnr"),
                "val_depth_rmse_m_raw": val_metrics.get("val_depth_rmse_m_raw"),
                "val_ssim": val_metrics.get("val_ssim"),
                "val_lpips": val_metrics.get("val_lpips"),
                "val_abs_rel": val_metrics.get("val_abs_rel"),
                "n_gaussians": self.gaussians.num_gaussians(),
                "enable_densification": self.unc_config.enable_densification,
            }
            if hasattr(self, "_dens_steps_count"):
                best_meta["densification_steps_count"] = self._dens_steps_count
                best_meta["total_cloned"] = getattr(self, "_total_cloned", 0)
                best_meta["total_pruned"] = getattr(self, "_total_pruned", 0)
            write_json(self.output_dir / "best_val_metrics.json", best_meta)

    def fit(self) -> Dict:
        self.setup()
        config = self.unc_config
        final_metrics = {}
        clip_active_count = 0
        clip_total_count = 0
        n_gaussians_initial = self.gaussians.num_gaussians()
        self._dens_steps_count = 0
        self._total_cloned = 0
        self._total_pruned = 0
        max_gaussians_hit = False

        for i in range(1, config.iterations + 1):
            self._apply_warmup(i)
            t0 = time.time()
            if config.enable_densification:
                step, context = self.train_step(i, return_context=True)
            else:
                step = self.train_step(i)
            step_time = time.time() - t0

            vram = 0.0
            if torch.cuda.is_available():
                vram = torch.cuda.max_memory_allocated() / 1024 ** 3
                torch.cuda.reset_peak_memory_stats()

            metrics = {
                "iter": i,
                "loss_total": round(step["loss_total"], 6),
                "photo_loss_weighted": round(step["photo_loss_weighted"], 8),
                "depth_loss_raw_m": round(step["depth_loss_raw_m"], 8),
                "depth_loss_weighted": round(step["depth_loss_weighted"], 8),
                "psnr": round(step["psnr"], 4),
                "depth_rmse_m_raw": round(step["depth_rmse_m_raw"], 6),
                "depth_mae_m_raw": round(step["depth_mae_m_raw"], 6),
                "abs_rel": round(step["abs_rel"], 6),
                "depth_valid_ratio": round(step["depth_valid_ratio"], 4),
                "n_gaussians": step["n_gaussians"],
                "iter_time_s": round(step_time, 4),
                "vram_gb": round(vram, 3),
            }

            for key in ("w_photo_mean", "w_photo_min", "w_photo_max",
                         "w_photo_p10", "w_photo_p50", "w_photo_p90",
                         "fraction_w_photo_at_min", "fraction_w_photo_at_one",
                         "w_photo_p90_minus_p10", "u_photo_mean", "p95_scale"):
                if key in step:
                    metrics[key] = round(step[key], 6)

            if config.log_grad_norm:
                metrics["grad_norm_before_clip"] = round(step["grad_norm_before_clip"], 6)
                metrics["grad_norm_after_clip"] = round(step["grad_norm_after_clip"], 6)

            clip_total_count += 1
            if config.clip_grad_norm and step["grad_norm_before_clip"] > config.max_grad_norm:
                clip_active_count += 1
            clip_active_ratio = clip_active_count / clip_total_count if clip_total_count > 0 else 0.0
            metrics["clip_active_ratio"] = round(clip_active_ratio, 4)

            # Densification step (M4-A2-1): train_step → optimizer.step → post-step clamp → densification → log
            if (config.enable_densification
                    and config.densify_from_iter <= i <= config.densify_until_iter
                    and i % config.densify_every == 0):
                dens_sel = self._densification_step(i, context, clip_active_ratio=clip_active_ratio)
                self._dens_steps_count += 1
                self._total_cloned += dens_sel.n_cloned
                self._total_pruned += dens_sel.n_pruned
                if dens_sel.max_gaussians_hit:
                    max_gaussians_hit = True
                step["n_gaussians"] = self.gaussians.num_gaussians()
                metrics["n_gaussians"] = step["n_gaussians"]
                metrics["densification_step_applied"] = True
            else:
                metrics["densification_step_applied"] = False

            self.metrics_logger.log(metrics)

            if i == 1:
                final_metrics["initial_loss_total"] = step["loss_total"]

            if i % config.log_every == 0:
                msg = (f"iter {i:5d}/{config.iterations}  "
                       f"loss={metrics['loss_total']:.6f}  "
                       f"photo={metrics['photo_loss_weighted']:.6f}  "
                       f"depth={metrics['depth_loss_raw_m']:.6f}  "
                       f"psnr={metrics['psnr']:.2f}  "
                       f"mean_w={metrics.get('w_photo_mean', 0):.4f}  "
                       f"gaussians={metrics['n_gaussians']}  "
                       f"time={metrics['iter_time_s']:.3f}s")
                if config.warmup_iters > 0:
                    factor = min(1.0, i / config.warmup_iters)
                    msg += f"  lr_scale={factor:.3f}"
                if config.log_grad_norm:
                    msg += f"  grad={metrics.get('grad_norm_before_clip', 0):.4f}"
                    msg += f"  clip_ratio={metrics.get('clip_active_ratio', 0):.3f}"
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
                w_str = ""
                if "val_w_photo_mean" in val_metrics:
                    w_str = f"  mean_w={val_metrics['val_w_photo_mean']:.4f}"
                print(f"VAL iter {i}: psnr={val_metrics['val_psnr']:.2f}  "
                      f"ssim={val_metrics['val_ssim']:.4f}  "
                      f"lpips={lpips_str}{depth_str}{w_str}")
                self._maybe_save_best_val(i, val_metrics)

            if i % config.ckpt_every == 0:
                self.save(i)

        final_metrics["final_loss_total"] = step["loss_total"]
        final_metrics["final_photo_loss_weighted"] = step["photo_loss_weighted"]
        final_metrics["n_gaussians"] = step["n_gaussians"]
        val_metrics = self._run_val(config.iterations)
        self._maybe_save_best_val(config.iterations, val_metrics)
        final_metrics.update(val_metrics)
        final_metrics["loss_decreased"] = final_metrics["final_loss_total"] < final_metrics["initial_loss_total"]
        final_metrics["iterations"] = config.iterations
        final_metrics["split_strategy"] = self.split_strategy
        final_metrics["lambda_depth"] = config.lambda_depth
        final_metrics["lambda_reg"] = config.lambda_reg
        final_metrics["alpha"] = config.alpha
        final_metrics["w_photo_min"] = config.w_photo_min
        final_metrics["variant"] = config.variant
        final_metrics["clip_grad_norm"] = config.clip_grad_norm
        final_metrics["max_grad_norm"] = config.max_grad_norm
        final_metrics["clip_active_ratio"] = round(clip_active_ratio, 4)
        final_metrics["warmup_iters"] = config.warmup_iters
        final_metrics["enable_densification"] = config.enable_densification
        final_metrics["densify_from_iter"] = config.densify_from_iter
        final_metrics["densify_every"] = config.densify_every
        final_metrics["densify_until_iter"] = config.densify_until_iter
        final_metrics["densify_depth_residual_threshold"] = config.densify_depth_residual_threshold
        final_metrics["densify_w_photo_threshold"] = config.densify_w_photo_threshold
        final_metrics["densify_max_clone_per_step"] = config.densify_max_clone_per_step
        final_metrics["densify_max_clone_fraction"] = config.densify_max_clone_fraction
        final_metrics["densify_max_gaussians"] = config.densify_max_gaussians
        final_metrics["prune_min_opacity"] = config.prune_min_opacity
        final_metrics["max_prune_fraction_per_step"] = config.max_prune_fraction_per_step
        final_metrics["clone_offset_scale_factor"] = config.clone_offset_scale_factor
        final_metrics["n_gaussians_initial"] = n_gaussians_initial
        final_metrics["n_gaussians_final"] = self.gaussians.num_gaussians()
        final_metrics["gaussian_growth_ratio"] = round(
            self.gaussians.num_gaussians() / max(1, n_gaussians_initial), 6
        )
        final_metrics["densification_steps_count"] = self._dens_steps_count
        final_metrics["total_cloned"] = self._total_cloned
        final_metrics["total_pruned"] = self._total_pruned
        final_metrics["max_gaussians_hit"] = max_gaussians_hit
        final_metrics["depth_semantics"] = "metric_meters"
        final_metrics["normalization_mode"] = "p95_detached"
        final_metrics["run_mode"] = getattr(self.unc_config, "run_mode", "production")
        final_metrics["gate_eligible"] = final_metrics["run_mode"] == "production"
        final_metrics["mask_used"] = config.variant in ("h2", "h3")
        if config.variant in ("h2", "h3"):
            avg_coverage = val_metrics.get("val_mask_coverage", 0.0)
            final_metrics["mask_coverage"] = avg_coverage
            final_metrics["mask_effective"] = avg_coverage >= 0.01
            if avg_coverage < 0.01:
                final_metrics["mask_interpretation"] = (
                    "H2 mask path executed, but mask coverage too low "
                    "to attribute gains to mask-aware weighting."
                )

        if self._best_val is not None:
            final_metrics["best_val_iter"] = self._best_val["iter"]
            final_metrics["best_val_metric"] = self.unc_config.best_val_metric
            final_metrics["best_val_metric_value"] = self._best_val["metric_value"]
            final_metrics["best_val_tiebreaker_value"] = self._best_val.get("tie_value")
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
            config=asdict(self.unc_config),
            backend_name=self.unc_config.backend,
            seed=self.unc_config.seed,
        )

    def _write_report(self, fm: Dict) -> None:
        lines = [
            "# Uncertainty-Weighted Run Report (M3)",
            "",
            f"## 1. Dataset Split (deterministic, seed={self.unc_config.seed})",
            f"- train: Experiment_1 frame 1-6 (n={len(self.train_entries)})",
            f"- val: Experiment_1 frame 7-8 (n={len(self.val_entries)})",
            "- split_strategy: " + self.split_strategy,
            "",
            "## 2. Environment",
            "See `environment.json` for full details.",
            "",
            "## 3. Run Summary",
            f"- iterations: {self.unc_config.iterations}",
            f"- variant: {self.unc_config.variant}",
            f"- initial_loss_total (iter 1): {fm.get('initial_loss_total', 0):.6f}",
            f"- final_loss_total (iter {self.unc_config.iterations}): {fm.get('final_loss_total', 0):.6f}",
            f"- loss_decreased: {fm.get('loss_decreased', False)}",
            f"- n_gaussians (final): {fm.get('n_gaussians', 0)}",
            f"- val_psnr: {fm.get('val_psnr', 0):.4f}",
            f"- val_ssim: {fm.get('val_ssim', 0):.4f}",
            f"- val_lpips: {fm.get('val_lpips', 'N/A')}",
            "",
            "## 4. Uncertainty Configuration",
            f"- lambda_depth: {self.unc_config.lambda_depth}",
            f"- lambda_reg: {self.unc_config.lambda_reg}",
            f"- alpha: {self.unc_config.alpha}",
            f"- w_photo_min: {self.unc_config.w_photo_min}",
            f"- mask_boost: {self.unc_config.mask_boost}",
            f"- variant: {self.unc_config.variant}",
            f"- mask_dir: {self.unc_config.mask_dir}",
            f"- depth_semantics: metric_meters",
            "",
            "## 5. Depth Loss Diagnostics",
        ]
        for key in ("val_depth_rmse_m_raw", "val_depth_rmse_m_clipped",
                     "val_depth_mae_m_raw", "val_depth_mae_m_clipped",
                     "val_abs_rel", "val_depth_valid_ratio",
                     "val_median_aligned_rmse_m"):
            if key in fm:
                lines.append(f"- {key}: {fm[key]}")
        lines.append(f"- num_depth_val_samples: {fm.get('val_num_depth_samples', 0)}")
        lines.append("")
        lines.append("## 5.5 Uncertainty Diagnostics")
        lines.append(f"- normalization_mode: {fm.get('normalization_mode', 'N/A')}")
        lines.append(f"- alpha: {self.unc_config.alpha}")
        lines.append(f"- w_photo_min: {self.unc_config.w_photo_min}")
        if fm.get("val_w_photo_mean") is not None:
            lines.append(f"- val_w_photo_mean: {fm['val_w_photo_mean']:.4f}")
            lines.append(f"- val_w_photo_min: {fm['val_w_photo_min']:.4f}")
            lines.append(f"- val_w_photo_max: {fm['val_w_photo_max']:.4f}")
            lines.append(f"- val_w_photo_p10: {fm['val_w_photo_p10']:.4f}")
            lines.append(f"- val_w_photo_p50: {fm['val_w_photo_p50']:.4f}")
            lines.append(f"- val_w_photo_p90: {fm['val_w_photo_p90']:.4f}")
            lines.append(f"- val_fraction_w_photo_at_min: {fm['val_fraction_w_photo_at_min']:.4f}")
            lines.append(f"- val_fraction_w_photo_at_one: {fm['val_fraction_w_photo_at_one']:.4f}")
            lines.append(f"- val_w_photo_p90_minus_p10: {fm['val_w_photo_p90_minus_p10']:.4f}")
        lines.append("")
        lines.append("## 5.6 Mask Effectiveness")
        lines.append(f"- mask_used: {fm.get('mask_used', False)}")
        if fm.get("mask_used"):
            lines.append(f"- mask_coverage: {fm.get('mask_coverage', 0):.4f}")
            lines.append(f"- mask_effective: {fm.get('mask_effective', False)}")
            if fm.get("val_w_photo_in_mask_mean") is not None:
                lines.append(f"- w_photo_in_mask_mean: {fm['val_w_photo_in_mask_mean']:.4f}")
                lines.append(f"- w_photo_out_mask_mean: {fm['val_w_photo_out_mask_mean']:.4f}")
            if fm.get("mask_interpretation"):
                lines.append(f"- interpretation: {fm['mask_interpretation']}")
        lines.append("")
        lines += [
            "## 6. Outputs",
            "- config.json: yes",
            "- environment.json: yes",
            "- metrics.jsonl: yes",
            "- checkpoints/: yes",
            "- renders/: yes",
            "- depth/: yes",
            "- uncertainty/: yes",
            "- final_metrics.json: yes",
            "",
            '## 7. Densification (M4-A2-1)',
            f'- enable_densification: {fm.get("enable_densification", False)}',
        ]
        if fm.get("enable_densification"):
            lines += [
                f'- n_gaussians_initial: {fm.get("n_gaussians_initial", "N/A")}',
                f'- n_gaussians_final: {fm.get("n_gaussians_final", "N/A")}',
                f'- gaussian_growth_ratio: {fm.get("gaussian_growth_ratio", "N/A")}',
                f'- densification_steps_count: {fm.get("densification_steps_count", 0)}',
                f'- total_cloned: {fm.get("total_cloned", 0)}',
                f'- total_pruned: {fm.get("total_pruned", 0)}',
                f'- max_gaussians_hit: {fm.get("max_gaussians_hit", False)}',
                '- mapping_mode: projection_based_approximate',
                '- Future improvement: true renderer contribution-aware densification requires exposing gsplat visibility/radii/meta outputs.',
            ]
        else:
            lines += [f"- Fixed {fm.get('n_gaussians', 20000)} Gaussians, densification off"]
        lines += [
            "- Learned mapper not included",
            "",
            "## 8. Next Steps",
        ]
        if fm.get("enable_densification"):
            lines += ["- M4-A2-1 evaluation: evaluate_m4_a2_1.py"]
        else:
            lines += ["- Milestone 4: multi-criteria density control"]
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

            mask = self._get_mask_for_entry(entry) if self.unc_config.variant in ("h2", "h3") else None
            _, uw_diag = uncertainty_weighted_photometric_l1(
                rgb_pred=out.rgb[..., :3],
                rgb_gt=gt_rgb,
                alpha=self.unc_config.alpha,
                w_min=self.unc_config.w_photo_min,
                mask=mask,
                mask_boost=self.unc_config.mask_boost,
            )

            residual = compute_photo_residual(out.rgb[..., :3], gt_rgb, detach_pred=True)
            scale = compute_p95_scale(residual)
            u_photo = compute_u_photo(residual, scale)
            if mask is not None:
                w_photo = compute_w_photo_with_mask(
                    u_photo, mask, alpha=self.unc_config.alpha,
                    w_min=self.unc_config.w_photo_min, mask_boost=self.unc_config.mask_boost,
                )
            else:
                w_photo = compute_w_photo(u_photo, alpha=self.unc_config.alpha, w_min=self.unc_config.w_photo_min)

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
            w_np = w_photo.detach().cpu().numpy()
            w_norm = np.clip((w_np - self.unc_config.w_photo_min) / (1.0 - self.unc_config.w_photo_min), 0, 1)
            w_color = cv2.applyColorMap((w_norm * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)

            panels = [
                cv2.cvtColor(gt_rgb_np, cv2.COLOR_RGB2BGR),
                cv2.cvtColor(pred_rgb_np, cv2.COLOR_RGB2BGR),
                cv2.applyColorMap(_norm_depth(gt_d_np), cv2.COLORMAP_INFERNO),
                cv2.applyColorMap(_norm_depth(pred_d_np), cv2.COLORMAP_INFERNO),
                cv2.applyColorMap(_norm_depth(error_map), cv2.COLORMAP_VIRIDIS),
                w_color,
            ]

            # Densification artifacts (M4-A2-1)
            if self.unc_config.enable_densification:
                depth_residual = np.abs(pred_d_np - gt_d_np)
                valid_mask = (gt_d_np > self.unc_config.depth_near_m) & (gt_d_np < self.unc_config.depth_far_m) & np.isfinite(gt_d_np)
                candidate_mask = np.zeros_like(depth_residual, dtype=np.uint8)
                if valid_mask.any():
                    candidate_mask[(depth_residual > self.unc_config.densify_depth_residual_threshold) & (w_np > self.unc_config.densify_w_photo_threshold) & valid_mask] = 255
                cand_color = cv2.applyColorMap(candidate_mask, cv2.COLORMAP_BONE)
                panels.append(cand_color)

                from surgtwin.cameras.projection import project_points_to_image
                K_np = camera.K.cpu().numpy()
                w2c_np = camera.w2c.cpu().numpy()
                means_np = self.gaussians.means.detach().cpu().numpy()
                uv_hom = w2c_np[:3, :3] @ means_np.T + w2c_np[:3, 3:4]
                z_cam = uv_hom[2]
                uv = uv_hom[:2] / np.where(z_cam > 0, z_cam, 1e-8)
                u = np.round(uv[0]).astype(int)
                v = np.round(uv[1]).astype(int)
                in_img = (u >= 0) & (u < entry["width"]) & (v >= 0) & (v < entry["height"]) & (z_cam > 0).squeeze()
                overlay = np.zeros((entry["height"], entry["width"], 3), dtype=np.uint8)
                overlay[v[in_img], u[in_img]] = [0, 255, 0]
                panels.append(overlay)

            h, w = gt_rgb_np.shape[:2]
            if len(panels) == 8:
                row1 = np.hstack([panels[0], panels[1]])
                row2 = np.hstack([panels[2], panels[3]])
                row3 = np.hstack([panels[4], panels[5]])
                row4 = np.hstack([panels[6], panels[7]])
                full = np.vstack([row1, row2, row3, row4])
            else:
                top = np.hstack([panels[0], panels[1]])
                mid = np.hstack([panels[2], panels[3]])
                bottom = np.hstack([panels[4], panels[5]])
                full = np.vstack([top, mid, bottom])
            panel_path = panel_dir / f"val_{entry['sample_id']}_panel.png"
            cv2.imwrite(str(panel_path), full)
