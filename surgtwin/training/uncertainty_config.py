from dataclasses import dataclass
from typing import Optional

_SUPPORTED_VAL_METRICS = ("depth_rmse", "psnr", "weighted_score")


@dataclass(frozen=True)
class UncertaintyConfig:
    iterations: int = 1000
    log_every: int = 10
    val_every: int = 100
    ckpt_every: int = 500
    lr_means: float = 1e-3
    lr_scales: float = 1e-5
    lr_quats: float = 1e-3
    lr_opacities: float = 5e-2
    lr_colors: float = 2.5e-3
    init_num_points: int = 20000
    enable_densification: bool = False
    # Densification schedule
    densify_from_iter: int = 200
    densify_every: int = 100
    densify_until_iter: int = 800
    # Densification thresholds
    densify_depth_residual_threshold: float = 0.02
    densify_w_photo_threshold: float = 0.3
    # Clone bounds
    densify_max_clone_per_step: int = 5000
    densify_max_clone_fraction: float = 0.15
    densify_max_gaussians: int = 50000
    # Prune
    prune_min_opacity: float = 0.005
    max_prune_fraction_per_step: float = 0.05
    # Clone offset
    clone_offset_scale_factor: float = 0.25
    seed: int = 42
    backend: str = "gsplat"

    variant: str = "h1"
    lambda_depth: float = 0.2
    lambda_reg: float = 0.0
    alpha: float = 2.0
    w_photo_min: float = 0.15
    mask_boost: float = 0.5
    depth_near_m: float = 0.02
    depth_far_m: float = 0.30
    clip_grad_norm: bool = True
    max_grad_norm: float = 1.0
    log_grad_norm: bool = True
    warmup_iters: int = 0

    mask_dir: Optional[str] = None
    depth_semantics_artifact_path: Optional[str] = None
    save_uncertainty_maps_every: int = 100

    # Best validation checkpoint
    best_val_enabled: bool = True
    best_val_metric: str = "depth_rmse"
    best_val_tiebreaker: str = "psnr"
    best_val_metric_mode: str = "min"
    best_val_tiebreaker_mode: str = "max"

    # Optimizer state migration (clone damping, §4 tandem movement)
    clone_means_exp_avg_scale: float = 0.5

    def __post_init__(self):
        if self.best_val_enabled and self.best_val_metric not in _SUPPORTED_VAL_METRICS:
            raise ValueError(
                f"Unsupported best_val_metric '{self.best_val_metric}'. "
                f"Supported: {_SUPPORTED_VAL_METRICS}."
            )
