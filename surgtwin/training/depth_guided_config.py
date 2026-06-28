from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class DepthGuidedConfig:
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
    seed: int = 42
    backend: str = "gsplat"
    lambda_depth: float = 0.2
    lambda_reg: float = 0.01
    reg_type: str = "scale_drift"
    depth_near_m: float = 0.02
    depth_far_m: float = 0.30
    clip_grad_norm: bool = True
    max_grad_norm: float = 1.0
    log_grad_norm: bool = True
    depth_semantics_artifact_path: Optional[str] = None
    save_depth_snapshot_every: int = 100
