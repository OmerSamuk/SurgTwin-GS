from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class BaselineConfig:
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
