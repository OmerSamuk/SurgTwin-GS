from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class CameraData:
    K: torch.Tensor
    c2w: torch.Tensor
    w2c: torch.Tensor
    height: int
    width: int
    near: float = 0.001
    far: float = 1.0
    convention: str = "opencv_c2w"
