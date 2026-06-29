from pathlib import Path
from typing import Optional

import numpy as np
import torch


def load_specular_mask(path: Path, device: Optional[torch.device] = None) -> torch.Tensor:
    arr = np.load(str(path))
    if arr.dtype != np.bool_:
        arr = arr.astype(bool)
    mask = torch.from_numpy(arr)
    if device is not None:
        mask = mask.to(device)
    return mask


def mask_coverage(mask: torch.Tensor) -> float:
    if mask.numel() == 0:
        return 0.0
    return mask.float().mean().item()
