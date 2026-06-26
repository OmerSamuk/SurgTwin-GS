from pathlib import Path

import numpy as np
import torch
from PIL import Image


def load_servct_depth(path: Path) -> torch.Tensor:
    img = Image.open(path)
    if img.mode != "I;16":
        raise ValueError(f"Expected 16-bit grayscale PNG, got mode {img.mode} from {path}")
    arr = np.array(img, dtype=np.float32)
    depth_m = arr / 256.0 / 1000.0
    return torch.from_numpy(depth_m)


def load_servct_occlusion(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    arr = np.array(img, dtype=np.uint8)
    return arr
