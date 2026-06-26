import torch
from surgtwin.data.depth_io import load_servct_depth
from pathlib import Path
import numpy as np
from PIL import Image


def test_servct_depth_to_meters(tmp_path):
    raw = np.array([[25600, 51200], [12800, 0]], dtype=np.uint16)
    p = tmp_path / "depth.png"
    Image.fromarray(raw, mode="I;16").save(p)

    depth = load_servct_depth(p)
    assert depth.dtype == torch.float32
    expected = torch.tensor([[0.1, 0.2], [0.05, 0.0]], dtype=torch.float32)
    assert torch.allclose(depth, expected, atol=1e-6)


def test_servct_depth_256_scale(tmp_path):
    raw = np.full((5, 5), 256, dtype=np.uint16)
    p = tmp_path / "depth.png"
    Image.fromarray(raw, mode="I;16").save(p)

    depth = load_servct_depth(p)
    assert torch.allclose(depth, torch.full((5, 5), 0.001), atol=1e-6)
