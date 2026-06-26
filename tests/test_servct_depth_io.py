import numpy as np
import torch
from PIL import Image
from surgtwin.data.depth_io import load_servct_depth


def test_load_servct_depth(tmp_path):
    H, W = 100, 200
    depth_mm = np.random.uniform(30, 150, (H, W)).astype(np.float32)
    raw = np.round(depth_mm * 256.0).astype(np.uint16)
    p = tmp_path / "depth.png"
    Image.fromarray(raw, mode="I;16").save(p)

    depth_m = load_servct_depth(p)
    assert depth_m.shape == (H, W)
    assert depth_m.dtype == torch.float32

    expected = torch.from_numpy(depth_mm / 1000.0)
    assert torch.allclose(depth_m, expected, atol=1e-4), "Depth conversion mismatch"


def test_load_servct_depth_scale_factor(tmp_path):
    H, W = 10, 10
    raw = np.full((H, W), 25600, dtype=np.uint16)
    p = tmp_path / "depth.png"
    Image.fromarray(raw, mode="I;16").save(p)

    depth_m = load_servct_depth(p)
    assert torch.allclose(depth_m, torch.full((H, W), 0.1), atol=1e-6)


def test_load_servct_depth_invalid_mode(tmp_path):
    rgb_arr = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
    p = tmp_path / "not_depth.png"
    Image.fromarray(rgb_arr).save(p)

    import pytest
    with pytest.raises(ValueError, match="16-bit grayscale"):
        load_servct_depth(p)
