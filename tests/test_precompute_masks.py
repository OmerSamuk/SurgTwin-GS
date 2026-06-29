import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from surgtwin.masks.io import load_specular_mask, mask_coverage


def test_load_specular_mask():
    mask_np = np.random.rand(16, 16) > 0.5
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
        np.save(f, mask_np)
        p = Path(f.name)
    loaded = load_specular_mask(p)
    assert loaded.shape == (16, 16)
    assert loaded.dtype == torch.bool
    p.unlink()


def test_load_specular_mask_non_bool():
    mask_int = np.ones((8, 8), dtype=np.int64)
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
        np.save(f, mask_int)
        p = Path(f.name)
    loaded = load_specular_mask(p)
    assert loaded.dtype == torch.bool
    p.unlink()


def test_mask_coverage_empty():
    import torch
    mask = torch.zeros(10, 10, dtype=torch.bool)
    cov = mask_coverage(mask)
    assert cov == 0.0


def test_mask_coverage_full():
    import torch
    mask = torch.ones(10, 10, dtype=torch.bool)
    cov = mask_coverage(mask)
    assert cov == 1.0


def test_mask_coverage_half():
    import torch
    mask = torch.zeros(4, 4, dtype=torch.bool)
    mask[:2, :] = True
    cov = mask_coverage(mask)
    assert cov == 0.5


def test_mask_coverage_zero_numel():
    import torch
    mask = torch.empty(0, dtype=torch.bool)
    cov = mask_coverage(mask)
    assert cov == 0.0
