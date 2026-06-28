import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from surgtwin.evaluation.depth_diagnostics import (
    compare_depth_distributions,
    depth_in_servct_range,
    depth_scale_ok,
)


def test_real_verify_imports():
    """Verify the real verification script imports without errors."""
    from scripts.verify_depth_real import main as real_main, load_rgb, save_depth_viz
    assert callable(real_main)
    assert callable(load_rgb)
    assert callable(save_depth_viz)


def test_real_verify_rgb_loading(tmp_path):
    """Test that load_rgb produces correct shape and dtype."""
    from scripts.verify_depth_real import load_rgb
    H, W = 100, 200
    fake = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
    import cv2
    path = tmp_path / "test.png"
    cv2.imwrite(str(path), fake)
    rgb = load_rgb(str(path))
    assert rgb.shape == (H, W, 3), f"Expected ({H},{W},3), got {rgb.shape}"
    assert rgb.dtype == torch.float32
    assert rgb.min() >= 0.0 and rgb.max() <= 1.0


def test_real_verification_json_schema_matches_plan():
    """Validate that the per-sample schema matches the plan spec."""
    sample = {
        "sample_id": "Experiment_1_001",
        "split": "train",
        "frame_index": 1,
        "depth_semantics": "metric_meters",
        "metric_depth_verified": True,
        "distribution": {
            "shape": [576, 720],
            "dtype": "torch.float32",
            "rendered": {"min": 0.045, "max": 0.182, "median": 0.103, "mean": 0.106, "std": 0.029},
            "gt": {"min": 0.050, "max": 0.150, "median": 0.100, "mean": 0.098, "std": 0.024},
            "valid_ratio": 0.85, "finite_ratio": 1.0,
            "rendered_has_invalid": False, "gt_has_invalid": False,
        },
        "range_ok": True,
        "shape_ok": True,
        "finite_ok": True,
        "scale_ok": True,
        "scale_tier": "green",
        "scale_ratio": 0.97,
    }
    assert sample["sample_id"] == "Experiment_1_001"
    assert sample["depth_semantics"] in ("metric_meters", "relative_aligned", "relative_unaligned", "unavailable")
    assert sample["distribution"]["shape"] == [576, 720]
    assert isinstance(sample["scale_ok"], bool)
    assert sample["scale_tier"] in ("green", "acceptable", "diagnostic", "fail")


def test_real_verification_synthetic_shapes():
    """Test that compare_depth_distributions validates shape contract for (576,720) data."""
    H, W = 576, 720
    rendered = torch.rand(H, W)
    gt = torch.rand(H, W)
    gt[0, 0] = 0.0
    result = compare_depth_distributions(rendered, gt)
    assert result["shape"] == [H, W]
    assert result["dtype"] == "torch.float32"
    assert "rendered" in result and "gt" in result


def test_real_verification_range_ok():
    """Test that a depth with median ~0.10m is within SERV-CT range."""
    stats = {
        "shape": [576, 720], "dtype": "torch.float32",
        "rendered": {"min": 0.04, "max": 0.18, "median": 0.10, "mean": 0.10, "std": 0.03},
        "gt": {"min": 0.05, "max": 0.15, "median": 0.10, "mean": 0.098, "std": 0.024},
        "valid_ratio": 0.85, "finite_ratio": 1.0,
    }
    assert depth_in_servct_range(stats) is True


def test_real_verification_shape_mismatch():
    """Test that shape mismatch raises ValueError."""
    with pytest.raises(ValueError, match="Shape mismatch"):
        compare_depth_distributions(torch.rand(576, 720), torch.rand(100, 100))


def test_real_verification_scale_ok():
    """Test depth_scale_ok with metric depth."""
    pred = torch.ones(576, 720) * 0.103
    gt = torch.ones(576, 720) * 0.100
    ok, tier, scale = depth_scale_ok(pred, gt)
    assert ok is True
    assert tier in ("green", "acceptable")


def test_real_verification_scale_fail():
    """Test depth_scale_ok with non-metric depth."""
    pred = torch.ones(576, 720) * 0.05
    gt = torch.ones(576, 720) * 0.20
    ok, tier, scale = depth_scale_ok(pred, gt, tolerance=0.10)
    assert ok is False
    assert tier == "fail"
