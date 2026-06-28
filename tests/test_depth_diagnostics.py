import torch
from surgtwin.evaluation.depth_diagnostics import (
    compare_depth_distributions,
    median_align_depth,
    classify_scale,
    depth_in_servct_range,
    depth_scale_ok,
)


def test_compare_depth_distributions_returns_expected_keys():
    H, W = 100, 200
    rendered = torch.rand(H, W)
    gt = torch.rand(H, W)
    result = compare_depth_distributions(rendered, gt)
    assert "shape" in result
    assert "rendered" in result
    assert "gt" in result
    assert "valid_ratio" in result
    assert "finite_ratio" in result
    assert result["shape"] == [H, W]


def test_compare_depth_distributions_shape_mismatch():
    import pytest
    with pytest.raises(ValueError, match="Shape mismatch"):
        compare_depth_distributions(torch.rand(10, 10), torch.rand(20, 20))


def test_compare_depth_distributions_with_mask():
    H, W = 50, 50
    rendered = torch.ones(H, W)
    gt = torch.ones(H, W)
    gt[0, 0] = 0.0
    mask = gt > 0
    result = compare_depth_distributions(rendered, gt, mask)
    assert result["valid_ratio"] < 1.0


def test_compare_depth_distributions_invalid_in_gt():
    H, W = 50, 50
    rendered = torch.ones(H, W)
    gt = torch.ones(H, W)
    gt[0, 0] = float("nan")
    result = compare_depth_distributions(rendered, gt)
    assert result["gt_has_invalid"] is True


def test_median_align_perfect_metric():
    pred = torch.ones(100, 100) * 0.1
    gt = torch.ones(100, 100) * 0.1
    aligned, scale, shift = median_align_depth(pred, gt)
    assert abs(scale - 1.0) < 1e-4
    assert abs(shift) < 1e-4
    assert torch.allclose(aligned, pred)


def test_median_align_scale_2x():
    pred = torch.ones(100, 100) * 0.05
    gt = torch.ones(100, 100) * 0.1
    aligned, scale, shift = median_align_depth(pred, gt)
    assert abs(scale - 2.0) < 1e-4


def test_median_align_few_valid_pixels():
    pred = torch.zeros(100, 100)
    gt = torch.zeros(100, 100)
    gt[0, 0] = 0.1
    pred[0, 0] = 0.05
    aligned, scale, shift = median_align_depth(pred, gt)
    assert scale == 1.0


def test_classify_scale_green():
    assert classify_scale(1.0) == "green"
    assert classify_scale(1.04) == "green"
    assert classify_scale(0.96) == "green"


def test_classify_scale_acceptable():
    assert classify_scale(1.08) == "acceptable"
    assert classify_scale(0.93) == "acceptable"


def test_classify_scale_diagnostic():
    assert classify_scale(1.15) == "diagnostic"
    assert classify_scale(0.85) == "diagnostic"


def test_classify_scale_fail():
    assert classify_scale(1.25) == "fail"
    assert classify_scale(0.50) == "fail"
    assert classify_scale(None) == "fail"


def test_depth_in_servct_range_ok():
    stats = {"rendered": {"median": 0.10}}
    assert depth_in_servct_range(stats) is True


def test_depth_in_servct_range_outside():
    stats = {"rendered": {"median": 0.50}}
    assert depth_in_servct_range(stats) is False


def test_depth_in_servct_range_missing():
    assert depth_in_servct_range({}) is False


def test_depth_scale_ok_perfect():
    pred = torch.ones(100, 100) * 0.1
    gt = torch.ones(100, 100) * 0.1
    ok, tier, scale = depth_scale_ok(pred, gt)
    assert ok is True
    assert tier == "green"


def test_depth_scale_ok_outside_tolerance():
    pred = torch.ones(100, 100) * 0.1
    gt = torch.ones(100, 100) * 0.2
    ok, tier, scale = depth_scale_ok(pred, gt, tolerance=0.10)
    assert ok is False
    assert tier == "fail"


def test_depth_scale_ok_acceptable():
    pred = torch.ones(100, 100) * 0.1
    gt = torch.ones(100, 100) * 0.108
    ok, tier, scale = depth_scale_ok(pred, gt, tolerance=0.10)
    assert ok is True
    assert tier == "acceptable"
