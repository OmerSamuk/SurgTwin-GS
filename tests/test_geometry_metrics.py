import torch

from surgtwin.evaluation.geometry_metrics import (
    abs_rel,
    depth_mae,
    depth_rmse,
    geometry_metrics_report,
    median_aligned_rmse,
    valid_depth_ratio,
)


def _make_valid(pred_val=0.10, gt_val=0.10, h=5, w=5):
    pred = torch.full((h, w), pred_val, dtype=torch.float32)
    gt = torch.full((h, w), gt_val, dtype=torch.float32)
    valid = (gt > 0) & torch.isfinite(gt)
    return pred, gt, valid


def test_depth_rmse_identical():
    pred, gt, valid = _make_valid()
    rmse = depth_rmse(pred, gt, valid, "metric_meters")
    assert abs(rmse.item()) < 1e-6


def test_depth_rmse_known():
    pred, gt, _ = _make_valid(0.20, 0.10)
    valid = (gt > 0) & torch.isfinite(gt)
    rmse = depth_rmse(pred, gt, valid, "metric_meters")
    assert abs(rmse.item() - 0.10) < 1e-6


def test_depth_mae_known():
    pred, gt, valid = _make_valid(0.15, 0.10)
    mae = depth_mae(pred, gt, valid, "metric_meters")
    assert abs(mae.item() - 0.05) < 1e-6


def test_abs_rel_known():
    pred, gt, valid = _make_valid(0.12, 0.10)
    ar = abs_rel(pred, gt, valid, "metric_meters")
    assert abs(ar.item() - 0.2) < 1e-4


def test_valid_depth_ratio():
    valid = torch.tensor([[True, True], [False, False]], dtype=torch.bool)
    ratio = valid_depth_ratio(valid)
    assert abs(ratio.item() - 0.5) < 1e-6


def test_non_metric_raises():
    pred, gt, valid = _make_valid()
    try:
        depth_rmse(pred, gt, valid, "relative_unaligned")
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_non_metric_raises_mae():
    pred, gt, valid = _make_valid()
    try:
        depth_mae(pred, gt, valid, "unavailable")
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_non_metric_raises_abs_rel():
    pred, gt, valid = _make_valid()
    try:
        abs_rel(pred, gt, valid, "relative_aligned")
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_all_invalid_returns_zero():
    pred = torch.full((3, 3), 0.10, dtype=torch.float32)
    gt = torch.full((3, 3), 999.0, dtype=torch.float32)
    valid = torch.zeros(3, 3, dtype=torch.bool)
    rmse = depth_rmse(pred, gt, valid, "metric_meters")
    assert rmse.item() == 0.0


def test_median_aligned_rmse():
    pred = torch.full((5, 5), 0.05, dtype=torch.float32)
    gt = torch.full((5, 5), 0.10, dtype=torch.float32)
    rmse = median_aligned_rmse(pred, gt)
    assert rmse.item() < 0.01


def test_geometry_metrics_report():
    pred = torch.full((4, 4), 0.10, dtype=torch.float32)
    gt = torch.full((4, 4), 0.10, dtype=torch.float32)
    report = geometry_metrics_report(pred, gt, "metric_meters")
    assert "depth_rmse_m_raw" in report
    assert "depth_mae_m_raw" in report
    assert "abs_rel" in report
    assert "depth_valid_ratio" in report
    assert "depth_semantics" in report
    assert report["depth_semantics"] == "metric_meters"


def test_report_median_aligned():
    pred = torch.full((4, 4), 0.10, dtype=torch.float32)
    gt = torch.full((4, 4), 0.10, dtype=torch.float32)
    report = geometry_metrics_report(pred, gt, "metric_meters")
    assert "median_aligned_rmse_m" in report
