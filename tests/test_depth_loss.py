import torch

from surgtwin.losses.depth import depth_l1


def _make_hw(h=10, w=10):
    gt = torch.full((h, w), 0.10, dtype=torch.float32)
    pred = torch.full((h, w), 0.10, dtype=torch.float32)
    return pred, gt


def test_identical_depth_zero_loss():
    pred, gt = _make_hw()
    loss, diag = depth_l1(pred, gt, "metric_meters")
    assert abs(loss.item()) < 1e-6
    assert "depth_loss_raw_m" in diag
    assert "depth_valid_ratio" in diag


def test_known_l1_error():
    pred = torch.full((5, 5), 0.10, dtype=torch.float32)
    gt = torch.full((5, 5), 0.12, dtype=torch.float32)
    loss, diag = depth_l1(pred, gt, "metric_meters")
    assert abs(loss.item() - 0.02) < 1e-6
    assert abs(diag["depth_rmse_m_raw"].item() - 0.02) < 1e-6


def test_invalid_gt_mask_excluded():
    pred = torch.full((5, 5), 0.10, dtype=torch.float32)
    gt = torch.full((5, 5), 999.0, dtype=torch.float32)
    loss, diag = depth_l1(pred, gt, "metric_meters")
    assert loss.item() == 0.0
    assert diag["depth_valid_ratio"].item() == 0.0


def test_partial_invalid_mask():
    pred = torch.full((4, 4), 0.10, dtype=torch.float32)
    gt = torch.full((4, 4), 0.10, dtype=torch.float32)
    gt[0, :] = 999.0
    gt[:, 0] = -1.0
    loss, diag = depth_l1(pred, gt, "metric_meters")
    assert loss.item() >= 0.0
    assert torch.isfinite(torch.tensor(loss.item()))
    assert diag["depth_valid_ratio"].item() < 1.0


def test_non_metric_depth_raises():
    pred, gt = _make_hw()
    try:
        depth_l1(pred, gt, "relative_unaligned")
        assert False, "Expected ValueError for non-metric depth"
    except ValueError:
        pass


def test_all_invalid_returns_zero():
    pred = torch.full((5, 5), 0.10, dtype=torch.float32)
    gt = torch.full((5, 5), -1.0, dtype=torch.float32)
    loss, diag = depth_l1(pred, gt, "metric_meters")
    assert loss.item() == 0.0
    assert diag["depth_rmse_m_raw"].item() == 0.0
    assert diag["depth_mae_m_raw"].item() == 0.0


def test_pred_zero_not_masked():
    pred = torch.zeros(5, 5, dtype=torch.float32)
    gt = torch.full((5, 5), 0.10, dtype=torch.float32)
    loss, diag = depth_l1(pred, gt, "metric_meters")
    assert loss.item() > 0.0
    assert diag["depth_valid_ratio"].item() > 0.0


def test_shape_mismatch():
    try:
        depth_l1(torch.rand(3, 3), torch.rand(5, 5), "metric_meters")
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_clamp_effect():
    pred = torch.full((3, 3), 0.50, dtype=torch.float32)
    gt = torch.full((3, 3), 0.10, dtype=torch.float32)
    loss_clamped, diag = depth_l1(pred, gt, "metric_meters", near_m=0.02, far_m=0.30)
    assert abs(diag["depth_rmse_m_raw"].item() - 0.40) < 1e-5
    assert abs(diag["depth_rmse_m_clipped"].item() - 0.20) < 1e-5


def test_diagnostics_contain_all_keys():
    pred, gt = _make_hw()
    _, diag = depth_l1(pred, gt, "metric_meters")
    expected_keys = [
        "depth_loss_raw_m", "depth_loss_weighted",
        "depth_rmse_m_raw", "depth_rmse_m_clipped",
        "depth_mae_m_raw", "depth_mae_m_clipped",
        "abs_rel", "depth_valid_ratio",
    ]
    for k in expected_keys:
        assert k in diag, f"Missing key: {k}"
