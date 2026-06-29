import torch
import pytest

from surgtwin.losses.uncertainty_weighted import uncertainty_weighted_photometric_l1


def test_loss_shape():
    pred = torch.rand(4, 4, 3)
    gt = torch.rand(4, 4, 3)
    loss, diag = uncertainty_weighted_photometric_l1(pred, gt)
    assert loss.numel() == 1
    assert loss > 0


def test_loss_diagnostics_keys():
    pred = torch.rand(4, 4, 3)
    gt = torch.rand(4, 4, 3)
    loss, diag = uncertainty_weighted_photometric_l1(pred, gt)
    for key in ("w_photo_mean", "w_photo_min", "w_photo_max",
                 "w_photo_p10", "w_photo_p50", "w_photo_p90",
                 "fraction_w_photo_at_min", "fraction_w_photo_at_one",
                 "w_photo_p90_minus_p10", "u_photo_mean",
                 "p95_scale", "normalization_mode", "mask_used"):
        assert key in diag


def test_loss_gradient_detached():
    pred = torch.rand(4, 4, 3, requires_grad=True)
    gt = torch.rand(4, 4, 3)
    loss, diag = uncertainty_weighted_photometric_l1(pred, gt)
    assert loss.requires_grad
    loss.backward()
    assert pred.grad is not None


def test_loss_with_mask():
    pred = torch.rand(4, 4, 3)
    gt = torch.rand(4, 4, 3)
    mask = torch.zeros(4, 4, dtype=torch.bool)
    mask[0, :] = True
    loss_with, diag_with = uncertainty_weighted_photometric_l1(pred, gt, mask=mask)
    loss_without, diag_without = uncertainty_weighted_photometric_l1(pred, gt, mask=None)
    assert loss_with.numel() == 1
    assert loss_without.numel() == 1
    assert diag_with["mask_used"] == True
    assert diag_without["mask_used"] == False


def test_loss_mask_diagnostics():
    pred = torch.rand(4, 4, 3)
    gt = torch.rand(4, 4, 3)
    mask = torch.zeros(4, 4, dtype=torch.bool)
    mask[0:2, :] = True
    loss, diag = uncertainty_weighted_photometric_l1(pred, gt, mask=mask)
    assert "w_photo_in_mask_mean" in diag
    assert "w_photo_out_mask_mean" in diag
    assert "mask_coverage" in diag


def test_loss_no_collapse_all_min():
    pred = torch.rand(4, 4, 3) * 0.5 + 0.25
    gt = torch.rand(4, 4, 3) * 0.5
    loss, diag = uncertainty_weighted_photometric_l1(pred, gt, w_min=0.15)
    assert diag["fraction_w_photo_at_min"] < 0.90


def test_loss_collapse_all_one():
    pred = torch.rand(4, 4, 3)
    gt = pred.clone()
    loss, diag = uncertainty_weighted_photometric_l1(pred, gt, w_min=0.15)
    assert diag["fraction_w_photo_at_one"] > 0.90
    assert diag["w_photo_p90_minus_p10"] < 0.05


def test_loss_normalization_mode():
    pred = torch.rand(4, 4, 3)
    gt = torch.rand(4, 4, 3)
    loss, diag = uncertainty_weighted_photometric_l1(pred, gt)
    assert diag["normalization_mode"] == "p95_detached"


def test_loss_shape_mismatch():
    with pytest.raises(ValueError, match="Shape mismatch"):
        uncertainty_weighted_photometric_l1(torch.rand(4, 4, 3), torch.rand(4, 3, 3))


def test_loss_is_normalized_weighted_mean():
    pred = torch.zeros(4, 4, 3)
    gt = torch.ones(4, 4, 3)
    loss, diag = uncertainty_weighted_photometric_l1(pred, gt, w_min=0.15)
    uniform_w = diag.get("w_photo_mean", 0)
    expected = 1.0
    assert loss.item() == pytest.approx(expected, abs=0.02)


def test_loss_eps_protects_zero_weight_division():
    pred = torch.ones(4, 4, 3)
    gt = torch.ones(4, 4, 3)
    loss, diag = uncertainty_weighted_photometric_l1(pred, gt, w_min=0.15)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_loss_equal_photometric_when_weights_one():
    pred = torch.rand(4, 4, 3)
    gt = torch.rand(4, 4, 3)
    loss, diag = uncertainty_weighted_photometric_l1(pred, gt, w_min=0.15)
    from surgtwin.losses.photometric import photometric_l1
    ref = photometric_l1(pred, gt)
    diff = abs(loss.item() - ref.item())
    assert diff < 1.0, f"Weighted loss {loss.item():.4f} should be near unweighted {ref.item():.4f}"
