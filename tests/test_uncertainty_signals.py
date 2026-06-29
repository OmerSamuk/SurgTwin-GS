import torch
import pytest

from surgtwin.uncertainty.signals import (
    compute_photo_residual,
    compute_p95_scale,
    compute_u_photo,
    compute_w_photo,
    compute_w_photo_with_mask,
    w_photo_distribution_stats,
)


def test_compute_photo_residual_shape():
    pred = torch.rand(4, 4, 3)
    gt = torch.rand(4, 4, 3)
    residual = compute_photo_residual(pred, gt)
    assert residual.shape == (4, 4)
    assert residual.dtype == torch.float32


def test_compute_photo_residual_detach():
    pred = torch.rand(4, 4, 3, requires_grad=True)
    gt = torch.rand(4, 4, 3)
    residual = compute_photo_residual(pred, gt, detach_pred=True)
    assert not residual.requires_grad
    loss = residual.mean()
    assert loss.grad_fn is None


def test_compute_photo_residual_no_detach():
    pred = torch.rand(4, 4, 3, requires_grad=True)
    gt = torch.rand(4, 4, 3)
    residual = compute_photo_residual(pred, gt, detach_pred=False)
    assert residual.requires_grad


def test_compute_photo_residual_shape_mismatch():
    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_photo_residual(torch.rand(4, 4, 3), torch.rand(4, 3, 3))


def test_compute_photo_residual_zero():
    pred = torch.ones(2, 2, 3)
    gt = torch.ones(2, 2, 3)
    residual = compute_photo_residual(pred, gt)
    assert residual.abs().max() < 1e-6


def test_p95_scale_basic():
    residual = torch.tensor([0.0, 0.1, 0.2, 0.3, 0.5, 1.0, 0.6, 0.4, 0.05, 0.15,
                              0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 0.99, 0.999])
    scale = compute_p95_scale(residual)
    assert scale > 0
    assert scale.item() < 1.0


def test_p95_scale_all_zero():
    residual = torch.zeros(10, 10)
    scale = compute_p95_scale(residual, eps=1e-4)
    assert scale.item() == pytest.approx(1e-4, abs=1e-5)


def test_p95_scale_detached():
    residual = torch.rand(10, requires_grad=True)
    scale = compute_p95_scale(residual)
    assert not scale.requires_grad


def test_p95_scale_eps_clamp():
    residual = torch.full((100,), 1e-8)
    scale = compute_p95_scale(residual, eps=1e-4)
    assert scale.item() == pytest.approx(1e-4, abs=1e-5)


def test_u_photo_range():
    residual = torch.rand(8, 8)
    scale = torch.tensor(0.5)
    u = compute_u_photo(residual, scale)
    assert u.min() >= 0.0
    assert u.max() <= 1.0


def test_w_photo_range():
    u = torch.tensor([[0.0, 0.5, 1.0]])
    w = compute_w_photo(u, alpha=2.0, w_min=0.15)
    assert w[0, 0] == 1.0
    assert w[0, 2] == 0.15
    assert w[0, 1] >= 0.15 and w[0, 1] <= 1.0
    assert w.min() >= 0.15
    assert w.max() <= 1.0


def test_w_photo_monotonic():
    u = torch.linspace(0, 1, 20)
    w = compute_w_photo(u, alpha=2.0, w_min=0.15)
    diffs = w[1:] - w[:-1]
    assert (diffs <= 0).all()


def test_w_photo_with_mask_shape():
    u = torch.rand(4, 4)
    mask = torch.rand(4, 4) > 0.5
    w = compute_w_photo_with_mask(u, mask)
    assert w.shape == (4, 4)
    assert w.min() >= 0.15


def test_w_photo_with_mask_boost():
    u = torch.zeros(4, 4)
    mask = torch.zeros(4, 4, dtype=torch.bool)
    mask[0, 0] = True
    w = compute_w_photo_with_mask(u, mask, alpha=2.0, w_min=0.15, mask_boost=0.5)
    assert w[0, 0] < w[0, 1]
    assert w[0, 1] == 1.0


def test_w_photo_with_mask_none():
    u = torch.rand(4, 4)
    w = compute_w_photo_with_mask(u, None)
    assert w.shape == (4, 4)


def test_w_photo_with_mask_shape_mismatch():
    with pytest.raises(ValueError, match="Mask shape"):
        compute_w_photo_with_mask(torch.rand(4, 4), torch.rand(3, 3))


def test_w_photo_distribution_stats():
    w = torch.ones(10, 10)
    stats = w_photo_distribution_stats(w, w_min=0.15)
    assert stats["w_photo_mean"] == 1.0
    assert stats["w_photo_min"] == 1.0
    assert stats["w_photo_max"] == 1.0
    assert stats["fraction_w_photo_at_min"] == 0.0
    assert stats["fraction_w_photo_at_one"] == 1.0


def test_w_photo_distribution_stats_all_min():
    w = torch.full((10, 10), 0.15)
    stats = w_photo_distribution_stats(w, w_min=0.15)
    assert stats["fraction_w_photo_at_min"] == 1.0
    assert stats["fraction_w_photo_at_one"] == 0.0


def test_w_photo_distribution_stats_mixed():
    w = torch.linspace(0.15, 1.0, 100).reshape(10, 10)
    stats = w_photo_distribution_stats(w, w_min=0.15)
    assert stats["w_photo_p90_minus_p10"] >= 0.2
