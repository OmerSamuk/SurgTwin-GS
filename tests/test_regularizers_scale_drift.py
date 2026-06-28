import torch

from surgtwin.losses.regularizers import scale_drift_regularizer, REGISTRY


def test_identical_scales_zero_loss():
    scales = torch.rand(100, 3) * 0.01 + 1e-5
    init = scales.clone()
    loss = scale_drift_regularizer(scales, init)
    assert abs(loss.item()) < 1e-6


def test_different_scales_positive_loss():
    scales = torch.full((10, 3), 0.01, dtype=torch.float32)
    init = torch.full((10, 3), 0.001, dtype=torch.float32)
    loss = scale_drift_regularizer(scales, init)
    assert loss.item() > 0.0
    assert torch.isfinite(loss)


def test_large_drift_larger_loss():
    small = torch.full((5, 3), 0.01, dtype=torch.float32)
    large = torch.full((5, 3), 0.001, dtype=torch.float32)
    loss_small = scale_drift_regularizer(torch.full((5, 3), 0.005), small)
    loss_large = scale_drift_regularizer(torch.full((5, 3), 0.05), small)
    assert loss_large.item() > loss_small.item()


def test_shape_mismatch():
    try:
        scale_drift_regularizer(torch.rand(5, 3), torch.rand(10, 3))
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_eps_guard():
    scales = torch.zeros(10, 3)
    init = torch.zeros(10, 3)
    loss = scale_drift_regularizer(scales, init, eps=1e-8)
    assert torch.isfinite(loss)


def test_registry_contains_scale_drift():
    assert "scale_drift" in REGISTRY
    assert REGISTRY["scale_drift"] is scale_drift_regularizer
