import torch

from surgtwin.training.uncertainty_config import UncertaintyConfig


def test_warmup_iters_default():
    c = UncertaintyConfig()
    assert c.warmup_iters == 0


def test_warmup_iters_custom():
    c = UncertaintyConfig(warmup_iters=200)
    assert c.warmup_iters == 200


def test_warmup_factor_at_iter1():
    warmup = 200
    factor = min(1.0, 1 / warmup)
    assert abs(factor - 0.005) < 1e-6


def test_warmup_factor_at_iter_full():
    warmup = 200
    factor = min(1.0, 200 / warmup)
    assert factor == 1.0


def test_warmup_factor_beyond():
    warmup = 200
    factor = min(1.0, 500 / warmup)
    assert factor == 1.0


def test_warmup_zero():
    warmup = 0
    assert warmup == 0


def test_warmup_grad_norm_config():
    c = UncertaintyConfig(warmup_iters=200, max_grad_norm=1.5)
    assert c.warmup_iters == 200
    assert c.max_grad_norm == 1.5
