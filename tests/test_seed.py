import torch
from surgtwin.training.seed import set_seed, DEFAULT_SEED


def test_default_seed_equals_42():
    assert DEFAULT_SEED == 42


def test_set_seed_produces_reproducible_rand():
    set_seed(42)
    a = torch.rand(10)
    set_seed(42)
    b = torch.rand(10)
    assert torch.allclose(a, b), "Same seed should produce same random values"


def test_different_seed_different_values():
    set_seed(1)
    a = torch.rand(100)
    set_seed(2)
    b = torch.rand(100)
    assert not torch.allclose(a, b), "Different seeds should produce different values"
