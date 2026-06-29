import pytest

from surgtwin.training.uncertainty_config import UncertaintyConfig
from surgtwin.training.uncertainty_trainer import UncertaintyTrainer


def test_config_attributes():
    c = UncertaintyConfig(variant="h2", mask_boost=0.7)
    assert c.variant == "h2"
    assert c.mask_boost == 0.7


def test_config_h1_variant():
    c = UncertaintyConfig(variant="h1")
    assert c.variant == "h1"
    assert c.lambda_reg == 0.0


def test_config_h3_variant():
    c = UncertaintyConfig(variant="h3")
    assert c.variant == "h3"


def test_config_mutable_defaults():
    c1 = UncertaintyConfig()
    c2 = UncertaintyConfig(mask_dir="/some/path")
    assert c1.mask_dir is None
    assert c2.mask_dir == "/some/path"


def test_config_lambda_reg_locked():
    c = UncertaintyConfig()
    assert c.lambda_reg == 0.0


@pytest.mark.skipif(not pytest.importorskip("torch").cuda.is_available(), reason="CUDA required")
def test_trainer_cuda_check():
    import torch
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    from surgtwin.training.uncertainty_trainer import UncertaintyTrainer
    assert UncertaintyTrainer is not None
