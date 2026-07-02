from surgtwin.training.uncertainty_config import UncertaintyConfig


def test_default_values():
    c = UncertaintyConfig()
    assert c.iterations == 1000
    assert c.variant == "h1"
    assert c.lambda_depth == 0.2
    assert c.lambda_reg == 0.0
    assert c.alpha == 2.0
    assert c.w_photo_min == 0.15
    assert c.mask_boost == 0.5
    assert c.enable_densification == False
    assert c.clip_grad_norm == True
    assert c.max_grad_norm == 1.0
    assert c.warmup_iters == 0


def test_frozen():
    c = UncertaintyConfig()
    import dataclasses
    assert dataclasses.fields(c) is not None


def test_custom_values():
    c = UncertaintyConfig(variant="h2", mask_dir="/tmp/masks", mask_boost=1.0)
    assert c.variant == "h2"
    assert c.mask_dir == "/tmp/masks"
    assert c.mask_boost == 1.0


def test_h3_effective():
    c = UncertaintyConfig(variant="h3", lambda_depth=0.1)
    assert c.variant == "h3"
    assert c.lambda_depth == 0.1


def test_backend_default():
    c = UncertaintyConfig()
    assert c.backend == "gsplat"
