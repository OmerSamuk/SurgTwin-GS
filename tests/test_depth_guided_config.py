from surgtwin.training.depth_guided_config import DepthGuidedConfig


def test_default_lambda_values():
    cfg = DepthGuidedConfig()
    assert cfg.lambda_depth == 0.2
    assert cfg.lambda_reg == 0.01
    assert cfg.reg_type == "scale_drift"


def test_default_depth_guards():
    cfg = DepthGuidedConfig()
    assert cfg.depth_near_m == 0.02
    assert cfg.depth_far_m == 0.30


def test_default_grad_clip():
    cfg = DepthGuidedConfig()
    assert cfg.clip_grad_norm is True
    assert cfg.max_grad_norm == 1.0


def test_default_densification_off():
    cfg = DepthGuidedConfig()
    assert cfg.enable_densification is False


def test_frozen_dataclass():
    cfg = DepthGuidedConfig(lambda_depth=0.5)
    assert cfg.lambda_depth == 0.5
    try:
        cfg.lambda_depth = 0.3
        assert False, "Expected dataclass frozen error"
    except AttributeError:
        pass


def test_override_values():
    cfg = DepthGuidedConfig(
        lambda_depth=0.1,
        lambda_reg=0.005,
        reg_type="scale_drift",
        enable_densification=True,
        clip_grad_norm=False,
    )
    assert cfg.lambda_depth == 0.1
    assert cfg.lambda_reg == 0.005
    assert cfg.enable_densification is True
    assert cfg.clip_grad_norm is False


def test_all_required_fields():
    cfg = DepthGuidedConfig()
    expected = [
        "iterations", "lambda_depth", "lambda_reg", "reg_type",
        "depth_near_m", "depth_far_m", "clip_grad_norm", "max_grad_norm",
        "enable_densification", "depth_semantics_artifact_path",
    ]
    for field in expected:
        assert hasattr(cfg, field), f"Missing field: {field}"
