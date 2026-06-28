import torch
from surgtwin.gaussian.renderer_interface import RenderOutput


def test_rgb_shape_contract():
    out = RenderOutput(rgb=torch.rand(100, 200, 3), aux={})
    assert out.rgb.shape == (100, 200, 3), f"Expected [H,W,3], got {out.rgb.shape}"
    assert out.rgb.dtype == torch.float32


def test_alpha_shape_contract():
    out = RenderOutput(rgb=torch.rand(100, 200, 3), alpha=torch.rand(100, 200), aux={})
    assert out.alpha.shape == (100, 200), f"alpha shape mismatch: {out.alpha.shape}"
    assert out.alpha.dtype == torch.float32
    assert out.alpha.min() >= 0.0
    assert out.alpha.max() <= 1.0


def test_depth_shape_contract_when_available():
    out = RenderOutput(rgb=torch.rand(100, 200, 3), depth=torch.rand(100, 200), aux={})
    assert out.depth.shape == (100, 200), f"depth shape mismatch: {out.depth.shape}"
    assert out.depth.dtype == torch.float32


def test_render_output_rgb_matches_target_shape():
    target_H, target_W = 576, 720
    fake_rgb = torch.zeros(target_H, target_W, 3)
    out = RenderOutput(rgb=fake_rgb)
    assert out.rgb.shape == (target_H, target_W, 3)
