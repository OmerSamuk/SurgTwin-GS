import torch

from surgtwin.gaussian.gaussian_model import GaussianModel
from surgtwin.gaussian.renderer_interface import RenderOutput
from surgtwin.training.config import BaselineConfig


class _DummyBackend:
    def __init__(self):
        self.name = "dummy"

    def render(self, gaussians, camera, image_height, image_width, render_depth=True):
        return RenderOutput(
            rgb=torch.rand(image_height, image_width, 3),
            depth=None,
            alpha=torch.rand(image_height, image_width),
            aux={"depth_semantics": "unavailable"},
        )


def test_baseline_config_defaults():
    cfg = BaselineConfig()
    assert cfg.iterations == 1000
    assert cfg.enable_densification is False
    assert cfg.seed == 42
    assert cfg.backend == "gsplat"


def test_baseline_config_custom():
    cfg = BaselineConfig(iterations=500, seed=7)
    assert cfg.iterations == 500
    assert cfg.seed == 7
