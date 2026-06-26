import torch
from surgtwin.gaussian.renderer_interface import RenderOutput, RendererBackend


def test_render_output_fields():
    rgb = torch.rand(100, 200, 3)
    depth = torch.rand(100, 200)
    alpha = torch.rand(100, 200)

    out = RenderOutput(rgb=rgb, depth=depth, alpha=alpha, aux={"key": "val"})
    assert out.rgb.shape == (100, 200, 3)
    assert out.depth.shape == (100, 200)
    assert out.alpha.shape == (100, 200)
    assert out.aux["key"] == "val"


def test_render_output_defaults():
    rgb = torch.rand(50, 50, 3)
    out = RenderOutput(rgb=rgb)
    assert out.depth is None
    assert out.alpha is None
    assert out.aux == {}


def test_renderer_backend_abstract():
    class TestBackend(RendererBackend):
        @property
        def name(self):
            return "test"

        def render(self, gaussians, camera, image_height, image_width, render_depth=True):
            return RenderOutput(rgb=torch.rand(image_height, image_width, 3))

    backend = TestBackend()
    assert backend.name == "test"
    out = backend.render(None, None, 10, 20)
    assert out.rgb.shape == (10, 20, 3)
