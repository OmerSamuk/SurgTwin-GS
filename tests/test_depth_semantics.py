from surgtwin.gaussian.renderer_interface import RenderOutput


def test_depth_semantics_recorded():
    out = RenderOutput(
        rgb=__import__("torch").rand(10, 10, 3),
        aux={"depth_semantics": "metric_meters", "backend": "gsplat"},
    )
    assert out.aux["depth_semantics"] in ("metric_meters", "relative_aligned", "relative_unaligned", "unavailable")


def test_depth_semantics_non_metric_rejected():
    out = RenderOutput(
        rgb=__import__("torch").rand(10, 10, 3),
        depth=None,
        aux={"depth_semantics": "unavailable"},
    )
    assert out.depth is None
    assert out.aux["depth_semantics"] == "unavailable"
