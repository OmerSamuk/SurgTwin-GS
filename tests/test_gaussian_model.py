import torch
from surgtwin.gaussian.gaussian_model import GaussianModel


def _make_dummy_gaussians(n=100, device="cpu"):
    return GaussianModel(
        means=torch.rand(n, 3, device=device),
        scales=torch.rand(n, 3, device=device).clamp(min=1e-5),
        quats=torch.rand(n, 4, device=device),
        opacities=torch.randn(n, device=device),
        colors=torch.rand(n, 3, device=device),
        reliability_logits=torch.zeros(n, device=device),
    )


def test_num_gaussians():
    g = _make_dummy_gaussians(50)
    assert g.num_gaussians() == 50


def test_state_dict_keys():
    g = _make_dummy_gaussians(10)
    sd = g.state_dict()
    expected_keys = {"means", "scales", "quats", "opacities", "colors", "reliability_logits"}
    assert set(sd.keys()) == expected_keys


def test_state_dict_load_roundtrip():
    g = _make_dummy_gaussians(20)
    sd = g.state_dict()
    g2 = GaussianModel.load_state_dict(sd)
    for key in sd:
        assert torch.allclose(g2.state_dict()[key], sd[key]), f"Mismatch in {key}"


def test_state_dict_load_without_reliability_logits():
    sd = _make_dummy_gaussians(5).state_dict()
    del sd["reliability_logits"]
    g = GaussianModel.load_state_dict(sd)
    assert g.reliability_logits.shape[0] == 5
    assert (g.reliability_logits == 0).all()


def test_to_device():
    if not torch.cuda.is_available():
        return
    g = _make_dummy_gaussians(10)
    g_cuda = g.to("cuda")
    assert g_cuda.means.is_cuda
    assert g_cuda.scales.is_cuda
