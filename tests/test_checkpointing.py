from pathlib import Path

import torch

from surgtwin.gaussian.gaussian_model import GaussianModel
from surgtwin.training.checkpointing import save_checkpoint, load_checkpoint, load_gaussians_from_checkpoint


def _make_dummy_gaussians(n=50):
    return GaussianModel(
        means=torch.rand(n, 3),
        scales=torch.rand(n, 3).clamp(min=1e-5),
        quats=torch.rand(n, 4),
        opacities=torch.randn(n),
        colors=torch.rand(n, 3),
        reliability_logits=torch.zeros(n),
    )


def test_save_load_checkpoint_roundtrip(tmp_path: Path):
    g = _make_dummy_gaussians(50)
    optim_state = {"param_groups": [{"lr": 0.001}], "state": {}}
    config = {"iterations": 1000, "seed": 42}

    ckpt_path = tmp_path / "test_ckpt.pt"
    save_checkpoint(ckpt_path, g, optim_state, iteration=500, config=config, backend_name="gsplat", seed=42)

    assert ckpt_path.exists()

    loaded = load_checkpoint(ckpt_path)
    assert loaded["iteration"] == 500
    assert loaded["backend_name"] == "gsplat"
    assert loaded["config"]["iterations"] == 1000
    assert loaded["seed"] == 42

    g_loaded = load_gaussians_from_checkpoint(ckpt_path)
    assert g_loaded.num_gaussians() == 50
    for key in g.state_dict():
        assert torch.allclose(g_loaded.state_dict()[key], g.state_dict()[key])


def test_save_checkpoint_extra(tmp_path: Path):
    g = _make_dummy_gaussians(10)
    ckpt_path = tmp_path / "extra.pt"
    save_checkpoint(ckpt_path, g, {}, iteration=0, config={}, backend_name="gsplat", extra={"note": "test"})
    data = load_checkpoint(ckpt_path)
    assert data["extra"]["note"] == "test"
