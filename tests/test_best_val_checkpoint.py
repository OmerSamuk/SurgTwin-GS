from dataclasses import asdict
from pathlib import Path

import torch

from surgtwin.gaussian.gaussian_model import GaussianModel
from surgtwin.training.checkpointing import load_checkpoint
from surgtwin.training.uncertainty_config import UncertaintyConfig


def _make_gaussians(n=100, device="cpu"):
    return GaussianModel(
        means=torch.randn(n, 3, device=device) * 0.02 + torch.tensor([0.0, 0.0, 0.15], device=device),
        scales=torch.full((n, 3), 0.005, device=device),
        quats=torch.rand(n, 4, device=device),
        opacities=torch.randn(n, device=device),
        colors=torch.rand(n, 3, device=device),
        reliability_logits=torch.zeros(n, device=device),
    )


def _make_config(**overrides):
    kwargs = dict(
        best_val_enabled=True,
        best_val_metric="depth_rmse",
        best_val_tiebreaker="psnr",
        best_val_metric_mode="min",
        best_val_tiebreaker_mode="max",
        enable_densification=False,
        seed=42,
    )
    kwargs.update(overrides)
    return UncertaintyConfig(**kwargs)


def _make_optimizer(gaussians, lr=1e-3):
    from torch.optim import Adam
    gd = {k: v for k, v in gaussians.__dict__.items() if isinstance(v, torch.Tensor)}
    return Adam([
        {"params": [gd["means"]], "lr": lr},
        {"params": [gd["scales"]], "lr": lr},
        {"params": [gd["quats"]], "lr": lr},
        {"params": [gd["opacities"]], "lr": lr},
        {"params": [gd["colors"]], "lr": lr},
    ])


def _fake_val_metrics(psnr=20.0, depth_rmse=0.04, ssim=0.8, lpips=0.3, abs_rel=0.05):
    return {
        "val_psnr": psnr,
        "val_depth_rmse_m_raw": depth_rmse,
        "val_ssim": ssim,
        "val_lpips": lpips,
        "val_abs_rel": abs_rel,
    }


class FakeTrainer:
    """Minimal trainer stub to exercise _maybe_save_best_val logic."""
    def __init__(self, config: UncertaintyConfig, gaussians: GaussianModel, output_dir: Path):
        self.unc_config = config
        self.gaussians = gaussians
        self.output_dir = output_dir
        self.optimizer = _make_optimizer(gaussians)
        self._best_val = None
        self._dens_steps_count = 0
        self._total_cloned = 0
        self._total_pruned = 0

    def _resolve_val_metric(self, metric_name: str) -> str:
        mapping = {"depth_rmse": "val_depth_rmse_m_raw", "psnr": "val_psnr"}
        return mapping.get(metric_name, metric_name)

    def _maybe_save_best_val(self, iter_idx, val_metrics):
        if not self.unc_config.best_val_enabled:
            return
        metric_key = self._resolve_val_metric(self.unc_config.best_val_metric)
        tie_key = self._resolve_val_metric(self.unc_config.best_val_tiebreaker)
        metric_value = val_metrics.get(metric_key)
        tie_value = val_metrics.get(tie_key)
        if metric_value is None:
            return
        if self._best_val is None:
            improved = True
        else:
            if self.unc_config.best_val_metric_mode == "min":
                improved = metric_value < self._best_val["metric_value"]
            else:
                improved = metric_value > self._best_val["metric_value"]
            if not improved and metric_value == self._best_val["metric_value"]:
                if self.unc_config.best_val_tiebreaker_mode == "max":
                    improved = tie_value > self._best_val.get("tie_value", float("-inf"))
                else:
                    improved = tie_value < self._best_val.get("tie_value", float("inf"))
        if improved:
            self._best_val = {
                "iter": iter_idx,
                "metric_value": metric_value,
                "tie_value": tie_value,
                "val_metrics": val_metrics,
            }
            from surgtwin.training.checkpointing import save_checkpoint
            save_checkpoint(
                path=self.output_dir / "checkpoints" / "best_val.pt",
                gaussians=self.gaussians,
                optimizer_state_dict=self.optimizer.state_dict(),
                iteration=iter_idx,
                config=asdict(self.unc_config),
                backend_name=self.unc_config.backend,
                seed=self.unc_config.seed,
                extra={
                    "best_val_metric": self.unc_config.best_val_metric,
                    "best_val_metric_mode": self.unc_config.best_val_metric_mode,
                    "best_val_score": metric_value,
                    "val_metrics": val_metrics,
                },
            )
            best_meta = {
                "best_iter": iter_idx,
                "best_val_metric": self.unc_config.best_val_metric,
                "best_val_metric_mode": self.unc_config.best_val_metric_mode,
                "best_val_metric_value": metric_value,
                "best_val_tiebreaker": self.unc_config.best_val_tiebreaker,
                "best_val_tiebreaker_value": tie_value,
                "val_psnr": val_metrics.get("val_psnr"),
                "val_depth_rmse_m_raw": val_metrics.get("val_depth_rmse_m_raw"),
                "val_ssim": val_metrics.get("val_ssim"),
                "val_lpips": val_metrics.get("val_lpips"),
                "val_abs_rel": val_metrics.get("val_abs_rel"),
                "n_gaussians": self.gaussians.num_gaussians(),
                "enable_densification": self.unc_config.enable_densification,
            }
            import json
            (self.output_dir / "best_val_metrics.json").write_text(json.dumps(best_meta))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_best_val_selects_lower_depth_rmse_when_depth_primary(tmp_path):
    config = _make_config(best_val_metric="depth_rmse", best_val_metric_mode="min")
    g = _make_gaussians()
    t = FakeTrainer(config, g, tmp_path)

    t._maybe_save_best_val(100, _fake_val_metrics(depth_rmse=0.05))
    assert t._best_val["iter"] == 100
    assert t._best_val["metric_value"] == 0.05

    t._maybe_save_best_val(200, _fake_val_metrics(depth_rmse=0.03))
    assert t._best_val["iter"] == 200
    assert t._best_val["metric_value"] == 0.03


def test_best_val_selects_higher_psnr_when_psnr_primary(tmp_path):
    config = _make_config(best_val_metric="psnr", best_val_metric_mode="max")
    g = _make_gaussians()
    t = FakeTrainer(config, g, tmp_path)

    t._maybe_save_best_val(100, _fake_val_metrics(psnr=18.0))
    assert t._best_val["iter"] == 100

    t._maybe_save_best_val(200, _fake_val_metrics(psnr=20.0))
    assert t._best_val["iter"] == 200


def test_best_val_uses_psnr_tiebreaker_when_depth_equal(tmp_path):
    config = _make_config(best_val_metric="depth_rmse", best_val_metric_mode="min",
                           best_val_tiebreaker="psnr", best_val_tiebreaker_mode="max")
    g = _make_gaussians()
    t = FakeTrainer(config, g, tmp_path)

    t._maybe_save_best_val(100, _fake_val_metrics(depth_rmse=0.04, psnr=19.0))
    assert t._best_val["iter"] == 100

    t._maybe_save_best_val(200, _fake_val_metrics(depth_rmse=0.04, psnr=20.5))
    assert t._best_val["iter"] == 200, "tiebreaker should select higher PSNR"


def test_best_val_checkpoint_contains_optimizer_state(tmp_path):
    config = _make_config()
    g = _make_gaussians()
    t = FakeTrainer(config, g, tmp_path)

    t._maybe_save_best_val(150, _fake_val_metrics())

    ckpt = load_checkpoint(tmp_path / "checkpoints" / "best_val.pt")
    assert "optimizer_state_dict" in ckpt
    assert "gaussian_state" in ckpt
    assert "means" in ckpt["gaussian_state"]
    assert ckpt["iteration"] == 150


def test_best_val_metrics_json_written(tmp_path):
    config = _make_config()
    g = _make_gaussians()
    t = FakeTrainer(config, g, tmp_path)

    t._maybe_save_best_val(200, _fake_val_metrics(depth_rmse=0.035, psnr=20.5))

    meta_path = tmp_path / "best_val_metrics.json"
    assert meta_path.exists()
    import json
    meta = json.loads(meta_path.read_text())
    assert meta["best_iter"] == 200
    assert meta["best_val_metric_value"] == 0.035
    assert meta["best_val_tiebreaker_value"] == 20.5
    assert "val_psnr" in meta
    assert "val_depth_rmse_m_raw" in meta


def test_best_val_does_not_update_when_metric_worse(tmp_path):
    config = _make_config(best_val_metric="depth_rmse", best_val_metric_mode="min")
    g = _make_gaussians()
    t = FakeTrainer(config, g, tmp_path)

    t._maybe_save_best_val(100, _fake_val_metrics(depth_rmse=0.03))
    assert t._best_val["iter"] == 100

    t._maybe_save_best_val(200, _fake_val_metrics(depth_rmse=0.05))
    assert t._best_val["iter"] == 100, "should NOT update with worse metric"


def test_best_val_handles_missing_depth_metric_gracefully(tmp_path):
    config = _make_config(best_val_metric="depth_rmse")
    g = _make_gaussians()
    t = FakeTrainer(config, g, tmp_path)

    val_metrics = _fake_val_metrics()
    del val_metrics["val_depth_rmse_m_raw"]

    t._maybe_save_best_val(100, val_metrics)
    assert t._best_val is None, "should not update when metric missing"
