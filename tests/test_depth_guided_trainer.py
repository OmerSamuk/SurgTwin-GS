import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from surgtwin.gaussian.gaussian_model import GaussianModel
from surgtwin.gaussian.renderer_interface import RenderOutput
from surgtwin.training.depth_guided_config import DepthGuidedConfig
from surgtwin.training.depth_guided_trainer import (
    DepthGuidedTrainer,
    _verify_m2a_artifact,
)


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _MetricDepthBackend:
    def __init__(self):
        self.name = "metric_dummy"

    def render(self, gaussians, camera, image_height, image_width, render_depth=True):
        dev = _device()
        return RenderOutput(
            rgb=torch.rand(image_height, image_width, 3, device=dev),
            depth=torch.rand(image_height, image_width, device=dev) * 0.10 + 0.05,
            alpha=torch.rand(image_height, image_width, device=dev),
            aux={
                "depth_semantics": "metric_meters",
                "supports_metric_depth": True,
            },
        )


class _NonMetricBackend:
    def __init__(self):
        self.name = "non_metric_dummy"

    def render(self, gaussians, camera, image_height, image_width, render_depth=True):
        dev = _device()
        return RenderOutput(
            rgb=torch.rand(image_height, image_width, 3, device=dev),
            depth=torch.rand(image_height, image_width, device=dev) * 0.10 + 0.05,
            alpha=torch.rand(image_height, image_width, device=dev),
            aux={
                "depth_semantics": "relative_unaligned",
                "supports_metric_depth": False,
            },
        )


def _make_fake_artifact(tmp_path: Path, verified: bool = True) -> Path:
    path = tmp_path / "artifact.json"
    path.write_text(json.dumps({"depth_semantics_verified": verified, "m2a_gate": "PASS" if verified else "FAIL"}))
    return path


def _make_fake_sample(tmp_path: Path, h: int = 32, w: int = 32) -> dict:
    rgb_np = (np.random.rand(h, w, 3) * 255).astype(np.uint8)
    rgb_path = tmp_path / "rgb.png"
    Image.fromarray(rgb_np).save(rgb_path)

    depth_raw = np.random.randint(100, 30000, (h, w), dtype=np.uint16)
    depth_gt_path = tmp_path / "depth.png"
    Image.fromarray(depth_raw, mode="I;16").save(depth_gt_path)

    return {
        "sample_id": "test_000",
        "sequence_id": "test",
        "frame_index": 0,
        "left_rgb_path": str(rgb_path),
        "right_rgb_path": str(rgb_path),
        "left_depth_path": str(depth_gt_path),
        "right_depth_path": None,
        "K_left": [[100.0, 0.0, 16.0], [0.0, 100.0, 16.0], [0.0, 0.0, 1.0]],
        "K_right": [[100.0, 0.0, 16.0], [0.0, 100.0, 16.0], [0.0, 0.0, 1.0]],
        "c2w_left": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        "w2c_left": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        "c2w_right": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        "w2c_right": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        "height": h,
        "width": w,
        "depth_unit": "meter",
        "depth_scale_applied": 0.000256,
        "split": "train",
    }


def test_m2a_artifact_present_and_true(tmp_path):
    path = _make_fake_artifact(tmp_path, verified=True)
    _verify_m2a_artifact(path)
    assert True


def test_m2a_artifact_missing(tmp_path):
    fake = tmp_path / "nonexistent.json"
    try:
        _verify_m2a_artifact(fake)
        assert False, "Expected RuntimeError"
    except RuntimeError:
        pass


def test_m2a_artifact_false(tmp_path):
    path = _make_fake_artifact(tmp_path, verified=False)
    try:
        _verify_m2a_artifact(path)
        assert False, "Expected RuntimeError"
    except RuntimeError:
        pass


def test_m2a_artifact_none():
    try:
        _verify_m2a_artifact(None)
        assert False, "Expected RuntimeError"
    except RuntimeError:
        pass


class TestDepthGuidedTrainerMethods:
    def _make_trainer(self, tmp_path, backend=None, config_override=None):
        sample = _make_fake_sample(tmp_path)
        if backend is None:
            backend = _MetricDepthBackend()

        cfg_kwargs = {
            "iterations": 2,
            "log_every": 1,
            "val_every": 10,
            "ckpt_every": 10,
            "init_num_points": 100,
            "depth_semantics_artifact_path": str(_make_fake_artifact(tmp_path, verified=True)),
            "clip_grad_norm": True,
            "max_grad_norm": 1.0,
        }
        if config_override:
            cfg_kwargs.update(config_override)
        cfg = DepthGuidedConfig(**cfg_kwargs)

        trainer = DepthGuidedTrainer(
            train_entries=[sample],
            val_entries=[sample],
            backend=backend,
            config=cfg,
            output_dir=tmp_path / "output",
        )
        return trainer, cfg

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Gaussian init")
    def test_init_scale_snapshot(self, tmp_path):
        trainer, _ = self._make_trainer(tmp_path)
        trainer._check_m2a_gate()
        trainer.setup()
        assert trainer.init_scales is not None
        assert trainer.init_scales.shape == trainer.gaussians.scales.shape

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Gaussian init")
    def test_non_metric_depth_hard_fail(self, tmp_path):
        trainer, _ = self._make_trainer(tmp_path, backend=_NonMetricBackend())
        trainer._check_m2a_gate()
        trainer.setup()
        try:
            trainer.train_step(1)
            assert False, "Expected RuntimeError for non-metric depth"
        except RuntimeError:
            pass

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Gaussian init")
    def test_grad_norm_logged(self, tmp_path):
        trainer, cfg = self._make_trainer(tmp_path)
        trainer._check_m2a_gate()
        trainer.setup()
        step = trainer.train_step(1)
        assert "grad_norm_before_clip" in step
        assert "grad_norm_after_clip" in step
        assert step["grad_norm_before_clip"] >= 0.0
        assert step["grad_norm_after_clip"] >= 0.0

    def test_scale_drift_config_allowed(self):
        cfg = DepthGuidedConfig(reg_type="scale_drift")
        assert cfg.reg_type == "scale_drift"

    def test_densification_disabled_by_default(self):
        cfg = DepthGuidedConfig()
        assert cfg.enable_densification is False

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Gaussian init")
    def test_depth_guided_trainer_inherits_baseline(self, tmp_path):
        sample = _make_fake_sample(tmp_path)
        cfg = DepthGuidedConfig(
            iterations=2,
            depth_semantics_artifact_path=str(_make_fake_artifact(tmp_path)),
            clip_grad_norm=True,
        )
        trainer = DepthGuidedTrainer(
            train_entries=[sample],
            val_entries=[sample],
            backend=_MetricDepthBackend(),
            config=cfg,
            output_dir=tmp_path / "output",
        )
        assert hasattr(trainer, "_init_from_first_sample")
        assert hasattr(trainer, "_build_optimizer")
        assert hasattr(trainer, "train_entries")
        assert hasattr(trainer, "val_entries")
        assert hasattr(trainer, "dg_config")
