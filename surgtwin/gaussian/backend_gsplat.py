from typing import Any, Dict, Optional

import torch

from surgtwin.gaussian.gaussian_model import GaussianModel
from surgtwin.gaussian.renderer_interface import RenderOutput, RendererBackend


def _resolve_gaussian_dict(gaussians: Any) -> Dict[str, torch.Tensor]:
    if isinstance(gaussians, GaussianModel):
        return gaussians.state_dict()
    if isinstance(gaussians, dict):
        return gaussians
    raise TypeError(f"Expected GaussianModel or dict, got {type(gaussians)}")


def _add_batch_v_dims(t: torch.Tensor) -> torch.Tensor:
    return t.unsqueeze(0).unsqueeze(0)


def _remove_batch_v_dims(t: torch.Tensor) -> torch.Tensor:
    while t.ndim > 3 and t.shape[0] == 1:
        t = t.squeeze(0)
    while t.ndim == 3 and t.shape[0] == 1:
        t = t.squeeze(0)
    while t.ndim == 3 and t.shape[-1] == 1:
        t = t.squeeze(-1)
    return t


class GsplatBackend(RendererBackend):
    def __init__(self):
        self._name = "gsplat"
        self._depth_verified = False
        self._depth_is_metric = False

    @property
    def name(self) -> str:
        return self._name

    def _verify_metric_depth(self, gaussians: Any) -> bool:
        gd = _resolve_gaussian_dict(gaussians)
        try:
            import gsplat

            device = gd["means"].device
            means = torch.tensor([[[[0.0, 0.0, 2.0]]]], dtype=torch.float32, device=device)
            quats = torch.tensor([[[[1.0, 0.0, 0.0, 0.0]]]], dtype=torch.float32, device=device)
            scales = torch.tensor([[[[0.01, 0.01, 0.01]]]], dtype=torch.float32, device=device)
            opacities = torch.tensor([[[1.0]]], dtype=torch.float32, device=device)
            colors = torch.tensor([[[[0.5, 0.5, 0.5]]]], dtype=torch.float32, device=device)

            K = torch.eye(3, dtype=torch.float32, device=device)[None, None, None]
            K[0, 0, 0, 0, 0] = 100.0
            K[0, 0, 0, 1, 1] = 100.0
            K[0, 0, 0, 0, 2] = 50.0
            K[0, 0, 0, 1, 2] = 50.0

            viewmat = torch.eye(4, dtype=torch.float32, device=device)[None, None, None]

            result = gsplat.rasterization(
                means=means,
                quats=quats,
                scales=scales,
                opacities=opacities,
                colors=colors,
                viewmats=viewmat,
                Ks=K,
                width=100,
                height=100,
                render_mode="RGB+ED",
            )

            render_colors, render_alphas, meta = result

            if render_colors.shape[-1] < 4:
                return False

            depth_channel = _remove_batch_v_dims(render_colors[..., 3])
            valid = (depth_channel > 0) & torch.isfinite(depth_channel)
            if valid.sum() < 10:
                return False

            median_depth = depth_channel[valid].median().item()
            return abs(median_depth - 2.0) < 0.05

        except Exception:
            return False

    def render(
        self,
        gaussians: Any,
        camera: Any,
        image_height: int,
        image_width: int,
        render_depth: bool = True,
    ) -> RenderOutput:
        import gsplat

        gd = _resolve_gaussian_dict(gaussians)

        if not self._depth_verified:
            self._depth_is_metric = self._verify_metric_depth(gd)
            self._depth_verified = True

        K = camera.K.to(dtype=torch.float32, device=gd["means"].device)
        w2c = camera.w2c.to(dtype=torch.float32, device=gd["means"].device)

        viewmat = w2c[None, None, None]
        K_batch = K[None, None, None]

        try:
            result = gsplat.rasterization(
                means=_add_batch_v_dims(gd["means"]),
                quats=_add_batch_v_dims(gd["quats"]),
                scales=_add_batch_v_dims(gd["scales"]),
                opacities=_add_batch_v_dims(torch.sigmoid(gd["opacities"])),
                colors=_add_batch_v_dims(gd["colors"]),
                viewmats=viewmat,
                Ks=K_batch,
                width=image_width,
                height=image_height,
                render_mode="RGB+ED" if render_depth else "RGB",
            )
        except Exception as e:
            raise RuntimeError(f"gsplat rasterization failed: {e}")

        render_colors, render_alphas, meta = result

        rgb = _remove_batch_v_dims(render_colors[..., :3]).clamp(0.0, 1.0)
        alpha = _remove_batch_v_dims(render_alphas).clamp(0.0, 1.0)

        depth: Optional[torch.Tensor] = None
        depth_semantics = "unavailable"

        if render_depth and render_colors.shape[-1] >= 4:
            depth_candidate = _remove_batch_v_dims(render_colors[..., 3])

            if self._depth_is_metric:
                depth = depth_candidate
                depth_semantics = "metric_meters"
            else:
                depth = depth_candidate
                depth_semantics = "relative_unaligned"

        aux = {
            "backend": "gsplat",
            "depth_semantics": depth_semantics,
            "supports_metric_depth": depth_semantics == "metric_meters",
            "metric_depth_verified": self._depth_is_metric,
            "depth_distribution_verified": False,
            "depth_verification_artifact_path": "",
            "supports_alpha": True,
            "supports_contrib": False,
            "supports_color_variance": False,
        }

        return RenderOutput(
            rgb=rgb,
            depth=depth,
            alpha=alpha,
            aux=aux,
        )
