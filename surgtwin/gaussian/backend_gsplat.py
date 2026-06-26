from typing import Any, Dict, Optional

import torch

from surgtwin.gaussian.renderer_interface import RenderOutput, RendererBackend


class GsplatBackend(RendererBackend):
    def __init__(self):
        self._name = "gsplat"
        self._depth_verified = False
        self._depth_is_metric = False

    @property
    def name(self) -> str:
        return self._name

    def _verify_metric_depth(self, gaussians: Dict[str, torch.Tensor]) -> bool:
        try:
            import gsplat
            means = torch.tensor([[[0.0, 0.0, 2.0]]], dtype=torch.float32, device=gaussians["means"].device)
            quats = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], dtype=torch.float32, device=gaussians["means"].device)
            scales = torch.tensor([[[0.01, 0.01, 0.01]]], dtype=torch.float32, device=gaussians["means"].device)
            opacities = torch.tensor([[1.0]], dtype=torch.float32, device=gaussians["means"].device)
            colors = torch.tensor([[[0.5, 0.5, 0.5, 0.5]]], dtype=torch.float32, device=gaussians["means"].device)

            K = torch.eye(3, dtype=torch.float32, device=gaussians["means"].device).unsqueeze(0)
            K[0, 0, 0] = 100.0
            K[0, 1, 1] = 100.0
            K[0, 0, 2] = 50.0
            K[0, 1, 2] = 50.0

            viewmat = torch.eye(4, dtype=torch.float32, device=gaussians["means"].device)[:3, :4].unsqueeze(0)

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
                render_mode="RGB+D",
            )

            rendered = result[0]
            if rendered.shape[-1] < 5:
                return False

            depth_channel = rendered[0, :, :, 4]
            valid = (depth_channel > 0) & torch.isfinite(depth_channel)
            if valid.sum() < 10:
                return False

            median_depth = depth_channel[valid].median().item()
            return abs(median_depth - 2.0) < 0.05

        except Exception:
            return False

    def render(
        self,
        gaussians: Dict[str, torch.Tensor],
        camera: Any,
        image_height: int,
        image_width: int,
        render_depth: bool = True,
    ) -> RenderOutput:
        import gsplat

        if not self._depth_verified:
            self._depth_is_metric = self._verify_metric_depth(gaussians)
            self._depth_verified = True

        K = camera.K.to(dtype=torch.float32, device=gaussians["means"].device)
        w2c = camera.w2c.to(dtype=torch.float32, device=gaussians["means"].device)

        viewmat = w2c[:3, :4].unsqueeze(0)
        K_batch = K.unsqueeze(0)

        colors_premul = torch.cat(
            [gaussians["colors"], torch.ones_like(gaussians["colors"][:, :1])], dim=-1
        )

        try:
            result = gsplat.rasterization(
                means=gaussians["means"].unsqueeze(0),
                quats=gaussians["quats"].unsqueeze(0),
                scales=gaussians["scales"].unsqueeze(0),
                opacities=torch.sigmoid(gaussians["opacities"]).unsqueeze(0),
                colors=colors_premul.unsqueeze(0),
                viewmats=viewmat,
                Ks=K_batch,
                width=image_width,
                height=image_height,
                render_mode="RGB+D",
            )
        except Exception as e:
            raise RuntimeError(f"gsplat rasterization failed: {e}")

        rendered = result[0]
        rgb = rendered[0, :, :, :3].clamp(0.0, 1.0)
        alpha = rendered[0, :, :, 3:4]
        depth: Optional[torch.Tensor] = None
        depth_semantics = "unavailable"

        if render_depth and rendered.shape[-1] >= 5 and self._depth_is_metric:
            depth = rendered[0, :, :, 4]
            depth_semantics = "metric_meters"
        elif render_depth and rendered.shape[-1] >= 5:
            depth = rendered[0, :, :, 4]
            depth_semantics = "relative_unaligned"

        aux = {
            "backend": "gsplat",
            "depth_semantics": depth_semantics,
            "supports_metric_depth": depth_semantics == "metric_meters",
            "metric_depth_verified": self._depth_is_metric,
            "supports_alpha": True,
            "supports_contrib": False,
            "supports_color_variance": False,
        }

        return RenderOutput(
            rgb=rgb,
            depth=depth,
            alpha=alpha.squeeze(-1),
            aux=aux,
        )
