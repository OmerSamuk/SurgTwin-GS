from typing import Any, Dict

import torch

from surgtwin.gaussian.renderer_interface import RenderOutput, RendererBackend


class GsplatBackend(RendererBackend):
    def __init__(self):
        self._name = "gsplat"

    @property
    def name(self) -> str:
        return self._name

    def render(
        self,
        gaussians: Dict[str, torch.Tensor],
        camera: Any,
        image_height: int,
        image_width: int,
        render_depth: bool = True,
    ) -> RenderOutput:
        import gsplat

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
        depth = None
        depth_semantics = "unavailable"
        if render_depth and rendered.shape[-1] >= 5:
            depth = rendered[0, :, :, 4]
            depth_semantics = "metric_meters"
        elif render_depth:
            depth = None
            depth_semantics = "unavailable"

        aux = {
            "backend": "gsplat",
            "depth_semantics": depth_semantics,
            "supports_metric_depth": depth_semantics == "metric_meters",
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
