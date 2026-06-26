# Renderer Backend Interface

## RendererBackend (ABC)

All renderer backends must implement `RendererBackend`:

```python
class RendererBackend(ABC):
    @property
    def name(self) -> str: ...

    def render(self, gaussians, camera, image_height, image_width, render_depth=True) -> RenderOutput: ...
```

## RenderOutput

```python
@dataclass
class RenderOutput:
    rgb: torch.Tensor       # [H, W, 3], float32, [0, 1]
    depth: torch.Tensor     # [H, W], float32, optional
    alpha: torch.Tensor     # [H, W], float32, [0, 1], optional
    aux: dict               # backend metadata
```

## Required aux fields

- `backend`: str
- `depth_semantics`: "metric_meters" | "relative_aligned" | "relative_unaligned" | "unavailable"
- `supports_metric_depth`: bool
- `supports_alpha`: bool
- `supports_contrib`: bool
- `supports_color_variance`: bool

## GsplatBackend

Primary Sprint 0 backend. Wraps `gsplat.rasterization` and normalizes outputs to `RenderOutput`.
