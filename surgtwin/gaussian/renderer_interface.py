from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch


@dataclass
class RenderOutput:
    rgb: torch.Tensor
    depth: Optional[torch.Tensor] = None
    alpha: Optional[torch.Tensor] = None
    aux: Dict[str, Any] = field(default_factory=dict)


class RendererBackend(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def render(
        self,
        gaussians: Any,
        camera: Any,
        image_height: int,
        image_width: int,
        render_depth: bool = True,
    ) -> RenderOutput:
        raise NotImplementedError
