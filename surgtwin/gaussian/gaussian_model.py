from dataclasses import dataclass
from typing import Dict

import torch


@dataclass
class GaussianModel:
    means: torch.Tensor
    scales: torch.Tensor
    quats: torch.Tensor
    opacities: torch.Tensor
    colors: torch.Tensor
    reliability_logits: torch.Tensor

    def num_gaussians(self) -> int:
        return self.means.shape[0]

    def to(self, device: torch.device) -> "GaussianModel":
        return GaussianModel(
            means=self.means.to(device),
            scales=self.scales.to(device),
            quats=self.quats.to(device),
            opacities=self.opacities.to(device),
            colors=self.colors.to(device),
            reliability_logits=self.reliability_logits.to(device),
        )

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "means": self.means,
            "scales": self.scales,
            "quats": self.quats,
            "opacities": self.opacities,
            "colors": self.colors,
            "reliability_logits": self.reliability_logits,
        }

    @classmethod
    def load_state_dict(cls, sd: Dict[str, torch.Tensor]) -> "GaussianModel":
        return cls(
            means=sd["means"],
            scales=sd["scales"],
            quats=sd["quats"],
            opacities=sd["opacities"],
            colors=sd["colors"],
            reliability_logits=sd.get(
                "reliability_logits", torch.zeros(sd["means"].shape[0])
            ),
        )
