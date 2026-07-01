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

    def clone_gaussians(self, indices: torch.Tensor, offsets: torch.Tensor) -> None:
        N_selected = indices.shape[0]
        if N_selected == 0:
            return
        self.means = torch.cat([self.means, self.means[indices] + offsets], dim=0)
        self.scales = torch.cat([self.scales, self.scales[indices]], dim=0)
        self.quats = torch.cat([self.quats, self.quats[indices]], dim=0)
        self.opacities = torch.cat([self.opacities, self.opacities[indices]], dim=0)
        self.colors = torch.cat([self.colors, self.colors[indices]], dim=0)
        self.reliability_logits = torch.cat(
            [self.reliability_logits, self.reliability_logits[indices]], dim=0
        )
        for field in ("means", "scales", "quats", "opacities", "colors", "reliability_logits"):
            t = getattr(self, field)
            setattr(self, field, t.detach().clone().requires_grad_(True))

    def remove_gaussians(self, keep_mask: torch.Tensor) -> None:
        self.means = self.means[keep_mask]
        self.scales = self.scales[keep_mask]
        self.quats = self.quats[keep_mask]
        self.opacities = self.opacities[keep_mask]
        self.colors = self.colors[keep_mask]
        self.reliability_logits = self.reliability_logits[keep_mask]
        for field in ("means", "scales", "quats", "opacities", "colors", "reliability_logits"):
            t = getattr(self, field)
            setattr(self, field, t.detach().clone().requires_grad_(True))
