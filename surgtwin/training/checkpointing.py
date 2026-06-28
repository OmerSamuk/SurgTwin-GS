from pathlib import Path
from typing import Any, Dict, Optional

import torch

from surgtwin.gaussian.gaussian_model import GaussianModel


def save_checkpoint(
    path: Path,
    gaussians: GaussianModel,
    optimizer_state_dict: Dict[str, Any],
    iteration: int,
    config: Dict[str, Any],
    backend_name: str,
    seed: int = 42,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    data = {
        "gaussian_state": gaussians.state_dict(),
        "optimizer_state_dict": optimizer_state_dict,
        "iteration": iteration,
        "config": config,
        "backend_name": backend_name,
        "seed": seed,
    }
    if extra is not None:
        data["extra"] = extra

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, path)


def load_checkpoint(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    data = torch.load(path, map_location="cpu", weights_only=False)
    required = ["gaussian_state", "optimizer_state_dict", "iteration", "config", "backend_name", "seed"]
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"Checkpoint {path} missing keys: {missing}")
    return data


def load_gaussians_from_checkpoint(path: Path) -> GaussianModel:
    data = load_checkpoint(path)
    return GaussianModel.load_state_dict(data["gaussian_state"])
