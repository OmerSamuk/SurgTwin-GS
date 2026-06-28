import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional


class JsonlLogger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, metrics: Dict[str, Any]) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps(metrics) + "\n")


def _safe_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, cwd=Path(__file__).resolve().parent.parent
        ).decode().strip()
    except Exception:
        return "unknown"


def collect_environment(backend_name: str = "gsplat") -> Dict[str, Any]:
    env = {
        "python_version": sys.version.split()[0],
        "torch_version": "unknown",
        "torch_cuda_version": "unknown",
        "cuda_available": False,
        "gpu_name": "unknown",
        "gpu_vram_gb": None,
        "gsplat_version": "unknown",
        "platform": sys.platform,
        "git_commit": _safe_git_commit(),
        "cloud_provider": os.environ.get("CLOUD_PROVIDER", "unknown"),
        "zone": os.environ.get("CLOUD_ZONE", "unknown"),
        "machine_type": os.environ.get("CLOUD_MACHINE_TYPE", "unknown"),
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV", "unknown"),
        "backend": backend_name,
    }
    try:
        import torch

        env["torch_version"] = torch.__version__
        env["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env["torch_cuda_version"] = torch.version.cuda
            env["gpu_name"] = torch.cuda.get_device_name(0)
            total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            env["gpu_vram_gb"] = round(total_vram, 1)
    except Exception:
        pass
    try:
        import gsplat
        env["gsplat_version"] = getattr(gsplat, "__version__", "unknown")
    except Exception:
        pass
    return env


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
