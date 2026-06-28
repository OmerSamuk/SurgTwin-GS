import torch


def scale_drift_regularizer(
    scales: torch.Tensor,
    init_scales: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    if scales.shape != init_scales.shape:
        raise ValueError(
            f"Shape mismatch: current scales {scales.shape} vs init {init_scales.shape}"
        )
    log_curr = torch.log(scales.clamp(min=eps) + eps)
    log_init = torch.log(init_scales.clamp(min=eps) + eps)
    return ((log_curr - log_init) ** 2).mean()


REGISTRY = {
    "scale_drift": scale_drift_regularizer,
}
