from typing import Optional

import torch


def psnr(pred: torch.Tensor, gt: torch.Tensor, max_val: float = 1.0) -> float:
    mse = torch.mean((pred - gt) ** 2).item()
    if mse < 1e-12:
        return float("inf")
    return float(20.0 * torch.log10(torch.tensor(max_val)) - 10.0 * torch.log10(torch.tensor(mse)))


def ssim(pred: torch.Tensor, gt: torch.Tensor) -> float:
    from skimage.metrics import structural_similarity

    pred_np = pred.detach().cpu().numpy()
    gt_np = gt.detach().cpu().numpy()
    if pred_np.ndim == 3 and pred_np.shape[-1] == 3:
        return float(structural_similarity(pred_np, gt_np, channel_axis=-1, data_range=1.0))
    return float(structural_similarity(pred_np, gt_np, data_range=1.0))


def lpips_score(pred: torch.Tensor, gt: torch.Tensor, device: torch.device) -> Optional[float]:
    try:
        import lpips

        net = lpips.LPIPS(net="alex", verbose=False).to(device)
        pred_f = pred.permute(2, 0, 1).unsqueeze(0).to(device) * 2.0 - 1.0
        gt_f = gt.permute(2, 0, 1).unsqueeze(0).to(device) * 2.0 - 1.0
        with torch.no_grad():
            score = net(pred_f, gt_f).item()
        return float(score)
    except Exception:
        return None
