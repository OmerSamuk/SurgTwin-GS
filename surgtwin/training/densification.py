import torch
from dataclasses import dataclass
from typing import TYPE_CHECKING

from surgtwin.cameras.projection import project_points_to_image
from surgtwin.training.uncertainty_config import UncertaintyConfig

if TYPE_CHECKING:
    from surgtwin.gaussian.gaussian_model import GaussianModel
    from surgtwin.cameras.camera_types import CameraData


@dataclass
class DensificationSelection:
    clone_indices: torch.Tensor
    clone_offsets: torch.Tensor
    prune_mask: torch.Tensor
    n_cloned: int
    n_pruned: int
    selected_candidate_count: int
    selected_candidate_ratio: float
    selected_mean_depth_residual: float
    selected_p50_depth_residual: float
    selected_p90_depth_residual: float
    selected_mean_w_photo: float
    selected_p10_w_photo: float
    opacity_mean: float
    opacity_std: float
    opacity_min: float
    opacity_max: float
    max_gaussians_hit: bool
    clone_offset_mode: str = "random_unit_vector"
    clone_offset_scale: float = 0.25
    mapping_mode: str = "projection_based_approximate"
    trigger_reason: str = (
        "depth_residual > densify_depth_residual_threshold "
        "AND w_photo > densify_w_photo_threshold "
        "AND valid_depth AND sufficient opacity/visibility"
    )
    sample_id: str = ""
    frame_id: str = ""
    split: str = "train"
    densification_entry_mode: str = "current_train_entry"


def select_densification_candidates(
    gaussians,
    rendered_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    rendered_rgb: torch.Tensor,
    gt_rgb: torch.Tensor,
    w_photo: torch.Tensor,
    camera,
    config: UncertaintyConfig,
    sample_id: str = "",
    frame_id: str = "",
    split: str = "train",
) -> DensificationSelection:
    H, W = rendered_depth.shape
    device = rendered_depth.device

    near_m = config.depth_near_m
    far_m = config.depth_far_m
    depth_thresh = config.densify_depth_residual_threshold
    w_photo_thresh = config.densify_w_photo_threshold
    max_clone_per_step = config.densify_max_clone_per_step
    max_clone_fraction = config.densify_max_clone_fraction
    max_gaussians = config.densify_max_gaussians
    prune_min_opacity = config.prune_min_opacity
    max_prune_fraction = config.max_prune_fraction_per_step

    n_gaussians = gaussians.means.shape[0]

    with torch.no_grad():
        depth_residual = (rendered_depth - gt_depth).abs()

        valid_depth_mask = (
            torch.isfinite(gt_depth)
            & (gt_depth >= near_m)
            & (gt_depth <= far_m)
            & torch.isfinite(rendered_depth)
        )

        # Step 1: Project Gaussian centers to image plane
        w2c = camera.w2c.to(device)
        K = camera.K.to(device)
        uv, z_cam = project_points_to_image(gaussians.means, K, w2c, H, W)
        uv = uv.long()
        in_image = (
            (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H) & (z_cam > 0)
        )
        u_clamped = uv[:, 1].clamp(0, H - 1)
        v_clamped = uv[:, 0].clamp(0, W - 1)

        # Step 2: Read per-Gaussian values at projected pixel
        gauss_depth_residual = depth_residual[u_clamped, v_clamped]
        gauss_w_photo = w_photo[u_clamped, v_clamped]
        gauss_valid = valid_depth_mask[u_clamped, v_clamped]

        # Step 3: Gaussian-level criterion
        sigmoid_opacity = torch.sigmoid(gaussians.opacities)
        candidate = (
            in_image
            & gauss_valid
            & (gauss_depth_residual > depth_thresh)
            & (gauss_w_photo > w_photo_thresh)
            & (sigmoid_opacity > prune_min_opacity)
        )
        candidate_indices = torch.where(candidate)[0]

        if candidate_indices.numel() == 0:
            return DensificationSelection(
                clone_indices=torch.tensor([], device=device, dtype=torch.long),
                clone_offsets=torch.tensor([], device=device),
                prune_mask=torch.ones(n_gaussians, device=device, dtype=torch.bool),
                n_cloned=0,
                n_pruned=0,
                selected_candidate_count=0,
                selected_candidate_ratio=0.0,
                selected_mean_depth_residual=0.0,
                selected_p50_depth_residual=0.0,
                selected_p90_depth_residual=0.0,
                selected_mean_w_photo=0.0,
                selected_p10_w_photo=0.0,
                opacity_mean=float(sigmoid_opacity.mean().item()),
                opacity_std=float(sigmoid_opacity.std().item()),
                opacity_min=float(sigmoid_opacity.min().item()),
                opacity_max=float(sigmoid_opacity.max().item()),
                max_gaussians_hit=False,
                sample_id=sample_id,
                frame_id=frame_id,
                split=split,
            )

        # Step 4: Score = depth_residual * w_photo
        candidate_scores = gauss_depth_residual[candidate] * gauss_w_photo[candidate]

        # Step 5: Prune mask
        prune_candidates = torch.where(
            (sigmoid_opacity < prune_min_opacity) & torch.isfinite(sigmoid_opacity)
        )[0]
        n_prune = min(prune_candidates.numel(), int(n_gaussians * max_prune_fraction))
        if n_prune > 0:
            prune_top = prune_candidates[
                torch.argsort(sigmoid_opacity[prune_candidates])[:n_prune]
            ]
            prune_mask = torch.ones(n_gaussians, device=device, dtype=torch.bool)
            prune_mask[prune_top] = False
        else:
            n_prune = 0
            prune_mask = torch.ones(n_gaussians, device=device, dtype=torch.bool)

        # Step 6: Growth bounds
        n_current = n_gaussians
        n_to_clone = min(
            candidate_indices.numel(),
            max_clone_per_step,
            int(n_current * max_clone_fraction),
        )
        if n_current + n_to_clone > max_gaussians:
            n_to_clone = max(0, max_gaussians - n_current)
            max_gaussians_hit = True
        else:
            max_gaussians_hit = False

        if n_to_clone == 0:
            return DensificationSelection(
                clone_indices=torch.tensor([], device=device, dtype=torch.long),
                clone_offsets=torch.tensor([], device=device),
                prune_mask=prune_mask,
                n_cloned=0,
                n_pruned=n_prune,
                selected_candidate_count=int(candidate_indices.numel()),
                selected_candidate_ratio=float(
                    candidate_indices.numel() / max(1, n_gaussians)
                ),
                selected_mean_depth_residual=0.0,
                selected_p50_depth_residual=0.0,
                selected_p90_depth_residual=0.0,
                selected_mean_w_photo=0.0,
                selected_p10_w_photo=0.0,
                opacity_mean=float(sigmoid_opacity.mean().item()),
                opacity_std=float(sigmoid_opacity.std().item()),
                opacity_min=float(sigmoid_opacity.min().item()),
                opacity_max=float(sigmoid_opacity.max().item()),
                max_gaussians_hit=max_gaussians_hit,
                sample_id=sample_id,
                frame_id=frame_id,
                split=split,
            )

        # Step 7: Topk selection
        topk_scores, topk_orig_indices = torch.topk(
            candidate_scores, k=n_to_clone, sorted=True
        )
        clone_indices = candidate_indices[topk_orig_indices]

        selected_residuals = gauss_depth_residual[clone_indices]
        selected_w_photos = gauss_w_photo[clone_indices]

        # Step 8: Clone offset (per-Gaussian scale mean)
        scale_mean = gaussians.scales[clone_indices].mean(dim=-1)
        offset_magnitude = config.clone_offset_scale_factor * scale_mean
        torch.manual_seed(config.seed)
        direction = torch.randn(n_to_clone, 3, device=device)
        direction = direction / direction.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        offset = offset_magnitude.unsqueeze(-1) * direction

        # Statistics
        sorted_scores, _ = torch.sort(candidate_scores, descending=True)
        p50_idx = min(n_to_clone - 1, int(n_to_clone * 0.5))
        p90_idx = min(n_to_clone - 1, int(n_to_clone * 0.9))
        sorted_residuals, _ = torch.sort(selected_residuals)
        p50_res = sorted_residuals[p50_idx].item() if n_to_clone > 0 else 0.0
        p90_res = sorted_residuals[p90_idx].item() if n_to_clone > 0 else 0.0
        sorted_w, _ = torch.sort(selected_w_photos)
        p10_idx = min(n_to_clone - 1, int(n_to_clone * 0.1))
        p10_w = sorted_w[p10_idx].item() if n_to_clone > 0 else 0.0

    return DensificationSelection(
        clone_indices=clone_indices,
        clone_offsets=offset,
        prune_mask=prune_mask,
        n_cloned=n_to_clone,
        n_pruned=n_prune,
        selected_candidate_count=int(candidate_indices.numel()),
        selected_candidate_ratio=float(candidate_indices.numel() / max(1, n_gaussians)),
        selected_mean_depth_residual=float(selected_residuals.mean().item()),
        selected_p50_depth_residual=float(p50_res),
        selected_p90_depth_residual=float(p90_res),
        selected_mean_w_photo=float(selected_w_photos.mean().item()),
        selected_p10_w_photo=float(p10_w),
        opacity_mean=float(sigmoid_opacity.mean().item()),
        opacity_std=float(sigmoid_opacity.std().item()),
        opacity_min=float(sigmoid_opacity.min().item()),
        opacity_max=float(sigmoid_opacity.max().item()),
        max_gaussians_hit=max_gaussians_hit,
        sample_id=sample_id,
        frame_id=frame_id,
        split=split,
    )
