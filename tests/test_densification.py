import torch
from surgtwin.gaussian.gaussian_model import GaussianModel
from surgtwin.training.densification import DensificationSelection, select_densification_candidates
from surgtwin.training.uncertainty_config import UncertaintyConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_gaussians(n=100, device="cpu", centered=True):
    if centered:
        means = torch.randn(n, 3, device=device) * 0.02 + torch.tensor([0.0, 0.0, 0.15], device=device)
    else:
        means = torch.rand(n, 3, device=device) * 10.0
    return GaussianModel(
        means=means,
        scales=torch.full((n, 3), 0.005, device=device),
        quats=torch.rand(n, 4, device=device),
        opacities=torch.randn(n, device=device),
        colors=torch.rand(n, 3, device=device),
        reliability_logits=torch.zeros(n, device=device),
    )


def _make_dummy_camera(device="cpu", H=64, W=64):
    from surgtwin.cameras.camera_types import CameraData
    K = torch.tensor([[100.0, 0.0, W / 2],
                      [0.0, 100.0, H / 2],
                      [0.0, 0.0, 1.0]], device=device)
    w2c = torch.eye(4, device=device)
    c2w = torch.eye(4, device=device)
    return CameraData(K=K, w2c=w2c, c2w=c2w, height=H, width=W)


def _dense_config(**overrides):
    kwargs = dict(
        densify_from_iter=0,
        densify_every=1,
        densify_until_iter=1000,
        densify_depth_residual_threshold=0.01,
        densify_w_photo_threshold=0.2,
        densify_max_clone_per_step=50,
        densify_max_clone_fraction=0.5,
        densify_max_gaussians=500,
        prune_min_opacity=0.01,
        max_prune_fraction_per_step=0.1,
        clone_offset_scale_factor=0.25,
        depth_near_m=0.0,
        depth_far_m=1.0,
        seed=42,
    )
    kwargs.update(overrides)
    return UncertaintyConfig(**kwargs)


# ===================================================================
# GaussianModel.clone_gaussians
# ===================================================================

def test_clone_gaussians_basic():
    g = _make_gaussians(100)
    n_before = g.num_gaussians()
    indices = torch.tensor([0, 5, 10, 20, 50])
    offsets = torch.randn(5, 3)
    g.clone_gaussians(indices, offsets)
    assert g.num_gaussians() == n_before + 5


def test_clone_gaussians_zero_indices():
    g = _make_gaussians(100)
    g.clone_gaussians(torch.tensor([], dtype=torch.long), torch.zeros(0, 3))
    assert g.num_gaussians() == 100


def test_clone_gaussians_leaf_tensors():
    g = _make_gaussians(50)
    g.clone_gaussians(torch.tensor([0, 1]), torch.randn(2, 3))
    for field in ("means", "scales", "quats", "opacities", "colors", "reliability_logits"):
        t = getattr(g, field)
        assert t.requires_grad, f"{field} missing requires_grad"
        assert not t.is_leaf or t.grad_fn is None


def test_clone_gaussians_offset_applied():
    g = _make_gaussians(10)
    orig_means = g.means[0].clone()
    offset = torch.tensor([[0.1, 0.2, 0.3]])
    g.clone_gaussians(torch.tensor([0]), offset)
    cloned_mean = g.means[-1]
    assert torch.allclose(cloned_mean, orig_means + offset[0])


def test_clone_gaussians_values_match_source():
    g = _make_gaussians(20)
    idx = 7
    g.clone_gaussians(torch.tensor([idx]), torch.zeros(1, 3))
    cloned = g.means[-1]
    assert torch.allclose(cloned, g.means[idx])


# ===================================================================
# GaussianModel.remove_gaussians
# ===================================================================

def test_remove_gaussians_basic():
    g = _make_gaussians(100)
    keep = torch.ones(100, dtype=torch.bool)
    keep[:20] = False
    g.remove_gaussians(keep)
    assert g.num_gaussians() == 80


def test_remove_gaussians_keep_all():
    g = _make_gaussians(50)
    g.remove_gaussians(torch.ones(50, dtype=torch.bool))
    assert g.num_gaussians() == 50


def test_remove_gaussians_remove_all():
    g = _make_gaussians(30)
    g.remove_gaussians(torch.zeros(30, dtype=torch.bool))
    assert g.num_gaussians() == 0


def test_remove_gaussians_leaf_tensors():
    g = _make_gaussians(20)
    keep = torch.ones(20, dtype=torch.bool)
    keep[0] = False
    g.remove_gaussians(keep)
    for field in ("means", "scales", "quats", "opacities", "colors", "reliability_logits"):
        t = getattr(g, field)
        assert t.requires_grad, f"{field} missing requires_grad"


# ===================================================================
# DensificationSelection dataclass
# ===================================================================

def test_densification_selection_fields():
    sel = DensificationSelection(
        clone_indices=torch.tensor([0, 1]),
        clone_offsets=torch.randn(2, 3),
        prune_mask=torch.ones(100, dtype=torch.bool),
        n_cloned=2,
        n_pruned=0,
        selected_candidate_count=5,
        selected_candidate_ratio=0.05,
        selected_mean_depth_residual=0.03,
        selected_p50_depth_residual=0.025,
        selected_p90_depth_residual=0.04,
        selected_mean_w_photo=0.6,
        selected_p10_w_photo=0.4,
        opacity_mean=0.5,
        opacity_std=0.2,
        opacity_min=0.01,
        opacity_max=0.99,
        max_gaussians_hit=False,
        sample_id="S001",
        frame_id="000001",
        split="train",
    )
    assert sel.n_cloned == 2
    assert sel.n_pruned == 0
    assert sel.sample_id == "S001"
    assert sel.split == "train"
    assert sel.clone_offset_mode == "random_unit_vector"
    assert sel.mapping_mode == "projection_based_approximate"
    assert "depth_residual" in sel.trigger_reason


# ===================================================================
# select_densification_candidates
# ===================================================================

def test_select_no_candidates():
    """All depth residuals zero + w_photo low → no clones, no prune."""
    config = _dense_config()
    device = "cpu"
    g = _make_gaussians(50, device)
    H, W = 64, 64
    gt_d = torch.full((H, W), 0.15, device=device)
    pred_d = gt_d.clone()
    gt_rgb = torch.rand(H, W, 3, device=device)
    pred_rgb = gt_rgb.clone()
    w = torch.full((H, W), 0.1, device=device)
    cam = _make_dummy_camera(device, H, W)
    out = select_densification_candidates(g, pred_d, gt_d, pred_rgb, gt_rgb, w, cam, config,
                                           sample_id="S001", frame_id="000001")
    assert out.n_cloned == 0
    assert out.n_pruned == 0
    assert out.selected_candidate_count == 0


def test_select_clones_basic():
    """High depth residual + high w_photo → clones produced."""
    config = _dense_config(densify_depth_residual_threshold=0.005,
                            densify_w_photo_threshold=0.1)
    device = "cpu"
    g = _make_gaussians(100, device)
    H, W = 64, 64
    gt_d = torch.full((H, W), 0.15, device=device)
    pred_d = torch.full((H, W), 0.20, device=device)
    gt_rgb = torch.rand(H, W, 3, device=device)
    pred_rgb = torch.rand(H, W, 3, device=device)
    w = torch.full((H, W), 0.6, device=device)
    cam = _make_dummy_camera(device, H, W)
    out = select_densification_candidates(g, pred_d, gt_d, pred_rgb, gt_rgb, w, cam, config)
    assert out.n_cloned > 0
    assert out.clone_indices.shape[0] == out.n_cloned
    assert out.clone_offsets.shape[0] == out.n_cloned
    assert not out.max_gaussians_hit
    assert out.selected_candidate_count > 0


def test_select_prune_low_opacity():
    """Gaussians with very low sigmoid opacity → pruned alongside clones."""
    config = _dense_config(prune_min_opacity=0.5, max_prune_fraction_per_step=0.3)
    device = "cpu"
    g = _make_gaussians(100, device)
    g.opacities[80:] = -10.0
    H, W = 64, 64
    gt_d = torch.full((H, W), 0.15, device=device)
    pred_d = torch.full((H, W), 0.20, device=device)
    gt_rgb = torch.rand(H, W, 3, device=device)
    pred_rgb = torch.rand(H, W, 3, device=device)
    w = torch.full((H, W), 0.6, device=device)
    cam = _make_dummy_camera(device, H, W)
    out = select_densification_candidates(g, pred_d, gt_d, pred_rgb, gt_rgb, w, cam, config)
    assert out.n_pruned > 0
    assert out.n_cloned > 0
    assert not out.prune_mask.all()


def test_select_max_gaussians_hit():
    """n_to_clone capped by max_gaussians (tight cap)."""
    config = _dense_config(densify_max_gaussians=103,
                            densify_max_clone_per_step=50,
                            densify_max_clone_fraction=1.0,
                            densify_depth_residual_threshold=0.001,
                            densify_w_photo_threshold=0.05)
    device = "cpu"
    g = _make_gaussians(100, device)
    H, W = 64, 64
    gt_d = torch.full((H, W), 0.15, device=device)
    pred_d = torch.full((H, W), 0.20, device=device)
    gt_rgb = torch.rand(H, W, 3, device=device)
    pred_rgb = torch.rand(H, W, 3, device=device)
    w = torch.full((H, W), 0.6, device=device)
    cam = _make_dummy_camera(device, H, W)
    out = select_densification_candidates(g, pred_d, gt_d, pred_rgb, gt_rgb, w, cam, config)
    assert out.max_gaussians_hit
    assert out.n_cloned == 3


def test_select_stats_populated():
    """Statistics fields are populated when cloning occurs."""
    config = _dense_config(densify_depth_residual_threshold=0.001,
                            densify_w_photo_threshold=0.05,
                            densify_max_clone_per_step=20)
    device = "cpu"
    g = _make_gaussians(50, device)
    H, W = 64, 64
    gt_d = torch.full((H, W), 0.10, device=device)
    pred_d = torch.full((H, W), 0.15, device=device)
    gt_rgb = torch.rand(H, W, 3, device=device)
    pred_rgb = torch.rand(H, W, 3, device=device)
    w = torch.full((H, W), 0.5, device=device)
    cam = _make_dummy_camera(device, H, W)
    out = select_densification_candidates(g, pred_d, gt_d, pred_rgb, gt_rgb, w, cam, config)
    assert out.n_cloned > 0
    assert out.selected_mean_depth_residual > 0
    assert out.selected_mean_w_photo > 0
    assert out.opacity_mean > 0
    assert out.opacity_std >= 0


# ===================================================================
# select_densification_candidates — edge cases
# ===================================================================

def test_select_out_of_image_gaussians():
    """Gaussians far from the camera center project outside image and are ignored."""
    config = _dense_config()
    device = "cpu"
    g = _make_gaussians(20, device, centered=False)
    H, W = 64, 64
    gt_d = torch.full((H, W), 0.15, device=device)
    pred_d = torch.full((H, W), 0.15, device=device)
    gt_rgb = torch.rand(H, W, 3, device=device)
    pred_rgb = gt_rgb.clone()
    w = torch.full((H, W), 0.5, device=device)
    cam = _make_dummy_camera(device, H, W)
    out = select_densification_candidates(g, pred_d, gt_d, pred_rgb, gt_rgb, w, cam, config)
    assert out.n_cloned == 0
    assert out.selected_candidate_count == 0


def test_select_no_valid_depth():
    """All depth invalid → no selection."""
    config = _dense_config()
    device = "cpu"
    g = _make_gaussians(50, device)
    H, W = 64, 64
    gt_d = torch.full((H, W), float("nan"), device=device)
    pred_d = torch.rand(H, W, device=device)
    gt_rgb = torch.rand(H, W, 3, device=device)
    pred_rgb = gt_rgb.clone()
    w = torch.full((H, W), 0.5, device=device)
    cam = _make_dummy_camera(device, H, W)
    out = select_densification_candidates(g, pred_d, gt_d, pred_rgb, gt_rgb, w, cam, config)
    assert out.n_cloned == 0


# ===================================================================
# Clone + prune integration (index remap scenario)
# ===================================================================

def test_clone_after_prune_index_remap():
    """Prune-before-clone: prune removes indices, then clone operates on new indices space."""
    g = _make_gaussians(100)
    n_before = g.num_gaussians()

    keep = torch.ones(100, dtype=torch.bool)
    keep[:10] = False
    g.remove_gaussians(keep)
    n_after_prune = g.num_gaussians()
    assert n_after_prune == 90

    idx_src = 5
    offset = torch.tensor([[0.01, 0.02, 0.03]])
    g.clone_gaussians(torch.tensor([idx_src]), offset)
    assert g.num_gaussians() == 91

    cloned = g.means[-1]
    expected_src = g.means[idx_src]
    assert torch.allclose(cloned, expected_src + offset[0])
