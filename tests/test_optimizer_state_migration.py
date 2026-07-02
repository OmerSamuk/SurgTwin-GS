import torch
from torch.optim import Adam

from surgtwin.gaussian.gaussian_model import GaussianModel
from surgtwin.training.uncertainty_config import UncertaintyConfig


def _make_gaussians(n=100, device="cpu"):
    return GaussianModel(
        means=torch.randn(n, 3, device=device) * 0.02 + torch.tensor([0.0, 0.0, 0.15], device=device),
        scales=torch.full((n, 3), 0.005, device=device),
        quats=torch.rand(n, 4, device=device),
        opacities=torch.randn(n, device=device),
        colors=torch.rand(n, 3, device=device),
        reliability_logits=torch.zeros(n, device=device),
    )


def _dense_config(**overrides):
    kwargs = dict(clone_means_exp_avg_scale=0.5, seed=42)
    kwargs.update(overrides)
    return UncertaintyConfig(**kwargs)


def _make_optimizer(gaussians, lr=1e-3):
    gd = {k: v for k, v in gaussians.__dict__.items() if isinstance(v, torch.Tensor)}
    return Adam([
        {"params": [gd["means"]], "lr": lr},
        {"params": [gd["scales"]], "lr": lr},
        {"params": [gd["quats"]], "lr": lr},
        {"params": [gd["opacities"]], "lr": lr},
        {"params": [gd["colors"]], "lr": lr},
    ])


def _migrate_optimizer_state(g, old_state_dict, clone_parent_map, keep_mask, remapped_clone, config, pre_prune_clone_indices=None):
    """Rebuild optimizer and migrate Adam states from old_state_dict."""
    param_keys = ["means", "scales", "quats", "opacities", "colors"]
    gd = {k: v for k, v in g.__dict__.items() if isinstance(v, torch.Tensor)}
    new_opt = Adam([
        {"params": [gd["means"]], "lr": 1e-3},
        {"params": [gd["scales"]], "lr": 1e-3},
        {"params": [gd["quats"]], "lr": 1e-3},
        {"params": [gd["opacities"]], "lr": 1e-3},
        {"params": [gd["colors"]], "lr": 1e-3},
    ])
    old_state = old_state_dict["state"]
    for pg_idx, field in enumerate(param_keys):
        if pg_idx >= len(new_opt.param_groups):
            break
        pg = new_opt.param_groups[pg_idx]
        if not pg["params"]:
            continue
        p = pg["params"][0]
        old_pid = sorted(old_state.keys())[pg_idx] if pg_idx < len(old_state) else None
        if old_pid is None:
            continue
        old_s = old_state[old_pid]
        old_exp_avg = old_s.get("exp_avg")
        old_exp_avg_sq = old_s.get("exp_avg_sq")
        if old_exp_avg is None:
            continue
        new_exp_avg = torch.zeros_like(p)
        new_exp_avg_sq = torch.zeros_like(p)
        if keep_mask is not None:
            old_kept_avg = old_exp_avg[keep_mask]
            old_kept_avg_sq = old_exp_avg_sq[keep_mask]
        else:
            old_kept_avg = old_exp_avg
            old_kept_avg_sq = old_exp_avg_sq
        n_kept = old_kept_avg.shape[0]
        new_exp_avg[:n_kept] = old_kept_avg
        new_exp_avg_sq[:n_kept] = old_kept_avg_sq
        if clone_parent_map is not None and clone_parent_map.numel() > 0:
            n_clone = clone_parent_map.shape[0]
            if n_clone > 0:
                parent_idx = (pre_prune_clone_indices if pre_prune_clone_indices is not None else remapped_clone)[:n_clone]
                parent_avg = old_exp_avg[parent_idx]
                parent_avg_sq = old_exp_avg_sq[parent_idx]
                clone_start = n_kept
                clone_end = n_kept + n_clone
                if field == "means":
                    new_exp_avg[clone_start:clone_end] = parent_avg * config.clone_means_exp_avg_scale
                else:
                    new_exp_avg[clone_start:clone_end] = parent_avg
                new_exp_avg_sq[clone_start:clone_end] = parent_avg_sq
        new_opt.state[p] = {
            "step": old_s.get("step", 0),
            "exp_avg": new_exp_avg,
            "exp_avg_sq": new_exp_avg_sq,
        }
    return new_opt


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_state_migration_preserves_existing_parent_states():
    g = _make_gaussians(100)
    opt = _make_optimizer(g)
    opt.zero_grad()
    for name in ("means", "scales", "quats", "opacities", "colors"):
        getattr(g, name).grad = torch.randn_like(getattr(g, name))
    opt.step()
    old_state = opt.state_dict()
    old_pid = sorted(old_state["state"].keys())[0]
    old_means_state = old_state["state"][old_pid]["exp_avg"][:10].clone()
    config = _dense_config()
    new_opt = _migrate_optimizer_state(g, old_state, None, None, None, config)
    new_sd = new_opt.state_dict()
    new_pid = sorted(new_sd["state"].keys())[0]
    assert torch.allclose(new_sd["state"][new_pid]["exp_avg"][:10], old_means_state), "First 10 rows should match"


def test_state_migration_adds_clone_states_with_correct_shape():
    g = _make_gaussians(50)
    opt = _make_optimizer(g)
    opt.zero_grad()
    for name in ("means", "scales", "quats", "opacities", "colors"):
        getattr(g, name).grad = torch.randn_like(getattr(g, name))
    opt.step()
    old_state = opt.state_dict()
    n_before = g.num_gaussians()
    clone_indices = torch.tensor([0, 5, 10])
    n_clone = clone_indices.shape[0]
    offsets = torch.randn(n_clone, 3, device=g.means.device)
    g.clone_gaussians(clone_indices, offsets)
    assert g.num_gaussians() == n_before + 3
    new_opt = _migrate_optimizer_state(g, old_state, clone_indices, None, clone_indices, _dense_config())
    new_sd = new_opt.state_dict()
    for pid, s in new_sd["state"].items():
        assert "exp_avg" in s
        assert "exp_avg_sq" in s
        assert s["exp_avg"].shape[0] == g.num_gaussians()


def test_clone_means_exp_avg_is_damped():
    g = _make_gaussians(20)
    opt = _make_optimizer(g)
    opt.zero_grad()
    for name in ("means", "scales", "quats", "opacities", "colors"):
        getattr(g, name).grad = torch.randn_like(getattr(g, name))
    opt.step()
    old_state = opt.state_dict()
    config = _dense_config(clone_means_exp_avg_scale=0.5)
    old_pid = sorted(old_state["state"].keys())[0]
    parent_exp_avg = old_state["state"][old_pid]["exp_avg"][0].clone()
    clone_indices = torch.tensor([0])
    n_clone = 1
    offsets = torch.randn(n_clone, 3, device=g.means.device)
    g.clone_gaussians(clone_indices, offsets)
    new_opt = _migrate_optimizer_state(g, old_state, clone_indices, None, clone_indices, config)
    new_sd = new_opt.state_dict()
    new_pid = sorted(new_sd["state"].keys())[0]
    clone_avg = new_sd["state"][new_pid]["exp_avg"][-1]
    assert torch.allclose(clone_avg, parent_exp_avg * 0.5), "Clone means exp_avg should be damped"


def test_clone_exp_avg_sq_copied_or_valid():
    g = _make_gaussians(20)
    opt = _make_optimizer(g)
    opt.zero_grad()
    for name in ("means", "scales", "quats", "opacities", "colors"):
        getattr(g, name).grad = torch.randn_like(getattr(g, name))
    opt.step()
    old_state = opt.state_dict()
    old_pid = sorted(old_state["state"].keys())[0]
    parent_exp_avg_sq = old_state["state"][old_pid]["exp_avg_sq"][0].clone()
    clone_indices = torch.tensor([0])
    n_clone = 1
    offsets = torch.randn(n_clone, 3, device=g.means.device)
    g.clone_gaussians(clone_indices, offsets)
    new_opt = _migrate_optimizer_state(g, old_state, clone_indices, None, clone_indices, _dense_config())
    new_sd = new_opt.state_dict()
    new_pid = sorted(new_sd["state"].keys())[0]
    clone_avg_sq = new_sd["state"][new_pid]["exp_avg_sq"][-1]
    assert torch.allclose(clone_avg_sq, parent_exp_avg_sq), "Clone exp_avg_sq should match parent"


def test_optimizer_step_after_state_migration_does_not_crash():
    g = _make_gaussians(30)
    opt = _make_optimizer(g)
    opt.zero_grad()
    for name in ("means", "scales", "quats", "opacities", "colors"):
        getattr(g, name).grad = torch.randn_like(getattr(g, name))
    opt.step()
    old_state = opt.state_dict()
    clone_indices = torch.tensor([0, 1])
    n_clone = 2
    offsets = torch.randn(n_clone, 3, device=g.means.device)
    g.clone_gaussians(clone_indices, offsets)
    new_opt = _migrate_optimizer_state(g, old_state, clone_indices, None, clone_indices, _dense_config())
    for name in ("means", "scales", "quats", "opacities", "colors"):
        getattr(g, name).grad = torch.randn_like(getattr(g, name))
    new_opt.step()


def test_state_migration_handles_zero_clone():
    g = _make_gaussians(20)
    opt = _make_optimizer(g)
    opt.zero_grad()
    for name in ("means", "scales", "quats", "opacities", "colors"):
        getattr(g, name).grad = torch.randn_like(getattr(g, name))
    opt.step()
    old_state = opt.state_dict()
    n_before = g.num_gaussians()
    new_opt = _migrate_optimizer_state(g, old_state, None, None, None, _dense_config())
    assert g.num_gaussians() == n_before
    new_sd = new_opt.state_dict()
    for pid, s in new_sd["state"].items():
        assert s["exp_avg"].shape[0] == n_before


def test_state_migration_fallback_rebuild_on_shape_mismatch():
    g = _make_gaussians(10)
    opt = _make_optimizer(g)
    opt.zero_grad()
    for name in ("means", "scales", "quats", "opacities", "colors"):
        getattr(g, name).grad = torch.randn_like(getattr(g, name))
    opt.step()
    old_state = opt.state_dict()
    config = _dense_config()
    clone_indices = torch.tensor([0, 1, 2])
    n_clone = 3
    offsets = torch.randn(n_clone, 3, device=g.means.device)
    g.clone_gaussians(clone_indices, offsets)
    new_opt = _migrate_optimizer_state(g, old_state, clone_indices, None, clone_indices, config)
    for name in ("means", "scales", "quats", "opacities", "colors"):
        getattr(g, name).grad = torch.randn_like(getattr(g, name))
    new_opt.step()


def test_parent_clone_do_not_share_state_tensor_storage():
    g = _make_gaussians(10)
    opt = _make_optimizer(g)
    opt.zero_grad()
    for name in ("means", "scales", "quats", "opacities", "colors"):
        getattr(g, name).grad = torch.randn_like(getattr(g, name))
    opt.step()
    old_state = opt.state_dict()
    old_pid = sorted(old_state["state"].keys())[0]
    old_storage_ptr = old_state["state"][old_pid]["exp_avg"][0].untyped_storage().data_ptr()
    clone_indices = torch.tensor([0])
    n_clone = 1
    offsets = torch.randn(n_clone, 3, device=g.means.device)
    g.clone_gaussians(clone_indices, offsets)
    new_opt = _migrate_optimizer_state(g, old_state, clone_indices, None, clone_indices, _dense_config())
    new_sd = new_opt.state_dict()
    new_pid = sorted(new_sd["state"].keys())[0]
    new_exp_avg = new_sd["state"][new_pid]["exp_avg"]
    new_storage_ptr = new_exp_avg[0].untyped_storage().data_ptr()
    assert new_storage_ptr != old_storage_ptr, "New state should not alias old state storage"


def test_state_migration_prune_and_clone_same_step_correct_parent():
    g = _make_gaussians(20)
    opt = _make_optimizer(g)
    opt.zero_grad()
    for name in ("means", "scales", "quats", "opacities", "colors"):
        getattr(g, name).grad = torch.randn_like(getattr(g, name))
    opt.step()
    old_state = opt.state_dict()
    old_pid = sorted(old_state["state"].keys())[0]
    parent_exp_avg_pre = old_state["state"][old_pid]["exp_avg"].clone()
    # Prune: keep 0..9, 15..19 (remove 10..14)
    keep_mask = torch.cat([torch.ones(10, dtype=torch.bool), torch.zeros(5, dtype=torch.bool), torch.ones(5, dtype=torch.bool)])
    g.remove_gaussians(keep_mask)
    assert g.num_gaussians() == 15
    old_n = 20
    new_n = 15
    new_indices = torch.arange(new_n)
    old_to_new = torch.zeros(old_n, dtype=torch.long)
    old_to_new[keep_mask] = new_indices
    # Pre-prune clone indices: [0, 15] (both in the kept set)
    pre_prune_clone_idx = torch.tensor([0, 15])
    # Post-prune compact indices: old 0->new 0, old 15->new 10
    remapped_clone = old_to_new[pre_prune_clone_idx]
    # Verify the remapping
    assert remapped_clone.tolist() == [0, 10], "remapped_clone must differ from pre-prune indices"
    n_clone = 2
    offsets = torch.randn(n_clone, 3, device=g.means.device)
    g.clone_gaussians(remapped_clone, offsets)
    assert g.num_gaussians() == 17
    config = _dense_config(clone_means_exp_avg_scale=0.5)
    new_opt = _migrate_optimizer_state(
        g, old_state, remapped_clone, keep_mask, remapped_clone, config,
        pre_prune_clone_indices=pre_prune_clone_idx,
    )
    new_sd = new_opt.state_dict()
    new_pid = sorted(new_sd["state"].keys())[0]
    new_exp_avg = new_sd["state"][new_pid]["exp_avg"]
    assert new_exp_avg.shape[0] == 17
    # Clone at post-prune index 10 = pre-prune index 15 (remapped from old 15)
    # old_exp_avg[pre_prune_clone_idx[0]=0] should be parent
    # old_exp_avg[pre_prune_clone_idx[1]=15] should be parent
    clone0 = new_exp_avg[-2]
    parent0_pre = parent_exp_avg_pre[0] * 0.5
    assert torch.allclose(clone0, parent0_pre), "Clone 0 should use parent from pre-prune index 0"
    clone1 = new_exp_avg[-1]
    parent1_pre = parent_exp_avg_pre[15] * 0.5
    assert torch.allclose(clone1, parent1_pre), "Clone 1 should use parent from pre-prune index 15"
