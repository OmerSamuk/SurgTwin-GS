# Rasterizer Contribution Feasibility (expert-answer-18 §5 / expert-answer-22 §8)

## Goal

Determine whether per-Gaussian contribution attribution from the GPU rasterizer
(gsplat) is feasible for improving densification candidate selection.

## Current State

- `backend_gsplat.py` hardcodes `supports_contrib: False` at line 154
- `uncertainty_trainer.py` (line 752) notes this as a future improvement
- Densification selection uses only w_photo and depth_residual — no per-Gaussian
  rasterization contribution signal

## gsplat `meta` Output

`gsplat.rasterization` returns `(render_colors, render_alphas, meta)`. The
`meta` dict already contains per-Gaussian projection data:

| Key | Shape | Description |
|---|---|---|
| `gaussian_ids` | `[nnz]` | Indices of Gaussians rasterized to each pixel |
| `radii` | `[nnz]` | 2D bounding box radius in pixels |
| `means2d` | `[nnz, 2]` | Projected 2D means |
| `depths` | `[nnz]` | Depth at projection |
| `conics` | `[nnz, 3]` | Inverse covariance (upper-triangular) |
| `opacities` | `[nnz]` | Sigmoid opacity after compensation |
| `camera_ids` | `[nnz]` | Camera index per projected Gaussian |

## Phase 1 Answer (§8 Question)

**Question:** *"Projection-based approximate mapping yerine gsplat veya alternatif rasterizer üzerinden per-Gaussian contribution / visibility / accumulated alpha bilgisi Python tarafına alınabilir mi?"*

### Answer: Short-Term Feasible (Partially)

| Aspect | Assessment |
|---|---|
| **Short-term feasible?** | **YES** — for visibility/radii/means2d (already in `meta`) |
| **Short-term NOT feasible?** | Per-Gaussian accumulated alpha (α_i × T_i) requires CUDA kernel change |
| **Requires backend modification?** | Visibility mask: no (meta.gaussian_ids). Contribution weight: yes (CUDA) |

### What CAN be done today (no backend changes)

1. **Visibility mask** — via `meta["gaussian_ids"]`: which Gaussians touch ≥1 pixel
   - Unique count of gaussian_ids → fraction of scene Gaussians visible in a view
   - Already usable without any library change
2. **Footprint proxy** — via `meta["radii"]`: larger radius = more pixels affected
   - Coarse contribution signal — a Gaussian with radius > threshold is "contributing"
3. **Screen position** — via `meta["means2d"]`: where on screen the Gaussian projects
   - Useful for center-vs-edge downweighting

### What requires gsplat CUDA modification

4. **Per-Gaussian alpha contribution** — the value `α_i × T_i` for each Gaussian at
   each pixel is computed inside the CUDA rasterizer kernel and **discarded** after
   pixel accumulation. Exposing this requires:
   - Fork/modify gsplat forward kernel to store per-Gaussian α_i × T_i
   - Modified kernel returns contribution buffer `[num_gaussians]` or `[nnz]`
   - **Cost:** 1–2 weeks (CUDA C++ + Python bindings)
   - **Risk:** breaks gsplat upgrade path (custom fork)

### Alternative: Post-Hoc Gradient Norm (Option B)

- Instead of direct alpha contribution, compute per-Gaussian **gradient norm**
  w.r.t. render loss
- Requires graph retention (`requires_grad` on means/quaternions) and per-Gaussian
  gradient slicing
- **Cost:** 3–5 days (pure Python + PyTorch, no CUDA)
- **Signal quality:** Gradient magnitude is a good proxy for contribution
- **Risk:** O(N) memory for gradient graph (500K → ~500 MB extra)

### Recommended Backend Path

| Phase | Approach | Timeline | Dependency |
|---|---|---|---|
| **Phase 2 (short-term)** | Option A: radii + gaussian_ids from meta | 1 day | None — pure Python |
| **Phase 3 (medium-term)** | Option B: gradient norm post-hoc | 3–5 days | Requires grad graph |
| **M5 (long-term)** | Option C: custom gsplat CUDA fork | 1–2 weeks | Fork decision |

### Option A Details (Phase 2 Candidate)

- Thread `meta["radii"]` and `meta["gaussian_ids"]` through `RenderOutput.aux`
- Add `contrib_radii_fraction` field to `DensificationSelection`
- Use as soft mask in `select_densification_candidates`:
  - Gaussians with radius < threshold → lower clone priority
  - Gaussians not in gaussian_ids = not visible → no clone
- Compatible with existing w_photo + depth_residual logic (multiply or threshold)

### Expected Engineering Cost

| Approach | Person-days | Code changes |
|---|---|---|
| Option A (radii mask) | 1 | `RenderOutput`, `densification.py`, backend |
| Option B (gradient norm) | 3–5 | Trainer forward loop, gradient accumulation, validation |
| Option C (CUDA fork) | 5–10 | gsplat fork, CUDA kernel, Python bindings |

### Risk Summary

1. **Partial redundancy with w_photo** — w_photo already captures photometric
   inconsistency. Gaussian contribution may not add new signal.
2. **Multi-view cost** — contribution from one view ≠ scene importance.
   Averaging over N views multiplies render cost by N.
3. **Gradient graph memory** — Option B doubles active memory during training.
4. **Fork maintenance** — Option C locks to specific gsplat commit; upstream
   syncs are manual.

## Decision

**Phase 1 decision:** Proceed with original M5 deferral for full contribution.
Phase 2 can implement Option A (radii mask, 1 day) as a lightweight improvement
if densification audit shows low-w_photo clones are still being created after
w_photo threshold tuning.

For Phase 2 planning: implement Option A as a soft gating mechanism in
`select_densification_candidates` to filter Gaussians with zero/negligible
screen footprint before they reach clone/prune logic.
