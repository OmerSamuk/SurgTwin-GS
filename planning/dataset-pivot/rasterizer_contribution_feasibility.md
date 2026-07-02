# Rasterizer Contribution Feasibility (expert-answer-18 §5)

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

## Feasibility Assessment

### What gsplat provides TODAY (via meta)

1. **Gaussian visibility mask**: Which Gaussians contribute to at least one pixel
   (gaussian_ids). Already usable without any library change.
2. **Per-Gaussian pixel footprint** (radii in meta): coarse proxy for
   contribution magnitude — larger radius → more pixels affected.
3. **Per-Gaussian 2D position** (means2d): which image region a Gaussian lands on.

### What gsplat does NOT expose

- Per-pixel Gaussian contribution _weights_ (the normalized w_i = alpha_i *
  T_i product used in alpha compositing). These are computed inside the CUDA
  rasterizer kernel and discarded after accumulation.
- Per-pixel per-Gaussian gradient breaks. The current backward kernel
  composites gradients per-pixel without exposing individual Gaussian
  contributions.

### Implementation options

| Option | Effort | Impact |
|---|---|---|
| **A. Use meta.radii only** | Low (few hours) | Weak signal — radius alone doesn't capture blending |
| **B. Post-hoc contribution via gradient** | Medium (3-5 days) | Compute contribution as norm of per-Gaussian gradient w.r.t. loss; requires forward-with-grad per Gaussian |
| **C. Modify gsplat CUDA kernel** | High (1-2 weeks) | Fork gsplat to expose per-pixel per-Gaussian alpha weights from the forward pass — maximum signal but breaks upgrade path |
| **D. Offline multi-pass rendering** | Medium (1 week) | Render each Gaussian individually and measure pixel overlap vs total — expensive for >100K Gaussians |

### Recommendation

**Option B (post-hoc gradient norm) is the most pragmatic near-term approach:**

- No library forks, no CUDA changes
- `meta["radii"]` + `meta["gaussian_ids"]` already filter Gaussians that land
  on screen
- A Gaussian's gradient w.r.t. the render loss naturally captures its
  contribution to the current view
- Can be implemented as an additional `contrib` field in `RenderOutput.aux`
  without breaking existing API
- Feasible to prototype in 3-5 days

### Risks

1. **Per-Gaussian gradient cost**: Computing per-Gaussian gradient norm
   requires retaining the computation graph for all Gaussians (O(N) memory).
   For 500K Gaussians this is ~500 MB extra for means alone.
2. **Multi-view accumulation**: Contribution from one view ≠ scene importance.
   Would need to avg/max over N random views per step.
3. **Interaction with w_photo**: w_photo already captures photometric
   consistency; per-Gaussian contrib may be partially redundant.

## Decision

Defer to M5 planning. Implement Option A (radii-based footprint filter) if
densification still shows low-w_photo clones after log audit (§5.2).

Option A cost: ~1 day to thread `meta["radii"]` through `RenderOutput.aux` and
use it as a soft mask in `select_densification_candidates`.
