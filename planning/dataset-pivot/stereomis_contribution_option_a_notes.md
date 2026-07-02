# StereoMIS Contribution Option A Feasibility

**Date:** 2026-07-02
**Status:** Pre-download assessment — to be confirmed after render smoke

## Background

Phase 1 `rasterizer_contribution_feasibility.md` identified **Option A** (radii/visibility
proxy via gsplat `meta`) as the short-term feasible path for contribution-aware
densification. Full CUDA contribution profile requires a gsplat fork.

## What We Need to Check

### gsplat meta inspection

The `GsplatBackend.render()` captures `meta` from `gsplat.rasterization()`:

```python
render_colors, render_alphas, meta = result
```

`meta` is a dict that typically contains:
- `radii` — per-Gaussian screen-space radii (Tensor, shape [N])
- `gaussian_ids` — per-pixel Gaussian IDs (Tensor, shape [H, W, K])

These are exactly what Option A needs: a per-Gaussian screen-space footprint
proxy that can be mapped back to Gaussian parameters.

### Current status

| Check | Result |
|---|---|
| `meta` available in `GsplatBackend.render()` | ✅ Captured from gsplat output |
| `meta` exposed in `RenderOutput.aux` | ❌ Currently discarded |
| `supports_contrib` flag in aux | `False` (hardcoded) |
| `out.aux["radii"]` or `out.aux["visibility_mask"]` | ❌ Not implemented |

### Changes required

To enable Option A prototype:

1. **`surgtwin/gaussian/backend_gsplat.py`** — Forward relevant meta fields:
   ```python
   if meta is not None:
       if "radii" in meta:
           out.aux["radii"] = _remove_batch_v_dims(meta["radii"])
   ```
2. **`surgtwin/gaussian/backend_gsplat.py`** — Set `supports_contrib = True` in
   the renderer info dict when meta is available.
3. **Smoke test** — After change, verify `radii` shape = `[N]` where N is the
   Gaussian count, and values are finite.

### Risk

| Risk | Severity | Mitigation |
|---|---|---|
| gsplat `radii` format changes between versions | Low | Pin gsplat version |
| `meta` may be None on first frame | Low | Add None guard |
| radii projection mapping to density requires calibration | Medium | Empirically determine scale factor in M5 |
| Option B (exact contribution) still requires CUDA fork | High | Phase 1 already confirmed — outside Option A scope |

## Decision

**Recommended: Defer to Phase 3 (M5 prototype)**

Rationale:
- Option A is ~1 day of engineering (backend changes + test) but is not needed
  for Phase 2 go/no-go.
- The render smoke in Phase 2 does NOT require meta forwarding.
- Phase 3 (M5 training loop integration) is the natural time to implement
  Option A and test with real StereoMIS data.
- The gsplat meta schema can be confirmed during Phase 2 render smoke
  (add a `--dump-meta` flag to `smoke_stereomis_render.py` if desired).

**Option A was already assessed as Feasible in Phase 1 ~1 day.**
**No new information in Phase 2 changes this conclusion.**
