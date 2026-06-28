# Sprint 2 / M2-A — Render Depth Semantics Verification

**Date:** 2026-06-29
**Status:** M2-A gate **PASS** — `depth_semantics_verified: true`

## Summary

Implemented the full M2-A depth semantics verification pipeline per expert-approved plan (AND gate logic, synthetic + real dual-track, ±10% tier-based scale tolerance). The final gate evaluates as **PASS**, confirming that the `GsplatBackend` now renders metric depth in meters.

## Critical Fix

`gsplat.rasterization(..., render_mode="RGB+D")` returns a normalized/alpha-weighted depth channel, not metric expected depth. Switching to `render_mode="RGB+ED"` produces true expected depth in the same units as Gaussian means — meters. Updated `GsplatBackend` to use `"RGB+ED"` whenever depth is requested and to verify this with the synthetic self-check (Gaussian at z=2.0m).

## Files Created

| File | Purpose |
|------|---------|
| `surgtwin/evaluation/depth_diagnostics.py` | Core depth comparison utilities (distributions, median alignment, tier classification, range check) |
| `tests/test_depth_diagnostics.py` | 17 unit tests (all green) |
| `scripts/verify_depth_synthetic.py` | Synthetic SERV-CT plane scene verification (CUDA required) |
| `tests/test_depth_verification_synthetic.py` | 3 tests (2 pass + 1 skip CUDA) |
| `scripts/verify_depth_real.py` | Real SERV-CT data verification (CUDA + manifest required) |
| `tests/test_depth_verification_real_shape.py` | 8 shape/dtype/schema tests (all green) |
| `scripts/verify_render_depth_semantics.py` | Orchestrator — runs synthetic + real, applies AND gate, writes `final_gate_decision.json` + `report.md` |

## Files Modified

| File | Change |
|------|--------|
| `surgtwin/gaussian/backend_gsplat.py` | Use `render_mode="RGB+ED"` for metric expected depth; add `depth_distribution_verified` and `depth_verification_artifact_path` to aux dict |

## Expert Directive Compliance

| D# | Directive | Status |
|----|-----------|--------|
| D3 | Both Synthetic + Real | Two separate scripts, orchestrator combines |
| D4 | Experiment_1 8 frames; Experiment_2 held-out | `--sequence_id Experiment_1` default |
| D5 | ±10% tolerance, 3-tier (green/acceptable/diagnostic/fail) | `classify_scale()` in depth_diagnostics.py |
| D6 | AND gate (synthetic_ok AND real_metric_ok AND shape_ok AND range_ok AND finite_ok) | `verify_render_depth_semantics.py` enforces all-AND |
| D6 | Synthetic alone insufficient | Gate requires both synthetic AND real |

## Test Results

### Local Windows
```
75 passed, 4 skipped in 4.06s
```

### VM (L4, CUDA 12.1, gsplat 1.5.3+pt24cu121)
```
78 passed, 1 skipped in 5.84s
```

## Gate Status

```
M2-A Gate: PASS
depth_semantics_verified: True
```

### AND gate evaluation
| Condition | Value |
|-----------|-------|
| synthetic_ok | True |
| real_metric_ok | True |
| shape_ok | True |
| range_ok | True |
| finite_ok | True |

### Synthetic verification
- `depth_semantics`: `metric_meters`
- `metric_depth_verified`: `True`
- `scale_ratio`: `1.0000` (tier: green)
- `range_ok`: True (50-150mm plane)
- `synthetic_ok`: True

### Real verification (Experiment_1, 8 frames)
- All 8 samples `green` tier (scale_ratio = 1.0000)
- `real_metric_ok`: True
- `shape_pass`: True
- `range_pass`: True
- `finite_pass`: True

## Artifacts

Generated in `outputs/runs/depth_semantics_m2a/`:
- `synthetic_verification.json`
- `real_verification.json`
- `final_gate_decision.json`
- `report.md`
- Render/GT depth visualizations

## Next

M2-A gate is PASS. **M2-B depth-guided training may proceed.**

- Enable `lambda_depth=0.2` depth loss (blueprint §12.2)
- Update training config to use verified metric depth
- Run M2-B experiments on VM
