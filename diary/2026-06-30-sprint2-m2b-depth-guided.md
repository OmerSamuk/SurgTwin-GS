# Sprint 2 / M2-B — Depth-Guided Gaussian Splatting

**Date:** 2026-06-30
**Status:** Implementation complete — 120 passed, 8 skipped (CUDA/LPIPS)

## Summary

Full M2-B implementation per expert-revised plan (v2). 3 critical revisions applied: PSNR dB threshold, GT-centered depth mask, scale drift regularizer. 38 new tests, 7 new modules, 2 CLI scripts.

## Expert Revisions Applied

| R# | Revision | Plan v1 | Plan v2 (final) |
|----|----------|---------|-----------------|
| R1 | PSNR eşiği | `≥ 0.95 * baseline_psnr` | `≥ baseline_val_psnr - 3.0 dB` |
| R2 | `valid_depth_mask` | pred_0 masked | GT-centered; pred_0 NOT masked, clamped for loss |
| R3 | L_reg | simple mean(shrink) | scale_drift: `mean((log(scale) - log(init_scale))^2)` |
| R4 | Ablation | default 4 runs required | optional, not blocker |
| R5 | Gradient clipping | not specified | `max_norm=1.0`, logged |
| R6 | M2-A artifact gate | report.md reference | JSON artifact hard-fail |
| R7 | Raw + clipped metrics | suggested | required in `metrics.jsonl` + `final_metrics.json` |

## Files Created (7 new)

| File | Purpose |
|------|---------|
| `surgtwin/losses/photometric.py` | `photometric_l1()` — mean L1 RGB loss |
| `surgtwin/losses/depth.py` | `depth_l1()` — GT-centered masked depth loss + diagnostics |
| `surgtwin/losses/regularizers.py` | `scale_drift_regularizer()` + `REGISTRY` |
| `surgtwin/losses/__init__.py` | Re-exports |
| `surgtwin/evaluation/geometry_metrics.py` | `depth_rmse`, `depth_mae`, `abs_rel`, `valid_depth_ratio`, `median_aligned_rmse`, `geometry_metrics_report` — all with metric semantic guards |
| `surgtwin/training/depth_guided_config.py` | `DepthGuidedConfig` dataclass (all fields locked) |
| `surgtwin/training/depth_guided_trainer.py` | `DepthGuidedTrainer(BaselineTrainer)` — M2-A gate, depth loss, scale drift, grad clip, depth snapshots |

## CLI Scripts Created (2 new)

| Script | Purpose |
|--------|---------|
| `scripts/train_depth_guided.py` | CLI entry: `--lambda_depth 0.2 --lambda_reg 0.01 --reg_type scale_drift --depth_semantics_artifact <path> --output_dir outputs/runs/depth_guided_m2b` |
| `scripts/compare_baseline_vs_depth_guided.py` | Comparison: baseline ↔ depth-guided, PSNR dB guard, per-metric table |

## Files Modified

| File | Change |
|------|--------|
| `surgtwin/evaluation/__init__.py` | Added geometry_metrics exports |
| `surgtwin/losses/__init__.py` | Added photometric, depth, regularizers exports |
| `tests/test_depth_loss.py` | 10 tests — valid/invalid mask, clamp effect, diagnostics keys |
| `tests/test_photometric_loss.py` | 4 tests — L1, shape, alpha |
| `tests/test_regularizers_scale_drift.py` | 6 tests — zero loss, drift, eps guard, registry |
| `tests/test_geometry_metrics.py` | 11 tests — RMSE/MAE/AbsRel, semantic guard, report |
| `tests/test_depth_guided_config.py` | 7 tests — defaults, frozen, overrides |
| `tests/test_depth_guided_trainer.py` | 6 tests — M2-A gate, non-metric fail, grad norm logging (4 CUDA-skip) |

## Loss Configuration (Locked)

```
L_total = L_photo + lambda_depth * L_depth + lambda_reg * L_reg
lambda_depth = 0.2
lambda_reg = 0.01
reg_type = scale_drift
depth_near_m = 0.02
depth_far_m = 0.30
clip_grad_norm = true (max_norm=1.0)
```

### valid_depth_mask
```
isfinite(gt) AND isfinite(pred) AND gt >= 0.02 AND gt <= 0.30
```
`pred == 0` is NOT masked; pred is clamped to `[near, far]` for loss.

### Metrics Logged Per Iter (metrics.jsonl)
`loss_total`, `photo_loss`, `depth_loss_raw_m`, `depth_loss_weighted`, `reg_loss_raw`, `reg_loss_weighted`, `psnr`, `depth_rmse_m_raw`, `depth_mae_m_raw`, `abs_rel`, `depth_valid_ratio`, `grad_norm_before_clip`, `grad_norm_after_clip`, `n_gaussians`, `iter_time_s`, `vram_gb`

## Test Results

```
120 passed, 8 skipped in 4.57s
```

- **4 skipped** (no CUDA): `test_depth_guided_trainer.py` — need GPU for Gaussian init
- **4 skipped** (pre-existing): LPIPS/skimage unavailable
- **0 regressions** from M1 + M2-A

## Next

- **VM production run**: 1000 iter on L4
  ```bash
  python scripts/train_depth_guided.py \
    --manifest data/processed/manifests/servct_manifest.jsonl \
    --iterations 1000 \
    --depth_semantics_artifact outputs/runs/depth_semantics_m2a/final_gate_decision.json \
    --output_dir outputs/runs/depth_guided_m2b
  ```
- **Comparison**: `scripts/compare_baseline_vs_depth_guided.py`
- **Milestone 3**: Uncertainty-weighted loss
