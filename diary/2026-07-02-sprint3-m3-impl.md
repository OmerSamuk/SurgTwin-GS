# Sprint 3 / M3 — Uncertainty-Weighted Photometric Loss

**Date:** 2026-07-02
**Status:** Implementation complete — 175 passed, 9 skipped (CUDA/optional deps)

## Summary

Full M3 implementation per expert-revised plan (revides-m3-plan.txt + expert-answer.txt). 3 expert revisions integrated: weight-collapse metrics, mask_effective reporting, realistic test count commitment. 9 new modules, 3 CLI scripts, 6 test suites. Zero regressions vs M1/M2-A/M2-B.

## Expert Revisions Applied

| R# | Revision | Implementation |
|----|----------|---------------|
| R1 | `w_photo` collapse controls | 3 new stats in diagnostics: `fraction_w_photo_at_min`, `fraction_w_photo_at_one`, `w_photo_p90_minus_p10`; AND gate with `≥0.05` spread + `<0.90` fraction caps |
| R2 | Mask coverage reporting | H2 always separate run; `mask_coverage`, `mask_effective: bool`, `mask_interpretation` string if coverage `<1%` |
| R3 | Test count realism | Promise: "All existing M1/M2-A/M2-B tests pass, plus all newly added M3 tests pass." |

### Open Question Answers Applied

1. **H3 auto-run**: conditional `if best(H1,H2).val_psnr < 56.20 AND depth_rmse ≤ 0.030` → `compare_m2b_vs_m3.py` gate_decision block
2. **Mask low coverage no fallback**: `mask_effective: false` with interpretation note
3. **Comparison table**: full 7-run history (M1, M2-B orig/R1/R3, M3 H1/H2/H3); primary PASS/FAIL uses M1 + M2-B R1
4. **Next milestone**: M4 — multi-criteria density control

## Files Created (9 new)

| File | Purpose |
|------|---------|
| `surgtwin/masks/specular.py` | HSV-based specular detection: `detect_specular_hsv()` |
| `surgtwin/masks/io.py` | Mask loading: `load_specular_mask()`, `mask_coverage()` |
| `surgtwin/uncertainty/signals.py` | Uncertainty computations: `compute_photo_residual()`, `compute_p95_scale()`, `compute_u_photo()`, `compute_w_photo()`, `compute_w_photo_with_mask()`, `w_photo_distribution_stats()` |
| `surgtwin/losses/uncertainty_weighted.py` | `uncertainty_weighted_photometric_l1()` — main loss with full diagnostics |
| `surgtwin/training/uncertainty_config.py` | `UncertaintyConfig` frozen dataclass |
| `surgtwin/training/uncertainty_trainer.py` | `UncertaintyTrainer(DepthGuidedTrainer)` — H1/H2/H3 variants, mask handling, w_photo stats |
| `scripts/precompute_masks.py` | CLI for specular mask generation + `mask_quality_report.json` |
| `scripts/train_uncertainty.py` | CLI entry: `--variant h1/h2/h3`, all locked params |
| `scripts/compare_m2b_vs_m3.py` | 7-run comparison: full history table + gate decision |

## Files Modified (4)

| File | Change |
|------|--------|
| `surgtwin/masks/__init__.py` | Exports for specular.py + io.py |
| `surgtwin/uncertainty/__init__.py` | Exports for signals.py |
| `surgtwin/losses/__init__.py` | Added `uncertainty_weighted_photometric_l1` export |
| `surgtwin/training/__init__.py` | (cleaned empty) |

## Locked Loss Configuration

```
L_total = mean(w_photo * |rgb_pred - rgb_gt|)
        + lambda_depth * mean(|depth_pred - depth_gt|[valid_depth_mask])

lambda_depth = 0.2
lambda_reg = 0.0
alpha = 2.0
w_photo_min = 0.15
densification = false
normalization_mode = p95_detached
```

### H1 Algorithm
```python
rgb_residual = abs(rgb_pred.detach() - rgb_gt).mean(dim=-1)
scale = quantile(rgb_residual.flatten(), 0.95).detach()
scale = clamp(scale, min=1e-4)
u_photo = clamp(rgb_residual / scale, 0.0, 1.0)
w_photo = clamp(exp(-alpha * u_photo), w_photo_min, 1.0)
```

### H2 Addition
```python
u_photo_combined = clip(u_photo + mask_boost * mask.float(), 0.0, 1.0)
w_photo = clamp(exp(-alpha * u_photo_combined), w_photo_min, 1.0)
```

## Diagnostics per Iter (64+ fields in metrics.jsonl)

`loss_total`, `photo_loss_weighted`, `depth_loss_raw_m`, `depth_loss_weighted`, `psnr`, `depth_rmse_m_raw`, `depth_mae_m_raw`, `abs_rel`, `depth_valid_ratio`, `w_photo_mean`, `w_photo_min`, `w_photo_max`, `w_photo_p10`, `w_photo_p50`, `w_photo_p90`, `fraction_w_photo_at_min`, `fraction_w_photo_at_one`, `w_photo_p90_minus_p10`, `u_photo_mean`, `p95_scale`, `n_gaussians`, `iter_time_s`, `vram_gb`, `grad_norm_before_clip`, `grad_norm_after_clip`

## Test Results

```
175 passed, 9 skipped in 4.49s
```

- **5 skipped** (CUDA): `test_depth_guided_trainer.py` (3), `test_depth_verification_synthetic.py` (1), `test_uncertainty_trainer.py` (1)
- **4 skipped** (pre-existing, optional deps): LPIPS/skimage
- **0 regressions** from M1 + M2-A + M2-B (120 → 175 tests, +55 new)

## Test Breakdown (6 new files, 55 new tests)

| Test file | Tests | Focus |
|-----------|-------|-------|
| `test_uncertainty_signals.py` | 19 | Detach constraint, p95 quantile, clamp, mask boost, distribution collapsing, NaN protection |
| `test_uncertainty_weighted_loss.py` | 10 | Loss shape, diagnostics keys, gradient flow, mask option, collapse detection, normalization mode |
| `test_uncertainty_config.py` | 5 | Defaults, frozen, H2/H3 variants |
| `test_uncertainty_trainer.py` | 6 | Config delegation, variants, lambda_reg locked, CUDA skip |
| `test_specular_mask.py` | 12 | HSV rules, dtype handling, morphology, coverage edge cases |
| `test_precompute_masks.py` | 5 | IO roundtrip, bool conversion, coverage edge cases |

## VM Run Plan (after approval)

```bash
# Step 1: precompute specular masks
python scripts/precompute_masks.py \
  --manifest data/processed/manifests/servct_manifest.jsonl \
  --output_dir data/processed/masks

# Step 2: M3-H1 (p95-detached, no masks)
python scripts/train_uncertainty.py \
  --manifest data/processed/manifests/servct_manifest.jsonl \
  --variant h1 --lambda_depth 0.2 --alpha 2.0 --w_photo_min 0.15 \
  --output_dir outputs/runs/uncertainty_m3_h1

# Step 3: M3-H2 (p95-detached + specular mask)
python scripts/train_uncertainty.py \
  --manifest data/processed/manifests/servct_manifest.jsonl \
  --variant h2 --lambda_depth 0.2 --alpha 2.0 --w_photo_min 0.15 \
  --mask_dir data/processed/masks \
  --output_dir outputs/runs/uncertainty_m3_h2

# Step 4: M3-H3 (conditional — only if min PASS fails)
# if best(H1,H2).val_psnr < 56.20 AND depth_rmse <= 0.030:
python scripts/train_uncertainty.py \
  --manifest data/processed/manifests/servct_manifest.jsonl \
  --variant h3 --lambda_depth 0.1 --alpha 2.0 --w_photo_min 0.15 \
  --mask_dir data/processed/masks \
  --output_dir outputs/runs/uncertainty_m3_h3

# Step 5: comparison table
python scripts/compare_m2b_vs_m3.py \
  --baseline outputs/runs/baseline_debug \
  --m2b_orig outputs/runs/depth_guided_m2b \
  --m2b_r1 outputs/runs/depth_guided_m2b_R1 \
  --m2b_r3 outputs/runs/depth_guided_m2b_R3 \
  --m3_h1 outputs/runs/uncertainty_m3_h1 \
  --m3_h2 outputs/runs/uncertainty_m3_h2 \
  --output_dir outputs/runs/m3_comparison
```

## Acceptance Checklist (built into compare_m2b_vs_m3.py gate_decision)

### Minimum PASS (13 AND conditions)
1. 1000 iterations complete
2. depth_semantics = metric_meters
3. densification = false
4. scale regularizer not used (lambda_reg=0.0)
5. val_psnr >= 56.20 dB
6. val_depth_rmse_m_raw <= 0.030 m
7. PSNR improves over M2-B R1
8. 0.15 <= mean_w_photo <= 0.95
9. fraction_w_photo_at_min < 0.90
10. fraction_w_photo_at_one < 0.90
11. p90_w_photo - p10_w_photo >= 0.05
12. w_photo distribution reported (min, max, p10, p50, p90)
13. normalization_mode = p95_detached

### Strong PASS (5 AND conditions)
1. val_psnr >= 56.20 dB
2. val_depth_rmse_m_raw <= 0.023 m or not worse than M2-B R1
3. val_abs_rel <= 0.18
4. 0.30 <= mean_w_photo <= 0.80
5. RGB-depth trade-off better than M2-B R1

## Next

VM production runs on L4 GPU with SERV-CT data. After VM results, gate decision via comparison table + handover to M4 planning.
