# Sprint 1 / Milestone 1 — Modular Refactor & Baseline Training

**Date:** 2026-06-28
**Status:** ✅ Complete — Baseline RGB-only L1 training runs on VM (L4)

## Executive Summary

Full Milestone 1 as per revised plan (6 expert revisions). 17 new files, 5 modified. Baseline training verified on VM (g2-standard-8, 1×L4) with SERV-CT synthetic data.

## Results

| Metric | Value |
|--------|-------|
| Initial loss | 0.3646 |
| Final loss (iter 1000) | 0.00088 |
| Val PSNR | 59.20 dB |
| Val SSIM | 0.9988 |
| Val LPIPS | 0.00014 |
| Gaussian count | 20,000 (sabit, densification yok) |
| Training time | ~20 saniye (L4 GPU) |
| VRAM | 0.053 GB |

## Expert Revisions Applied

1. **O3 düzeltmesi** — render depth NOT_USED (Sprint 0'da `relative_unaligned`)
2. **Render shape contract tests** (tests/test_render_shapes.py)
3. **Deterministic Experiment_1 split** (6 train / 2 val)
4. **configs/ erteleme onayı** — BaselineConfig dataclass yeterli
5. **LPIPS optional**; PSNR/SSIM numeric zorunlu
6. **pass/TODO kuralı** — `__init__.py` muaf, test skip gerekçeli kabul

## Fixes Applied (VM debugging)

- **gsplat wheel URL**: 404 → doğru `pt24cu121` pre-built wheel
- **backend_gsplat.py**: `_add_batch_v_dims` / `_remove_batch_v_dims` for v1.5.3 batch API (B, V, C dims)
- **backend_gsplat.py v2**: `_remove_batch_v_dims` `t[0,0]` → squeeze loop (SSIM shape mismatch fix)
- **trainer.py**: `out.rgb[..., :3]` (RGBA→RGB), `requires_grad_(True)`, optimizer uses `__dict__`
- **initialization.py**: `opacities.expand` (overlapping memory) → `torch.full`

## Files Created (17 new)

| File | Purpose |
|------|---------|
| `surgtwin/gaussian/gaussian_model.py` | GaussianModel dataclass |
| `surgtwin/training/seed.py` | set_seed() |
| `surgtwin/training/config.py` | BaselineConfig |
| `surgtwin/training/logging_utils.py` | JsonlLogger, collect_environment (cloud!) |
| `surgtwin/training/checkpointing.py` | save/load checkpoint |
| `surgtwin/training/trainer.py` | BaselineTrainer |
| `surgtwin/evaluation/image_metrics.py` | PSNR/SSIM/LPIPS |
| `scripts/train_baseline.py` | CLI entry point |
| `tests/test_gaussian_model.py` | 5 tests |
| `tests/test_seed.py` | 3 tests |
| `tests/test_checkpointing.py` | 2 tests |
| `tests/test_logging_utils.py` | 3 tests |
| `tests/test_image_metrics.py` | 5 tests |
| `tests/test_render_shapes.py` | 4 tests |
| `tests/test_trainer.py` | 2 tests |

## Acceptance Criteria (10/10 ✅)

| # | Criterion | Status |
|---|-----------|--------|
| 1 | Output in `outputs/runs/baseline_debug/` | ✅ |
| 2 | `metrics.jsonl` with per-iter loss/psnr | ✅ |
| 3 | `config.json` | ✅ |
| 4 | `environment.json` (cloud fields) | ✅ |
| 5 | `final_metrics.json` (PSNR numeric, SSIM numeric, LPIPS optional) | ✅ |
| 6 | Loss decreases | ✅ |
| 7 | Checkpoint at final iteration | ✅ (ckpt_001000.pt) |
| 8 | Validation renders saved | ✅ (2 val frames × 10 val cycles) |
| 9 | `report.md` 7 sections | ✅ |
| 10 | All tests pass | ✅ (51 passed) |

## Test Results

```
Local (Windows):  35 passed, 3 skipped  (skimage, LPIPS — graceful)
VM (L4, CUDA):    51 passed, 7 warnings (5.5s)
Sprint 0 tests:   all 6 green (backward compat verified)
```

## Next

- Milestone 2: depth-guided GS with lambda_depth=0.2
- Render depth semantics verification
