# Sprint 1 / Milestone 1 — Baseline RGB-only L1 Training

**Date:** 2026-06-28
**Status:** ✅ Complete

## Summary

Modüler refactor + baseline eğitim loop'u tamamlandı. Gaussian Splatting ile sentetik SERV-CT verisi üzerinde RGB-only L1 eğitimi çalışıyor.

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

## Fixes Applied

- **gsplat wheel URL**: 404 → doğru `pt24cu121` pre-built wheel
- **backend_gsplat.py**: `_add_batch_v_dims` / `_remove_batch_v_dims` for v1.5.3 batch API (B, V, C dims)
- **trainer.py**: `out.rgb[..., :3]` (RGBA→RGB), `requires_grad_(True)`, optimizer uses `__dict__`
- **initialization.py**: `opacities.expand` (overlapping memory) → `torch.full`
- **backend_gsplat.py v2**: `_remove_batch_v_dims` `t[0,0]` → squeeze loop (SSIM shape mismatch fix)

## Acceptance Criteria Check

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

## File Changes

- `surgtwin/gaussian/backend_gsplat.py` — batch dim fix, squeeze fix
- `surgtwin/training/trainer.py` — n_gaussians in final_metrics
- `surgtwin/gaussian/initialization.py` — torch.full fix

## Next

- Milestone 2: depth-guided GS with lambda_depth=0.2
- Render depth semantics verification
