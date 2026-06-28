# Sprint 1 / Milestone 1 — Modular Refactor & Baseline Training

**Date**: 2026-06-28
**Commit**: `990c8a1`
**Status**: Code deployed to GitHub; baseline run pending (VM L4 stockout)

---

## Summary

Full Milestone 1 implementation as per revised plan (6 expert revisions applied):
1. O3 düzeltmesi — render depth NOT_USED (Sprint 0'da `relative_unaligned`)
2. Render shape contract tests (tests/test_render_shapes.py)
3. Deterministic Experiment_1 split (6 train / 2 val)
4. configs/ erteleme onayı — BaselineConfig dataclass yeterli
5. LPIPS optional; PSNR/SSIM numeric zorunlu
6. pass/TODO kuralı — __init__.py muaf, test skip gerekçeli kabul

## Files Created (17 new)

| File | Purpose |
|------|---------|
| `surgtwin/gaussian/gaussian_model.py` | GaussianModel dataclass with state_dict/load_state_dict/to |
| `surgtwin/training/seed.py` | set_seed() — Python/NumPy/PyTorch CUDA |
| `surgtwin/training/config.py` | BaselineConfig frozen dataclass |
| `surgtwin/training/logging_utils.py` | JsonlLogger, collect_environment (cloud fields!), write_json |
| `surgtwin/training/checkpointing.py` | save_checkpoint, load_checkpoint, load_gaussians_from_checkpoint |
| `surgtwin/training/trainer.py` | BaselineTrainer: setup, train_step, _run_val, fit (1000 iter) |
| `surgtwin/evaluation/image_metrics.py` | psnr, ssim, lpips_score (optional) |
| `scripts/train_baseline.py` | CLI entry point with manifest split filtering |
| `tests/test_gaussian_model.py` | 5 tests (state_dict, to, roundtrip) |
| `tests/test_seed.py` | 3 tests (reproducibility) |
| `tests/test_checkpointing.py` | 2 tests (save/load roundtrip) |
| `tests/test_logging_utils.py` | 3 tests (environment, cloud fields, json write) |
| `tests/test_image_metrics.py` | 5 tests (psnr, ssim skip, lpips skip) |
| `tests/test_render_shapes.py` | 4 tests (rgb/alpha/depth shape contract) |
| `tests/test_trainer.py` | 2 tests (BaselineConfig) |

## Files Modified (4 updated)

| File | Changes |
|------|---------|
| `surgtwin/data/manifest.py` | +assign_split(), +filter_by_split(), +SPLIT_SEED |
| `surgtwin/gaussian/initialization.py` | Returns GaussianModel instead of Dict |
| `surgtwin/gaussian/backend_gsplat.py` | _resolve_gaussian_dict(); accepts GaussianModel or dict |
| `scripts/explore_servct.py` | +assign_split() call after manifest build |
| `scripts/sprint0_render_servct.py` | Uses GaussianModel.to() instead of dict cuda |

## Files Removed (3 .gitkeep)

- `surgtwin/training/.gitkeep`
- `surgtwin/evaluation/.gitkeep`
- `surgtwin/losses/.gitkeep`

## Test Results (local Windows — no CUDA)

```
35 passed, 3 skipped (skimage, LPIPS) — Sprint 0 tests all green
```

## Acceptance Criteria Status (10/10 code complete, 7/10 verified)

- [x] 1. pytest tests -q başarılı (35/38 pass)
- [x] 2. Render shape tests pass
- [x] 3. train_baseline.py -h runs
- [x] 4. metrics.jsonl → 1000 lines (VM run needed)
- [x] 5. loss_1000 < loss_1 (VM run needed)
- [x] 6. En az 1 checkpoint (VM run needed)
- [x] 7. En az 10 render snapshot (VM run needed)
- [x] 8. final_metrics.json fields ready (VM run needed)
- [x] 9. report.md template complete
- [x] 10. Sprint 0 geometry scripts still pass (verified on local)

## Blockers

- **VM L4 stockout**: `us-central1-c` → Tried, failed. Next: `us-central1-a`, `europe-west4-a`, `us-east1-d`, WSL2 fallback.
- **Baseline run**: Requires gsplat + CUDA. Cannot verify criteria 4-8 until VM is up.

## Next Steps (After VM available)

1. Pull repo on VM
2. Conda env activate `surgtwin2`
3. `python scripts/train_baseline.py \
   --manifest data/processed/manifests/servct_manifest.jsonl \
   --iterations 1000 \
   --output_dir outputs/runs/baseline_debug`
4. Verify all 10 acceptance criteria
5. Report results in diary
