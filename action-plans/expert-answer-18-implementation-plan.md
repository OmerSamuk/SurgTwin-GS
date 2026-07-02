# Expert-Answer-18 Implementation Plan

**Source:** `projects/surgtwin-gs/expert-feedback/expert-answer-18.txt`
**Status:** Approved by user (2026-07-02)
**Base branch:** `m4_a2_densification`

---

## Summary

SERV-CT hatti kapanmistir. Official sonuc degismemektedir:

> **CONDITIONAL PARTIAL POSITIVE / CONTROLLED 2.5D BENCHMARK COMPLETE**

Bu plan, expert-answer-18'deki 16 bolumu 5 feature branch + 1 local edit + planning dokumanlari olarak uygular.

---

## Feature Branches (tumu `m4_a2_densification` uzerinden)

| # | Branch | Icerik | Section |
|---|--------|--------|---------|
| B-C-D | `feat/best-checkpoint` | best_val altyapisi + config + 7 unit test | §3, §11 |
| C | `feat/state-migration` | Optimizer state migration + config + 8 test | §4, §11 |
| D | `feat/log-audit` | w_photo audit log field'lari + 5 test | §5 |
| E | `feat/rasterizer-contrib-feasibility` | Fizibilite dokumani (planning/doc) | §8 |
| F | `feat/dataset-pivot-planning` | Dataset pivot planning dokumanlari | §9, §10, §13 |

**VM'e ertelenen:** §6 overlay artifact'leri, §12 iter 800 reproducibility.

---

## A. SERV-CT Report Local Edits (PR disi)

**Dosya:** `outputs/runs/m4_a2_1_densify/m4_a2_1_review_report.md`

- Tarih: `2025-07-01` → `2026-07-01`
- Status: `SERV_CT_PARTIAL_POSITIVE` → `SERV_CT_PARTIAL_POSITIVE / CONTROLLED 2.5D BENCHMARK COMPLETE`
- Depth RMSE yorumu: bilimsel dili yumusat (n=2 uyarisi)
- Kapanis paragrafi ekle (expert-answer-18 §1)
- Overfitting riski: `Low` → `UNKNOWN under n=2 validation`
- iter 800 / iter 1000 ayrimi: best observation vs official result
- Pruning dili: expert-answer-18 §7 ifadesi

---

## B. feat/best-checkpoint — Kod Degisiklikleri

### B1. `surgtwin/training/uncertainty_config.py`

```python
# Best validation checkpoint
best_val_enabled: bool = True
best_val_metric: str = "depth_rmse"       # M4-A2 geometry-focused default
best_val_tiebreaker: str = "psnr"
best_val_metric_mode: str = "min"
best_val_tiebreaker_mode: str = "max"
```

### B2. `surgtwin/training/uncertainty_trainer.py`

- `__init__` (`:97` sonrasi): `self._best_val_metric: str = None` (runtime resolved name)
- `_resolve_val_metric(metric_name)` → `"depth_rmse"` → `"val_depth_rmse_m_raw"` mapping
- `_maybe_save_best_val(iter_idx, val_metrics)` → validation sonrasi cagrilir
  - `best_val_enabled=False` ise skip
  - Metric mode (min/max) + tiebreaker karsilastirmasi
  - Iyilesme varsa: `checkpoints/best_val.pt` yaz (save_checkpoint + extra metrikler)
  - `best_val_metrics.json` yaz (16 alan)
- `fit()` (`:566-580` ve `:588`): her `_run_val` sonrasi `_maybe_save_best_val` cagir
- `fit()` final_metrics (`:585-636`): `best_val_*` key'leri ekle
- `save()` (`:645-657`): dokunma (zaten calisiyor)

### B3. `tests/test_best_val_checkpoint.py` (yeni)

7 test:
1. `best_val_selects_lower_depth_rmse_when_depth_primary`
2. `best_val_uses_psnr_tiebreaker_when_depth_equal`
3. `best_val_selects_higher_psnr_when_psnr_primary`
4. `best_val_checkpoint_contains_optimizer_state`
5. `best_val_metrics_json_written`
6. `best_val_does_not_update_when_metric_worse`
7. `best_val_handles_missing_depth_metric_gracefully`

Helper pattern: module-level `_make_gaussians`, `_dense_config` (mirror `test_densification.py`).

---

## C. feat/state-migration — Kod Degisiklikleri

### C1. `surgtwin/training/uncertainty_config.py`

```python
clone_means_exp_avg_scale: float = 0.5   # §4 tandem movement riski damping
```

### C2. `surgtwin/gaussian/gaussian_model.py`

- `clone_gaussians(self, indices, offsets, return_parent_mapping=False)`:
  - `return_parent_mapping=True` iken `(None, parent_mapping)` donebilir
  - Veya her zaman parent mapping donebilir (basit tensor: `n_cloned` uzunlugunda, her clone icin parent index)
  - backward-compat: mevcut cagrilar `return_parent_mapping=False` (default)

### C3. `surgtwin/training/uncertainty_trainer.py`

- Yeni `_migrate_optimizer_state(clone_parent_map, keep_mask)` metodu:
  - Clone oncesi optimizer state yakala (`state_dict()['state']`)
  - Parent state'leri koru (shape remap)
  - Clone state'lerini parent'tan turet:
    - `means.exp_avg` → `parent * clone_means_exp_avg_scale`
    - `means.exp_avg_sq` → parent'tan kopya
    - scales/quats/opacities/colors → parent'tan kopya (exp_avg + exp_avg_sq)
  - Storage sharing kontrolu (`.clone()`)
  - Shape mismatch → fallback `_build_optimizer()` + warning
- `_densification_step` (`:306-307`): `self._build_optimizer()` → `self._migrate_optimizer_state(...)` (with fallback)

### C4. `tests/test_optimizer_state_migration.py` (yeni)

8 test:
1. `test_state_migration_preserves_existing_parent_states`
2. `test_state_migration_adds_clone_states_with_correct_shape`
3. `test_clone_means_exp_avg_is_damped`
4. `test_clone_exp_avg_sq_copied_or_valid`
5. `test_optimizer_step_after_state_migration_does_not_crash`
6. `test_state_migration_handles_zero_clone`
7. `test_state_migration_fallback_rebuild_on_shape_mismatch`
8. `test_parent_clone_do_not_share_state_tensor_storage`

---

## D. feat/log-audit — Kod Degisiklikleri

### D1. `surgtwin/training/densification.py`

`DensificationSelection` dataclass'ina 9 yeni field:
```python
selected_min_w_photo: float = 0.0
selected_p01_w_photo: float = 0.0
selected_p05_w_photo: float = 0.0
# selected_p10_w_photo / selected_mean_w_photo zaten var
w_photo_threshold: float = 0.3
w_photo_leak_count: int = 0
w_photo_near_threshold_count: int = 0
w_photo_threshold_margin_min: float = 0.0
```

`select_densification_candidates` icinde (`:199-200` sonrasi):
- `selected_w_photos` uzerinden min/p01/p05 hesapla
- `w_photo_threshold` config'ten al
- `w_photo_leak_count`: `selected_w_photos <= threshold` sayisi
- `w_photo_near_threshold_count`: `threshold < w_photo <= threshold+0.05` sayisi
- `w_photo_threshold_margin_min`: `min(selected_w_photo - threshold)`
- Tum return bloklarinda bu alanlari doldur

### D2. `surgtwin/training/uncertainty_trainer.py`

`_densification_step` log writer'inda (`:310-341`):
- 9 yeni alani `log_entry`'ye ekle

### D3. Testler

`tests/test_densification.py`'ye eklenecek veya yeni `tests/test_densification_logging.py`:

5 test:
1. `test_w_photo_leak_count_zero_when_all_above_threshold`
2. `test_w_photo_leak_count_detects_threshold_violation`
3. `test_w_photo_near_threshold_count`
4. `test_w_photo_threshold_margin_min`
5. `test_zero_candidate_still_logs_audit_fields`

---

## E. feat/rasterizer-contrib-feasibility — Doc

**Dosya:** `planning/dataset-pivot/rasterizer_contribution_feasibility.md`

Icerik (expert-answer-18 §8 basliklari):
- Mevcut backend durumu (gsplat 1.5.3, RenderOutput)
- gsplat forward/meta ciktisi: radii, visibility, accumulated alpha, per-Gaussian contribution
- diff-gaussian-rasterization karsilastirmasi
- projection_based_approximate failure kosullari
- Multi-view angular-diversity riski
- Minimum backend API degisikligi
- Implementation path, riskler, performans/memory etkisi, oneri

---

## F. feat/dataset-pivot-planning — Docs

**Klasor:** `planning/dataset-pivot/`

Dosyalar (expert-answer-18 §9, §10, §13):

1. **`stereomis_feasibility.md`**
   - Erisim/lisans durumu
   - Data format envanteri
   - RGB/depth/pose/intrinsics availability
   - Frame/sequence count
   - Kalibrasyon kalitesi
   - Manifest feasibility
   - Val seti >=4 frame criteria
   - Kamera geometry audit checklist (§10)

2. **`hamlyn_feasibility.md`**
   - Ayni basliklar
   - Vendor loader referanslari (UC-NeRF `data/hamlyn.py`, Forplane `hamlyn_datasets.py`)
   - Depth genellikle relative/monocular → `depth_semantics=relative_unaligned`

3. **`dataset_pivot_decision_matrix.md`**
   - StereoMIS vs Hamlyn karsilastirma matrisi

4. **`servct_closure_notes.md`**
   - SERV-CT kapanis dili (§13, EN+TR)
   - Real-pose densification default'lari (§11)
   - Funding/concept paper paragrafi

---

## Calisma Sirasi

| Sirada | Ne | Branch/Location |
|--------|----|-----------------|
| 1 | action-plans/expert-answer-18-implementation-plan.md | Git'siz (plan doc) |
| 2 | A. Report local edits | `outputs/` (gitignore, PR disi) |
| 3 | B. feat/best-checkpoint | Branch: `feat/best-checkpoint` |
| 4 | C. feat/state-migration | Branch: `feat/state-migration` |
| 5 | D. feat/log-audit | Branch: `feat/log-audit` |
| 6 | E. feat/rasterizer-contrib-feasibility | Branch: `feat/rasterizer-contrib-feasibility` |
| 7 | F. feat/dataset-pivot-planning | Branch: `feat/dataset-pivot-planning` |

---

## PR / Test Gate (§15)

Her feature branch PR'sinde:
- Yeni test dosyasi gecer
- `pytest tests/test_densification.py`
- `pytest tests/test_evaluate_m4_a2_1.py`
- `pytest tests/test_evaluate_m4_a2_0.py`
- Mumkunse: `pytest tests/`
- PR aciklamasinda: *"Does not change official SERV-CT M4-A2-1 result."*
