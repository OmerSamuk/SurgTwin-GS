# StereoMIS Metric Policy

**Date:** 2026-07-02
**Status:** Planning — subject to revision during Phase 2 execution

## Rationale

StereoMIS has **no metric depth ground truth**. The existing SERV-CT depth evaluation
(RMSE, MAE, AbsRel, δ thresholds) cannot be directly applied. This document defines
which metrics are available, which are unavailable, and which candidates require
further analysis.

## Unavailable Metrics

| Metric | Reason |
|---|---|
| Depth RMSE (meters) | No GT depth map |
| Depth MAE (meters) | No GT depth map |
| AbsRel (|d−d*|/d*) | No GT depth map |
| δ<1.25, δ<1.25², δ<1.25³ | No GT depth map |
| Mean relative error (MRE) | No GT depth map |
| Any metric-depth comparison | No GT depth map |

## Available / Candidate Metrics

### Primary (low-risk, directly computable)

| Metric | Source | Notes |
|---|---|---|
| PSNR (RGB) | Rendered vs captured left RGB | Standard image quality |
| SSIM (RGB) | Rendered vs captured left RGB | Structural similarity |
| LPIPS (RGB) | Rendered vs captured left RGB | Perceptual similarity (if available) |

### Photometric (w_photo integration)

| Metric | Source | Notes |
|---|---|---|
| w_photo mean | Training stats | Measures photo-consistency weight evolution |
| w_photo variance | Training stats | Indicates convergence stability |
| Photometric loss value | Training stats | Baseline for convergence check |

### Pose / Trajectory Sanity

| Metric | Source | Notes |
|---|---|---|
| RPE (relative pose error) | FK vs estimated trajectory | Requires SLAM or COLMAP estimated poses |
| ATE (absolute trajectory error) | FK vs estimated trajectory | After alignment; FK must be usable as reference |
| Rotation drift per frame | FK vs estimated | Diagnostic for long-sequence consistency |
| Translation drift per frame | FK vs estimated | Diagnostic for long-sequence consistency |

**Important:** FK = forward kinematics (robot arm encoders). FK is not metric-perfect
but provides a strong reference trajectory for relative error computation.

### Stereo Consistency / Pseudo-Depth

| Metric | Source | Notes |
|---|---|---|
| Stereo disparity error | Left-right consistency | Measures stereo rectification quality |
| Pseudo-depth from stereo | SGBM / RAFT-Stereo | Can fill gap if depth-like metric needed |
| Pseudo-depth vs rendered depth | Rendered vs SGBM | Diagnostic for geometry quality |
| Pseudo-depth AbsRel | Rendered vs SGBM | Not a true metric — only relative |

### Qualitative

- Novel-view rendering panels (side-by-side with reference)
- Render sequence video (temporal consistency)
- Depth map visualization (colorized, no GT comparison)

## Metric Decision Matrix

| Phase | Metric Set | Implementation Cost |
|---|---|---|
| Initial (M5 prototype) | PSNR + SSIM + w_photo stats | Low |
| Mid | + LPIPS + RPE/ATE with FK | Medium |
| Advanced | + Pseudo-depth (SGBM) + consistency | Medium-High |
| Full | + SLAM trajectory + COLMAP SfM baseline | High |

## Recommended Policy

1. **Phase 2 (this sprint):** No training — smoke only. Metric policy documented.
2. **Phase 3 (M5 prototype):** Train with PSNR/SSIM/w_photo. No depth RMSE allowed.
   Use `weighted_score` with w_photo and PSNR only.
3. **Phase 4:** Add LPIPS and pseudo-depth diagnostic. Evaluate RPE/ATE feasibility.
4. **Conditional depth:** If pseudo-depth pipeline is mature, add it as a diagnostic
   (never as GT).

## Open Questions

- Should `weighted_score` in M4 config accept `w_photo` + `psnr` without depth RMSE?
  → Yes, `_SUPPORTED_VAL_METRICS` already includes `"weighted_score"`.
  → Trainer currently expects `depth_rmse` in weighted_score computation.
  → M5 trainer must handle missing depth gracefully.
- Is FK accurate enough for ATE/RPE reference?
  → da Vinci FK is typically ~mm-accurate. Need to verify on StereoMIS data.
- Can COLMAP/SfM run on StereoMIS to provide estimated poses for RPE?
  → Feasible but outside current sprint scope.
