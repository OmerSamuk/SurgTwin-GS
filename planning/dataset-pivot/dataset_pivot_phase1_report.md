# Dataset Pivot Sprint Phase 1 Report

**Date:** 2026-07-02
**Branch:** `dataset-pivot-phase1`
**Status:** READ-ONLY audit — no data downloaded.

## Executive Summary

This sprint evaluated StereoMIS and Hamlyn datasets for their suitability as a secondary benchmark alongside SERV-CT. Both datasets lack metric depth ground truth, so quantitative evaluation must rely on pose accuracy (RPE/ATE) and photometric consistency (w_photo, PSNR, SSIM) rather than absolute depth metrics.

**Key finding:** StereoMIS is the recommended next dataset for generalization testing. It offers the best balance of license clarity (CC BY-NC-SA 4.0), dataset size (~63K frames, 11 sequences), predefined train/test split, camera forward kinematics, tool masks, and low ethical risk (porcine subjects). Hamlyn is deferred to Phase 3 due to human patient data ethical concerns and license ambiguity.

## StereoMIS Feasibility

| Criterion | Status |
|---|---|
| License | ✅ **CC BY-NC-SA 4.0** — clear, research-compatible |
| Download | ✅ 11.2 GB zip from Zenodo |
| RGB | ✅ 720×576, video → frame extraction needed |
| Depth | ❌ No metric GT — pseudo-depth via stereo matching possible |
| Pose | ✅ Forward kinematics, 4×4 SE(3) per frame |
| Intrinsics | ✅ Provided with dataset |
| Tool masks | ✅ Auto-generated (left frame) |
| Frames | ~63,272 frames across 11 sequences |
| Split | Predefined: P1 (train), P2/P3 (test) |
| Ethical risk | 🟢 Low (porcine) |
| Angular diversity | 🟢 High (FK trajectories from 1323mm total path length) |

**Primary risk:** No metric depth GT → metric depth evaluation impossible.
**Mitigation:** Use pose metrics (RPE/ATE), w_photo, photometric render quality.

## Hamlyn Feasibility

| Criterion | Status |
|---|---|
| License | ❌ **Ambiguous** — original: informal permission; rectified: GPL-3.0 |
| Download | ✅ 3.05 GB from Hugging Face (rectified version) |
| RGB | ✅ Already extracted JPEG pairs |
| Depth | ❌ No metric GT — Libelas pseudo-depth available (weak signal) |
| Pose | ✅ SfM poses (relative, no absolute reference) |
| Intrinsics | ✅ Provided |
| Ethical risk | 🔴 **High** — human in-vivo patient data, consent unverified |
| Preprocessing cost | 🟢 Low — images are extracted |
| Split | ✅ Provided by EndoDepthAndMotion |

**Primary risk:** Human patient data ethics require institutional verification before any download/use.

## Camera Geometry Audit Plan

**Script:** `scripts/audit_camera_geometry.py`

The audit script computes camera pose diversity metrics from a manifest or .npy file of c2w matrices:

| Metric | Description |
|---|---|
| `unique_camera_pose_count` | Number of distinct camera poses (rotation + translation threshold) |
| `rotation_variation_deg` | Min/max/mean angular distance from mean rotation |
| `translation_variation_m` | Min/max/mean distance from mean camera position |
| `identity_camera_ratio` | Fraction of poses near identity matrix |
| `zero_angular_diversity_flag` | True if mean rotation variation < threshold (default 5°) |
| `approximate_mean_baseline_m` | Mean pairwise baseline between camera centers |
| `valid_validation_frame_count` | Number of frames in val set (from train/val comparison) |
| `train/val pose diversity comparison` | Ratio of val diversity to train diversity |

**Threshold Checks (§6):**
- Valid val frames ≥ 4 (preferably ≥ 6) ✓
- Unique camera poses ≥ 3 OR rotation variation ≥ 5° ✓
- Identity camera ratio low ✓
- Non-trivial translation/parallax evidence preferred ✓

**Status:** 23 unit tests pass (rotation angle, quaternion, identity ratio, full report, manifest loading, train/val comparison).

## Manifest Feasibility

Both datasets are compatible with a SERV-CT-like manifest schema:

| Field | StereoMIS | Hamlyn |
|---|---|---|
| `sample_id` | sequence + frame index | sequence + frame index |
| `sequence_id` | P1/P2\_0–P2\_8/P3 | sequence name |
| `frame_id` | extracted frame number | frame number |
| `split` | train/test (predefined) | provided by EndoDepthAndMotion |
| `left_rgb_path` | extracted from video vertical stack | JPEG image path |
| `right_rgb_path` | extracted from vertical stack | JPEG image path (rectified) |
| `depth_path` | N/A (no depth) | Libelas pseudo-depth (uint16 PNG) |
| `mask_path` | auto-generated PNG | N/A |
| `intrinsics_path` | from intrinsics file | from calibration file |
| `c2w` | from FK file (4×4) | from SfM file |
| `width/height` | 720×576 | varies (320×240 – 720×576) |
| `dataset_name` | "stereomis" | "hamlyn" |

**Recommendation:** Implement manifest as JSONL with the same schema as SERV-CT but with `depth_path` and `right_rgb_path` optional. No new dataclass needed in Phase 1.

## Contribution-Aware Feasibility (Updated)

See `rasterizer_contribution_feasibility.md` for full assessment.

**Short-term answer:** Visibility mask and radii proxy are feasible today via `meta["gaussian_ids"]` and `meta["radii"]` (Option A, ~1 day). Full per-Gaussian contribution weights require gsplat CUDA fork (Option C, 1–2 weeks).

**Phase 2 recommendation:** Implement Option A (radii mask) as a lightweight improvement in `select_densification_candidates` — filter Gaussians with zero/negligible screen footprint before clone/prune.

## Recommended Next Dataset

**StereoMIS** — based on:
1. ✅ Clear CC BY-NC-SA license
2. ✅ ~63K frames from 11 sequences (sufficient for statistical significance)
3. ✅ Predefined train/test split
4. ✅ Camera FK per frame for pose evaluation
5. ✅ Tool masks for occlusion robustness experiments
6. ✅ Breathing/deformation present (generalization challenge)
7. 🟢 Low ethical risk (porcine subjects)

## Recommended Next Implementation Task

**Dataset Pivot Sprint Phase 2 — StereoMIS Download + Pipeline Integration**

1. Download StereoMIS (11.2 GB from Zenodo) on VM
2. Implement video → frame extraction pipeline (ffmpeg, vertical split)
3. Build manifest JSONL for all 11 sequences
4. Run `audit_camera_geometry.py` on actual FK poses to verify pose diversity
5. Render smoke test (1–2 samples) using existing 3DGS pipeline
6. Document results + go/no-go for full training

## Blockers

| Blocker | Impact | Resolution |
|---|---|---|
| No metric depth GT on either dataset | Cannot compute depth RMSE/MAE/AbsRel | Use pose metrics + photometric metrics |
| StereoMIS video → frame extraction | 1–2 hours preprocessing | Automate with script |
| Hamlyn patient data ethics | 🔴 Cannot download without verification | Defer to Phase 3 |
| Zenodo download size (11.2 GB) | Bandwidth on VM needed | Schedule download on VM |
| Vehicle FK → c2w transform | Need to verify coordinate convention | Parse one sequence FK file first |

## Go/No-Go Decision

### Current Decision: **CONDITIONAL GO**

**Conditions for Go:**
1. ✅ Planning docs complete
2. ✅ Audit script draft exists and passes 23 tests
3. 🟡 **Pending:** Download StereoMIS on VM + verify FK format
4. 🟡 **Pending:** Run `audit_camera_geometry.py` on actual data
5. 🟢 Not blocked by license or ethics

**Phase 2 begins when the above conditions are met.**

---

*End of Dataset Pivot Sprint Phase 1*
