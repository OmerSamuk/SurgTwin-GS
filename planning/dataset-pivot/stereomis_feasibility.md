# StereoMIS Feasibility Assessment

**Date:** 2026-07-02
**Status:** READ-ONLY audit — no data downloaded.

## Source

| Field | Value |
|---|---|
| URL | https://zenodo.org/records/8154924 |
| DOI | 10.5281/zenodo.8154924 |
| Publisher | Zenodo (CERN) |
| Authors | Hayoz Michel, Allan Max et al. |
| Paper | Hayoz et al., IJCARS 2023, "Learning how to robustly estimate camera pose in endoscopic videos" |
| Repository | https://github.com/aimi-lab/robust-pose-estimator |

## License

**Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)**

Permitted: sharing, adaptation, research use.
Restrictions: no commercial use, ShareAlike derivatives, attribution required.

## Access / Download Procedure

| Property | Detail |
|---|---|
| Registration | **None required** — direct Zenodo download |
| Download method | Zenodo web UI or REST API (`wget`/`curl`) |
| File | `StereoMIS_0_0_1.zip` (11.2 GB) |
| Checksum | md5: `e014067d348ad81fff94ac039b97abe8` |
| Mirror | None — Zenodo primary |

## Expected Storage Size

| Stage | Size |
|---|---|
| Compressed archive | 11.2 GB (one `.zip`) |
| Extracted (raw video + masks + calibration) | ~15–20 GB (estimated) |
| Extracted frames (PNG, per-sequence) | ~50–100 GB (depending on res/codec) |
| Manifest (JSONL) | ~1–5 MB |

## RGB Availability

| Property | Detail |
|---|---|
| Modality | **Stereo video** (da Vinci Xi, vertically stacked) |
| Resolution | 720 × 576 (per camera, from paper) |
| Color | RGB (24-bit, from video) |
| Format | Video file (MP4/AVI container), vertically stacked = top half left, bottom half right |
| Preprocessing required | **Yes** — frame extraction + vertical split to get left/right views |
| Frames extracted | ~26–37 fps after extraction per sequence |

## Depth Availability

| Property | Detail |
|---|---|
| Metric depth GT | **NO** |
| Relative depth | Not provided |
| Pseudo-depth | Could be generated via stereo matching (OpenCV) |

## Pose Availability

| Property | Detail |
|---|---|
| Pose source | **Camera forward kinematics** (da Vinci Xi robot) |
| Format | 4×4 transformation matrices (SE(3)) — w2c or c2w |
| Synchronization | Synchronized with video feed at extracted frame rate |
| Coordinate convention | Robot base frame (not camera optical center) — requires transformation |
| Pose quality | Absolute pose may have drift (accumulated FK error); relative motion accurate |
| Trajectory length (total) | ~9.9 m (P2: P2_0–P2_8 sum of listed lengths) |

## Camera Intrinsics / Extrinsics Availability

| Property | Detail |
|---|---|
| Intrinsics | Provided (Zenodo bundle) |
| Format | Calibration file (likely matrix + distortion) |
| Extrinsics | Derived from forward kinematics |
| Baseline | Fixed stereo baseline (da Vinci Xi endoscope) |

## Tool Mask Availability

| Property | Detail |
|---|---|
| Type | **Auto-generated** tool segmentation masks |
| Frame target | Left video frame only |
| Format | PNG mask files (binary or multi-class) |
| Quality | Auto-generated; expected quality not guaranteed |
| Coverage | All sequences with tools present |

## Sequence / Frame Count

| Sequence | Subject | Duration | Frames (approx) | Split |
|---|---|---|---|---|
| P1 | Porcine 1 | 5min 19s | 9,570 | **train** |
| P2_0 | Porcine 2 | 3min 28s | 5,472 | test |
| P2_1 | Porcine 2 | 2min 36s | 4,243 | test |
| P2_2 | Porcine 2 | 2min 37s | 4,271 | test |
| P2_3 | Porcine 2 | 2min 35s | 4,212 | test |
| P2_4 | Porcine 2 | 2min 37s | 4,271 | test |
| P2_5 | Porcine 2 | 3min 18s | 5,371 | test |
| P2_6 | Porcine 2 | 52s | 1,355 | test |
| P2_7 | Porcine 2 | 2min 26s | 5,402 | test |
| P2_8 | Porcine 2 | 3min 55s | 9,165 | test |
| P3 | Porcine 3 | 2min 58s | 5,340 | test |
| **Total** | | **~33 min** | **~63,272 frames** | |

**Note:** The full StereoMIS dataset also includes 3 human subjects (H1, H2, H3) described in the paper (Hayoz et al. IJCARS 2023). The Zenodo package appears to contain only the porcine subset.

## Validation Split Feasibility

| Aspect | Assessment |
|---|---|
| Predefined split | **Yes** — P1 = train, P2 + P3 = test (11 splits total: 1 train, 10 test) |
| Train frames | ~9,570 (P1) |
| Validation from test | Can reserve 1–2 P2 sequences as val: e.g., P2_7 (~5,402 frames) or P2_0 (~5,472) |
| Sequence-level split | Recommended: hold out entire P2_0 or P2_7 as val set |
| Frame-level split | Possible but risks temporal leakage |
| Multi-sequence val | Better: train on P1, val on P2_0 + P2_7, test on remaining 8 sequences |

## Manifest Feasibility

| Aspect | Assessment |
|---|---|
| Schema structure | **Feasible** — can adapt SERV-CT manifest format |
| Per-row mapping | sample_id (sequence+frame), split, left_rgb_path (extracted frame), mask_path (auto-generated), intrinsics_path, c2w_path (forward kinematics) |
| Missing fields | right_rgb_path (needs vertical split), depth_path (none) |
| Conversion cost | Video → frames → manifest generation: ~1–2 hours pipeline |

## Minimum Render Smoke Feasibility

| Aspect | Assessment |
|---|---|
| Step 1: Download | 11.2 GB zip → ~2–15 min (depends on bandwidth) |
| Step 2: Extract frames | ffmpeg vertical split → ~30 min |
| Step 3: Build manifest | Script to enumerate frames + parse CSV/camera data → ~1 hour |
| Step 4: Smoke render | Point existing render pipeline at manifest; test 1–2 samples |
| Total smoke cost | ~2–3 hours (mostly unattended download + extraction) |
| Early blocker? | Need to confirm forward kinematics format (4×4 file vs CSV) |

## Risk: Metrics Without Depth GT

Since StereoMIS has **no metric depth ground truth**, direct depth metrics (RMSE, MAE, AbsRel) cannot be computed.

**Alternative metrics:**

| Metric | Data Required | Feasibility |
|---|---|---|
| **Pose accuracy** (RPE/ATE) | Predicted vs FK camera poses | **Yes** — FK provided per frame |
| **Stereo consistency** (disparity error) | Left/right RGB + pred depth | **Yes** — stereo pairs available |
| **Photometric consistency** (w_photo residual) | RGB frames only | **Yes** — w_photo already implemented |
| **Relative depth error** | Pseudo-depth from stereo matching | Medium — requires prior stereo matching |
| **Visual odometry metrics** | Sequential pose + RGB | **Yes** — FK provides baseline |
| **Render quality** (PSNR/SSIM/LPIPS) | Left RGB vs rendered from 3DGS | **Yes** — no GT needed for baseline comparison |

**Primary risk:** Cannot validate absolute metric depth. All evaluation is geometric (pose) or photometric (render quality, consistency).

## Risk Summary

| Risk | Level | Mitigation |
|---|---|---|
| No metric depth GT | 🔴 High | Use pose metrics (RPE/ATE), photometric consistency, render quality |
| Tool masks auto-generated | 🟡 Medium | Validate mask quality on a subset; can use raw RGB as fallback |
| Preprocessing overhead (video→frames) | 🟡 Medium | Automate with script; one-time cost |
| FK absolute pose drift | 🟡 Medium | Use relative pose metrics (RPE) instead of absolute (ATE) |
| Zenodo bandwidth limit | 🟢 Low | ~11 GB single download; no registration |
| Human subset not in archive | 🟡 Medium | Porcine-only available; human subjects may require separate request |

## Recommendation

**StereoMIS is the recommended next dataset** for generalization testing:
- Clear CC BY-NC-SA license
- Predefined train/test split (P1 train, P2/P3 test)
- Camera forward kinematics per frame (pose evaluation possible)
- Tool masks available (masking experiments)
- Breathing/deformation present (generalization challenge)
- ~63K frames from 11 sequences (sufficient for statistical significance)
