# M4-0 Camera/Data Geometry Audit

**Date:** 2026-07-03
**Status:** ✅ COMPLETE

## Summary

All 16 frames in the SERV-CT manifest (`data/manifest.jsonl`) use the **same identity camera pose** for both left and right views. There is **zero angular diversity** across temporal frames. This confirms the current setup is a controlled 2.5D / depth-supervised smoke benchmark, not a full multi-view 3D reconstruction problem.

## Detailed Findings

### c2w_left (all 16 frames)
```
[[1.0, 0.0, 0.0, 0.0],
 [0.0, 1.0, 0.0, 0.0],
 [0.0, 0.0, 1.0, 0.0],
 [0.0, 0.0, 0.0, 1.0]]
```
**Identity matrix** — no rotation, no translation. Identical across Experiment_1 (frames 001–008) and Experiment_2 (frames 009–016).

### c2w_right (all 16 frames)
```
[[1.0, 0.0, 0.0, baseline_m],
 [0.0, 1.0, 0.0, 0.0],
 [0.0, 0.0, 1.0, 0.0],
 [0.0, 0.0, 0.0, 1.0]]
```
Only X-translation differs per experiment:
- Experiment_1 (frames 001–008): 0.00545m (5.45mm baseline)
- Experiment_2 (frames 009–016): 0.00550m (5.50mm baseline)

### Intrinsics (K)
- Experiment_1: K = [[996.4, 0, cx], [0, 996.4, cy], [0, 0, 1]], identical across all 8 frames
- Experiment_2: K = [[934.7, 0, cx], [0, 934.7, cy], [0, 0, 1]], identical across all 8 frames

### Angular Diversity
| Measurement | Value |
|---|---|
| Unique left camera poses | **1** (identity) |
| Unique right camera poses | **2** (one per experiment, X-translation only) |
| Temporal pose variation | **Zero** |
| Rotation variation | **Zero** |
| Baseline (stereo) | ~5.5 mm (≤2× pixel at 576×720) |

### Code Verification

`servct_calibration.py:47–54` (hardcoded identity):
```python
c2w_left = np.eye(4, dtype=np.float32)
w2c_left = np.eye(4, dtype=np.float32)
c2w_right = np.eye(4, dtype=np.float32)
c2w_right[0, 3] = baseline_m
w2c_right = np.eye(4, dtype=np.float32)
w2c_right[0, 3] = -baseline_m
```

## Benchmark Label

Per expert-answer-7/8:

> **Current SERV-CT setup is a "controlled 2.5D / depth-supervised smoke benchmark" with zero angular diversity.**
> Results should NOT be interpreted as full multi-view 3D reconstruction performance.

## Implications for M4-A1

1. **Stereopsis is minimal** — 5.5mm baseline at ~576×720 resolution provides weak multi-view geometric constraint.
2. **Densification alone may not close the depth RMSE gap** — multi-view angular diversity is needed for true 3D geometric consistency, which is absent here.
3. **Improvements from capacity scaling (50K) are tested under 2.5D conditions** — any positive result is valid for this benchmark but may not generalize to full multi-view endoscopic reconstruction.
4. **If M4-A1 fails to improve depth, the bottleneck may be these camera geometry constraints, not just representational capacity.**
