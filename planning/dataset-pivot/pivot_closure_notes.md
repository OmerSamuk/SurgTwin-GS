# Dataset Pivot — Closure Notes

**Date:** 2026-07-02
**Status:** Planning only — no data downloaded, no code changed.

## Context

Expert-answer-18 §16 recommended evaluating dataset pivot options (replace or
augment SERV-CT) for better generalization to _in-vivo_ tissue with breathing
and deformation. This document records the findings.

## Summary

| Question | Answer |
|---|---|
| Can we replace SERV-CT for metric depth eval? | **No** — no public endoscopic dataset provides metric depth GT |
| Should we augment with StereoMIS? | **Yes** (next sprint planning) |
| Should we augment with Hamlyn? | **Maybe** (lower priority, licensing concerns) |
| Is the pivot worth doing before M4-A2-1 completion? | **No** — does not affect current milestone |
| Should we download data now? | **No** — access/rights verified only |

## Rationale

- **M4-A2-1** evaluates whether w_photo-driven densification improves depth
  quality on SERV-CT. Adding a new dataset mid-milestone adds confounding
  variables (domain shift, no depth GT, preprocessing pipeline changes).
- Dataset pivot is an M5 concern (generalization phase), not M4
  (proof-of-concept on a single dataset).
- Both StereoMIS and Hamlyn lack metric depth GT → cannot evaluate the same
  metrics (depth_rmse, depth_mae, abs_rel). A separate eval protocol would
  be needed (e.g., relative depth error, point-cloud consistency).

## Next Steps (M5+)

1. Create `data/datasets/stereomis/` directory structure
2. Write `surgtwin/data/stereomis_loader.py` (video frame extraction +
   calibration parsing)
3. Add depth pseudo-GT via stereo matching (e.g., RAFT-Stereo or
   CRE-Stereo) for qualitative assessment
4. Implement cross-dataset eval script
5. (Optional) Add Hamlyn after StereoMIS pipeline stabilizes

## Caveats

- No code or data changes were made in this audit
- The access/license audit is based on public information and may require
  verification with dataset authors for specific use cases
- Hugging Face vs original Hamlyn license discrepancy (GPL-3.0 vs informal)
  needs resolution before any Hamlyn code is committed
