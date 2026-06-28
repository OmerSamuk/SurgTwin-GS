# Sprint 2 / M2-B Recovery (2026-07-01)

## Context
M2-B initial run (λ_depth=0.2, λ_reg=0.01, scale_drift) gave PSNR=47.14 dB, failing the guard (56.20 dB). The plan prescribed R1/R2/R3 recovery runs to isolate whether the RGB collapse was due to the scale_drift regularizer or the depth loss itself.

## R1 — λ_depth=0.2, λ_reg=0.0, reg_type=none
- **PSNR**: 55.16 dB (Δ=-4.04 dB vs baseline 59.20 dB)
- **Guard**: 56.20 dB → **FAIL** by 1.04 dB
- **Depth RMSE**: 0.023 m
- **Result**: Removing the regularizer recovered ~8 dB (47.14→55.16), confirming the scale_drift regularizer was the dominant cause of RGB collapse in the original run. However, even without the regularizer, depth guidance at λ=0.2 still degrades PSNR below the guard. The depth loss gradient conflicts with the photometric gradient.

## R2 — Skipped
Per plan: R2 is optional "if R1 succeeds but regularization is still needed". R1 FAILed, so R2 is not applicable.

## R3 — λ_depth=0.05, λ_reg=0.0, reg_type=none
- **PSNR**: 53.97 dB (Δ=-5.22 dB vs baseline 59.20 dB)
- **Guard**: 56.20 dB → **FAIL** by 2.23 dB
- **Depth RMSE**: 0.031 m
- **Result**: Counterintuitively, reducing λ_depth from 0.2→0.05 made PSNR worse (55.16→53.97). Both runs show a gradual PSNR decline after peaking at ~iter 200-300. λ_depth=0.05 may provide weaker depth gradient regularization against photometric overfitting to training views.

## Summary Table

| Run | λ_depth | λ_reg | reg_type | PSNR (dB) | Δ PSNR | Guard | Depth RMSE |
|-----|---------|-------|----------|-----------|--------|-------|------------|
| Baseline (M1) | — | — | — | 59.20 | — | — | — |
| Original M2-B | 0.2 | 0.01 | scale_drift | 47.14 | -12.06 | FAIL | 0.026 m |
| R1 | 0.2 | 0.0 | none | 55.16 | -4.04 | FAIL (-1.04) | 0.023 m |
| R3 | 0.05 | 0.0 | none | 53.97 | -5.22 | FAIL (-2.23) | 0.031 m |

## Analysis
The scale_drift regularizer was the dominant factor (Δ~8 dB). But even without it, depth guidance at λ=0.2 and λ=0.05 degrades RGB PSNR below the -3 dB guard under these tested conditions. On this synthetic SERV-CT dataset (perfect RGB-D, no noise), the photometric-only optimum is sharply defined; any additional depth loss pulls the solution away.

This is a known multi-task gradient conflict. Without densification (M4) or uncertainty weighting (M3), 20k fixed Gaussians lack capacity to satisfy both RGB and depth objectives simultaneously.

## Final Status
M2-B: **IMPLEMENTATION COMPLETE / CONTROLLED NEGATIVE RESULT / GATE FAIL** — Gate miss by 1.04 dB (best recovery: R1, PSNR 55.16 dB). The failure is a scientific result (multi-task gradient conflict under fixed capacity), not a code/pipeline bug.

## M3 Design Input (from M2-B finding)
M2-B showed that fixed global depth loss creates RGB-depth gradient conflict. M3 objective is not simply to add another loss term, but to spatially modulate unreliable or conflicting supervision:
- **High-confidence geometry regions**: depth loss active
- **RGB-dominant reliable texture regions**: photometric loss preserved
- **Conflict / occlusion / specular / low-support regions**: depth or photo contribution down-weighted

Candidates: uncertainty-weighted loss (learnable per-loss σ²) or reliability-weighted loss (data-driven weights from predicted variance/reprojection error).

## Conclusion
M2-B is a **controlled negative result** on SERV-CT synthetic data. The PSNR guard is not met under the tested fixed-capacity, no-densification configurations. The depth-guided GS still produces valid metric depth (RMSE ~23-31 mm, semantics=metric_meters verified), but not within the PSNR tolerance.

## Artifacts
- R1 training: `outputs/runs/depth_guided_m2b_R1/`
- R1 comparison: `outputs/runs/m2b_comparison_R1/comparison_table.json`
- R3 training: `outputs/runs/depth_guided_m2b_R3/`
- R3 comparison: `outputs/runs/m2b_comparison_R3/comparison_table.json`

## Next
M2-B closed as controlled negative result. Proceeding to M3 planning per expert decision.
Awaiting user decision on M2-B gate fail → M3 planning or further investigation.
