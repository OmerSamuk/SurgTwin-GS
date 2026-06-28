# M2-B Recovery Report — Depth-Guided Gaussian Splatting

## Gate Criterion
`depth_guided_val_psnr >= baseline_val_psnr - 3.0 dB`
Threshold: **56.20 dB** (baseline = 59.20 dB)

## Recovery Runs

### R1: λ_depth=0.2, λ_reg=0.0, reg_type=none
- **Result**: FAIL — PSNR 55.16 dB (Δ=-4.04 dB, guard miss by 1.04 dB)
- Key finding: Removing scale_drift regularizer recovered ~8 dB (from 47.14), proving the regularizer was the dominant cause of RGB collapse.
- Depth RMSE: 0.023 m

### R3: λ_depth=0.05, λ_reg=0.0, reg_type=none
- **Result**: FAIL — PSNR 53.97 dB (Δ=-5.22 dB, guard miss by 2.23 dB)
- Key finding: Even near-negligible depth weight (0.05) degrades PSNR more than λ=0.2 (55.16 > 53.97). Depth guidance at 0.2 acts as a mild implicit regularizer against photometric overfitting; at 0.05 it is too weak to prevent gradual PSNR decay.
- Depth RMSE: 0.031 m

## Final Verdict

**M2-B status: IMPLEMENTATION COMPLETE / CONTROLLED NEGATIVE RESULT / GATE FAIL**

```
M2-B status: IMPLEMENTATION COMPLETE / EXPERIMENTAL GATE FAIL

Reason:
- Depth loss implementation, metric-depth guard, logging, snapshots and comparison pipeline completed.
- M2-A metric depth semantics remained valid.
- However, depth-guided training failed the PSNR preservation guard under tested configurations.
- Best recovery run: R1, lambda_depth=0.2, lambda_reg=0.0, PSNR=55.16 dB.
- Required threshold: 56.20 dB.
- Gate miss: 1.04 dB.
```

The multi-task gradient conflict between RGB and depth objectives cannot be resolved with fixed 20k Gaussians without densification or uncertainty weighting. Under the tested fixed-capacity, no-densification configurations, depth loss at λ=0.2 and λ=0.05 degraded PSNR below the -3 dB guard.

### Summary Table

| Run | λ_depth | λ_reg | reg_type | PSNR (dB) | Guard (56.20) | Depth RMSE |
|-----|---------|-------|----------|-----------|---------------|------------|
| Baseline | — | — | — | 59.20 | — | — |
| Original | 0.2 | 0.01 | scale_drift | 47.14 | FAIL (-12.06) | 0.026 m |
| R1 | 0.2 | 0.0 | none | 55.16 | FAIL (-1.04) | 0.023 m |
| R3 | 0.05 | 0.0 | none | 53.97 | FAIL (-2.23) | 0.031 m |

### What worked ✅
- Depth semantics: metric_meters verified (M2-A gate pass)
- Depth geometry: ~23-31 mm RMSE on SERV-CT
- All code paths ran clean (0 failures, 127+ tests passing)
- Scale_drift regularizer → use only with very low weight if at all

### What didn't ❌
- Under the tested fixed-capacity, no-densification configurations, depth loss at λ=0.2 and λ=0.05 degraded PSNR below the -3 dB guard on synthetic SERV-CT
- Scale_drift regularizer at λ=0.01 dominates the loss and causes severe RGB collapse

### Recommendations for M3/M4
- Consider **uncertainty-weighted loss** (M3) to reduce gradient conflict — may allow depth guidance without RGB degradation
- Consider **densification** (M4) to increase capacity for multi-objective optimization
- On real surgical data (noisy depth from monocular/stereo), depth guidance may be less competitive with RGB, changing the trade-off

## Artifacts
- `outputs/runs/depth_guided_m2b_R1/` — R1 training (ckpt, renders, metrics)
- `outputs/runs/m2b_comparison_R1/` — R1 comparison table
- `outputs/runs/depth_guided_m2b_R3/` — R3 training
- `outputs/runs/m2b_comparison_R3/` — R3 comparison table
