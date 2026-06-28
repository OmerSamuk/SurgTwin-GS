# Sprint 0 Setup — 26 June 2026

## Completed

- Full SurgTwin-GS codebase created from scratch (SERV-CT oriented)
- 47 files committed: core library, scripts, tests, documentation
- Pushed to `github.com/OmerSamuk/SurgTwin-GS` (2 commits)

## Sprint 0A — Expert Mandated Revisions

| Rev | Change | Status |
|---|---|---|
| 1 | `reprojection_overlay.png` added to `validate_servct_geometry.py` | ✅ |
| 2 | Hard-coded 576×720 removed; `explore_servct.py` reads from actual RGB file | ✅ |
| 3 | `backend_gsplat.py`: `gsplat.rasterization()` → `(render_colors, render_alphas, meta)` unpack | ✅ |
| 4 | CUDA fail-fast in `sprint0_render_servct.py` (RuntimeError, not warning) | ✅ |
| 5 | `sprint0_revision_notes.md` deleted (both vault-root and proje copies) | ✅ |

## VM Stage (Incomplete)

- Old VM `qwmo-phase1` (n2-highcpu-16, no GPU, TERMINATED) deleted from `qwmo-framework`
- New `SurgTwin-GS` project blocked: project quota full
  - 2 auto-generated projects deleted; still in 30-day grace period
  - Continuing with `qwmo-framework` project
- VM creation blocked: `GPUS_ALL_REGIONS` global quota = 0
- **NEXT step:** Request GPU quota increase at:
  `https://console.cloud.google.com/apis/api/compute.googleapis.com/quotas?project=qwmo-framework`
  → Set `GPUS_ALL_REGIONS` from 0 to 1

## Planned VM Config

- **Name:** `surgtwin-gs-phase1`
- **Zone:** `europe-west4-a`
- **Machine type:** `g2-standard-8` (8 vCPU, 32 GB RAM)
- **GPU:** 1 × NVIDIA L4 (24 GB VRAM)
- **Boot disk:** 200 GB SSD, Ubuntu 22.04 LTS
- **Image:** `ubuntu-2204-jammy-v20260623`

## Repo State

```
38a99af  Sprint 0A: backend_gsplat.py API interpretation fix
f6a79b6  Sprint 0A: Expert mandated revisions applied
2338193  Sprint 0: SERV-CT pipeline MVP
```

All 23 unit tests passing.
