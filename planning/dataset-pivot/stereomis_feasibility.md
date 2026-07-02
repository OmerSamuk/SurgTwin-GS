# StereoMIS Access & License Audit

**Date:** 2026-07-02
**Status:** READ-ONLY audit — no data downloaded.

## Source

| Field | Value |
|---|---|
| URL | https://zenodo.org/records/7727691 |
| DOI | 10.5281/zenodo.7727691 |
| Publisher | Zenodo (CERN) |
| Authors | Hayoz Michel, Allan Max et al. |
| Paper | Hayoz et al., IJCARS 2023 |

## License

**Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)**

### Permitted
- Sharing, reproduction, redistribution
- Adaptation and transformation
- Research and academic use (non-commercial)

### Restrictions
- **No commercial use**: cannot use for commercial products/services
- **ShareAlike**: derivative works must carry the same license
- Attribution required: cite Hayoz et al. IJCARS 2023

## Dataset Contents

| Property | Detail |
|---|---|
| Modality | Stereo video (da Vinci Xi) |
| Subjects | 3 in-vivo porcine |
| Sequences | 11 surgical sequences |
| Resolution | Not specified (likely 720p or 1080p) |
| Format | Vertically stacked stereo video files |
| Annotations | Tool masks (left frame, auto-generated), camera forward kinematics, calibration |
| Ground truth | Camera poses (forward kinematics), NO depth ground truth |
| Duration | ~30 min total |
| FPS | 26-60 fps captured, extracted at 26-37 fps |

## Relevancy for SurgTwin-GS

| Criterion | Assessment |
|---|---|
| Stereo pairs | Yes (vertically stacked video) |
| Metric depth GT | **No** — only pose GT |
| Calibration | Yes (camera intrinsics provided) |
| Tissue diversity | Moderate (porcine, 3 subjects) |
| Tool presence | Yes (tool masks provided) |
| Breathing/deformation | Yes (explicitly includes these) |

## Access

- **Free download**: Yes, no registration required
- **File size**: ~15-20 GB (estimated from video content)
- **Format compatibility**: Requires frame extraction from video (preprocessing needed)

## Risk Assessment

- License (CC BY-NC-SA) compatible with academic research
- No depth ground truth → cannot directly replace SERV-CT for metric depth evaluation
- Could use for generalization testing if pseudo-depth generated
- Tool masks are auto-generated (quality not guaranteed)
