# Hamlyn Dataset Feasibility Assessment

**Date:** 2026-07-02
**Status:** READ-ONLY audit — no data downloaded.

## Source

| Field | Value |
|---|---|
| Original URL | http://hamlyn.doc.ic.ac.uk/vision/ (currently unreachable) |
| DOI (TIB entry) | 10.57702/loc3hl5d |
| Publisher | Hamlyn Centre, Imperial College London |
| Contact | stamatia.giannarou03 "at" imperial.ac.uk |
| Rectified version | https://davidrecasens.github.io/EndoDepthAndMotion/ |
| HF mirror | https://huggingface.co/datasets/vslamlab/Hamlyn_Rectified_Dataset |
| Original paper | Stoyanov et al., MICCAI 2010 |
| Rectified paper | Recasens et al., RA-L 2021 |

## License / Access Status

### Original Hamlyn

| Aspect | Detail |
|---|---|
| Formal license | **None** — website states: *"Permission is given to use and publish all data on this website."* |
| Type | Informal permission statement (not a standard OS license) |
| Commercial use | Ambiguous — no explicit restriction but no formal license terms |
| Redistribution | Permitted by statement |
| Attribution | Requested — cite relevant Hamlyn papers |

### Rectified Version (EndoDepthAndMotion)

| Aspect | Detail |
|---|---|
| License | **GPL-3.0** |
| Source | https://huggingface.co/datasets/vslamlab/Hamlyn_Rectified_Dataset |
| Implications | Copyleft — derivatives must be GPL-3.0; acceptable for academic research |
| Restriction | May conflict with proprietary deployment (not applicable currently) |

### GPL-3.0 Implications

- Academic research: **acceptable**
- Derivative code: must be GPL-3.0 licensed (copyleft)
- Project compatibility: SurgTwin-GS is research-only; GPL-3.0 is compatible
- Risk for future: If code+depth model bundled as single artifact, GPL-3.0 applies

### Human In-Vivo Ethical / Data Usage Risk

| Risk | Level | Detail |
|---|---|---|
| Patient data | 🔴 High | In-vivo human surgical video — identifiable content possible |
| Original consent | 🔴 Unknown | Consent terms not publicly documented |
| Redistribution | 🟡 Medium | Original permission permits redistribution; no explicit patient consent waiver cited |
| Institutional approval | 🟡 Medium | May require ethics board confirmation for new training |
| De-identification | 🟡 Medium | Videos may contain identifiable anatomical features; no explicit de-identification confirmed |
| **Bottom line** | 🔴 High | Ethical burden significantly higher than porcine-only StereoMIS |

## RGB Availability

| Property | Detail |
|---|---|
| Modality | Stereo video (rectified version available) |
| Original sequences | 10+ sequences (porcine + human cardiac + abdominal) |
| Resolution | Varies: 320×240, 360×288, 640×480, 720×288, 720×576 |
| Color | RGB (24-bit) |
| Format (original) | Video files (container varies by sequence) |
| Format (rectified) | JPEG image pairs (uint8) |
| Preprocessing | Original: frame extraction needed. Rectified: images ready |

## Depth Availability

| Property | Detail |
|---|---|
| Metric depth GT | **NO** |
| Pseudo-depth | **Yes** — Libelas stereo matching provides depth maps (uint16 PNG, mm) |
| Depth quality | Stereo matching GT — errors in textureless regions; not metric-grade |
| Target accuracy | Semi-dense, approximate metric scale from stereo baseline |

## Pose Availability

| Property | Detail |
|---|---|
| Pose source | **Structure-from-Motion** (Recasens et al.) |
| Format | c2w or w2c matrices (from EndoDepthAndMotion) |
| Quality | SfM poses — no absolute reference frame; relative accuracy only |
| Coverage | Per-frame for rectified sequences |
| Scale | Unknown/ambiguous without reference object |

## Intrinsics / Extrinsics Availability

| Property | Detail |
|---|---|
| Intrinsics | **Provided** — calibration files included in rectified package |
| Extrinsics | Fixed stereo baseline from calibration |
| Rectified | Yes — images are rectified with known baseline |

## Sequence / Frame Count

| Sequence | Modality | Resolution | Content | Approx Frames |
|---|---|---|---|---|
| Seq 1–10 (Hamlyn original) | Stereo / Mono | 320×240 – 720×576 | Porcine, human cardiac, abdominal | Varies per seq |
| Rectified subset (EndoDepthAndMotion) | Stereo (rectified) | Varies | Human in-vivo | ~3.05 GB total |

Detailed per-sequence frame counts require download. The HF mirror (3.05 GB) contains rectified stereo pairs from at least 4–6 surgical sequences.

## Validation Split Feasibility

| Aspect | Assessment |
|---|---|
| Predefined split | **Yes** — EndoDepthAndMotion provides train/val splits (available as download) |
| Train/val coverage | Splits created by Recasens et al. — sequence-level |
| Adaptation for 3DGS | Train on >30K frames (estimated 80%), val on remaining 20% |
| Risk | Small dataset size limits reliable val split |

## Manifest Feasibility

| Aspect | Assessment |
|---|---|
| Schema structure | **Feasible** — can adapt SERV-CT manifest format |
| Per-row mapping | sample_id (sequence+frame), split, left_rgb_path, right_rgb_path (rectified pairs), depth_path (Libelas pseudo-GT), intrinsics_path, c2w_path (SfM) |
| Missing fields | Mask_path (none for original; possible from Endo-Depth repo) |
| Conversion cost | Low — rectified images are already extracted; cost is mapping → JSONL (~30 min) |

## Minimum Render Smoke Feasibility

| Aspect | Assessment |
|---|---|
| Step 1: Download | 3.05 GB from HF → ~5–10 min |
| Step 2: Extract images | Already extracted (JPEG) → no preprocessing |
| Step 3: Build manifest | Already organized by sequence → ~30 min |
| Step 4: Smoke render | Point render pipeline at manifest; test 1–2 samples |
| Total smoke cost | ~1 hour |
| Early blocker? | GPL-3.0 terms for depth model weights need review |

## Qualitative-Only Risk

Since Hamlyn has **no metric depth GT** and its pseudo-depth is from stereo matching:

| Risk | Detail |
|---|---|
| No absolute metric eval | Cannot compute depth RMSE, MAE, AbsRel for validation |
| Pseudo-depth bias | Libelas depth is baseline — improvements relative to it = unknown real-world gain |
| Photometric-only eval | w_photo, PSNR, SSIM, LPIPS are the only hard metrics |
| Mesh quality qualitative | Rendered output must be reviewed by surgical expert for anatomical plausibility |
| **Result** | **Qualitative-only risk: MEDIUM** — photometric metrics provide some signal; mesh quality remains subjective |

## Overall Risk Summary

| Risk | Level | Detail |
|---|---|---|
| License ambiguity (original) | 🟡 Medium | Informal permission; rectified GPL-3.0 acceptable |
| Human patient data | 🔴 High | Ethical burden; verification of consent/anonymization needed |
| No metric depth | 🟡 Medium | Photometric + pose metrics only |
| Pseudo-depth bias | 🟡 Medium | Libelas is baseline — innovation measured against it |
| Dataset size (rectified) | 🟡 Medium | 3.05 GB estimated; may be smaller than ideal for 3DGS |
| Original site unreachable | 🟢 Low | HF mirror + rectified version available |
| SfM pose accuracy | 🟡 Medium | No absolute reference — relative metrics only |

## Recommendation

**Hamlyn is a tertiary candidate:**

- Better for **qualitative / human-tissue demonstration** than quantitative evaluation
- GPL-3.0 on rectified version is acceptable for research but limits future distribution
- Patient data ethics require institutional verification
- Pseudo-depth provides a weak signal for evaluation
- Prefer StereoMIS as primary pivot; Hamlyn as second-step human tissue check
