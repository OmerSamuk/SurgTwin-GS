# Dataset Pivot Decision Matrix

## Comparison Table

| Criterion | SERV-CT (current) | StereoMIS | Hamlyn |
|---|---|---|---|
| **License** | Research-only† | CC BY-NC-SA 4.0 | Informal / GPL-3.0 |
| **Metric depth GT** | Yes (CT, mm) | No | No (pseudo-depth via Libelas) |
| **Stereo rectified** | Yes | Yes (vertical stack, split needed) | Yes (EndoDepthAndMotion) |
| **Resolution** | 720×576 | 720×576 | Varies (320×240 – 720×576) |
| **Frames** | 16 | ~63,000 (11 sequences) | ~10K–20K estimated (3.05 GB) |
| **Subjects** | Cadaver (human) | Porcine (in-vivo) | Human (in-vivo) |
| **Tool occlusion** | No | Yes (auto masks, left frame) | Varies by sequence |
| **Tissue deformation** | Minimal (cadaver) | Yes (breathing, motion) | Yes |
| **Camera poses** | No | Yes (FK, 4×4 SE(3)) | Yes (SfM, relative) |
| **Calibration provided** | Yes (P1/P2/Q) | Yes (intrinsics + FK) | Yes (rectified version) |
| **Depth unit** | mm (metric) | N/A | mm (pseudo, Libelas) |
| **Data format** | PNG frames (extracted) | Video preprocessing needed | JPEG pairs (rectified, ready) |
| **Download size** | ~200 MB | 11.2 GB (zip) | 3.05 GB (HF mirror) |
| **Eval compatibility** | Native (metric) | Pose (RPE/ATE) + photometric | Photometric + pseudo-depth |
| **Preprocessing cost** | None (already extracted) | High (~1–2 hrs: extract + split) | Low (images ready) |
| **Validation split** | Predefined (8/8) | Predefined (P1 train, P2/P3 test) | Provided by Recasens et al. |
| **Human data ethical risk** | 🟢 Low (cadaver) | 🟢 Low (porcine) | 🔴 High (patient data) |
| **Angular diversity** | 🟡 Low (fixed boom) | 🟢 High (FK trajectory) | 🟢 High (SfM trajectory) |

† SERV-CT was provided directly; verify specific license terms.

## Phase 1 Updated Findings

### StereoMIS Strengths
- Clear CC BY-NC-SA license
- Predefined 11-sequence split (1 train, 10 test)
- Camera forward kinematics per frame at ~27 fps → **pose evaluation tractable**
- ~63K frames — sufficient for training
- Tool masks available for occlusion robustness tests

### StereoMIS Weaknesses
- Video → frame extraction required (one-time ~1–2 hrs)
- No metric depth GT → can only evaluate pose and photometric quality
- FK absolute pose drift → use relative pose metrics (RPE)
- Human subjects described in paper but not in Zenodo archive

### Hamlyn Strengths
- Images already extracted (rectified version)
- Pseudo-depth from Libelas available
- Human tissue — higher clinical relevance
- EndoDepthAndMotion provides predefined splits

### Hamlyn Weaknesses
- 🔴 Human in-vivo patient data — ethical burden, consent unclear
- License ambiguity (original) + GPL-3.0 (rectified)
- Dataset size smaller (3.05 GB)
- SfM poses are relative, no absolute reference
- Qualcommity-only risk if pseudo-depth is biased

## Updated Recommendation

| Priority | Dataset | Rationale |
|---|---|---|
| **1st (Critical)** | **SERV-CT** | Only metric depth GT for quantitative evaluation |
| **2nd (High)** | **StereoMIS** | Best balance: clear license, ~63K frames, FK poses, tool masks, porcine (low ethical risk) |
| **3rd (Medium)** | **Hamlyn** | Human tissue for qualitative check; higher ethical/license burden |

## Recommended Next Steps

1. **Do NOT pivot** away from SERV-CT for metric depth evaluation
2. **Add StereoMIS** as secondary generalization benchmark (Phase 2 candidate)
3. **Defer Hamlyn** to Phase 3 (after StereoMIS pipeline is validated)
4. **Both require pseudo-depth** generation (stereo matching) for any depth-based evaluation
5. **Camera geometry audit** must verify minimum pose diversity on candidate dataset
