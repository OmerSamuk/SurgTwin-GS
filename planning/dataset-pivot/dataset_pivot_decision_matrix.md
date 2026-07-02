# Dataset Pivot Decision Matrix

## Comparison

| Criterion | SERV-CT (current) | StereoMIS | Hamlyn |
|---|---|---|---|
| **License** | Research-only† | CC BY-NC-SA 4.0 | Informal / GPL-3.0 |
| **Metric depth GT** | Yes (CT, mm) | No | No |
| **Stereo rectified** | Yes | Yes (vertical stack) | Yes (via EndoDepthAndMotion) |
| **Resolution** | 720×576 | ~720p | Varies |
| **Frames** | 16 | ~50K extracted | Varies |
| **Subjects** | Cadaver (human) | Porcine (in-vivo) | Human (in-vivo) |
| **Tool occlusion** | No | Yes (auto masks) | Yes |
| **Tissue deformation** | Minimal (cadaver) | Yes (breathing, motion) | Yes |
| **Camera poses** | No | Yes (kinematics) | Yes (SfM) |
| **Calibration provided** | Yes (P1/P2/Q) | Yes | Yes |
| **Depth unit** | mm (metric) | N/A | N/A |
| **Data format** | PNG frames | Video (preprocessing needed) | PNG / video |
| **Download size** | ~200 MB | ~15-20 GB | ~10+ GB |
| **Eval compatibility** | Native (metric) | Requires pseudo-depth | Requires pseudo-depth |

† SERV-CT was provided directly; verify specific license terms.

## Findings

1. **StereoMIS** is the strongest candidate for _generalization testing_:
   - CC BY-NC-SA is the clearest and most permissive license
   - Large number of frames, porcine tissue with breathing/deformation
   - Tool masks available for masking experiments
   - No depth GT — can only test _relative_ geometry

2. **Hamlyn** offers human tissue but:
   - License ambiguity on original; GPL-3.0 on rectified version
   - No depth GT (same limitation as StereoMIS)
   - Higher ethical/compliance burden (patient data)

3. **SERV-CT remains the only dataset with metric depth GT**:
   - No replacement for quantitative depth evaluation
   - 16 frames only — generalization limited
   - Cadaver tissue → no breathing/deformation

## Recommendation

| Use Case | Dataset | Priority |
|---|---|---|
| Metric depth evaluation | SERV-CT (current) | Critical (no substitute) |
| Generalization / tissue deformation | StereoMIS | High |
| Human tissue qualitative check | Hamlyn | Medium |
| Tool occlusion robustness | StereoMIS | Medium |

## Action Items

- [ ] Do NOT pivot away from SERV-CT for metric evaluation
- [ ] Add StereoMIS as secondary generalization benchmark (next sprint)
- [ ] Add Hamlyn as tertiary qualitative set (time permitting)
- [ ] Both StereoMIS and Hamlyn require pseudo-depth generation (e.g. stereo matching)
