# StereoMIS Access Log

**Date:** 2026-07-02
**Purpose:** Pre-download access/license verification

## Source Metadata

| Field | Value |
|---|---|
| Official URL | https://zenodo.org/records/7727691 |
| DOI | 10.5281/zenodo.7727691 |
| Publisher | Zenodo (CERN) |
| Authors | Hayoz Michel, Allan Max, Johnathan Bursztyn, Raphael Sznitman |
| Paper | Michel Hayoz et al., "Learning how to robustly estimate camera pose from endoscopic surgical videos", IJCARS 2023 |
| Paper DOI | 10.1007/s11548-023-02860-0 |

## License

**Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)**

| Check | Result |
|---|---|
| Research allowed | Yes |
| Commercial use allowed | No |
| Share-alike required | Yes |
| Attribution required | Yes (cite Hayoz et al. IJCARS 2023) |
| Redistribution allowed | Yes, with same license |
| License file present in dataset | TBD on download |

## Dataset Details

| Property | Detail |
|---|---|
| Modality | Stereo video (da Vinci Xi) |
| Subjects | 3 in-vivo porcine |
| Sequences | 11 surgical sequences |
| Expected download size (zip) | 11.2 GB |
| Frames | ~63,000 (11 sequences × ~26–60 fps) |
| Resolution | 640×512 (estimated from stereo vertical stack) |
| Format | MP4 vertically stacked stereo video |
| Annotations | Tool masks (left, auto-generated), camera FK poses, calibration |
| Depth GT | None |
| Download method | Direct HTTP (Zenodo) — no registration required |

## Risk Assessment

| Risk | Status |
|---|---|
| License clarity (CC BY-NC-SA) | Clear |
| Ethical (porcine, not human) | Low |
| FK availability | Confirmed |
| Intrinsics availability | Confirmed |
| Tool masks | Auto-generated, quality variable |

## Local Storage

| Path | Purpose |
|---|---|
| `data/external/stereomis/raw/` | Raw zip + extracted video files |
| `data/processed/stereomis/frames/` | Extracted left/right PNG frames |
| `data/processed/stereomis/masks/` | Extracted tool masks (paired) |
| `data/processed/manifests/stereomis_manifest.jsonl` | Final manifest |
| `outputs/logs/stereomis_download.log` | Download log |
| `outputs/reports/` | Extraction / audit / smoke reports |

## Go Decision

- [x] License verified (CC BY-NC-SA)
- [x] Research use permitted
- [x] Ethical risk low (porcine)
- [x] Size manageable (11.2 GB)
- [ ] Download pending (Phase 2, Step 3)
- [ ] Checksum verification pending
