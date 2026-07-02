# Hamlyn Dataset Access & License Audit

**Date:** 2026-07-02
**Status:** READ-ONLY audit — no data downloaded.

## Source

| Field | Value |
|---|---|
| URL | http://hamlyn.doc.ic.ac.uk/vision/ |
| DOI | 10.57702/loc3hl5d (2024 TIB entry) |
| Publisher | Hamlyn Centre, Imperial College London |
| Contact | stamatia.giannarou03 "at" imperial.ac.uk |
| Rectified version | https://davidrecasens.github.io/EndoDepthAndMotion/ |

## License

**No formal license** on original Hamlyn page. Statement: *"Permission is given to use and publish all data on this website."*

Rectified version (EndoDepthAndMotion): dataset files on Hugging Face carry **GPL-3.0**.

### Permitted (original)
- Use for research and publication
- Redistribution (permission statement)

### Restrictions (original)
- No formal license terms defined
- No commercial use clause (ambiguous without formal license)
- Attribution requested: cite relevant Hamlyn papers

### GPL-3.0 implications (rectified version)
- Copyleft: any derivative work must be GPL-3.0
- May conflict with proprietary/commercial deployment
- Acceptable for academic research

## Dataset Contents (Rectified Version)

| Property | Detail |
|---|---|
| Modality | Stereo video (laparoscopic/endoscopic) |
| Subjects | In-vivo patient data |
| Sequences | Multiple surgical procedures |
| Resolution | Varies by sequence |
| Format | Rectified stereo image pairs |
| Annotations | Camera calibration |
| Ground truth | Camera poses (from structure-from-motion), NO metric depth |
| Known papers | Recasens et al., Endo-Depth-and-Motion |

## Relevancy for SurgTwin-GS

| Criterion | Assessment |
|---|---|
| Stereo pairs | Yes (rectified version available) |
| Metric depth GT | **No** |
| Calibration | Yes |
| Tissue diversity | High (multiple procedures, human tissue) |
| Tool presence | Varies by sequence |
| Clinical relevance | Higher than porcine (human in-vivo data) |

## Access

- Free download from Hamlyn website (no registration)
- Rectified version via EndoDepthAndMotion page
- Hugging Face mirror (GPL-3.0): https://huggingface.co/datasets/vslamlab/Hamlyn_Rectified_Dataset

## Risk Assessment

- Original license ambiguity (informal permission statement)
- GPL-3.0 on rectified version imposes copyleft obligations
- No depth ground truth — cannot directly evaluate metric depth
- Patient data — ethical considerations for redistribution
- Best used for qualitative / generalization assessment only
