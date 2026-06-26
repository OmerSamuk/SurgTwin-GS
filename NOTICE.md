# NOTICE

This project develops an original modular research codebase for SurgTwin-GS.

External packages and repositories may be used as dependencies, methodological references, benchmark implementations, or backend components where licensing permits.

## Primary external references/components

- **gsplat**: Gaussian Splatting rasterization backend dependency.
- **3D Gaussian Splatting / diff-gaussian-rasterization**: reference and potential fallback backend.
- **UC-NeRF**: uncertainty-aware loss and cross-view pseudo-GT conceptual reference.
- **EndoNeRF and ForPlane**: baseline comparison references.
- **Endoscopy Corruptions**: future robustness testing reference.
- **SAM2/SAM3D/Mobile-SAM**: optional offline mask precomputation references.
- **EVO**: future pose/trajectory evaluation reference.
- **SERV-CT / servcttk**: primary dataset and calibration toolkit reference.

## Dataset

SERV-CT (Surgical Endoscopic Videos with CT) is available from UCL under a research license.
https://www.ucl.ac.uk/interventional-surgical-sciences/serv-ct

No external repository is used as the main project skeleton.
Any copied or adapted code must include file-level attribution and license compatibility review.
