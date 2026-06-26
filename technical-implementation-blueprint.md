# SurgTwin-GS Technical Implementation Blueprint v0.1

**Project:** SurgTwin-GS  
**Full title:** Uncertainty-Weighted Gaussian Splatting for Decision-Support-Oriented 3D Surgical Scene Reconstruction from Endoscopic Images  
**Document type:** Engineering execution blueprint  
**Phase:** Phase 1 — laboratory research prototype  
**Revision date:** 26 June 2026  
**Review integration:** Gemini approved without changes; DeepSeek mandatory revisions integrated: inference-safe mapper features, realistic pose tolerances, and explicit depth-output semantics.  
**Primary language for code/comments:** English  
**Primary language for project documentation:** Turkish or English, depending on downstream funding document  
**Execution target:** opencode / coding agent / human developer  

---

## 0. Non-Negotiable Instruction to the Coding Model

This document is the source of truth for the first implementation phase. The coding model must not fill architectural gaps with its own assumptions.

The model must obey the following rules:

1. Do not replace the selected development strategy.
2. Do not turn SurgTwin-GS into a fork of any existing repository.
3. Do not integrate a different renderer backend unless it is one of the approved backends listed in this file.
4. Do not run SAM2, SAM3D, Mobile-SAM, or any segmentation model inside the training loop.
5. Do not move the lightweight confidence/risk mapper to Phase 2. It is part of Phase 1.
6. Do not hard-code local machine paths. Use the default paths defined in this file and allow CLI overrides.
7. Do not create runnable code containing `TODO`, `pass`, empty placeholder functions, or silent fallbacks. Abstract interfaces may raise `NotImplementedError`, but executable scripts must either complete or fail with a clear error message.
8. Do not silently ignore missing RGB, depth, pose, mask, or calibration files.
9. Do not change coordinate conventions without updating all projection/unprojection tests.
10. Do not report success unless the acceptance criteria of the relevant sprint are satisfied.
11. If a required dependency fails to install, stop and report the failing command, Python version, CUDA version, PyTorch version, and complete error trace. Do not choose a new dependency unless this document already lists it as a fallback.
12. All code copied or adapted from external repositories must be attributed in `NOTICE.md` and must comply with the corresponding license.
13. The confidence/risk mapper must not use ground-truth RGB residuals, ground-truth depth residuals, or held-out right-view errors as inference-time input features. These values may be used only for training targets, training-time loss diagnostics, or pseudo-GT generation.
14. Depth outputs from any renderer backend must explicitly state their semantics: `metric_meters`, `relative_aligned`, `relative_unaligned`, or `unavailable`. Metric depth losses and metric depth metrics must not consume non-metric depth.

---

## 1. Scope and Research Boundary

SurgTwin-GS Phase 1 is a controlled laboratory research prototype. It is not a clinical product, not a medical device, not an autonomous surgery system, and not a real-time intraoperative deployment.

The Phase 1 objective is to test whether uncertainty signals can be integrated into Gaussian Splatting optimization, Gaussian density control, and inference-time confidence/risk map generation for endoscopic 3D reconstruction.

Phase 1 must produce:

1. A reproducible SERV-CT-compatible data pipeline.
2. A minimal Gaussian Splatting rendering baseline.
3. A depth-guided Gaussian Splatting baseline.
4. An uncertainty-weighted photometric/geometric loss module.
5. A multi-criteria densification/pruning policy.
6. A cross-view pseudo-ground-truth pipeline.
7. A lightweight inference-time confidence/risk mapper.
8. A metric suite covering visual quality, geometry, uncertainty calibration, cross-view risk, and runtime performance.
9. Ablation results showing the contribution of each module.

Phase 1 must not attempt:

1. Full endoscopic SLAM.
2. Monocular pose estimation from scratch.
3. Clinical validation.
4. Regulatory validation.
5. Autonomous surgical decision-making.
6. Real-time robotic control.
7. Full CUDA kernel development from scratch.

---

## 2. Locked Architectural Decisions

| Area | Locked Decision | Rationale |
|---|---|---|
| Codebase strategy | SurgTwin-GS will be written as an original modular research codebase. | Enables ablation, reproducibility, ownership, and funding credibility. |
| Use of external repositories | External repositories are references, baselines, or backend components; none becomes the main project skeleton. | Prevents hidden coupling and spaghetti modifications. |
| Primary rasterization backend | `gsplat` is the primary Sprint 0 backend. | It provides PyTorch-facing Gaussian Splatting rasterization and is more suitable for rapid modular prototyping than directly embedding the original 3DGS repository. |
| Fallback backend | A forked `diff-gaussian-rasterization` backend may be added later behind the same renderer interface. | Protects the project if `gsplat` lacks required intermediate outputs or has installation/runtime issues. |
| Future backend | A custom SurgTwin CUDA backend may be created in later phases only after the algorithmic prototype is validated. | Avoids premature CUDA engineering. |
| Renderer abstraction | Mandatory `RendererBackend` interface. | Prevents the training loop, loss functions, and uncertainty modules from depending on one rasterizer implementation. |
| Development strategy | Code-first, refactor-second. | First produce a small working pipeline; then modularize. |
| SERV-CT strategy | Phase 1 starts with calibrated RGB-D/stereo data (SERV-CT rectified stereo). Pose estimation is not a Phase 1 claim. | Isolates uncertainty-aware reconstruction from camera tracking. |
| Mask strategy | Tool, occlusion, and specular masks are precomputed offline and loaded during training. | Protects VRAM and runtime. |
| UC-NeRF use | Use uncertainty/loss and cross-view pseudo-GT ideas only; do not copy the NeRF architecture. | UC-NeRF is implicit/MLP-based; SurgTwin-GS is explicit Gaussian primitive-based. |
| Uncertainty representation | Combine per-pixel rasterization signals and optional per-Gaussian reliability attributes. | Matches the explicit Gaussian representation. |
| Lightweight mapper | Phase 1 includes a PyTorch MLP mapper. `tiny-cuda-nn` acceleration is postponed to Phase 2. | Inference-time risk mapping is part of the main hypothesis. |
| Configuration system | Sprint 0 uses Python constants/argparse. YAML config begins after the first working renderer exists. | Avoids premature abstraction. |
| Main ablation order | Baseline GS → Depth-guided GS → Uncertainty-weighted GS → Density-control GS → Full SurgTwin-GS. | Isolates the contribution of each component. |

---

## 3. Approved External Repository Roles

The coding model must keep these roles fixed.

| Repository / Package | Role in SurgTwin-GS | Integration rule |
|---|---|---|
| `gsplat` | Primary Gaussian rasterization backend for Sprint 0 and early MVP. | Install as a dependency; wrap behind `RendererBackend`. |
| Original 3DGS / `diff-gaussian-rasterization` | Reference and possible fallback backend. | Do not make it the project skeleton. If used, fork and wrap behind `RendererBackend`. |
| UC-NeRF | Uncertainty-aware loss and cross-view uncertainty concept reference. | Do not copy NeRF-specific architecture. Adapt formulas to per-pixel/per-Gaussian signals. |
| EndoNeRF | Baseline comparison. | Keep in separate environment or external benchmark folder; do not embed into core code. |
| ForPlane | Baseline comparison for deformation-aware neural rendering. | Keep external; do not embed into core code. |
| Endoscopy Corruptions | Robustness testing under specular/blur/occlusion-type corruptions. | Add to evaluation pipeline after baseline and uncertainty modules work. |
| SAM2 / SAM3D / Mobile-SAM | Offline tool/occlusion mask precomputation. | Never call inside the training loop. |
| EVO | Phase 2 camera trajectory evaluation. | Not required for Sprint 0-4. |
| `tiny-cuda-nn` | Future optimization for lightweight mapper. | Not used in Phase 1 MVP unless all core Phase 1 modules already work. |

The following notes expand the integration intent for each approved external repository, derived from the project's vault README analyses. These notes are conceptual and do not introduce code-level requirements.

### 3.1 UC-NeRF Integration Notes

- Photometric confidence → uncertainty conversion (cost-volume variance) provides the most concrete realization of the cross-view disagreement signal in WP3.
- The two-branch (base/adapt) MLP blending pattern serves as a comparison reference: UC-NeRF's hard network gating vs SurgTwin-GS's soft down-weighting will be isolated as a single ablation in the Phase 1 matrix.
- Scale-and-shift-invariant depth loss from UC-NeRF is transferable as a principle for GS depth loss design under scale ambiguity in endoscopic scenes.
- The SCARED data loader skeleton is a direct starting point for Phase 1 SERV-CT pipeline (now replaced by SERV-CT as primary dataset).
- Pretrained CasMVSNet weights may be reused as an independent, frozen uncertainty estimator without fine-tuning, matching the charter's "soft down-weighting with lower bound" mitigation strategy.

### 3.2 EndoNeRF Integration Notes

- Hard masking (zeroing loss in tool regions) is the exact opposite of SurgTwin-GS's soft down-weighting strategy; the difference will be tested in the ablation matrix as Baseline (no mask) vs Hard masking (EndoNeRF) vs Soft re-weighting (SurgTwin-GS).
- The DirectTemporalNeRF canonical + deformation architecture is the reference for Phase 2 Temporal Gaussian Splatting: canonical GS primitives + time-deformed position/rotation.
- The DaVinci endoscopic data loader in LLFF format is the skeleton to adapt to SERV-CT rectified stereo data structure.
- Stereo depth iterative refinement is the conceptual reference for GS depth consistency loss design.
- Bilateral depth filter from point-cloud reconstruction may be reused for noise reduction on GS point clouds.

### 3.3 ForPlane Integration Notes

- The full regularization set (PlaneTV, TimeSmoothness, L1TimePlanes, HistogramLoss, DistortionLoss, DepthLossHuber, MonoDepthLoss) is the most comprehensive reference for Phase 1 + Phase 2 regularization design.
- The 4D (x,y,z,t) factorization via 6 orthogonal 2D planes is the conceptual reference for Phase 2 Temporal GS static/dynamic decomposition.
- Mask-guided importance sampling (maskIS) sits between EndoNeRF's hard masking and SurgTwin-GS's soft re-weighting; closer to the latter.
- The multi-component loss architecture (image + depth + multiple regularization terms) is the reference for WP4 multi-criterion density control.

### 3.4 SAM3D Integration Notes

- The ViT-B encoder + 3D U-Net decoder architecture provides offline tool/occlusion mask precomputation, consistent with the "never call inside training loop" rule.
- Mask quality control from the SAM3D training pipeline can contribute to the mask_quality_report.json acceptance criteria.

### 3.5 Endoscopy Corruptions Integration Notes

- The 16 corruption functions × 5 severity levels will be added to the evaluation pipeline after baseline and uncertainty modules work.
- Specular simulation specifically targets the pseudo-GT quality test concern identified in the charter.
- Robustness metrics from corruption testing are integrated with SurgTwin-GS's ECE, AUSE, AUROC, and AUPRC uncertainty calibration metrics.

### 3.6 EVO Integration Notes

- ATE/RPE metrics apply to Phase 2 camera trajectory evaluation; not required for Sprint 0-4.
- TUM/KITTI/EuRoC format support enables future monocular pose estimation benchmarks.

---

## 4. Target Environment

### 4.1 Primary Development Environment

The primary development environment is Linux or WSL2 Ubuntu.

Native Windows CUDA development is not the default Sprint 0 target because Gaussian rasterization packages with custom CUDA extensions are more fragile on Windows. If the user insists on native Windows later, create a separate `docs/WINDOWS_NATIVE_NOTES.md`; do not alter Sprint 0 defaults.

### 4.2 Required System Components

| Component | Required value |
|---|---|
| OS | Ubuntu 22.04 LTS or WSL2 Ubuntu 22.04 |
| GPU | NVIDIA CUDA-capable GPU, minimum 8 GB VRAM; 12 GB+ preferred |
| NVIDIA driver | Must support the selected CUDA runtime |
| Python | 3.10.x |
| Package manager | Conda or Mamba for environment; pip for Python packages |
| CUDA runtime for PyTorch | CUDA 12.1 wheel by default |
| Compiler utilities | `ninja`, `gcc/g++`, Python build tools |
| Git | Required |

### 4.3 Environment Creation Commands

Use these commands for Sprint 0.

```bash
conda create -n surgtwin python=3.10 -y
conda activate surgtwin
python -m pip install --upgrade pip setuptools wheel ninja
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
python -m pip install gsplat opencv-python pillow numpy scipy tqdm rich matplotlib imageio scikit-image lpips pyyaml pytest
```

### 4.4 Environment Verification Commands

Run immediately after installation.

```bash
python - <<'PY'
import sys
import torch
print('python', sys.version)
print('torch', torch.__version__)
print('cuda_available', torch.cuda.is_available())
print('cuda_version', torch.version.cuda)
if torch.cuda.is_available():
    print('gpu', torch.cuda.get_device_name(0))
PY

python - <<'PY'
import gsplat
print('gsplat_import_ok', gsplat.__name__)
PY
```

Acceptance criteria:

1. `torch.cuda.is_available()` prints `True`.
2. `gsplat` imports without error.
3. If `gsplat` JIT-compiles on first use, the first rasterization call may be slow; this is acceptable.

---

## 5. Repository Layout

Sprint 0 begins with a small number of scripts. After the first render works, refactor into the following structure.

```text
surgtwin-gs/
│
├── README.md
├── NOTICE.md
├── LICENSE
├── requirements.txt
├── pyproject.toml
├── .gitignore
│
├── configs/
│   ├── servct_baseline.py
│   ├── servct_depth_guided.py
│   ├── servct_uncertainty_loss.py
│   ├── servct_density_control.py
│   └── servct_full_surgtwin.py
│
├── data/
│   ├── raw/
│   │   └── SERV-CT/
│   ├── processed/
│   │   ├── manifests/
│   │   ├── masks/
│   │   ├── debug/
│   │   └── cache/
│   └── README.md
│
├── docs/
│   ├── DATA_CONTRACT.md
│   ├── COORDINATE_CONVENTIONS.md
│   ├── BACKEND_INTERFACE.md
│   ├── EXPERIMENT_MATRIX.md
│   └── TROUBLESHOOTING.md
│
├── scripts/
│   ├── explore_servct.py
│   ├── validate_servct_geometry.py
│   ├── sprint0_render_servct.py
│   ├── precompute_masks.py
│   ├── train_baseline.py
│   ├── train_depth_guided.py
│   ├── train_uncertainty.py
│   ├── train_full_surgtwin.py
│   ├── generate_cross_view_pseudogt.py
│   ├── train_confidence_mapper.py
│   ├── evaluate.py
│   └── run_ablation.py
│
├── surgtwin/
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── servct_dataset.py
│   │   ├── manifest.py
│   │   ├── sample_types.py
│   │   └── preprocessing.py
│   │
│   ├── cameras/
│   │   ├── __init__.py
│   │   ├── camera_types.py
│   │   ├── camera_utils.py
│   │   ├── projection.py
│   │   └── coordinate_conversions.py
│   │
│   ├── gaussian/
│   │   ├── __init__.py
│   │   ├── gaussian_model.py
│   │   ├── renderer_interface.py
│   │   ├── backend_gsplat.py
│   │   ├── backend_diffgaussian.py
│   │   ├── density_control.py
│   │   └── initialization.py
│   │
│   ├── losses/
│   │   ├── __init__.py
│   │   ├── photometric.py
│   │   ├── depth.py
│   │   ├── regularizers.py
│   │   ├── cross_view.py
│   │   └── uncertainty_weighted.py
│   │
│   ├── uncertainty/
│   │   ├── __init__.py
│   │   ├── signals.py
│   │   ├── pseudo_gt.py
│   │   ├── confidence_mapper.py
│   │   ├── calibration.py
│   │   └── visualization.py
│   │
│   ├── masks/
│   │   ├── __init__.py
│   │   ├── specular.py
│   │   ├── tool_mask_io.py
│   │   └── mask_quality.py
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   ├── checkpointing.py
│   │   ├── logging_utils.py
│   │   └── seed.py
│   │
│   └── evaluation/
│       ├── __init__.py
│       ├── image_metrics.py
│       ├── geometry_metrics.py
│       ├── uncertainty_metrics.py
│       ├── performance_metrics.py
│       └── report.py
│
├── tests/
│   ├── test_camera_projection.py
│   ├── test_manifest.py
│   ├── test_depth_scale.py
│   ├── test_renderer_interface.py
│   ├── test_uncertainty_signals.py
│   └── test_loss_weighting.py
│
└── outputs/
    ├── debug/
    ├── runs/
    ├── renders/
    ├── metrics/
    ├── pseudogt/
    ├── mapper/
    └── reports/
```

---

## 6. Data Contract

### 6.1 Default Dataset Path

Default raw dataset root:

```text
data/raw/SERV-CT/
```

Default processed manifest path:

```text
data/processed/manifests/servct_manifest.jsonl
```

All scripts must accept CLI overrides:

```bash
--dataset_root data/raw/SERV-CT
--manifest data/processed/manifests/servct_manifest.jsonl
--output_dir outputs/debug
```

### 6.2 Normalized Manifest Format

The loader must create and consume a JSONL manifest. Each line represents one stereo/RGB-D sample.

Required fields:

```json
{
  "sample_id": "Experiment_1_001",
  "sequence_id": "Experiment_1",
  "frame_index": 1,
  "left_rgb_path": "data/raw/SERV-CT/Experiment_1/Left_rectified/001.png",
  "right_rgb_path": "data/raw/SERV-CT/Experiment_1/Right_rectified/001.png",
  "left_depth_path": "data/raw/SERV-CT/Experiment_1/Ground_truth_CT/DepthL/001.png",
  "right_depth_path": null,
  "left_tool_mask_path": null,
  "right_tool_mask_path": null,
  "left_specular_mask_path": "data/processed/masks/Experiment_1_001_left_specular.npy",
  "right_specular_mask_path": "data/processed/masks/Experiment_1_001_right_specular.npy",
  "K_left": [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
  "K_right": [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
  "c2w_left": [[...], [...], [...], [...]],
  "c2w_right": [[...], [...], [...], [...]],
  "w2c_left": [[...], [...], [...], [...]],
  "w2c_right": [[...], [...], [...], [...]],
  "height": 1024,
  "width": 1280,
  "depth_unit": "meter",
  "depth_scale_applied": 0.000256,
  "split": "train"
}
```

Rules:

1. `sample_id` must be unique.
2. All paths must be stored relative to repository root when possible.
3. Depth values consumed by the training code must be in meters.
4. Both `c2w` and `w2c` must be stored.
5. `w2c @ c2w` must be numerically close to identity using the split tolerance rule in Section 8.5: rotation error `< 1e-4`, translation error `< 1e-3` meters.
6. Missing optional masks are allowed only before mask precomputation. Missing RGB, depth, intrinsics, or pose for a selected training sample is not allowed.
7. If right depth is unavailable, `right_depth_path` may be `null`; right RGB and right pose are still required for cross-view pseudo-GT.

### 6.3 Sample Data Types

Implement these types in `surgtwin/data/sample_types.py` and `surgtwin/cameras/camera_types.py`.

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import torch

@dataclass(frozen=True)
class CameraData:
    K: torch.Tensor          # shape [3, 3], float32
    c2w: torch.Tensor        # shape [4, 4], float32
    w2c: torch.Tensor        # shape [4, 4], float32
    height: int
    width: int
    near: float = 0.001
    far: float = 1.0
    convention: str = "opencv_c2w"

@dataclass(frozen=True)
class FrameSample:
    sample_id: str
    sequence_id: str
    frame_index: int
    left_rgb_path: Path
    right_rgb_path: Path
    left_depth_path: Path
    right_depth_path: Optional[Path]
    left_camera: CameraData
    right_camera: CameraData
    left_tool_mask_path: Optional[Path]
    right_tool_mask_path: Optional[Path]
    left_specular_mask_path: Optional[Path]
    right_specular_mask_path: Optional[Path]
    split: str
```

---

## 7. Coordinate and Depth Conventions

### 7.1 Internal Camera Convention

SurgTwin-GS internal camera convention for Phase 1:

```text
OpenCV camera coordinates:
+X = right
+Y = down
+Z = forward
```

Pose convention stored in `CameraData`:

```text
c2w = camera-to-world transform
w2c = world-to-camera transform
```

All projection/unprojection utilities must explicitly state this convention.

### 7.2 Depth Convention

1. All depth tensors used by training, loss, metrics, and rendering must be in meters.
2. If the raw dataset stores depth in millimeters, apply `depth_m = depth_raw * 0.001`.
   - SERV-CT stores depth as 16-bit PNG with scale factor ×256, i.e., `depth_mm = uint16_value / 256.0`. The full conversion is `depth_m = uint16_value / 256.0 / 1000.0` (effective scale 0.000256).
3. Invalid depth values are `NaN`, `Inf`, `<= 0`, or outside configured `[near, far]` range.
4. Default near/far for endoscopic Phase 1:

```text
near = 0.001 meter
far = 1.0 meter
```

### 7.3 Projection Utilities

Implement in `surgtwin/cameras/projection.py`:

```python
def unproject_depth_to_points(depth_m, K, c2w, valid_mask=None):
    """Convert depth map to world-space 3D points using OpenCV camera convention."""

def project_points_to_image(points_world, K, w2c, height, width):
    """Project world-space 3D points to image coordinates and depth."""
```

Required validation:

1. Unproject a set of valid depth pixels.
2. Project the resulting 3D points back into the same camera.
3. The reprojected pixel coordinates must match original coordinates within tolerance.
4. Initial tolerance: mean reprojection error `< 1e-3 px` for exact synthetic test; `< 1.0 px` for real dataset sanity checks.

---

## 8. Sprint 0 — Ugly but Working MVP

### 8.1 Sprint 0 Objective

Create a minimal working pipeline that reads a SERV-CT sample, validates camera/depth consistency, initializes Gaussians from valid depth pixels, renders RGB/depth using `gsplat`, and writes debug outputs.

Sprint 0 is not a full training system. It is a pipeline sanity milestone.

### 8.2 Sprint 0 Files to Create

Create these files first:

```text
scripts/explore_servct.py
scripts/validate_servct_geometry.py
scripts/sprint0_render_servct.py
surgtwin/data/sample_types.py
surgtwin/cameras/camera_types.py
surgtwin/cameras/projection.py
surgtwin/gaussian/renderer_interface.py
surgtwin/gaussian/backend_gsplat.py
surgtwin/gaussian/initialization.py
```

### 8.3 Sprint 0 Command Sequence

Command 1: inspect dataset and build manifest.

```bash
python scripts/explore_servct.py \
  --dataset_root data/raw/SERV-CT \
  --output_manifest data/processed/manifests/servct_manifest.jsonl
```

Command 2: validate camera/depth geometry for one sample.

```bash
python scripts/validate_servct_geometry.py \
  --manifest data/processed/manifests/servct_manifest.jsonl \
  --sample_index 0 \
  --output_dir outputs/debug/sprint0_geometry
```

Command 3: render one minimal Gaussian scene.

```bash
python scripts/sprint0_render_servct.py \
  --manifest data/processed/manifests/servct_manifest.jsonl \
  --sample_index 0 \
  --num_points 20000 \
  --output_dir outputs/debug/sprint0_render
```

### 8.4 Sprint 0 Output Files

The following files must be produced:

```text
outputs/debug/sprint0_geometry/left_rgb.png
outputs/debug/sprint0_geometry/left_depth_color.png
outputs/debug/sprint0_geometry/reprojection_overlay.png
outputs/debug/sprint0_geometry/geometry_report.json
outputs/debug/sprint0_render/render_rgb.png
outputs/debug/sprint0_render/render_depth_color.png
outputs/debug/sprint0_render/render_alpha.png
outputs/debug/sprint0_render/render_report.json
```

### 8.5 Sprint 0 Acceptance Criteria

Sprint 0 is successful only if all conditions are met:

1. The manifest exists and contains at least one valid sample.
2. RGB image loads as float32 `[H, W, 3]` in `[0, 1]`.
3. Depth image loads as float32 `[H, W]` in meters.
4. Camera intrinsics are loaded as `[3, 3]` float32.
5. `c2w` and `w2c` are `[4, 4]` float32 matrices.
6. Pose inverse consistency is checked with split tolerances: for `E = w2c @ c2w - I`, `max_abs(E[:3, :3]) < 1e-4` for the rotation block and `max_abs(E[:3, 3]) < 1e-3` meters for the translation column. This tolerance reflects float32 storage and dataset calibration precision; do not use a single stricter global tolerance that fails valid SERV-CT samples due to translation rounding.
7. Unproject/project sanity check completes.
8. `geometry_report.json` includes valid depth ratio, min/max/median depth, pose identity error, and reprojection error.
9. `gsplat` renders an RGB image without runtime error.
10. Rendered RGB, depth, and alpha files are written.
11. `render_report.json` records backend name, number of Gaussians, image size, GPU name, VRAM allocated, render time, `depth_semantics`, and whether metric depth is available.

If any criterion fails, the model must fix the issue before moving to Milestone 1.

---

## 9. Gaussian Initialization for Sprint 0

Implement in `surgtwin/gaussian/initialization.py`.

### 9.1 Input

1. RGB image: float32 tensor `[H, W, 3]`, range `[0, 1]`.
2. Depth map: float32 tensor `[H, W]`, meters.
3. Camera intrinsics `K`.
4. Camera pose `c2w`.
5. Number of points `N`, default `20000`.

### 9.2 Procedure

1. Compute valid depth mask.
2. Uniformly sample up to `N` valid pixels.
3. Unproject sampled pixels into world-space points.
4. Assign RGB colors from sampled pixels.
5. Initialize Gaussian scales from depth:

```python
scale = torch.clamp(depth_values * 0.002, min=1e-5, max=3e-3)
```

6. Initialize rotations as identity quaternions.
7. Initialize opacity logits such that initial opacity is 0.1:

```python
opacity = 0.1
opacity_logit = torch.logit(torch.tensor(opacity))
```

8. Use spherical harmonics degree 0 for Sprint 0. Do not implement higher SH degree in Sprint 0.

### 9.3 Output Gaussian Parameters

The Gaussian container must expose:

```text
means: [N, 3] float32, world coordinates
scales: [N, 3] float32
quats: [N, 4] float32
opacities: [N] float32 or logits, depending on backend wrapper
colors: [N, 3] float32
reliability_logits: [N] float32, initialized to 0.0, optional but reserved
```

---

## 10. Renderer Backend Interface

Implement in `surgtwin/gaussian/renderer_interface.py`.

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional
import torch

@dataclass
class RenderOutput:
    rgb: torch.Tensor              # [H, W, 3], float32, [0, 1]
    depth: Optional[torch.Tensor]  # [H, W], float32
    alpha: Optional[torch.Tensor]  # [H, W], float32, [0, 1]
    aux: Dict[str, Any]

class RendererBackend(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def render(self, gaussians, camera, image_height: int, image_width: int, render_depth: bool = True) -> RenderOutput:
        raise NotImplementedError
```

### 10.1 Required RenderOutput Semantics

The backend wrapper must normalize output shapes and semantics to the contract below.

Required `RenderOutput.rgb`:

```text
shape: [H, W, 3]
dtype: float32
range: [0, 1]
semantics: rendered RGB
```

Required `RenderOutput.alpha`:

```text
shape: [H, W]
dtype: float32
range: [0, 1]
semantics: accumulated opacity / alpha accumulation
```

Required `RenderOutput.depth` when `render_depth=True`:

```text
shape: [H, W]
dtype: float32
preferred semantics: metric depth in meters along the internal OpenCV camera +Z axis
```

The wrapper must set this string in `RenderOutput.aux`:

```python
aux["depth_semantics"] = one_of(
    "metric_meters",
    "relative_aligned",
    "relative_unaligned",
    "unavailable",
)
```

Rules:

1. Training code must call only `RendererBackend.render()`.
2. The backend wrapper must normalize output shapes to the contract above.
3. If the selected backend can produce metric expected depth, `depth_semantics` must be `"metric_meters"` and depth may be used for `L_depth`, Depth RMSE, Depth MAE, AbsRel, and depth instability features.
4. If the backend returns only relative depth, the wrapper must not pretend it is metric. It must set `"relative_unaligned"` or `"relative_aligned"`.
5. `relative_aligned` depth may be produced only by explicit median alignment against GT depth for debugging or diagnostic reporting. It must not be used as the default metric-depth training signal unless a later experiment explicitly declares an aligned-depth ablation.
6. If the backend cannot produce depth at all, set `depth=None` and `depth_semantics="unavailable"`.
7. Metric depth-guided training in Milestone 2 must not begin until either:
   - `GsplatBackend` provides `depth_semantics="metric_meters"`, or
   - the fallback `DiffGaussianBackend` is implemented and provides `depth_semantics="metric_meters"`, or
   - a documented SurgTwin expected-depth computation is implemented inside the backend wrapper and validated against GT depth on Sprint 0 sanity checks.
8. If depth semantics are not metric in Sprint 0, Sprint 0 may still save `render_depth_color.png` as a diagnostic visualization, but `render_report.json` must clearly mark that depth as non-metric.
9. Any metric report consuming non-metric depth is invalid and must fail loudly.

### 10.2 Gsplat Backend Requirements

`GsplatBackend` must be implemented in `surgtwin/gaussian/backend_gsplat.py`.

The implementation must:

1. Call the installed `gsplat` rasterization API through a wrapper, not directly from training code.
2. Attempt to request RGB, alpha, and depth according to the installed `gsplat` version's supported API.
3. Inspect and record whether returned depth is metric, relative, or unavailable.
4. Populate `aux` with backend metadata:

```python
aux["backend"] = "gsplat"
aux["depth_semantics"] = "metric_meters"  # or other allowed value
aux["supports_metric_depth"] = True        # bool
aux["supports_alpha"] = True               # bool
aux["supports_contrib"] = False            # bool unless verified
aux["supports_color_variance"] = False     # bool unless verified
```

If the installed `gsplat` version cannot expose Gaussian contribution lists, per-pixel contribution weights, or color variance, this is not a failure. The wrapper must set the corresponding support flags to `False` and the uncertainty code must use the approved fallback features in Section 13.4.

### 10.3 Fallback Backend Trigger

Implementing `DiffGaussianBackend` is not required for Sprint 0 unless `gsplat` cannot produce any valid RGB/alpha render. However, `DiffGaussianBackend` must be activated before Milestone 2 if `GsplatBackend` cannot provide metric depth or a validated metric expected-depth wrapper.

The file `surgtwin/gaussian/backend_diffgaussian.py` may exist during Sprint 0, but if called before implementation it must raise:

```python
NotImplementedError("DiffGaussianBackend is an approved fallback but has not yet been implemented. Use GsplatBackend or implement this backend behind RendererBackend.")
```

### 10.4 Depth Alignment Diagnostics

Implement a diagnostic helper in `surgtwin/evaluation/geometry_metrics.py`:

```python
def median_align_depth(pred_depth, gt_depth, valid_mask, eps=1e-6):
    """Return pred_depth scaled by median(gt)/median(pred) for diagnostic aligned-depth metrics only."""
```

Rules:

1. Median alignment is allowed only for diagnostic reporting or explicit aligned-depth ablations.
2. Median-aligned metrics must be named with the prefix `median_aligned_`.
3. Median-aligned depth must not be mixed with metric depth metrics.
4. The report must include `depth_semantics` for every depth metric.
---

## 11. Offline Mask Precomputation

### 11.1 Goal

Generate and save masks before training. Training reads masks from disk and never calls segmentation models.

### 11.2 Mask Priority Order

Tool mask source priority:

1. Dataset-provided tool masks, if available.
2. Manually or semi-automatically generated masks for a small validation subset.
3. SAM2 / SAM3D / Mobile-SAM masks with visual QA.
4. Heuristic masks only for debugging; never for final metrics.

Specular mask source priority:

1. Conservative threshold-based mask.
2. Threshold-based mask plus morphology.
3. Dataset-specific or learned specular detector in later phases.

### 11.3 Specular Mask Rule

Implement in `surgtwin/masks/specular.py`:

1. Convert RGB to HSV.
2. Candidate specular pixels:

```text
V > 220/255 AND S < 40/255
```

3. Add optional RGB whiteness rule:

```text
R > 220/255 AND G > 220/255 AND B > 220/255
```

4. Apply small morphological opening to remove isolated noise.
5. Save as boolean `.npy` and visual `.png` overlay.

### 11.4 Precompute Command

```bash
python scripts/precompute_masks.py \
  --manifest data/processed/manifests/servct_manifest.jsonl \
  --output_dir data/processed/masks \
  --mask_types specular \
  --save_overlays true
```

SAM/tool segmentation is disabled by default in Sprint 0 and Milestone 1 unless dataset-provided tool masks exist.

### 11.5 Acceptance Criteria

1. For every selected sample, specular mask is saved as `.npy`.
2. Mask dtype is boolean.
3. Mask shape matches RGB height and width.
4. Training loader can read masks without invoking any segmentation model.
5. `mask_quality_report.json` contains mask coverage percentage per frame.

---

## 12. Baseline Training Milestones

### 12.1 Milestone 1 — Modular Refactor and Baseline Renderer

Weeks: 3-4

Goal:

1. Move Sprint 0 code into modular files.
2. Keep all Sprint 0 commands working.
3. Add minimal baseline optimization loop.

Baseline loss:

```text
L_baseline = mean(abs(rgb_pred - rgb_gt))
```

Command:

```bash
python scripts/train_baseline.py \
  --manifest data/processed/manifests/servct_manifest.jsonl \
  --sequence_id dataset_001_keyframe_001 \
  --iterations 1000 \
  --output_dir outputs/runs/baseline_debug
```

Acceptance criteria:

1. Loss decreases over 1000 iterations on a small sequence.
2. Rendered image is saved every 100 iterations.
3. Checkpoint is saved.
4. Metrics JSON contains L1 loss, PSNR, render time, VRAM, and number of Gaussians.

### 12.2 Milestone 2 — Depth-Guided GS

Weeks: 5-6

Goal:

Add depth consistency without uncertainty weighting.

Loss:

```text
L_depth_guided = L_photo + lambda_depth * L_depth + lambda_reg * L_reg
```

Defaults:

```text
lambda_depth = 0.2
lambda_reg = 0.01
```

Depth loss:

```text
L_depth = mean(valid_depth_mask * abs(depth_pred - depth_gt))
```

Acceptance criteria:

1. Depth loss is computed only where GT depth is valid.
2. Invalid GT depth does not contribute to loss.
3. Rendered depth used in `L_depth` has `depth_semantics == "metric_meters"`. If not, Milestone 2 must stop and either implement metric expected-depth in the active backend or activate the approved fallback backend.
4. Depth-guided run uses the same train/val split as baseline.
5. Evaluation report compares baseline and depth-guided models and records `depth_semantics` for every depth metric.

---

## 13. Uncertainty Signal Design

Implement in `surgtwin/uncertainty/signals.py`.

The project uses two different signal groups. They must not be confused.

1. **Training-time residual signals** may use GT RGB and GT depth because they are used for loss weighting, diagnostics, and ablation during supervised/offline training.
2. **Inference-safe mapper features** must not use GT RGB, GT depth, or held-out right-view information. They must be computable from the current rendered view, current camera, current Gaussian state, offline masks available for the current frame, and renderer auxiliary outputs.

### 13.1 Training-Time Residual Signals

For each rendered training view, compute the following signals when GT is available:

| Signal | Symbol | Definition | Range | Inference allowed? |
|---|---|---|---|---|
| Photometric residual | `r_photo` | Mean absolute RGB error per pixel: `mean(abs(rgb_pred - rgb_gt), dim=-1)` | `[0, 1]` before normalization | No |
| Depth residual | `r_depth` | Absolute depth error where GT depth is valid and rendered depth is metric | meters | No |
| Alpha uncertainty | `u_alpha` | `1 - clamp(alpha, 0, 1)` | `[0, 1]` | Yes |
| Specular flag | `m_specular` | Boolean or float mask | `{0,1}` | Yes if precomputed/current-frame mask exists |
| Tool/occlusion flag | `m_tool` | Boolean or float mask | `{0,1}` | Yes if precomputed/current-frame mask exists |
| View coverage proxy | `coverage` | Backend-provided contribution/visibility proxy if available; otherwise alpha proxy | `[0, +inf)` | Yes |
| Gaussian contribution proxy | `contrib` | Accumulated contribution if backend exposes it; otherwise alpha proxy | `[0, +inf)` | Yes |

Rules:

1. `r_photo` and `r_depth` are training-time residuals only.
2. `r_depth` may be computed only when `RenderOutput.aux["depth_semantics"] == "metric_meters"` and GT depth is valid.
3. These residuals may influence uncertainty-weighted loss during training, but they must not be included as mapper input features at inference.
4. If GT depth is unavailable or rendered depth is not metric, exclude `r_depth` from the combined training uncertainty and renormalize the remaining weights.

### 13.2 Normalization for Training-Time Uncertainty

Default normalizers:

```text
tau_photo = 0.20
tau_depth = 0.010 meter
coverage_ref = 0.95
```

Normalized signals:

```python
u_photo = clamp(r_photo / tau_photo, 0, 1)
u_depth = clamp(r_depth / tau_depth, 0, 1)
u_alpha = 1 - clamp(alpha, 0, 1)
u_coverage = 1 - clamp(coverage / coverage_ref, 0, 1)
u_mask = maximum(m_specular, m_tool)
```

### 13.3 Combined Training-Time Pixel Uncertainty

Default combination for uncertainty-weighted loss during training:

```python
u_train = (
    0.25 * u_depth +
    0.20 * u_photo +
    0.20 * u_alpha +
    0.20 * u_coverage +
    0.15 * u_mask
)
u_train = clamp(u_train, 0, 1)
```

If GT depth is unavailable for a pixel, exclude `u_depth` from the weighted sum and renormalize remaining weights. If GT RGB is unavailable, `u_photo` cannot be computed and the training sample must not be used for photometric training.

Acceptance criteria:

1. `u_train` has shape `[H, W]`.
2. `u_train` is finite everywhere.
3. `u_train` lies within `[0, 1]`.
4. Specular/tool regions show higher uncertainty on average than normal tissue regions in debug reports.

### 13.4 Inference-Safe Render-Time Features

These features are used by the lightweight mapper during inference. They must not depend on GT RGB, GT depth, or held-out views.

Required inference-safe features per pixel:

| Feature | Symbol | Definition | Source |
|---|---|---|---|
| Alpha uncertainty | `f_alpha_unc` | `1 - clamp(alpha, 0, 1)` | current render |
| View coverage uncertainty | `f_coverage_unc` | `1 - clamp(coverage / coverage_ref, 0, 1)` | renderer aux or alpha fallback |
| Gaussian contribution proxy | `f_contrib` | normalized accumulated contribution | renderer aux or alpha fallback |
| Covariance/scale proxy | `f_scale` | normalized projected covariance trace or mean visible Gaussian scale | renderer aux or Gaussian state |
| Color variance / instability | `f_color_var` | contribution-weighted color variance if available; otherwise deterministic micro-perturbation color instability | renderer aux or current Gaussian state |
| Depth instability | `f_depth_instab` | absolute depth change under deterministic micro camera perturbation | current render plus micro-perturbed render |
| Tool mask flag | `f_tool` | current-frame tool/occlusion mask | offline mask or current-frame mask module |
| Specular mask flag | `f_specular` | current-frame specular mask | offline/current-frame threshold mask |

The default mapper input dimension is therefore 8:

```text
[f_alpha_unc, f_coverage_unc, f_contrib, f_scale, f_color_var, f_depth_instab, f_tool, f_specular]
```

### 13.5 Color Variance / Color Instability Helper

Implement:

```python
def compute_color_variance_feature(render_output, gaussians, camera, backend, perturbation_degrees=0.1):
    """Return [H, W] inference-safe color variance/instability feature in [0, 1]."""
```

Primary path:

1. If the backend exposes per-pixel Gaussian contribution weights and contributing Gaussian colors, compute weighted RGB variance:

```python
mean_c = sum_i(w_i * c_i) / (sum_i(w_i) + eps)
var_c = sum_i(w_i * ||c_i - mean_c||_1) / (sum_i(w_i) + eps)
f_color_var = clamp(var_c / tau_color_var, 0, 1)
```

2. Default `tau_color_var = 0.20`.

Fallback path when per-Gaussian contribution lists are unavailable:

1. Render the same scene from two deterministic micro-perturbed cameras: yaw `+0.1°` and yaw `-0.1°` around the current camera center.
2. Warp is not required for Sprint 0-4. Compare the same pixel coordinates as a conservative local instability proxy.
3. Compute:

```python
f_color_var = clamp(mean(abs(rgb_yaw_plus - rgb_yaw_minus), dim=-1) / tau_color_instability, 0, 1)
```

4. Default `tau_color_instability = 0.10`.
5. Record in the mapper report whether `color_feature_mode` is `"contribution_weighted_variance"` or `"micro_perturbation_instability"`.

### 13.6 Depth Instability Helper

Implement:

```python
def compute_depth_instability_feature(gaussians, camera, backend, image_height, image_width, perturbation_degrees=0.1):
    """Return [H, W] inference-safe depth instability in [0, 1]."""
```

Procedure:

1. Render base depth from the current camera.
2. Render depth from yaw `+0.1°` and yaw `-0.1°` micro-perturbed cameras around the same camera center.
3. Use only renders whose `depth_semantics == "metric_meters"`.
4. Compute:

```python
instability = 0.5 * (abs(depth_plus - depth_base) + abs(depth_minus - depth_base))
f_depth_instab = clamp(instability / tau_depth_instability, 0, 1)
```

5. Default `tau_depth_instability = 0.010 meter`.

Rules:

1. If metric depth is unavailable, set `f_depth_instab` to zeros and record `depth_instability_available=false` in the mapper report.
2. Do not use GT depth to compute this feature.
3. Do not use held-out right-view depth to compute this feature.
4. Micro-perturbation renders are allowed in Phase 1 inference evaluation because they do not require GT or cross-view supervision. Their runtime cost must be reported separately.

### 13.7 Mapper Feature Contract

Implement:

```python
def build_mapper_features(render_output, gaussians, camera, masks, backend, image_height, image_width):
    """Return mapper feature tensor [H, W, 8] using only inference-safe features."""
```

Rules:

1. `build_mapper_features()` must never accept `rgb_gt`, `depth_gt`, `right_rgb`, `right_depth`, or `pseudo_gt_error` as arguments.
2. If a feature cannot be computed, use the explicit fallback specified above and record it in `aux`/report.
3. Returned features must be finite and normalized to `[0, 1]`.
4. Feature order must match Section 13.4 exactly.
---

## 14. Uncertainty-Weighted Loss

Implement in `surgtwin/losses/uncertainty_weighted.py`.

### 14.1 Main Formula

```text
L_total = mean(w_photo(u) * L_photo_pixel) + lambda_depth * L_depth + lambda_cv * L_cv + lambda_reg * L_reg
```

Photometric weight:

```text
w_photo(u) = max(1 - u, w_min)
```

Default:

```text
w_min = 0.15
lambda_depth = 0.2
lambda_cv = 0.1
lambda_reg = 0.01
```

### 14.2 Gradient Starvation Rule

`w_photo` must never be zero. The minimum allowed value is `0.15` unless an ablation explicitly changes it.

### 14.3 Debug Logging

Every training run using uncertainty weighting must log:

1. Mean `u`.
2. Median `u`.
3. Mean `w_photo`.
4. Mean `w_photo` inside specular mask.
5. Mean `w_photo` outside specular/tool masks.
6. Depth residual mean.
7. Photometric residual mean.

Acceptance criteria:

1. `w_photo` lies within `[0.15, 1.0]`.
2. Tool/specular regions have lower `w_photo` on average than normal tissue regions.
3. Loss remains finite for all iterations.
4. Training does not collapse to all-high uncertainty.

---

## 15. Multi-Criteria Density Control

Implement in `surgtwin/gaussian/density_control.py`.

### 15.1 Baseline Densification Problem

Standard 3DGS often densifies based on high photometric or view-space gradient. In endoscopy, tool edges, specular highlights, smoke, blood, and glare can produce high gradients that are not anatomical detail.

### 15.2 SurgTwin-GS Densification Rule

A Gaussian may be split/cloned only if all required conditions are satisfied:

```text
grad_norm > grad_threshold
AND view_coverage > coverage_min
AND uncertainty_score < uncertainty_max
AND gaussian_contribution > contribution_min
AND mask_contamination < mask_contamination_max
```

Default threshold starting points:

```text
grad_threshold = backend_default_or_0.0002
coverage_min = 0.50
uncertainty_max = 0.65
contribution_min = 0.01
mask_contamination_max = 0.50
```

These thresholds are backend-specific tunables. They must be stored in config files and must not be hard-coded inside `density_control.py`. `grad_threshold` in particular is expected to differ between `gsplat`, `diff-gaussian-rasterization`, and any future custom backend. If backend-specific gradient scale differs, record the scale in the run report and tune only through config overrides.

### 15.3 Pruning Rule

A Gaussian may be pruned if the following persistent conditions hold for `prune_patience` iterations:

```text
contribution < contribution_prune_threshold
AND view_support < view_support_min
AND grad_norm < grad_low_threshold
AND depth_alignment_error > depth_error_high_threshold
```

Default:

```text
prune_patience = 500
contribution_prune_threshold = 0.005
view_support_min = 0.20
grad_low_threshold = 0.00005
depth_error_high_threshold = 0.020 meter
```

Do not prune solely because uncertainty is high. High uncertainty may mean the region is difficult but important.

### 15.4 Acceptance Criteria

1. Density-control decisions are logged.
2. Number of split, clone, and prune operations is reported.
3. Mask-contaminated/specular regions show reduced over-densification relative to baseline.
4. Gaussian count and VRAM are tracked.
5. The density-control variant can be disabled for ablation.

---

## 16. Cross-View Pseudo-Ground Truth

Implement in `surgtwin/uncertainty/pseudo_gt.py` and `scripts/generate_cross_view_pseudogt.py`.

### 16.1 Goal

Generate an approximate error/risk target for mapper training and uncertainty validation.

### 16.2 Procedure

For each stereo sample:

1. Train or optimize scene primarily using left views.
2. Render the scene from the right camera pose.
3. Compare rendered right RGB to true right RGB.
4. Apply valid masks and conservative specular exclusion.
5. Save normalized error map.

Error map:

```python
error_rgb = mean(abs(rgb_right_pred - rgb_right_gt), dim=-1)
error_map = clamp(error_rgb / tau_photo, 0, 1)
```

If LPIPS is used, it is used as an additional image-level or patch-level signal, not the only pixel target.

### 16.3 Output

```text
outputs/pseudogt/{run_id}/{sample_id}_right_error.npy
outputs/pseudogt/{run_id}/{sample_id}_right_error_color.png
outputs/pseudogt/{run_id}/pseudogt_report.json
```

### 16.4 Acceptance Criteria

1. Error map shape matches rendered right image resolution.
2. Error map values lie within `[0, 1]`.
3. Specular exclusion mask is saved and reported.
4. Pearson and Spearman correlation between current uncertainty map and pseudo-GT error map is computed.
5. Pseudo-GT is explicitly labeled as pseudo-GT, not ground-truth hallucination.

---

## 17. Lightweight Confidence/Risk Mapper

Implement in `surgtwin/uncertainty/confidence_mapper.py` and `scripts/train_confidence_mapper.py`.

### 17.1 Phase 1 Requirement

The confidence/risk mapper is mandatory in Phase 1. It must not be postponed.

### 17.2 Model

Initial model: two-layer PyTorch MLP.

```text
Input dimension: 8
Hidden dimension: 128
Activation: ReLU
Output dimension: 1
Output activation: Sigmoid
```

Input feature vector per pixel must be inference-safe and must follow the exact order defined in Section 13.4:

```text
1. alpha uncertainty
2. view coverage uncertainty
3. Gaussian contribution proxy
4. normalized covariance/scale proxy
5. color variance / render-time color instability
6. depth instability from micro-perturbed metric-depth renders
7. tool mask flag
8. specular mask flag
```

Forbidden mapper input features:

```text
- photometric residual against GT RGB
- depth residual against GT depth
- held-out right-view error
- pseudo-GT error map
- any feature that requires ground-truth values at inference time
```

These forbidden values may be used only as training targets, training diagnostics, or ablation analysis outputs. They must not be passed into `build_mapper_features()` or the mapper forward pass.

Output:

```text
risk_score in [0, 1]
```

### 17.3 Training Target

Primary target:

```text
cross-view pseudo-GT error map
```

Loss:

```text
L_mapper = MSE(risk_score, error_map)
```

Optional auxiliary binary loss:

```text
high_error = error_map > 0.5
L_bce = BCE(risk_score, high_error)
```

Default:

```text
L_mapper_total = L_mse + 0.1 * L_bce
```

### 17.4 Train/Test Skew Guard

The mapper training loop must use the same feature builder as inference:

```python
features = build_mapper_features(render_output, gaussians, camera, masks, backend, H, W)
risk_score = mapper(features)
loss = mse(risk_score, pseudo_gt_error_map)
```

Rules:

1. Pseudo-GT error maps are labels only, never inputs.
2. GT RGB/depth residuals are not mapper inputs.
3. The feature tensor used during mapper training must have the same feature order, normalization, and fallback behavior as inference.
4. `mapper_report.json` must include feature availability flags:

```json
{
  "color_feature_mode": "contribution_weighted_variance | micro_perturbation_instability",
  "depth_instability_available": true,
  "depth_semantics": "metric_meters | relative_aligned | relative_unaligned | unavailable",
  "uses_gt_residual_inputs": false
}
```

### 17.5 Acceptance Criteria

1. Mapper trains without changing Gaussian model parameters.
2. Mapper output aligns with rendered image resolution.
3. Mapper inference does not require held-out right view.
4. Mapper inference does not require GT RGB or GT depth.
5. Mapper report includes MSE, AUROC, AUPRC, ECE, AUSE, Pearson, and Spearman.
6. Risk maps are saved as `.npy` and color overlays.
7. `uses_gt_residual_inputs` is recorded as `false`.
---

## 18. Evaluation Matrix

### 18.1 Required Variants

| Config ID | Variant | Components enabled |
|---|---|---|
| C1 | Baseline GS | Photometric loss only |
| C2 | Depth-guided GS | C1 + depth loss |
| C3 | Uncertainty-weighted loss GS | C2 + uncertainty signal + soft photometric weighting |
| C4 | Density-control GS | C3 + multi-criteria densification/pruning |
| C5 | Full SurgTwin-GS | C4 + cross-view pseudo-GT + lightweight mapper |

### 18.2 Required Metrics

Visual metrics:

```text
PSNR
SSIM
LPIPS
```

Geometry metrics:

```text
Depth RMSE
Depth MAE
AbsRel
Valid-depth ratio
Chamfer distance if point/mesh reference exists
Hausdorff distance if point/mesh reference exists
F-score if point/mesh reference exists
Surface normal error if normal reference exists
```

Cross-view risk metrics:

```text
Held-out right-view L1 error
Held-out right-view PSNR/SSIM/LPIPS
Mask-conditioned error gap
View-consistency error
```

Uncertainty metrics:

```text
ECE
UCE
Reliability diagram bins
AUSE
AUROC for high-error detection
AUPRC for high-error detection
Pearson correlation uncertainty vs measured error
Spearman correlation uncertainty vs measured error
```

Performance metrics:

```text
FPS
single-frame render latency
training iteration time
VRAM allocated
VRAM reserved
number of Gaussians
checkpoint size
```

### 18.3 Success Interpretation

Do not require every metric to improve simultaneously.

The expected Phase 1 success pattern is:

1. Full SurgTwin-GS improves uncertainty-error correlation over C1/C2.
2. Full SurgTwin-GS improves ECE/AUSE/AUROC/AUPRC over uncertainty-free baselines.
3. A modest PSNR reduction is acceptable if confidence/risk maps become meaningfully better calibrated.
4. Gaussian count and VRAM should not grow uncontrolled in specular/tool regions.
5. Failure cases must be reported, not hidden.

No hard percentage improvement is required in the blueprint. Percentage targets may be used internally but not as mandatory proof before experiments exist.

---

## 19. Logging, Output, and Reproducibility

Every run must create:

```text
outputs/runs/{run_id}/
├── config.json
├── environment.json
├── metrics.jsonl
├── final_metrics.json
├── checkpoints/
├── renders/
├── depth/
├── uncertainty/
├── masks/
├── logs/
└── report.md
```

### 19.1 `environment.json`

Must include:

```json
{
  "python_version": "...",
  "torch_version": "...",
  "torch_cuda_version": "...",
  "cuda_available": true,
  "gpu_name": "...",
  "gsplat_version": "...",
  "platform": "...",
  "git_commit": "..."
}
```

### 19.2 Random Seeds

Default seed:

```text
seed = 42
```

Set seeds for:

```text
Python random
NumPy
PyTorch CPU
PyTorch CUDA
```

### 19.3 Checkpoints

Checkpoint must include:

1. Gaussian parameters.
2. Optimizer state.
3. Iteration number.
4. Active configuration.
5. Backend name.
6. Random seed.

---

## 20. Testing Requirements

Tests must be implemented before declaring a milestone complete.

### 20.1 Required Unit Tests

```text
tests/test_camera_projection.py
tests/test_manifest.py
tests/test_depth_scale.py
tests/test_renderer_interface.py
tests/test_depth_semantics.py
tests/test_uncertainty_signals.py
tests/test_loss_weighting.py
tests/test_mapper_features_inference_safe.py
```

### 20.2 Test Commands

```bash
pytest tests -q
```

### 20.3 Minimum Test Expectations

1. Projection/unprojection synthetic test passes.
2. Depth conversion test passes.
3. Manifest parser rejects incomplete samples.
4. Renderer output contract is respected.
5. Uncertainty values are finite and within `[0, 1]`.
6. `w_photo` lower bound is enforced.
7. Renderer depth semantics are recorded and non-metric depth is rejected by metric depth losses/metrics.
8. Mapper feature builder rejects GT RGB, GT depth, held-out right-view data, and pseudo-GT error maps as input arguments.

---

## 21. Twelve-Week Execution Plan

| Week | Milestone | Deliverable |
|---|---|---|
| 1 | Environment + manifest scanner | Environment verified, SERV-CT manifest generated |
| 2 | Sprint 0 render | Camera/depth sanity check and first `gsplat` render |
| 3 | Modular refactor | Renderer interface, dataset classes, projection utilities |
| 4 | Baseline training | C1 baseline run with metrics |
| 5 | Depth-guided training | C2 run with depth loss |
| 6 | Offline masks | Specular masks and optional tool masks saved and loaded |
| 7 | Uncertainty signals | Pixel uncertainty maps generated and visualized |
| 8 | Uncertainty-weighted loss | C3 run with soft loss weighting |
| 9 | Density control | C4 run with multi-criteria densification/pruning |
| 10 | Cross-view pseudo-GT | Right-view error maps generated |
| 11 | Lightweight mapper | MLP mapper trained and inference risk maps saved |
| 12 | Ablation report | C1-C5 comparison report generated |

---

## 22. Explicit opencode Task Prompt

Use the following task prompt when handing this blueprint to opencode.

```text
You are implementing SurgTwin-GS Phase 1 according to TECHNICAL_IMPLEMENTATION_BLUEPRINT_v0.1.md.

Do not ask architectural questions. Do not change locked decisions. Do not use any external repository as the main project skeleton. Create an original modular codebase.

Start with Sprint 0 only.

Required Sprint 0 deliverables:
1. Create the repository structure needed for Sprint 0.
2. Create environment/verification documentation.
3. Implement SERV-CT manifest creation in scripts/explore_servct.py.
4. Implement camera/depth validation in scripts/validate_servct_geometry.py.
5. Implement projection and unprojection utilities.
6. Implement RendererBackend interface.
7. Implement GsplatBackend wrapper.
8. Implement Gaussian initialization from RGB-D.
9. Implement scripts/sprint0_render_servct.py.
10. Write debug outputs exactly as specified.
11. In `render_report.json`, explicitly record `depth_semantics` and whether metric depth is available.
12. Use split pose inverse tolerances: rotation `< 1e-4`, translation `< 1e-3` meters.
13. Add tests for camera projection, depth scale, manifest parsing, renderer output contract, and depth semantics reporting.
14. Do not leave TODO, pass, or placeholder code in runnable modules.
15. If a dataset path is missing, fail with a clear error message explaining the expected path and command-line override. Do not invent data.
16. If gsplat fails, report the full dependency and CUDA diagnostics. Do not choose a new renderer unless using the pre-approved fallback plan.

Stop after Sprint 0 acceptance criteria pass. Do not implement later milestones until Sprint 0 is verified.
```

---

## 23. Definition of Done

A milestone is done only when:

1. The required command runs from a clean shell.
2. Outputs are written to the specified paths.
3. Metrics or debug reports are written in JSON/JSONL.
4. Unit tests for that milestone pass.
5. No runnable function contains `TODO`, `pass`, or silent placeholder behavior.
6. The run records environment information.
7. Any external code usage is documented in `NOTICE.md`.
8. The result is reproducible with the same seed and config.

---

## 24. `NOTICE.md` Policy

Create `NOTICE.md` during Sprint 0.

It must contain:

```text
# NOTICE

This project develops an original modular research codebase for SurgTwin-GS.

External packages and repositories may be used as dependencies, methodological references, benchmark implementations, or backend components where licensing permits.

Primary external references/components:
- gsplat: Gaussian Splatting rasterization backend dependency.
- 3D Gaussian Splatting / diff-gaussian-rasterization: reference and potential fallback backend.
- UC-NeRF: uncertainty-aware loss and cross-view pseudo-GT conceptual reference.
- EndoNeRF and ForPlane: baseline comparison references.
- Endoscopy Corruptions: future robustness testing reference.
- SAM2/SAM3D/Mobile-SAM: optional offline mask precomputation references.
- EVO: future pose/trajectory evaluation reference.

No external repository is used as the main project skeleton.
Any copied or adapted code must include file-level attribution and license compatibility review.
```

---

## 25. Final Locked Summary

The implementation strategy is locked as follows:

```text
1. Write original SurgTwin-GS codebase from scratch.
2. Use gsplat as the first rasterization backend.
3. Hide all renderer calls behind RendererBackend.
4. Keep diff-gaussian-rasterization as reference/fallback/future fork only.
5. Use code-first, refactor-second.
6. Sprint 0 = SERV-CT manifest + camera/depth sanity + minimal RGB-D initialized gsplat render.
7. Precompute masks offline.
8. Transfer uncertainty concepts from UC-NeRF, not NeRF architecture.
9. Model uncertainty using per-pixel rasterization signals and optional per-Gaussian reliability attributes.
10. Train the lightweight confidence/risk mapper only on inference-safe features; do not use GT residuals as mapper inputs.
11. Keep lightweight confidence/risk mapper in Phase 1.
12. Require explicit depth semantics from every renderer backend before using depth in losses or metrics.
13. Keep tiny-cuda-nn in Phase 2 (locked after expert review).
14. Preserve ablation order: Baseline → Depth-guided → Uncertainty-weighted → Density-control → Full SurgTwin-GS.
15. Do not claim clinical readiness.
16. Do not advance to later milestones until Sprint 0 acceptance criteria pass.
```
