# Data Contract

## Primary Dataset: SERV-CT

SERV-CT provides 16 rectified stereo pairs (720×576) with ground-truth CT depth maps.

### Directory Layout

```
SERV-CT/
├── Experiment_1/              (frames 001-008)
│   ├── Left_rectified/        left rectified images (720x576 24-bit colour PNG)
│   ├── Right_rectified/       right rectified images
│   ├── Ground_truth_CT/
│   │   ├── DepthL/            left depth maps (16-bit PNG, mm × 256)
│   │   ├── DepthR/            right depth maps
│   │   ├── OcclusionL/        occlusion masks (colour-coded)
│   │   └── OcclusionR/
│   └── Rectified_calibration/  JSON with P1, P2, Q matrices (rectified stereo)
└── Experiment_2/              (frames 009-016)
    └── ...
```

### Calibration

Rectified stereo: P1, P2 are 3×4 projection matrices, Q is 4×4 reprojection matrix.
Left camera is identity pose. Right camera has X-axis baseline translation.

### Depth Encoding

- Format: 16-bit PNG (mode I;16)
- Scale: `depth_mm = uint16_value / 256.0`
- Internal: `depth_m = uint16_value / 256.0 / 1000.0`
- Valid range: ~20–200 mm (0.02–0.2 m)

### Occlusion Masks

Colour-coded:
- Yellow: non-overlap region
- Blue: outside reference surface
- Red: not visible in right image
