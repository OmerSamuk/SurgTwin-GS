# Data Directory

## Raw Data

Place SERV-CT dataset here:

```
data/raw/SERV-CT/
├── Experiment_1/     (frames 001-008)
│   ├── Left_rectified/
│   ├── Right_rectified/
│   ├── Ground_truth_CT/
│   │   ├── DepthL/
│   │   ├── DepthR/
│   │   ├── OcclusionL/
│   │   └── OcclusionR/
│   └── Rectified_calibration/
└── Experiment_2/     (frames 009-016)
    └── ...
```

### Download

SERV-CT is available from UCL:
https://www.ucl.ac.uk/interventional-surgical-sciences/serv-ct

Also available on OpenDataLab.

### Synthetic Data

To generate synthetic data for testing:

```bash
python scripts/generate_synthetic_servct.py --output_dir data/raw/SERV-CT
```

## Processed Data

- `manifests/` — JSONL manifest files created by `scripts/explore_servct.py`
- `masks/` — Precomputed tool/specular/occlusion masks
- `debug/` — Debug outputs from validation scripts
- `cache/` — Cached intermediate data
