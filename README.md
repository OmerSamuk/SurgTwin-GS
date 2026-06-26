# SurgTwin-GS

Uncertainty-Weighted Gaussian Splatting for Decision-Support-Oriented 3D Surgical Scene Reconstruction from Endoscopic Images.

**Phase:** Phase 1 — Laboratory research prototype

## Quick Start

```bash
conda create -n surgtwin python=3.10 -y
conda activate surgtwin
pip install -r requirements.txt
```

## Primary Dataset: SERV-CT

SERV-CT (Surgical Endoscopic Videos with CT) provides 16 calibrated stereo pairs with ground-truth CT-derived depth maps. The dataset uses rectified stereo with P1/P2/Q calibration matrices.

### Data Pipeline

1. Download SERV-CT from https://www.ucl.ac.uk/interventional-surgical-sciences/serv-ct
2. Place in `data/raw/SERV-CT/`
3. Build manifest:
   ```bash
   python scripts/explore_servct.py --dataset_root data/raw/SERV-CT --output_manifest data/processed/manifests/servct_manifest.jsonl
   ```
4. Validate geometry:
   ```bash
   python scripts/validate_servct_geometry.py --manifest data/processed/manifests/servct_manifest.jsonl
   ```

### Sprint 0 Render

```bash
python scripts/sprint0_render_servct.py --manifest data/processed/manifests/servct_manifest.jsonl
```

## Project Structure

```
surgtwin-gs/
├── surgtwin/          # Core library
│   ├── data/          # Data loaders and calibration
│   ├── cameras/       # Camera types and projection
│   ├── gaussian/      # Gaussian splatting implementation
│   ├── losses/        # Loss functions
│   ├── uncertainty/   # Uncertainty signals and mapper
│   ├── masks/         # Mask precomputation
│   ├── training/      # Training infrastructure
│   └── evaluation/    # Metrics and reporting
├── scripts/           # Runnable scripts
├── tests/             # Unit tests
├── data/              # Dataset storage
├── configs/           # Configuration files
├── docs/              # Documentation
└── outputs/           # Run outputs
```

## Tests

```bash
pytest tests -q
```

## License

MIT License — see LICENSE for details.
