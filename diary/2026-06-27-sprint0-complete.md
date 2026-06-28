# Sprint 0 Completed

**Date**: 2026-06-27  
**VM**: `surgtwin-gs-phase1` (us-central1-c, g2-standard-8, 1×L4 GPU) — **stopped**

## Sprint 0 Pipeline Results

| Step | Status | Details |
|------|--------|---------|
| `explore_servct.py` | ✅ | Manifest: 16 entries (8 per experiment) |
| `validate_servct_geometry.py` | ✅ | 100% valid depth, 0.062-0.093m range, ~0 px reprojection error |
| `sprint0_render_servct.py` | ✅ | 20K Gaussians, 0.81s render, 0.013 GB VRAM |
| `pytest tests -q` | ✅ | 23/23 passed |

## Bugs Fixed During Sprint 0

1. **`surgtwin/data/servct_calibration.py`** — OpenCV JSON matrix format: calibrasyon dosyaları `data`+`rows`+`cols` dict formatında, direkt array değil. `_parse_opencv_matrix()` helper eklendi.
2. **`surgtwin/gaussian/backend_gsplat.py`** — gsplat 1.5.3 API değişikliği: `viewmats` shape `[..., C, 4, 4]` formatında (C = camera sayısı). `batch_dims` `means.shape[:-2]`'den türetiliyor. Verification ve render fonksiyonları güncellendi.
3. **`scripts/sprint0_render_servct.py`** — Depth tensörü `.squeeze()` ile 2D'ye indirgenmeli, yoksa OpenCV `applyColorMap` hata veriyor.

## Environment Notes

- **Final env**: `surgtwin2` (CPython 3.10.20), PyTorch 2.5.1+cu121
- **gsplat**: 1.5.3+pt24cu121 pre-built wheel from GitHub Releases (JIT compilation bypass edildi)
- **CUDA**: Driver 580.159.03, toolkit 12.1 via conda (`cuda-nvcc` package)
- **GCC**: conda-forge `gxx_impl_linux-64=11` (PyTorch ABI uyumu için GCC 11.4)
- **Broken env `surgtwin`**: conda-forge `cuda-toolkit` metapackage graalpy dependency olarak çekip CPython'ı ezdi. Silinebilir.

## Dataset

- **Source**: OpenDataLab (`OpenDataLab/SERV-CT`)
- **Location**: `data/raw/SERV-CT/SERV-CT-ALL/`
- **Structure**: `Experiment_1/{Left_rectified, Right_rectified, Ground_truth_CT, Rectified_calibration}`, same for Experiment_2
- **Size**: 399 MB (SERV-CT-ALL.zip), 16 frames total

## Next Steps

- Clean up old env: `conda env remove -n surgtwin`
- Push SurgTwin-GS repo to GitHub (bug fixes)
- Rotate OpenDataLab AK/SK credentials
- Plan Sprint 1
