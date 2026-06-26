# Sprint 0 Mandatory Revision Notes

These are the mandatory fixes required before Sprint 0A can be considered accepted on the VM.

## 1. Add `reprojection_overlay.png` generation

**Status:** Mandatory

`validate_servct_geometry.py` must write:

```text
outputs/debug/sprint0_geometry/reprojection_overlay.png
```

The overlay must visualize sampled original depth pixels and their reprojected pixel locations on the left RGB image. This is required because Sprint 0/Sprint 0A acceptance depends not only on numeric reprojection error but also on visual camera/depth sanity checking.

**Acceptance condition:**

```text
left_rgb.png
left_depth_color.png
reprojection_overlay.png
geometry_report.json
```

must all be produced by the geometry validation command.

---

## 2. Remove hard-coded SERV-CT image size

**Status:** Mandatory

`explore_servct.py` must not hard-code:

```python
height, width = 576, 720
```

The image size must be read from the actual left RGB file. The manifest must reflect the real image dimensions.

Required behavior:

```python
img = cv2.imread(str(left_rgb))
if img is None:
    raise FileNotFoundError(...)
height, width = img.shape[:2]
```

**Acceptance condition:**

The manifest `height` and `width` values must match the actual RGB and depth file shapes for every selected sample.

---

## 3. Fix `GsplatBackend` depth/alpha semantics verification

**Status:** Mandatory

`backend_gsplat.py` must not assume that channel index 3 is alpha and channel index 4 is metric depth solely from `render_mode="RGB+D"` and tensor shape.

The wrapper must explicitly inspect the installed `gsplat` API return format and normalize it to the `RenderOutput` contract:

```text
rgb: [H, W, 3], float32, [0, 1]
alpha: [H, W], float32, [0, 1], or None if unavailable
depth: [H, W], float32, or None if unavailable
aux["depth_semantics"]: "metric_meters" | "relative_aligned" | "relative_unaligned" | "unavailable"
aux["supports_metric_depth"]: bool
```

Metric depth may be reported only if the wrapper verifies that the returned depth is metric camera-depth in meters under the project convention. If this cannot be verified, set:

```python
depth_semantics = "unavailable"
supports_metric_depth = False
```

or use an explicitly documented non-metric status.

**Acceptance condition:**

`render_report.json` must contain truthful depth semantics. Non-verified depth must not be labeled as `metric_meters`.

---

## 4. Make CUDA handling fail clearly

**Status:** Mandatory

`sprint0_render_servct.py` currently warns if CUDA is unavailable but then calls CUDA-specific operations. This must be changed to a clear fail-fast behavior.

Required behavior:

```python
if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is required for Sprint 0 gsplat rendering. "
        "Check NVIDIA driver, CUDA runtime, PyTorch CUDA build, and GPU availability."
    )
```

CUDA-only calls such as the following must not run unless CUDA is available:

```python
torch.cuda.reset_peak_memory_stats()
tensor.cuda()
torch.cuda.max_memory_allocated()
```

**Acceptance condition:**

If CUDA is unavailable, the script exits with a clear diagnostic message instead of partially continuing and failing later with an unrelated error.
