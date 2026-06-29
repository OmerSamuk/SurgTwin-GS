import numpy as np
import cv2


def detect_specular_hsv(
    rgb: np.ndarray,
    v_threshold: float = 220.0,
    s_threshold: float = 40.0,
    whiteness_r: float = 220.0,
    whiteness_g: float = 220.0,
    whiteness_b: float = 220.0,
    use_whiteness: bool = True,
    morph_kernel_size: int = 3,
) -> np.ndarray:
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected RGB [H, W, 3], got shape {rgb.shape}")
    if rgb.dtype == np.float32 or rgb.dtype == np.float64:
        if rgb.max() > 1.0 or rgb.min() < 0.0:
            raise ValueError(f"Float RGB must be in [0, 1], got range [{rgb.min()}, {rgb.max()}]")
        img_255 = (rgb * 255).astype(np.uint8)
    elif rgb.dtype == np.uint8:
        img_255 = rgb
    else:
        raise ValueError(f"Unsupported dtype: {rgb.dtype}. Expected uint8 or float32 in [0, 1].")

    hsv = cv2.cvtColor(img_255, cv2.COLOR_RGB2HSV)
    h = hsv[:, :, 0].astype(np.float32)
    s = hsv[:, :, 1].astype(np.float32)
    v = hsv[:, :, 2].astype(np.float32)

    v_high = v > v_threshold
    s_low = s < s_threshold
    mask_hsv = v_high & s_low

    mask = mask_hsv.copy()
    if use_whiteness:
        r, g, b = img_255[:, :, 0], img_255[:, :, 1], img_255[:, :, 2]
        mask_white = (r > whiteness_r) & (g > whiteness_g) & (b > whiteness_b)
        mask = mask_hsv | mask_white

    if morph_kernel_size > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel).astype(bool)

    return mask
