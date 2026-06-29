import numpy as np
import pytest

from surgtwin.masks.specular import detect_specular_hsv


def test_detect_specular_hsv_returns_bool():
    rgb = np.ones((64, 64, 3), dtype=np.float32)
    mask = detect_specular_hsv(rgb)
    assert mask.dtype == bool
    assert mask.shape == (64, 64)


def test_bright_white_pixel_detected():
    rgb = np.zeros((32, 32, 3), dtype=np.float32)
    rgb[15:18, 15:18] = [1.0, 1.0, 1.0]
    mask = detect_specular_hsv(rgb, v_threshold=200.0, s_threshold=50.0, morph_kernel_size=1)
    assert mask[16, 16]


def test_dark_pixel_not_detected():
    rgb = np.zeros((32, 32, 3), dtype=np.float32)
    mask = detect_specular_hsv(rgb, v_threshold=200.0, s_threshold=50.0)
    assert not mask.any()


def test_uint8_input():
    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    rgb[15:18, 15:18] = [255, 255, 255]
    mask = detect_specular_hsv(rgb, morph_kernel_size=1)
    assert mask[16, 16]


def test_invalid_dtype():
    with pytest.raises(ValueError, match="Unsupported dtype"):
        detect_specular_hsv(np.ones((4, 4, 3), dtype=np.int32))


def test_hsv_condition():
    rgb = np.zeros((16, 16, 3), dtype=np.float32)
    rgb[8, 8] = [0.9, 0.2, 0.2]
    mask = detect_specular_hsv(rgb, v_threshold=200.0, s_threshold=50.0, use_whiteness=False)
    assert not mask[8, 8]


def test_whiteness_condition():
    rgb = np.zeros((16, 16, 3), dtype=np.uint8)
    rgb[7:10, 7:10] = [240, 240, 240]
    mask = detect_specular_hsv(rgb, use_whiteness=True, morph_kernel_size=1)
    assert mask[8, 8]


def test_whiteness_disabled():
    rgb = np.ones((16, 16, 3), dtype=np.uint8) * 240
    mask_no_white = detect_specular_hsv(rgb, use_whiteness=False)
    mask_with_white = detect_specular_hsv(rgb, use_whiteness=True)
    assert mask_with_white.sum() > 0


def test_morphology_kernel_one():
    rgb = np.zeros((16, 16, 3), dtype=np.uint8)
    rgb[8, 8] = [255, 255, 255]
    mask = detect_specular_hsv(rgb, morph_kernel_size=1)
    assert mask[8, 8]


def test_all_white_full_coverage():
    rgb = np.ones((10, 10, 3), dtype=np.float32)
    mask = detect_specular_hsv(rgb)
    assert mask.sum() > 80


def test_float_range_check():
    rgb = np.random.rand(10, 10, 3).astype(np.float32) * 2.0
    with pytest.raises(ValueError, match="Float RGB must be in"):
        detect_specular_hsv(rgb)
