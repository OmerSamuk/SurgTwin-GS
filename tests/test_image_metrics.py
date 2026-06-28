import torch
from surgtwin.evaluation.image_metrics import psnr, ssim


def test_psnr_identical():
    img = torch.rand(100, 200, 3)
    assert psnr(img, img) == float("inf")


def test_psnr_finite():
    a = torch.ones(100, 200, 3) * 0.5
    b = torch.ones(100, 200, 3) * 0.6
    p = psnr(a, b)
    assert p < float("inf")
    assert p > 0


def test_ssim_identical():
    try:
        from skimage.metrics import structural_similarity  # noqa
    except ImportError:
        import pytest
        pytest.skip("skimage not installed")
    img = torch.rand(50, 50, 3)
    score = ssim(img, img)
    assert abs(score - 1.0) < 1e-6


def test_ssim_different():
    try:
        from skimage.metrics import structural_similarity  # noqa
    except ImportError:
        import pytest
        pytest.skip("skimage not installed")
    a = torch.zeros(50, 50, 3)
    b = torch.ones(50, 50, 3)
    score = ssim(a, b)
    assert 0 <= score < 1.0


def test_lpips_graceful_skip():
    try:
        from surgtwin.evaluation.image_metrics import lpips_score
        a = torch.rand(100, 100, 3)
        b = torch.rand(100, 100, 3)
        result = lpips_score(a, b, device="cpu")
        if result is None:
            import pytest
            pytest.skip("LPIPS weights not cached; skip is acceptable")
        assert isinstance(result, float)
        assert 0 <= result <= 2
    except ImportError:
        import pytest
        pytest.skip("lpips package not installed; skip is acceptable")
