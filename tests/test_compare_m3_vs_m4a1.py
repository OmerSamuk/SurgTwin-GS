from scripts.compare_m3_vs_m4a1 import _compute_gate, _vram_tier, _delta


def _m3_metrics(psnr=20.21, depth_rmse=0.0364):
    return {
        "val_psnr": psnr,
        "val_ssim": 0.838,
        "val_lpips": 0.557,
        "val_depth_rmse_m_raw": depth_rmse,
        "val_depth_mae_m_raw": 0.0316,
        "n_gaussians": 20000,
        "variant": "h1",
    }


def _m4_metrics(psnr=20.21, depth_rmse=0.0364, n_gaussians=50000):
    return {
        "val_psnr": psnr,
        "val_ssim": 0.838,
        "val_lpips": 0.557,
        "val_depth_rmse_m_raw": depth_rmse,
        "val_depth_mae_m_raw": 0.0316,
        "val_w_photo_mean": 0.52,
        "n_gaussians": n_gaussians,
        "enable_densification": False,
    }


def test_full_pass_depth_improved():
    m3 = _m3_metrics(20.21, 0.0364)
    m4 = _m4_metrics(20.50, 0.0280, 50000)
    gate = _compute_gate(m3, m4, vram={"max_vram_gb": 14.0, "mean_vram_gb": 12.0, "n_samples": 100})
    assert gate["status"] == "FULL PASS"
    assert gate["depth_rmse_pass"] is True
    assert gate["psnr_pass"] is True
    assert gate["n_gaussians_ok"] is True
    assert gate["vram_tier"] == "green"
    assert gate["m4_a1b_trigger"] is True


def test_partial_positive_depth_improved_but_not_030():
    m3 = _m3_metrics(20.21, 0.0364)
    m4 = _m4_metrics(20.30, 0.0330, 50000)
    gate = _compute_gate(m3, m4, vram={"max_vram_gb": 14.0, "mean_vram_gb": 12.0, "n_samples": 100})
    assert gate["status"] == "PARTIAL POSITIVE"
    assert gate["improved_over_m3_h1"] is True
    assert gate["depth_rmse_pass"] is False
    assert gate["m4_a1b_trigger"] is True
    assert gate["relative_improvement_pct"] is not None
    assert gate["relative_improvement_pct"] > 0


def test_improvement_below_8pct_not_over_0335_no_trigger():
    m3 = _m3_metrics(20.21, 0.0364)
    m4 = _m4_metrics(20.25, 0.0340, 50000)
    gate = _compute_gate(m3, m4, vram={"max_vram_gb": 14.0, "mean_vram_gb": 12.0, "n_samples": 100})
    assert gate["status"] == "PARTIAL POSITIVE"
    assert gate["improved_over_m3_h1"] is True
    assert gate["relative_improvement_pct"] is not None
    assert gate["m4_a1b_trigger"] is False  # 0.0340 > 0.0335, 6.6% < 8% → no trigger


def test_no_improvement_negative():
    m3 = _m3_metrics(20.21, 0.0364)
    m4 = _m4_metrics(20.21, 0.0364, 50000)
    gate = _compute_gate(m3, m4, vram={"max_vram_gb": 14.0, "mean_vram_gb": 12.0, "n_samples": 100})
    assert gate["status"] == "NEGATIVE (no improvement)"
    assert gate["improved_over_m3_h1"] is False
    assert gate["m4_a1b_trigger"] is False


def test_worse_depth_negative():
    m3 = _m3_metrics(20.21, 0.0364)
    m4 = _m4_metrics(20.10, 0.0390, 50000)
    gate = _compute_gate(m3, m4, vram={"max_vram_gb": 14.0, "mean_vram_gb": 12.0, "n_samples": 100})
    assert gate["status"] == "NEGATIVE (no improvement)"
    assert gate["improved_over_m3_h1"] is False
    assert gate["m4_a1b_trigger"] is False


def test_psnr_fails_gate():
    m3 = _m3_metrics(20.21, 0.0364)
    m4 = _m4_metrics(16.0, 0.0364, 50000)
    gate = _compute_gate(m3, m4, vram={"max_vram_gb": 14.0, "mean_vram_gb": 12.0, "n_samples": 100})
    assert gate["status"] == "FAIL"
    assert gate["psnr_pass"] is False
    assert "PSNR" in " ".join(gate.get("blocking_reasons", []))


def test_n_gaussians_mismatch():
    m3 = _m3_metrics(20.21, 0.0364)
    m4 = _m4_metrics(20.30, 0.0350, 20000)
    gate = _compute_gate(m3, m4, vram={"max_vram_gb": 14.0, "mean_vram_gb": 12.0, "n_samples": 100})
    assert gate["n_gaussians_ok"] is False
    assert "n_gaussians" in " ".join(gate.get("blocking_reasons", []))


def test_vram_tier_green():
    assert _vram_tier(12.0) == "green"


def test_vram_tier_acceptable():
    assert _vram_tier(18.0) == "acceptable"


def test_vram_tier_warning():
    assert _vram_tier(21.0) == "warning"


def test_vram_tier_fail():
    assert _vram_tier(22.0) == "fail"


def test_vram_tier_none():
    assert _vram_tier(None) == "N/A"


def test_delta_higher_better():
    result = _delta(20.0, 25.0, higher_is_better=True)
    assert "▲" in result
    assert "+5.0000" in result


def test_delta_lower_better():
    result = _delta(0.0364, 0.0280, higher_is_better=False)
    assert "▲" in result
    assert "-0.0084" in result


def test_delta_none():
    assert _delta(None, 10.0) == ""
    assert _delta(10.0, None) == ""
