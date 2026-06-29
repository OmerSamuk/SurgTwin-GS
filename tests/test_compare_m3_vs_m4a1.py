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
        "val_fraction_w_photo_at_min": 0.30,
        "n_gaussians": n_gaussians,
        "enable_densification": False,
        "loss_decreased": True,
        "depth_semantics": "metric_meters",
    }


def _snapshot(sha="abc123"):
    return {"manifest_sha256": sha}


def test_full_pass_depth_improved():
    m3 = _m3_metrics(20.21, 0.0364)
    m4 = _m4_metrics(20.50, 0.0280, 50000)
    snap = _snapshot()
    gate = _compute_gate(m3, m4, vram={"max_vram_gb": 14.0, "mean_vram_gb": 12.0, "n_samples": 100},
                         m3_snapshot=snap, m4_snapshot=snap)
    assert gate["status"] == "FULL PASS"
    assert gate["depth_rmse_pass"] is True
    assert gate["psnr_pass"] is True
    assert gate["n_gaussians_ok"] is True
    assert gate["vram_tier"] == "green"
    assert gate["m4_a1b_trigger"] is True
    assert gate["manifest_sha256_match"] is True
    assert gate["loss_decreased"] is True
    assert gate["densification_off"] is True
    assert gate["depth_semantics_ok"] is True
    assert gate["w_photo_mean_in_range"] is True
    assert gate["fraction_at_min_ok"] is True


def test_partial_positive_depth_improved_but_not_030():
    m3 = _m3_metrics(20.21, 0.0364)
    m4 = _m4_metrics(20.30, 0.0330, 50000)
    snap = _snapshot()
    gate = _compute_gate(m3, m4, vram={"max_vram_gb": 14.0, "mean_vram_gb": 12.0, "n_samples": 100},
                         m3_snapshot=snap, m4_snapshot=snap)
    assert gate["status"] == "PARTIAL POSITIVE"
    assert gate["improved_over_m3_h1"] is True
    assert gate["depth_rmse_pass"] is False
    assert gate["m4_a1b_trigger"] is True
    assert gate["relative_improvement_pct"] is not None
    assert gate["relative_improvement_pct"] > 0


def test_improvement_below_8pct_not_over_0335_no_trigger():
    m3 = _m3_metrics(20.21, 0.0364)
    m4 = _m4_metrics(20.25, 0.0340, 50000)
    snap = _snapshot()
    gate = _compute_gate(m3, m4, vram={"max_vram_gb": 14.0, "mean_vram_gb": 12.0, "n_samples": 100},
                         m3_snapshot=snap, m4_snapshot=snap)
    assert gate["status"] == "PARTIAL POSITIVE"
    assert gate["improved_over_m3_h1"] is True
    assert gate["relative_improvement_pct"] is not None
    assert gate["m4_a1b_trigger"] is False  # 0.0340 > 0.0335, 6.6% < 8% → no trigger


def test_no_improvement_negative():
    m3 = _m3_metrics(20.21, 0.0364)
    m4 = _m4_metrics(20.21, 0.0364, 50000)
    snap = _snapshot()
    gate = _compute_gate(m3, m4, vram={"max_vram_gb": 14.0, "mean_vram_gb": 12.0, "n_samples": 100},
                         m3_snapshot=snap, m4_snapshot=snap)
    assert gate["status"] == "NEGATIVE (no improvement)"
    assert gate["improved_over_m3_h1"] is False
    assert gate["m4_a1b_trigger"] is False


def test_worse_depth_negative():
    m3 = _m3_metrics(20.21, 0.0364)
    m4 = _m4_metrics(20.10, 0.0390, 50000)
    snap = _snapshot()
    gate = _compute_gate(m3, m4, vram={"max_vram_gb": 14.0, "mean_vram_gb": 12.0, "n_samples": 100},
                         m3_snapshot=snap, m4_snapshot=snap)
    assert gate["status"] == "NEGATIVE (no improvement)"
    assert gate["improved_over_m3_h1"] is False
    assert gate["m4_a1b_trigger"] is False


def test_psnr_fails_gate():
    m3 = _m3_metrics(20.21, 0.0364)
    m4 = _m4_metrics(16.0, 0.0364, 50000)
    snap = _snapshot()
    gate = _compute_gate(m3, m4, vram={"max_vram_gb": 14.0, "mean_vram_gb": 12.0, "n_samples": 100},
                         m3_snapshot=snap, m4_snapshot=snap)
    assert gate["status"] == "FAIL"
    assert gate["psnr_pass"] is False
    assert "PSNR" in " ".join(gate.get("blocking_reasons", []))


def test_n_gaussians_mismatch():
    m3 = _m3_metrics(20.21, 0.0364)
    m4 = _m4_metrics(20.30, 0.0350, 20000)
    snap = _snapshot()
    gate = _compute_gate(m3, m4, vram={"max_vram_gb": 14.0, "mean_vram_gb": 12.0, "n_samples": 100},
                         m3_snapshot=snap, m4_snapshot=snap)
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


def test_manifest_mismatch_blocks():
    m3 = _m3_metrics(20.21, 0.0364)
    m4 = _m4_metrics(20.50, 0.0280, 50000)
    gate = _compute_gate(m3, m4, vram={"max_vram_gb": 14.0, "mean_vram_gb": 12.0, "n_samples": 100},
                         m3_snapshot=_snapshot("sha3"), m4_snapshot=_snapshot("sha4"))
    assert gate["manifest_sha256_match"] is False
    assert "manifest sha256 mismatch" in " ".join(gate.get("blocking_reasons", []))


def test_manifest_missing_ok():
    m3 = _m3_metrics(20.21, 0.0364)
    m4 = _m4_metrics(20.50, 0.0280, 50000)
    gate = _compute_gate(m3, m4, vram={"max_vram_gb": 14.0, "mean_vram_gb": 12.0, "n_samples": 100},
                         m3_snapshot=None, m4_snapshot=None)
    assert gate["manifest_sha256_match"] is False
    assert "manifest sha256 mismatch" in " ".join(gate.get("blocking_reasons", []))


def test_loss_not_decreased_blocks():
    m3 = _m3_metrics()
    m4 = _m4_metrics(20.50, 0.0280)
    m4["loss_decreased"] = False
    snap = _snapshot()
    gate = _compute_gate(m3, m4, vram={"max_vram_gb": 14.0, "mean_vram_gb": 12.0, "n_samples": 100},
                         m3_snapshot=snap, m4_snapshot=snap)
    assert gate["loss_decreased"] is False
    assert "loss_decreased" in " ".join(gate.get("blocking_reasons", []))


def test_densification_on_blocks():
    m3 = _m3_metrics()
    m4 = _m4_metrics(20.50, 0.0280)
    m4["enable_densification"] = True
    snap = _snapshot()
    gate = _compute_gate(m3, m4, vram={"max_vram_gb": 14.0, "mean_vram_gb": 12.0, "n_samples": 100},
                         m3_snapshot=snap, m4_snapshot=snap)
    assert gate["densification_off"] is False
    assert "enable_densification" in " ".join(gate.get("blocking_reasons", []))


def test_depth_semantics_wrong_blocks():
    m3 = _m3_metrics()
    m4 = _m4_metrics(20.50, 0.0280)
    m4["depth_semantics"] = "mm"
    snap = _snapshot()
    gate = _compute_gate(m3, m4, vram={"max_vram_gb": 14.0, "mean_vram_gb": 12.0, "n_samples": 100},
                         m3_snapshot=snap, m4_snapshot=snap)
    assert gate["depth_semantics_ok"] is False
    assert "depth_semantics" in " ".join(gate.get("blocking_reasons", []))


def test_w_photo_mean_out_of_range_blocks():
    m3 = _m3_metrics()
    m4 = _m4_metrics(20.50, 0.0280)
    m4["val_w_photo_mean"] = 0.05
    snap = _snapshot()
    gate = _compute_gate(m3, m4, vram={"max_vram_gb": 14.0, "mean_vram_gb": 12.0, "n_samples": 100},
                         m3_snapshot=snap, m4_snapshot=snap)
    assert gate["w_photo_mean_in_range"] is False
    assert "w_photo_mean" in " ".join(gate.get("blocking_reasons", []))


def test_fraction_at_min_too_high_blocks():
    m3 = _m3_metrics()
    m4 = _m4_metrics(20.50, 0.0280)
    m4["val_fraction_w_photo_at_min"] = 0.95
    snap = _snapshot()
    gate = _compute_gate(m3, m4, vram={"max_vram_gb": 14.0, "mean_vram_gb": 12.0, "n_samples": 100},
                         m3_snapshot=snap, m4_snapshot=snap)
    assert gate["fraction_at_min_ok"] is False
    assert "fraction" in " ".join(gate.get("blocking_reasons", []))
