from scripts.compare_m2b_vs_m3 import _compute_gate_decisions


def _m1_metrics(psnr=59.2):
    return {"val_psnr": psnr, "val_ssim": 0.98, "val_lpips": 0.01}


def _m2b_metrics(psnr=55.16):
    return {"val_psnr": psnr, "val_ssim": 0.97, "val_lpips": 0.02,
            "val_depth_rmse_m_raw": 0.023, "val_depth_mae_m_raw": 0.015}


def _m3_row(run_id, psnr=50.0, depth_rmse=0.025):
    return {"run_id": run_id, "val_psnr": psnr, "val_ssim": 0.95,
            "val_lpips": 0.03, "val_depth_rmse_m_raw": depth_rmse,
            "val_depth_mae_m_raw": 0.015, "val_w_photo_mean": 0.65,
            "lambda_depth": 0.2}


def test_m1_missing_returns_fail_closed():
    gate = _compute_gate_decisions([_m3_row("M3-H1")], m1={}, m2b_r1=_m2b_metrics())
    assert gate["status"] == "INVALID_FOR_GATE"
    assert gate["reason"] == "m1_missing"
    assert gate["minimum_pass"] is False


def test_m1_none_psnr_returns_fail_closed():
    gate = _compute_gate_decisions([_m3_row("M3-H1")], m1={"val_psnr": None}, m2b_r1=_m2b_metrics())
    assert gate["status"] == "INVALID_FOR_GATE"
    assert gate["reason"] == "m1_missing"


def test_m2b_r1_missing_returns_fail_closed():
    gate = _compute_gate_decisions([_m3_row("M3-H1")], m1=_m1_metrics(), m2b_r1={})
    assert gate["status"] == "INVALID_FOR_GATE"
    assert gate["reason"] == "m2b_r1_missing"


def test_both_present_runs_normal_gate():
    m1 = _m1_metrics(59.2)
    m2b = _m2b_metrics(55.16)
    table = [_m3_row("M3-H1", 50.0), _m3_row("M3-H2", 52.0)]
    gate = _compute_gate_decisions(table, m1=m1, m2b_r1=m2b)
    assert gate["status"] == "OK"
    assert gate["best_m3_run_id"] == "M3-H2"
    assert gate["best_m3_val_psnr"] == 52.0
    assert gate["best_h12_run_id"] == "M3-H2"


def test_h3_trigger_psnr_low_rmse_low():
    m1 = _m1_metrics(59.2)
    m2b = _m2b_metrics(55.16)
    table = [_m3_row("M3-H1", 50.0, 0.025), _m3_row("M3-H2", 52.0, 0.020)]
    gate = _compute_gate_decisions(table, m1=m1, m2b_r1=m2b)
    assert gate["h3_should_run"] is True
    assert gate["h3_trigger_reason"] is not None


def test_h3_trigger_psnr_high_rmse_low():
    m1 = _m1_metrics(59.2)
    m2b = _m2b_metrics(55.16)
    table = [_m3_row("M3-H1", 58.0, 0.025), _m3_row("M3-H2", 59.0, 0.020)]
    gate = _compute_gate_decisions(table, m1=m1, m2b_r1=m2b)
    assert gate["h3_should_run"] is False


def test_h3_trigger_psnr_low_rmse_high():
    m1 = _m1_metrics(59.2)
    m2b = _m2b_metrics(55.16)
    table = [_m3_row("M3-H1", 50.0, 0.050)]
    gate = _compute_gate_decisions(table, m1=m1, m2b_r1=m2b)
    assert gate["h3_should_run"] is False


def test_threshold_is_dynamic():
    m1 = _m1_metrics(60.0)
    m2b = _m2b_metrics(55.0)
    table = [_m3_row("M3-H1", 57.5)]
    gate = _compute_gate_decisions(table, m1=m1, m2b_r1=m2b)
    assert gate["status"] == "OK"
    assert gate["m1_val_psnr"] == 60.0
