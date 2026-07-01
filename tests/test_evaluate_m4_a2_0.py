import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.evaluate_m4_a2_0 import (
    validate_m4_a2_0_config,
    classify_clip_ratio,
    compute_gate,
    REQUIRED_CONFIG,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_CONFIG = dict(REQUIRED_CONFIG)


def _make_final_metrics(overrides=None):
    base = {
        "val_psnr": 20.0,
        "val_ssim": 0.7583,
        "val_lpips": 0.6002,
        "val_depth_rmse_m_raw": 0.036,
        "val_depth_mae_m_raw": 0.030,
        "val_abs_rel": 0.28,
        "val_median_aligned_rmse_m": 0.022,
        "clip_active_ratio": 0.855,
        "loss_decreased": True,
        "warmup_iters": 200,
        "enable_densification": False,
        "run_mode": "production",
        "m2a_gate": "PASS",
    }
    if overrides:
        base.update(overrides)
    return base


def _make_early_metrics(losses_1_to_10):
    """losses_1_to_10: list of 10 loss_total values for iter 1..10."""
    result = {}
    for i, loss in enumerate(losses_1_to_10, start=1):
        result[i] = {"iter": i, "loss_total": loss, "psnr": 20.0 - loss * 10}
    return result


def _dummy_run_dir():
    """Return a dummy Path — compute_gate only uses run_dir for ablation output_dir path."""
    return Path("/dummy/run_dir")


# ---------------------------------------------------------------------------
# validate_m4_a2_0_config
# ---------------------------------------------------------------------------


def test_config_guard_rejects_50k():
    cfg = dict(_VALID_CONFIG)
    cfg["init_num_points"] = 50000
    result = validate_m4_a2_0_config(cfg)
    assert result["config_valid"] is False
    reasons = " ".join(result["blocking_reasons"])
    assert "init_num_points" in reasons


def test_config_guard_rejects_densification_on():
    cfg = dict(_VALID_CONFIG)
    cfg["enable_densification"] = True
    result = validate_m4_a2_0_config(cfg)
    assert result["config_valid"] is False
    reasons = " ".join(result["blocking_reasons"])
    assert "enable_densification" in reasons


def test_config_guard_rejects_wrong_variant():
    cfg = dict(_VALID_CONFIG)
    cfg["variant"] = "h2"
    result = validate_m4_a2_0_config(cfg)
    assert result["config_valid"] is False
    reasons = " ".join(result["blocking_reasons"])
    assert "variant" in reasons


def test_config_guard_accepts_valid():
    result = validate_m4_a2_0_config(_VALID_CONFIG)
    assert result["config_valid"] is True


def test_config_guard_accepts_float_tolerance():
    cfg = dict(_VALID_CONFIG)
    cfg["max_grad_norm"] = 1.500000001
    result = validate_m4_a2_0_config(cfg)
    assert result["config_valid"] is True


def test_config_guard_rejects_float_outside_tolerance():
    cfg = dict(_VALID_CONFIG)
    cfg["max_grad_norm"] = 1.51
    result = validate_m4_a2_0_config(cfg)
    assert result["config_valid"] is False


# ---------------------------------------------------------------------------
# classify_clip_ratio
# ---------------------------------------------------------------------------


def test_clip_healthy():
    assert classify_clip_ratio(0.3) == "healthy"
    assert classify_clip_ratio(0.5) == "healthy"


def test_clip_warning():
    assert classify_clip_ratio(0.6) == "warning"
    assert classify_clip_ratio(0.8) == "warning"


def test_clip_bound():
    assert classify_clip_ratio(0.85) == "clip_bound"
    assert classify_clip_ratio(1.0) == "clip_bound"


def test_clip_unknown():
    assert classify_clip_ratio(None) == "unknown"


# ---------------------------------------------------------------------------
# compute_gate — PASS
# ---------------------------------------------------------------------------


def test_full_pass():
    fm = _make_final_metrics()
    early = _make_early_metrics([0.036] * 10)
    gate = compute_gate(fm, early, _VALID_CONFIG, _dummy_run_dir())
    assert gate["status"] == "PASS"
    assert gate["psnr_pass"] is True
    assert gate["depth_rmse_pass"] is True
    assert gate["loss_decreased"] is True
    assert gate["unrecovered_jump_fail"] is False
    assert "proceed" in gate["recommendation"].lower()


# ---------------------------------------------------------------------------
# compute_gate — recovery-aware jump
# ---------------------------------------------------------------------------


def test_recovery_pass_warmup_hump():
    """iter2 spikes (2.5x) but recovers by iter5 — should PASS."""
    losses = [0.036, 0.092, 0.098, 0.095, 0.035, 0.034, 0.034, 0.034, 0.034, 0.034]
    fm = _make_final_metrics()
    early = _make_early_metrics(losses)
    gate = compute_gate(fm, early, _VALID_CONFIG, _dummy_run_dir())
    assert gate["early_recovery_pass"] is True
    assert gate["unrecovered_jump_fail"] is False
    assert gate["iter2_jump_ratio"] == round(0.092 / 0.036, 4)
    # Legacy fields (deprecated)
    assert gate["catastrophic_jump_1p5"] is False
    assert gate["catastrophic_jump_2p0"] is False


def test_unrecovered_jump_fail():
    """iter2 spikes and never recovers — should FAIL."""
    losses = [0.036, 0.092, 0.090, 0.088, 0.087, 0.087, 0.087, 0.087, 0.087, 0.087]
    fm = _make_final_metrics()
    early = _make_early_metrics(losses)
    gate = compute_gate(fm, early, _VALID_CONFIG, _dummy_run_dir())
    assert gate["early_recovery_pass"] is False
    assert gate["unrecovered_jump_fail"] is True
    assert gate["status"] == "CLEAR_FAIL"


# ---------------------------------------------------------------------------
# compute_gate — MARGINAL FAIL
# ---------------------------------------------------------------------------


def test_marginal_fail_psnr_narrow():
    """PSNR 19.4 (Δ0.1 dB ≤ 0.2), depth OK — should be MARGINAL."""
    fm = _make_final_metrics({"val_psnr": 19.4, "val_depth_rmse_m_raw": 0.036})
    early = _make_early_metrics([0.036] * 10)
    gate = compute_gate(fm, early, _VALID_CONFIG, _dummy_run_dir())
    assert gate["status"] == "MARGINAL_FAIL"
    assert gate["psnr_pass"] is False
    assert gate["depth_rmse_pass"] is True
    assert "marginal" in gate
    assert gate["marginal"]["ablation_config"]["max_grad_norm"] == 2.0


def test_marginal_fail_depth_narrow():
    """Depth 0.0383 (Δ0.3 mm ≤ 0.001 m), PSNR OK — should be MARGINAL."""
    fm = _make_final_metrics({"val_psnr": 20.0, "val_depth_rmse_m_raw": 0.0383})
    early = _make_early_metrics([0.036] * 10)
    gate = compute_gate(fm, early, _VALID_CONFIG, _dummy_run_dir())
    assert gate["status"] == "MARGINAL_FAIL"
    assert gate["psnr_pass"] is True
    assert gate["depth_rmse_pass"] is False
    assert "marginal" in gate


def test_clear_fail_psnr_wide_gap():
    """PSNR 17.0 (Δ2.5 dB > 0.2) — should be CLEAR."""
    fm = _make_final_metrics({"val_psnr": 17.0})
    early = _make_early_metrics([0.036] * 10)
    gate = compute_gate(fm, early, _VALID_CONFIG, _dummy_run_dir())
    assert gate["status"] == "CLEAR_FAIL"
    assert "marginal" not in gate


# ---------------------------------------------------------------------------
# compute_gate — clip does NOT block
# ---------------------------------------------------------------------------


def test_clip_bound_passes_gate():
    """clip_active_ratio=0.85 → clip_bound warning, but gate still PASS."""
    fm = _make_final_metrics({"clip_active_ratio": 0.85})
    early = _make_early_metrics([0.036] * 10)
    gate = compute_gate(fm, early, _VALID_CONFIG, _dummy_run_dir())
    assert gate["status"] == "PASS"
    assert gate["clip_health"] == "clip_bound"


# ---------------------------------------------------------------------------
# compute_gate — loss_decreased=False → CLEAR
# ---------------------------------------------------------------------------


def test_loss_not_decreased_clear_fail():
    fm = _make_final_metrics({"loss_decreased": False})
    early = _make_early_metrics([0.036] * 10)
    gate = compute_gate(fm, early, _VALID_CONFIG, _dummy_run_dir())
    assert gate["status"] == "CLEAR_FAIL"
    assert gate["loss_decreased"] is False


# ---------------------------------------------------------------------------
# compute_gate — diagnostic fields
# ---------------------------------------------------------------------------


def test_diagnostic_fields_present():
    fm = _make_final_metrics()
    early = _make_early_metrics([0.036] * 10)
    gate = compute_gate(fm, early, _VALID_CONFIG, _dummy_run_dir())
    assert gate["median_aligned_rmse_m"] == 0.022
    assert gate["val_ssim"] == 0.7583
    assert gate["val_lpips"] == 0.6002
    assert gate["val_abs_rel"] == 0.28
    assert gate["clip_health"] == "clip_bound"
    assert gate["gate_eligible_run_mode"] == "production"
    assert gate["m2a_gate_confirmed"] == "PASS"


# ---------------------------------------------------------------------------
# CLI-level: INVALID_FOR_GATE
# ---------------------------------------------------------------------------


def _write_run_dir(base, config, final_metrics, early_losses):
    """Write a minimal run directory with config.json, final_metrics.json, metrics.jsonl."""
    (base / "config.json").write_text(json.dumps(config))
    (base / "final_metrics.json").write_text(json.dumps(final_metrics))
    lines = "\n".join(
        json.dumps({"iter": i, "loss_total": loss})
        for i, loss in enumerate(early_losses, start=1)
    )
    (base / "metrics.jsonl").write_text(lines)


# ---------------------------------------------------------------------------
# compute_gate — ablation no repeat recommendation (Phase 0 bug fix)
# ---------------------------------------------------------------------------


def test_ablation_no_repeat_recommendation():
    """allow_ablation=True → recommendation says 'No further ablation'."""
    fm = _make_final_metrics({"val_psnr": 19.4, "val_depth_rmse_m_raw": 0.0383})
    early = _make_early_metrics([0.036] * 10)
    gate = compute_gate(fm, early, _VALID_CONFIG, _dummy_run_dir(), allow_ablation=True)
    assert gate["status"] == "MARGINAL_FAIL"
    assert "No further" in gate["recommendation"]
    assert "ablation_output_dir" not in gate["marginal"] or gate["marginal"]["ablation_output_dir"] is None
    assert gate.get("gate_profile") == "M4-A2-0_ABLATION_GRAD2"
    assert gate.get("canonical_gate") is False


def test_gate_profile_fields():
    """Canonical run → gate_profile=M4-A2-0_CANONICAL, canonical_gate=true."""
    fm = _make_final_metrics()
    early = _make_early_metrics([0.036] * 10)
    gate = compute_gate(fm, early, _VALID_CONFIG, _dummy_run_dir(), allow_ablation=False)
    assert gate.get("gate_profile") == "M4-A2-0_CANONICAL"
    assert gate.get("canonical_gate") is True


def test_invalid_config_cli_writes_gate_and_exits_2():
    """50K config → evaluator writes gate JSON with INVALID_FOR_GATE, exit code 2."""
    cfg = dict(_VALID_CONFIG)
    cfg["init_num_points"] = 50000
    fm = _make_final_metrics()
    early = [0.036] * 10

    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "run"
        run_dir.mkdir()
        _write_run_dir(run_dir, cfg, fm, early)

        evaluator = str(Path(__file__).resolve().parent.parent / "scripts" / "evaluate_m4_a2_0.py")
        result = subprocess.run(
            [sys.executable, evaluator, "--run_dir", str(run_dir)],
            capture_output=True, text=True
        )

        assert result.returncode == 2, f"expected exit 2, got {result.returncode}: {result.stderr}"

        gate_path = run_dir / "m4_a2_0_gate.json"
        assert gate_path.exists(), "gate JSON not written"
        gate = json.loads(gate_path.read_text())
        assert gate["status"] == "INVALID_FOR_GATE"
        assert gate["config_valid"] is False
        reasons = " ".join(gate["blocking_reasons"])
        assert "init_num_points" in reasons
        # Ensure no KeyError in _write_and_report output
        assert "Status: INVALID_FOR_GATE" in result.stdout
