import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.evaluate_m4_a2_1 import (
    validate_m4_a2_1_config,
    compute_gate,
    REQUIRED_CONFIG,
)

_VALID_CONFIG = dict(REQUIRED_CONFIG)


def _make_fm(overrides=None):
    base = {
        "val_psnr": 20.5,
        "val_ssim": 0.76,
        "val_lpips": 0.59,
        "val_depth_rmse_m_raw": 0.028,
        "val_depth_mae_m_raw": 0.022,
        "val_abs_rel": 0.25,
        "val_median_aligned_rmse_m": 0.025,
        "loss_decreased": True,
        "enable_densification": True,
        "n_gaussians_initial": 20000,
        "n_gaussians_final": 25000,
        "gaussian_growth_ratio": 1.25,
        "densification_steps_count": 3,
        "total_cloned": 6000,
        "total_pruned": 1000,
        "max_gaussians_hit": False,
    }
    if overrides:
        base.update(overrides)
    return base


def _make_early(overrides_1=None, overrides_2=None):
    b1 = {"loss_total": 0.050}
    b2 = {"loss_total": 0.045}
    if overrides_1:
        b1.update(overrides_1)
    if overrides_2:
        b2.update(overrides_2)
    return {1: b1, 2: b2}


# ---------------------------------------------------------------------------
# Config Guard
# ---------------------------------------------------------------------------

def test_config_guard_accepts_valid():
    assert validate_m4_a2_1_config(_VALID_CONFIG)["config_valid"]


def test_config_guard_rejects_densification_off():
    cfg = dict(_VALID_CONFIG)
    cfg["enable_densification"] = False
    assert not validate_m4_a2_1_config(cfg)["config_valid"]


def test_config_guard_rejects_wrong_iterations():
    cfg = dict(_VALID_CONFIG)
    cfg["iterations"] = 300
    assert not validate_m4_a2_1_config(cfg)["config_valid"]


def test_config_guard_rejects_wrong_variant():
    cfg = dict(_VALID_CONFIG)
    cfg["variant"] = "h3"
    assert not validate_m4_a2_1_config(cfg)["config_valid"]


def test_config_guard_rejects_wrong_lambda():
    cfg = dict(_VALID_CONFIG)
    cfg["lambda_depth"] = 0.5
    assert not validate_m4_a2_1_config(cfg)["config_valid"]


# ---------------------------------------------------------------------------
# 3-Tier Gate — FULL_SUCCESS
# ---------------------------------------------------------------------------

def test_full_success():
    fm = _make_fm()
    early = _make_early()
    gate = compute_gate(fm, early, _VALID_CONFIG)
    assert gate["status"] == "FULL_SUCCESS"
    assert gate["growth_occurred"]
    assert gate["clone_happened"]
    assert gate["psnr_pass"]
    assert gate["depth_rmse_pass"]
    assert gate["loss_decreased"]


def test_full_success_more_gaussians():
    fm = _make_fm({"n_gaussians_final": 48000, "gaussian_growth_ratio": 2.4})
    early = _make_early()
    gate = compute_gate(fm, early, _VALID_CONFIG)
    assert gate["status"] == "FULL_SUCCESS"


# ---------------------------------------------------------------------------
# 3-Tier Gate — SERV_CT_PARTIAL_POSITIVE
# ---------------------------------------------------------------------------

def test_partial_positive_depth_marginal():
    fm = _make_fm({"val_depth_rmse_m_raw": 0.035})
    early = _make_early()
    gate = compute_gate(fm, early, _VALID_CONFIG)
    assert gate["status"] == "SERV_CT_PARTIAL_POSITIVE"
    assert "depth_rmse" in gate["marginal_notes"][0]
    assert gate["growth_occurred"]
    assert gate["clone_happened"]
    assert gate["loss_decreased"]


def test_partial_positive_psnr_marginal():
    fm = _make_fm({"val_psnr": 18.0})
    early = _make_early()
    gate = compute_gate(fm, early, _VALID_CONFIG)
    assert gate["status"] == "SERV_CT_PARTIAL_POSITIVE"
    assert "psnr" in gate["marginal_notes"][0]


def test_partial_positive_both_marginal():
    fm = _make_fm({"val_psnr": 18.0, "val_depth_rmse_m_raw": 0.040})
    early = _make_early()
    gate = compute_gate(fm, early, _VALID_CONFIG)
    assert gate["status"] == "SERV_CT_PARTIAL_POSITIVE"
    assert len(gate["marginal_notes"]) == 2


# ---------------------------------------------------------------------------
# 3-Tier Gate — CONTROLLED_NEGATIVE
# ---------------------------------------------------------------------------

def test_negative_densification_disabled():
    fm = _make_fm({"enable_densification": False})
    early = _make_early()
    gate = compute_gate(fm, early, _VALID_CONFIG)
    assert gate["status"] == "CONTROLLED_NEGATIVE"
    assert any("enable_densification=False" in r for r in gate["blocking_reasons"])


def test_negative_no_growth():
    fm = _make_fm({"n_gaussians_final": 20000, "gaussian_growth_ratio": 1.0})
    early = _make_early()
    gate = compute_gate(fm, early, _VALID_CONFIG)
    assert gate["status"] == "CONTROLLED_NEGATIVE"
    assert any("n_gaussians_final" in r for r in gate["blocking_reasons"])


def test_negative_no_clones():
    fm = _make_fm({"total_cloned": 0})
    early = _make_early()
    gate = compute_gate(fm, early, _VALID_CONFIG)
    assert gate["status"] == "CONTROLLED_NEGATIVE"
    assert any("total_cloned = 0" in r for r in gate["blocking_reasons"])


def test_negative_loss_not_decreased():
    fm = _make_fm({"loss_decreased": False})
    early = _make_early()
    gate = compute_gate(fm, early, _VALID_CONFIG)
    assert gate["status"] == "CONTROLLED_NEGATIVE"
    assert any("loss_decreased" in r for r in gate["blocking_reasons"])


def test_negative_unrecovered_jump():
    early = _make_early({"loss_total": 0.050}, {"loss_total": 0.150})
    fm = _make_fm()
    gate = compute_gate(fm, early, _VALID_CONFIG)
    assert gate["status"] == "CONTROLLED_NEGATIVE"
    assert any("unrecovered_jump_fail" in r for r in gate["blocking_reasons"])


def test_negative_multiple_blockers():
    fm = _make_fm({"enable_densification": False, "loss_decreased": False, "total_cloned": 0})
    early = _make_early()
    gate = compute_gate(fm, early, _VALID_CONFIG)
    assert gate["status"] == "CONTROLLED_NEGATIVE"
    assert len(gate["blocking_reasons"]) >= 2


# ---------------------------------------------------------------------------
# Gate fields output
# ---------------------------------------------------------------------------

def test_gate_fields_present():
    fm = _make_fm()
    early = _make_early()
    gate = compute_gate(fm, early, _VALID_CONFIG)
    for key in ("densification_enabled", "growth_occurred", "clone_happened",
                 "n_gaussians_initial", "n_gaussians_final", "gaussian_growth_ratio",
                 "densification_steps_count", "total_cloned", "total_pruned",
                 "max_gaussians_hit", "loss_decreased", "status", "label"):
        assert key in gate, f"Missing key: {key}"


def test_gate_label_and_line():
    fm = _make_fm()
    early = _make_early()
    gate = compute_gate(fm, early, _VALID_CONFIG)
    assert "M4-A2-1" in gate["label"]
    assert "SERV-CT" in gate["project_line"]


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------

def test_cli_invalid_config_writes_gate_and_exits_2():
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        cfg = dict(_VALID_CONFIG)
        cfg["enable_densification"] = False
        (run_dir / "config.json").write_text(json.dumps(cfg))
        early = {1: {"iter": 1, "loss_total": 0.05}, 2: {"iter": 2, "loss_total": 0.045}}
        (run_dir / "metrics.jsonl").write_text(
            "\n".join(json.dumps(early[k]) for k in sorted(early)) + "\n"
        )
        (run_dir / "final_metrics.json").write_text(json.dumps(_make_fm()))
        result = subprocess.run(
            [sys.executable, str(Path("scripts/evaluate_m4_a2_1.py")),
             f"--run_dir={run_dir}"],
            capture_output=True, text=True, timeout=15,
        )
        assert result.returncode == 2
        gate_path = run_dir / "m4_a2_1_gate.json"
        assert gate_path.exists()
        gate = json.loads(gate_path.read_text())
        assert gate["status"] == "CONTROLLED_NEGATIVE"
        assert not gate["config_valid"]


def test_cli_full_success_exit_0():
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        (run_dir / "config.json").write_text(json.dumps(_VALID_CONFIG))
        early = {1: {"iter": 1, "loss_total": 0.05}, 2: {"iter": 2, "loss_total": 0.045}}
        (run_dir / "metrics.jsonl").write_text(
            "\n".join(json.dumps(early[k]) for k in sorted(early)) + "\n"
        )
        (run_dir / "final_metrics.json").write_text(json.dumps(_make_fm()))
        result = subprocess.run(
            [sys.executable, str(Path("scripts/evaluate_m4_a2_1.py")),
             f"--run_dir={run_dir}"],
            capture_output=True, text=True, timeout=15,
        )
        assert result.returncode == 0
        gate_path = run_dir / "m4_a2_1_gate.json"
        assert gate_path.exists()
        gate = json.loads(gate_path.read_text())
        assert gate["status"] == "FULL_SUCCESS"


def test_cli_partial_positive_exit_1():
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        (run_dir / "config.json").write_text(json.dumps(_VALID_CONFIG))
        early = {1: {"iter": 1, "loss_total": 0.05}, 2: {"iter": 2, "loss_total": 0.045}}
        lines = "\n".join(json.dumps(early[k]) for k in sorted(early))
        (run_dir / "metrics.jsonl").write_text(lines + "\n")
        fm = _make_fm({"val_psnr": 18.0, "val_depth_rmse_m_raw": 0.040})
        (run_dir / "final_metrics.json").write_text(json.dumps(fm))
        result = subprocess.run(
            [sys.executable, str(Path("scripts/evaluate_m4_a2_1.py")),
             f"--run_dir={run_dir}"],
            capture_output=True, text=True, timeout=15,
        )
        assert result.returncode == 1
        gate = json.loads((run_dir / "m4_a2_1_gate.json").read_text())
        assert gate["status"] == "SERV_CT_PARTIAL_POSITIVE"


def test_cli_controlled_negative_exit_1():
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        (run_dir / "config.json").write_text(json.dumps(_VALID_CONFIG))
        early = {1: {"iter": 1, "loss_total": 0.05}, 2: {"iter": 2, "loss_total": 0.045}}
        lines = "\n".join(json.dumps(early[k]) for k in sorted(early))
        (run_dir / "metrics.jsonl").write_text(lines + "\n")
        fm = _make_fm({"loss_decreased": False})
        (run_dir / "final_metrics.json").write_text(json.dumps(fm))
        result = subprocess.run(
            [sys.executable, str(Path("scripts/evaluate_m4_a2_1.py")),
             f"--run_dir={run_dir}"],
            capture_output=True, text=True, timeout=15,
        )
        assert result.returncode == 1
        gate = json.loads((run_dir / "m4_a2_1_gate.json").read_text())
        assert gate["status"] == "CONTROLLED_NEGATIVE"
