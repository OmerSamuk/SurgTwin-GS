import argparse
import json
import sys
from pathlib import Path

# --- Onaylanan M4-A2-1 gate eşikleri (expert-answer-17) ---
PSNR_FULL_SUCCESS = 20.20
PSNR_PARTIAL_POSITIVE = 20.0
DEPTH_RMSE_FULL_SUCCESS = 0.0364
DEPTH_RMSE_PARTIAL_POSITIVE = 0.038
CATASTROPHIC_JUMP_RATIO = 1.5

# --- Onaylanan config guard (expert-answer-17) ---
REQUIRED_CONFIG = {
    "init_num_points": 20000,
    "enable_densification": True,
    "iterations": 1000,
    "warmup_iters": 200,
    "lr_opacities": 1e-2,
    "max_grad_norm": 2.0,
    "variant": "h1",
    "lambda_depth": 0.2,
    "lambda_reg": 0.0,
    "densify_from_iter": 200,
    "densify_every": 100,
    "densify_until_iter": 800,
    "densify_depth_residual_threshold": 0.02,
    "densify_w_photo_threshold": 0.3,
    "densify_max_clone_per_step": 5000,
    "densify_max_clone_fraction": 0.15,
    "densify_max_gaussians": 50000,
    "prune_min_opacity": 0.005,
    "max_prune_fraction_per_step": 0.05,
    "clone_offset_scale_factor": 0.25,
}

# --- Büyüme kontrolü için makul üst sınır ---
GROWTH_CONTROLLED_MAX_RATIO = 3.0


def validate_m4_a2_1_config(cfg):
    errors = []
    for key, expected in REQUIRED_CONFIG.items():
        actual = cfg.get(key)
        if isinstance(expected, bool):
            if actual is not expected:
                errors.append(f"{key}: expected {expected}, got {actual}")
        elif isinstance(expected, float):
            if not isinstance(actual, (int, float)) or abs(actual - expected) > 1e-8:
                errors.append(f"{key}: expected {expected}, got {actual}")
        elif actual != expected:
            errors.append(f"{key}: expected {expected}, got {actual}")
    if errors:
        return {"config_valid": False, "blocking_reasons": errors}
    return {"config_valid": True, "blocking_reasons": []}


def _load_early_metrics(metrics_path):
    early = {}
    for line in metrics_path.read_text().strip().splitlines():
        obj = json.loads(line)
        it = obj.get("iter")
        if it is None:
            continue
        if 1 <= it <= 10:
            early[it] = obj
        if it > 10:
            break
    return early


def _check_logs_complete(run_dir, dens_steps_count):
    """densification_log.jsonl var ve satır sayısı >= densification_steps_count."""
    log_path = run_dir / "densification_log.jsonl"
    if not log_path.exists():
        return False, "densification_log.jsonl not found"
    n_lines = sum(1 for _ in log_path.open())
    if n_lines < dens_steps_count:
        return False, f"densification_log.jsonl has {n_lines} entries, expected >= {dens_steps_count}"
    return True, ""


def compute_gate(fm, early_metrics, config, run_dir=None):
    iter1_loss = early_metrics[1]["loss_total"] if 1 in early_metrics else None
    iter2_loss = early_metrics[2]["loss_total"] if 2 in early_metrics else None
    losses_2_to_10 = [early_metrics[i]["loss_total"]
                      for i in range(2, 11) if i in early_metrics] if early_metrics else []

    loss_decreased = fm.get("loss_decreased", False)
    iter2_jump_ratio = (iter2_loss / iter1_loss) if (iter1_loss and iter1_loss > 0) else None
    min_recovery_loss = min(losses_2_to_10) if losses_2_to_10 else None
    early_recovery_pass = (
        min_recovery_loss is not None and iter1_loss is not None
        and min_recovery_loss <= CATASTROPHIC_JUMP_RATIO * iter1_loss
    )
    unrecovered_jump_fail = (
        iter2_jump_ratio is not None
        and iter2_loss > CATASTROPHIC_JUMP_RATIO * iter1_loss
        and not early_recovery_pass
    )

    psnr = fm.get("val_psnr", 0.0)
    depth_rmse = fm.get("val_depth_rmse_m_raw", float("inf"))

    enable_densification = fm.get("enable_densification", False)
    n_initial = fm.get("n_gaussians_initial", 0)
    n_final = fm.get("n_gaussians_final", 0)
    growth_ratio = fm.get("gaussian_growth_ratio", 0.0)
    dens_steps = fm.get("densification_steps_count", 0)
    total_cloned = fm.get("total_cloned", 0)
    total_pruned = fm.get("total_pruned", 0)
    max_hit = fm.get("max_gaussians_hit", False)

    growth_occurred = n_final > n_initial
    clone_happened = total_cloned > 0
    growth_controlled = (
        growth_occurred
        and not max_hit
        and growth_ratio <= GROWTH_CONTROLLED_MAX_RATIO
    )

    logs_complete = True
    logs_note = ""
    if run_dir is not None:
        logs_complete, logs_note = _check_logs_complete(run_dir, dens_steps)

    psnr_full = psnr >= PSNR_FULL_SUCCESS
    psnr_partial = psnr >= PSNR_PARTIAL_POSITIVE
    depth_full = depth_rmse <= DEPTH_RMSE_FULL_SUCCESS
    depth_partial = depth_rmse <= DEPTH_RMSE_PARTIAL_POSITIVE

    gate = {
        "config_valid": True,
        "label": "M4-A2-1 Densification Controlled Test",
        "project_line": "SERV-CT",
        "densification_enabled": enable_densification,
        "n_gaussians_initial": n_initial,
        "n_gaussians_final": n_final,
        "gaussian_growth_ratio": growth_ratio,
        "growth_occurred": growth_occurred,
        "growth_controlled": growth_controlled,
        "clone_happened": clone_happened,
        "densification_steps_count": dens_steps,
        "total_cloned": total_cloned,
        "total_pruned": total_pruned,
        "max_gaussians_hit": max_hit,
        "logs_complete": logs_complete,
        "logs_note": logs_note,
        "iter1_loss": iter1_loss,
        "iter2_loss": iter2_loss,
        "iter2_jump_ratio": round(iter2_jump_ratio, 4) if iter2_jump_ratio is not None else None,
        "min_loss_iter2_to_iter10": min_recovery_loss,
        "early_recovery_pass": early_recovery_pass,
        "unrecovered_jump_fail": unrecovered_jump_fail,
        "loss_decreased": loss_decreased,
        "val_psnr": psnr,
        "psnr_full_success_threshold": PSNR_FULL_SUCCESS,
        "psnr_partial_positive_threshold": PSNR_PARTIAL_POSITIVE,
        "psnr_full": psnr_full,
        "psnr_partial": psnr_partial,
        "val_depth_rmse_m_raw": depth_rmse,
        "depth_rmse_full_success_threshold": DEPTH_RMSE_FULL_SUCCESS,
        "depth_rmse_partial_positive_threshold": DEPTH_RMSE_PARTIAL_POSITIVE,
        "depth_full": depth_full,
        "depth_partial": depth_partial,
        "val_ssim": fm.get("val_ssim"),
        "val_lpips": fm.get("val_lpips"),
        "val_depth_mae_m_raw": fm.get("val_depth_mae_m_raw"),
        "val_abs_rel": fm.get("val_abs_rel"),
        "median_aligned_rmse_m": fm.get("val_median_aligned_rmse_m"),
    }

    # --- 3-tier gate decision (expert-answer-17 karar ağacı) ---
    full_success = (
        enable_densification
        and psnr_full
        and depth_full
        and loss_decreased
        and not unrecovered_jump_fail
        and clone_happened
        and growth_controlled
        and logs_complete
    )

    partial_positive = (
        enable_densification
        and psnr_partial
        and depth_partial
        and loss_decreased
        and not unrecovered_jump_fail
        and growth_occurred
        and growth_controlled
        and clone_happened
    )

    if full_success:
        gate["status"] = "FULL_SUCCESS"
        gate["recommendation"] = (
            "M4-A2-1 full success. Densification ran, Gaussians grew in controlled range, "
            f"depth RMSE {depth_rmse:.4f}m <= {DEPTH_RMSE_FULL_SUCCESS}m and "
            f"PSNR {psnr:.2f}dB >= {PSNR_FULL_SUCCESS}dB. "
            "SERV-CT line may be considered complete contingent on expert review."
        )
    elif partial_positive:
        gate["status"] = "SERV_CT_PARTIAL_POSITIVE"
        marginal_notes = []
        if not psnr_full:
            marginal_notes.append(f"psnr {psnr:.2f}dB < {PSNR_FULL_SUCCESS}dB (full threshold)")
        if not depth_full:
            marginal_notes.append(f"depth_rmse {depth_rmse:.4f}m > {DEPTH_RMSE_FULL_SUCCESS}m (full threshold)")
        if not logs_complete:
            marginal_notes.append(f"logs incomplete: {logs_note}")
        gate["marginal_notes"] = marginal_notes
        gate["recommendation"] = (
            "M4-A2-1 partial positive. Densification mechanics executed (Gaussians grew, "
            f"{total_cloned} cloned, {total_pruned} pruned) and metrics meet partial thresholds "
            f"(PSNR >= {PSNR_PARTIAL_POSITIVE}, RMSE <= {DEPTH_RMSE_PARTIAL_POSITIVE}). "
            "Consider expert review for SERV-CT line closure with caveats."
        )
    else:
        gate["status"] = "CONTROLLED_NEGATIVE"
        blocking = []
        if not enable_densification:
            blocking.append("enable_densification=False, expected True")
        if not psnr_partial:
            blocking.append(f"psnr {psnr:.2f}dB < {PSNR_PARTIAL_POSITIVE}dB (partial threshold)")
        if not depth_partial:
            blocking.append(f"depth_rmse {depth_rmse:.4f}m > {DEPTH_RMSE_PARTIAL_POSITIVE}m (partial threshold)")
        if not loss_decreased:
            blocking.append("loss_decreased=False, expected True")
        if unrecovered_jump_fail:
            blocking.append(f"unrecovered_jump_fail: iter2_jump_ratio={iter2_jump_ratio:.2f}")
        if not growth_occurred:
            blocking.append(f"n_gaussians_final ({n_final}) <= n_gaussians_initial ({n_initial})")
        if not growth_controlled and growth_occurred:
            if max_hit:
                blocking.append("max_gaussians_hit=True, growth not controlled")
            elif growth_ratio > GROWTH_CONTROLLED_MAX_RATIO:
                blocking.append(f"growth_ratio {growth_ratio:.2f} > {GROWTH_CONTROLLED_MAX_RATIO} (uncontrolled)")
        if not clone_happened:
            blocking.append("total_cloned = 0, no clones produced")
        if not logs_complete:
            blocking.append(f"logs incomplete: {logs_note}")
        gate["blocking_reasons"] = blocking
        gate["recommendation"] = (
            "M4-A2-1 controlled negative. Metrics do not meet partial thresholds or "
            "densification did not execute as expected. "
            "Do NOT close SERV-CT line positively. Expert intervention required."
        )

    return gate


def main():
    parser = argparse.ArgumentParser(description="M4-A2-1 Densification Gate Evaluation (3-Tier)")
    parser.add_argument("--run_dir", type=str, required=True, help="Path to M4-A2-1 run directory")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for gate decision")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    config_path = run_dir / "config.json"
    metrics_path = run_dir / "metrics.jsonl"
    final_metrics_path = run_dir / "final_metrics.json"

    for p, label in [(config_path, "config.json"), (metrics_path, "metrics.jsonl"),
                     (final_metrics_path, "final_metrics.json")]:
        if not p.exists():
            print(f"FAIL: {label} not found at {p}")
            sys.exit(2)

    with open(config_path) as f:
        cfg = json.load(f)
    with open(final_metrics_path) as f:
        fm = json.load(f)

    merged_cfg = dict(cfg)
    for k in REQUIRED_CONFIG:
        if k not in merged_cfg and k in fm:
            merged_cfg[k] = fm[k]

    guard = validate_m4_a2_1_config(merged_cfg)
    if not guard["config_valid"]:
        gate = {
            "status": "CONTROLLED_NEGATIVE",
            "label": "M4-A2-1 Config Invalid",
            "config_valid": False,
            "blocking_reasons": guard["blocking_reasons"],
        }
        _write_and_report(gate, run_dir, args.output_dir)
        sys.exit(2)

    early = _load_early_metrics(metrics_path)
    if 1 not in early or 2 not in early:
        print(f"FAIL: iter1/iter2 metrics not found in {metrics_path}")
        sys.exit(2)

    gate = compute_gate(fm, early, merged_cfg, run_dir=run_dir)
    exit_code = 0 if gate["status"] == "FULL_SUCCESS" else 1
    _write_and_report(gate, run_dir, args.output_dir)
    sys.exit(exit_code)


def _write_and_report(gate, run_dir, output_dir_arg):
    output_dir = Path(output_dir_arg) if output_dir_arg else run_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    gate_path = output_dir / "m4_a2_1_gate.json"
    gate_path.write_text(json.dumps(gate, indent=2))

    print("=== M4-A2-1 Densification Gate (3-Tier) ===")
    print(f"  Label: {gate.get('label', 'M4-A2-1')}")
    print(f"  Status: {gate['status']}")
    print(f"  enable_densification: {gate.get('densification_enabled', 'N/A')}")
    print(f"  growth_occurred: {gate.get('growth_occurred', 'N/A')} "
          f"({gate.get('n_gaussians_initial', 0)} -> {gate.get('n_gaussians_final', 0)}, "
          f"ratio {gate.get('gaussian_growth_ratio', 0):.3f})")
    print(f"  growth_controlled: {gate.get('growth_controlled', 'N/A')}")
    print(f"  clone_happened: {gate.get('clone_happened', 'N/A')} "
          f"(cloned={gate.get('total_cloned', 0)}, pruned={gate.get('total_pruned', 0)}, "
          f"steps={gate.get('densification_steps_count', 0)})")
    print(f"  max_gaussians_hit: {gate.get('max_gaussians_hit', False)}")
    print(f"  logs_complete: {gate.get('logs_complete', 'N/A')}")
    print(f"  loss_decreased: {gate.get('loss_decreased', 'N/A')}")
    jr = gate.get("iter2_jump_ratio")
    if jr is not None:
        print(f"  iter2_jump_ratio: {jr:.4f}")
        print(f"  unrecovered_jump_fail: {gate.get('unrecovered_jump_fail', 'N/A')}")
    psnr = gate.get("val_psnr", "N/A")
    if psnr != "N/A":
        print(f"  val_PSNR: {psnr:.2f} dB (full >= {PSNR_FULL_SUCCESS}, partial >= {PSNR_PARTIAL_POSITIVE})")
    rmse = gate.get("val_depth_rmse_m_raw", "N/A")
    if rmse != "N/A":
        print(f"  val_depth_RMSE: {rmse:.4f} m (full <= {DEPTH_RMSE_FULL_SUCCESS}, "
              f"partial <= {DEPTH_RMSE_PARTIAL_POSITIVE})")

    if gate.get("blocking_reasons"):
        for r in gate["blocking_reasons"]:
            print(f"    X {r}")
    if gate.get("marginal_notes"):
        for r in gate["marginal_notes"]:
            print(f"    ~ {r}")
    print(f"\n  Recommendation: {gate.get('recommendation', 'N/A')}")
    print(f"  Gate artifact: {gate_path.resolve()}")


if __name__ == "__main__":
    main()
