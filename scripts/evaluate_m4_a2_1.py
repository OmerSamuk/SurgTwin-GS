import argparse
import json
import sys
from pathlib import Path

PSNR_THRESHOLD = 19.5
DEPTH_RMSE_THRESHOLD = 0.030
CATASTROPHIC_JUMP_RATIO = 1.5

REQUIRED_CONFIG = {
    "enable_densification": True,
    "iterations": 500,
    "warmup_iters": 200,
    "lr_opacities": 1e-2,
    "max_grad_norm": 1.5,
    "variant": "h1",
    "lambda_depth": 0.2,
    "lambda_reg": 0.0,
}


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


def compute_gate(fm, early_metrics, config):
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
    psnr_pass = psnr >= PSNR_THRESHOLD
    depth_rmse = fm.get("val_depth_rmse_m_raw", float("inf"))
    depth_rmse_pass = depth_rmse <= DEPTH_RMSE_THRESHOLD

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

    gate = {
        "config_valid": True,
        "label": "M4-A2-1 Gateless Controlled Test — No Entry to Upper Gateline",
        "project_line": "SERV-CT",
        "densification_enabled": enable_densification,
        "n_gaussians_initial": n_initial,
        "n_gaussians_final": n_final,
        "gaussian_growth_ratio": growth_ratio,
        "growth_occurred": growth_occurred,
        "clone_happened": clone_happened,
        "densification_steps_count": dens_steps,
        "total_cloned": total_cloned,
        "total_pruned": total_pruned,
        "max_gaussians_hit": max_hit,
        "iter1_loss": iter1_loss,
        "iter2_loss": iter2_loss,
        "iter2_jump_ratio": round(iter2_jump_ratio, 4) if iter2_jump_ratio is not None else None,
        "min_loss_iter2_to_iter10": min_recovery_loss,
        "early_recovery_pass": early_recovery_pass,
        "unrecovered_jump_fail": unrecovered_jump_fail,
        "loss_decreased": loss_decreased,
        "val_psnr": psnr,
        "psnr_threshold": PSNR_THRESHOLD,
        "psnr_pass": psnr_pass,
        "val_depth_rmse_m_raw": depth_rmse,
        "depth_rmse_threshold": DEPTH_RMSE_THRESHOLD,
        "depth_rmse_pass": depth_rmse_pass,
        "val_ssim": fm.get("val_ssim"),
        "val_lpips": fm.get("val_lpips"),
        "val_depth_mae_m_raw": fm.get("val_depth_mae_m_raw"),
        "val_abs_rel": fm.get("val_abs_rel"),
        "median_aligned_rmse_m": fm.get("val_median_aligned_rmse_m"),
    }

    # --- 3-tier gate decision ---
    blocking = []

    if not enable_densification:
        blocking.append("enable_densification=False, expected True")
    if not growth_occurred:
        blocking.append(f"n_gaussians_final ({n_final}) <= n_gaussians_initial ({n_initial})")
    if not clone_happened:
        blocking.append("total_cloned = 0, no clones produced")
    if not loss_decreased:
        blocking.append("loss_decreased=False, expected True")
    if unrecovered_jump_fail:
        blocking.append(
            f"unrecovered_jump_fail: iter2_jump_ratio={iter2_jump_ratio:.2f}"
        )

    if len(blocking) > 0:
        gate["status"] = "CONTROLLED_NEGATIVE"
        gate["blocking_reasons"] = blocking
        gate["recommendation"] = (
            "M4-A2-1 controlled negative. Densification did not execute as expected. "
            "Do NOT close SERV-CT line positively. Review densification config, render output, "
            "or data pipeline. Expert intervention required."
        )
        return gate

    depth_ok = depth_rmse_pass
    psnr_ok = psnr_pass

    if depth_ok and psnr_ok:
        gate["status"] = "FULL_SUCCESS"
        gate["recommendation"] = (
            "M4-A2-1 full success. Densification ran, Gaussians grew, "
            f"depth RMSE {depth_rmse:.4f}m and PSNR {psnr:.2f}dB meet thresholds. "
            "SERV-CT line may be considered complete contingent on expert review."
        )
    else:
        partial_notes = []
        if not depth_ok:
            partial_notes.append(f"depth_rmse {depth_rmse:.4f}m > {DEPTH_RMSE_THRESHOLD}m")
        if not psnr_ok:
            partial_notes.append(f"psnr {psnr:.2f}dB < {PSNR_THRESHOLD}dB")
        gate["status"] = "SERV_CT_PARTIAL_POSITIVE"
        gate["marginal_notes"] = partial_notes
        gate["recommendation"] = (
            "M4-A2-1 partial positive. Densification mechanics executed (Gaussians grew, "
            f"{total_cloned} cloned, {total_pruned} pruned) but metric thresholds not fully met. "
            "Consider expert review for SERV-CT line closure with caveats."
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

    gate = compute_gate(fm, early, merged_cfg)
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
    print(f"  clone_happened: {gate.get('clone_happened', 'N/A')} "
          f"(cloned={gate.get('total_cloned', 0)}, pruned={gate.get('total_pruned', 0)}, "
          f"steps={gate.get('densification_steps_count', 0)})")
    print(f"  max_gaussians_hit: {gate.get('max_gaussians_hit', False)}")
    print(f"  loss_decreased: {gate.get('loss_decreased', 'N/A')}")
    jr = gate.get("iter2_jump_ratio")
    if jr is not None:
        print(f"  iter2_jump_ratio: {jr:.4f}")
        print(f"  unrecovered_jump_fail: {gate.get('unrecovered_jump_fail', 'N/A')}")
    psnr = gate.get("val_psnr", "N/A")
    if psnr != "N/A":
        print(f"  val_PSNR: {psnr:.2f} dB (threshold {PSNR_THRESHOLD} dB, "
              f"pass={gate.get('psnr_pass', False)})")
    rmse = gate.get("val_depth_rmse_m_raw", "N/A")
    if rmse != "N/A":
        print(f"  val_depth_RMSE: {rmse:.4f} m (threshold {DEPTH_RMSE_THRESHOLD} m, "
              f"pass={gate.get('depth_rmse_pass', False)})")

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
