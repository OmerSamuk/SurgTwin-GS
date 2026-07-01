import argparse
import json
import sys
from pathlib import Path

PSNR_THRESHOLD = 19.5
DEPTH_RMSE_THRESHOLD = 0.038
CATASTROPHIC_JUMP_RATIO = 1.5
FLOAT_TOLERANCE = 1e-8

REQUIRED_CONFIG = {
    "init_num_points": 20000,
    "enable_densification": False,
    "iterations": 200,
    "warmup_iters": 200,
    "lr_opacities": 1e-2,
    "max_grad_norm": 1.5,
    "variant": "h1",
    "lambda_depth": 0.2,
    "lambda_reg": 0.0,
}


def validate_m4_a2_0_config(cfg):
    errors = []
    for key, expected in REQUIRED_CONFIG.items():
        actual = cfg.get(key)
        if isinstance(expected, bool):
            if actual is not expected:
                errors.append(f"{key}: expected {expected}, got {actual}")
        elif isinstance(expected, float):
            if not isinstance(actual, (int, float)) or abs(actual - expected) > FLOAT_TOLERANCE:
                errors.append(f"{key}: expected {expected}, got {actual}")
        elif actual != expected:
            errors.append(f"{key}: expected {expected}, got {actual}")
    if errors:
        return {"config_valid": False, "blocking_reasons": errors}
    return {"config_valid": True, "blocking_reasons": []}


def classify_clip_ratio(ratio):
    if ratio is None:
        return "unknown"
    if ratio <= 0.5:
        return "healthy"
    elif ratio <= 0.8:
        return "warning"
    return "clip_bound"


def _load_early_metrics(metrics_path):
    early = {}
    for line in metrics_path.read_text().strip().splitlines():
        obj = json.loads(line)
        it = obj.get("iter")
        if it is not None and 1 <= it <= 10:
            early[it] = obj
        if it is not None and it > 10:
            break
    return early


def compute_gate(fm, early_metrics, config, run_dir, allow_ablation=False):
    iter1_loss = early_metrics[1]["loss_total"]
    iter2_loss = early_metrics[2]["loss_total"]
    # Recovery check uses iter2-10 (excludes iter1 — otherwise min trivially passes)
    losses_2_to_10 = [early_metrics[i]["loss_total"] for i in range(2, 11) if i in early_metrics]

    loss_decreased = fm.get("loss_decreased", False)

    iter2_jump_ratio = iter2_loss / iter1_loss if iter1_loss > 0 else None
    min_recovery_loss = min(losses_2_to_10) if losses_2_to_10 else None
    early_recovery_pass = (
        min_recovery_loss is not None
        and min_recovery_loss <= CATASTROPHIC_JUMP_RATIO * iter1_loss
    )
    unrecovered_jump_fail = (
        iter2_jump_ratio is not None
        and iter2_loss > CATASTROPHIC_JUMP_RATIO * iter1_loss
        and not early_recovery_pass
    )
    legacy_catastrophic_1p5 = iter2_loss <= CATASTROPHIC_JUMP_RATIO * iter1_loss
    legacy_catastrophic_2p0 = iter2_loss <= 2.0 * iter1_loss

    psnr = fm.get("val_psnr", 0.0)
    psnr_pass = psnr >= PSNR_THRESHOLD
    psnr_delta = psnr - PSNR_THRESHOLD

    depth_rmse = fm.get("val_depth_rmse_m_raw", float("inf"))
    depth_rmse_pass = depth_rmse <= DEPTH_RMSE_THRESHOLD
    depth_rmse_delta = DEPTH_RMSE_THRESHOLD - depth_rmse

    clip_active_ratio = fm.get("clip_active_ratio")
    clip_active_ratio_reported = clip_active_ratio is not None
    clip_health = classify_clip_ratio(clip_active_ratio)

    warmup_iters = fm.get("warmup_iters", 0)
    densification_off = fm.get("enable_densification") is False

    median_aligned_rmse = fm.get("val_median_aligned_rmse_m")
    val_abs_rel = fm.get("val_abs_rel")
    val_ssim = fm.get("val_ssim")
    val_lpips = fm.get("val_lpips")
    val_depth_mae = fm.get("val_depth_mae_m_raw")

    gate = {
        "config_valid": True,
        "config_guard": {k: config.get(k) for k in REQUIRED_CONFIG},
        "allow_ablation": allow_ablation,
        "gate_eligible_run_mode": fm.get("run_mode", "unknown"),
        "m2a_gate_confirmed": fm.get("m2a_gate", "unknown"),
        "iter1_loss": iter1_loss,
        "iter2_loss": iter2_loss,
        "iter2_jump_ratio": round(iter2_jump_ratio, 4) if iter2_jump_ratio is not None else None,
        "min_loss_iter2_to_iter10": min_recovery_loss,
        "early_recovery_pass": early_recovery_pass,
        "unrecovered_jump_fail": unrecovered_jump_fail,
        "catastrophic_jump_1p5": legacy_catastrophic_1p5,
        "catastrophic_jump_2p0": legacy_catastrophic_2p0,
        "_deprecated": "catastrophic_jump_1p5/2p0 are legacy; use unrecovered_jump_fail for gate",
        "loss_decreased": loss_decreased,
        "val_psnr": psnr,
        "psnr_threshold": PSNR_THRESHOLD,
        "psnr_pass": psnr_pass,
        "psnr_delta": round(psnr_delta, 3),
        "val_depth_rmse_m_raw": depth_rmse,
        "depth_rmse_threshold": DEPTH_RMSE_THRESHOLD,
        "depth_rmse_pass": depth_rmse_pass,
        "depth_rmse_delta": round(depth_rmse_delta, 4),
        "val_ssim": val_ssim,
        "val_lpips": val_lpips,
        "val_depth_mae_m_raw": val_depth_mae,
        "val_abs_rel": val_abs_rel,
        "median_aligned_rmse_m": median_aligned_rmse,
        "median_aligned_rmse_note": "diagnostic shape-quality indicator; primary gate uses raw depth RMSE",
        "clip_active_ratio": clip_active_ratio,
        "clip_active_ratio_reported": clip_active_ratio_reported,
        "clip_health": clip_health,
        "warmup_iters": warmup_iters,
        "densification_off": densification_off,
    }

    primary_failures = []
    marginal_failures = []

    if not loss_decreased:
        primary_failures.append("loss_decreased=False, expected True")
    if unrecovered_jump_fail:
        primary_failures.append(
            f"unrecovered_jump_fail: iter2_jump_ratio={iter2_jump_ratio:.2f}, "
            f"min_recovery_loss={min_recovery_loss:.5f}, no recovery in iter2-10"
        )
    if not psnr_pass:
        marginal_failures.append(f"val_psnr {psnr:.2f} < {PSNR_THRESHOLD} dB")
    if not depth_rmse_pass:
        marginal_failures.append(f"val_depth_rmse_m_raw {depth_rmse:.4f} > {DEPTH_RMSE_THRESHOLD} m")
    if not clip_active_ratio_reported:
        gate["clip_warning"] = "clip_active_ratio not reported in final_metrics"

    if len(primary_failures) > 0:
        gate["status"] = "CLEAR_FAIL"
        gate["blocking_reasons"] = primary_failures
        gate["recommendation"] = (
            "M4-A2-0 clear fail. Do NOT proceed to M4-A2-1. "
            "Consider SERV-CT line closure/pivot discussion."
        )
    elif len(marginal_failures) == 0:
        gate["status"] = "PASS"
        gate["recommendation"] = "M4-A2-0 passed. Proceed to M4-A2-1 densification run."
    else:
        psnr_marginal = not psnr_pass and abs(psnr_delta) <= 0.2
        depth_marginal = not depth_rmse_pass and abs(depth_rmse_delta) <= 0.001
        if loss_decreased and not unrecovered_jump_fail and (psnr_marginal or depth_marginal):
            gate["status"] = "MARGINAL_FAIL"
            gate["blocking_reasons"] = marginal_failures
            gate["marginal"] = {
                "psnr_marginal": psnr_marginal,
                "depth_rmse_marginal": depth_marginal,
                "ablation_config": {"max_grad_norm": 2.0},
                "ablation_output_dir": str(run_dir) + "_abl_grad2",
            }
            gate["recommendation"] = (
                "M4-A2-0 marginal fail. Run one ablation with max_grad_norm=2.0 "
                f"(output: {gate['marginal']['ablation_output_dir']}). "
                "Ablation will NOT be a canonical gate (expert review required)."
            )
        else:
            gate["status"] = "CLEAR_FAIL"
            gate["blocking_reasons"] = marginal_failures
            gate["recommendation"] = (
                "M4-A2-0 clear fail. Do NOT proceed to M4-A2-1. "
                "Consider SERV-CT line closure/pivot discussion."
            )

    return gate


def main():
    parser = argparse.ArgumentParser(description="M4-A2-0 Smoke Test Gate Evaluation")
    parser.add_argument("--run_dir", type=str, required=True, help="Path to M4-A2-0 run directory")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for gate decision")
    parser.add_argument("--allow_ablation", action="store_true",
                        help="Skip config guard for ablation runs (requires expert review)")
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

    # Merge config.json + final_metrics.json for config guard
    # (BaselineTrainer.__init__ overwrites config.json with 13 BaselineConfig fields;
    #  final_metrics.json contains the full UncertaintyConfig fields set by
    #  UncertaintyTrainer.fit.)
    merged_cfg = dict(cfg)
    for k in REQUIRED_CONFIG:
        if k not in merged_cfg and k in fm:
            merged_cfg[k] = fm[k]
    # Also add run_mode for gate_eligible check
    merged_cfg.setdefault("run_mode", fm.get("run_mode"))

    # --- Config Guard ---
    if not args.allow_ablation:
        guard = validate_m4_a2_0_config(merged_cfg)
        if not guard["config_valid"]:
            gate = {
                "status": "INVALID_FOR_GATE",
                "label": "M4-A2-0 Config Invalid — Not Eligible for Gate",
                "config_valid": False,
                "blocking_reasons": guard["blocking_reasons"],
            }
            _write_and_report(gate, run_dir, args.output_dir)
            print(f"  Exit code 2: INVALID_FOR_GATE")
            sys.exit(2)

    # --- Early metrics ---
    early = _load_early_metrics(metrics_path)
    if 1 not in early or 2 not in early:
        print(f"FAIL: iter1/iter2 metrics not found in {metrics_path}")
        sys.exit(2)

    gate = compute_gate(fm, early, cfg, run_dir, allow_ablation=args.allow_ablation)
    exit_code = 0 if gate["status"] == "PASS" else 1
    _write_and_report(gate, run_dir, args.output_dir)
    sys.exit(exit_code)


def _write_and_report(gate, run_dir, output_dir_arg):
    output_dir = Path(output_dir_arg) if output_dir_arg else run_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    gate_path = output_dir / "m4_a2_0_gate.json"
    gate_path.write_text(json.dumps(gate, indent=2))

    print("=== M4-A2-0 Smoke Gate ===")
    print(f"  Status: {gate['status']}")
    ld = gate.get("loss_decreased", "N/A")
    if isinstance(ld, bool):
        print(f"  loss_decreased: {ld}")
    jr = gate.get("iter2_jump_ratio")
    if jr is not None:
        print(f"  iter2_jump_ratio: {jr:.3f} (threshold {CATASTROPHIC_JUMP_RATIO})")
        mrl = gate.get("min_loss_iter2_to_iter10")
        if mrl is not None:
            print(f"  min_recovery_loss: {mrl:.5f}")
        print(f"  early_recovery_pass: {gate.get('early_recovery_pass', 'N/A')}")
        print(f"  unrecovered_jump_fail: {gate.get('unrecovered_jump_fail', 'N/A')}")
    psnr = gate.get("val_psnr")
    if psnr is not None:
        print(f"  val_PSNR: {psnr:.2f} dB (threshold {PSNR_THRESHOLD} dB, "
              f"delta {gate.get('psnr_delta', 0):+.2f})")
    rmse = gate.get('val_depth_rmse_m_raw')
    if rmse is not None:
        print(f"  val_depth_RMSE: {rmse:.4f} m (threshold {DEPTH_RMSE_THRESHOLD} m, "
              f"delta {gate.get('depth_rmse_delta', 0):+.4f})")
    capr = gate.get('clip_active_ratio', 'N/A')
    if capr != 'N/A' or 'clip_health' in gate:
        print(f"  clip_active_ratio: {capr} ({gate.get('clip_health', 'unknown')})")
    wi = gate.get("warmup_iters")
    if wi is not None:
        print(f"  warmup_iters: {wi}")
    do = gate.get("densification_off")
    if do is not None:
        print(f"  densification_off: {do}")
    mar = gate.get("median_aligned_rmse_m")
    if mar is not None:
        print(f"  median_aligned_RMSE (diagnostic): {mar:.4f} m")
    if gate.get("blocking_reasons"):
        for r in gate["blocking_reasons"]:
            print(f"    X {r}")
    print(f"\n  Recommendation: {gate.get('recommendation', 'N/A')}")
    print(f"  Gate artifact: {gate_path.resolve()}")
    if "marginal" in gate:
        print(f"  Ablation target: {gate['marginal']['ablation_output_dir']}")


if __name__ == "__main__":
    main()
