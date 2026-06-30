import argparse
import json
import sys
from pathlib import Path

PSNR_THRESHOLD = 19.5
DEPTH_RMSE_THRESHOLD = 0.038
CATASTROPHIC_JUMP_1P5 = 1.5
CATASTROPHIC_JUMP_2P0 = 2.0


def main():
    parser = argparse.ArgumentParser(description="M4-A2-0 Smoke Test Gate Evaluation")
    parser.add_argument("--run_dir", type=str, required=True, help="Path to M4-A2-0 run directory")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for gate decision")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    metrics_path = run_dir / "metrics.jsonl"
    final_metrics_path = run_dir / "final_metrics.json"

    if not metrics_path.exists():
        print(f"FAIL: metrics.jsonl not found at {metrics_path}")
        sys.exit(1)
    if not final_metrics_path.exists():
        print(f"FAIL: final_metrics.json not found at {final_metrics_path}")
        sys.exit(1)

    with open(final_metrics_path) as f:
        fm = json.load(f)

    metrics_lines = metrics_path.read_text().strip().splitlines()
    iter1_metrics = None
    iter2_metrics = None
    for line in metrics_lines:
        obj = json.loads(line)
        if obj.get("iter") == 1:
            iter1_metrics = obj
        if obj.get("iter") == 2:
            iter2_metrics = obj

    gate = {}

    # 1. loss_decreased
    loss_decreased = fm.get("loss_decreased", False)
    gate["loss_decreased"] = loss_decreased

    # 2. catastrophic jump
    catastrophic_1p5 = True
    catastrophic_2p0 = True
    if iter1_metrics is not None and iter2_metrics is not None:
        l1 = iter1_metrics.get("loss_total", 0.0)
        l2 = iter2_metrics.get("loss_total", 0.0)
        catastrophic_1p5 = l2 <= CATASTROPHIC_JUMP_1P5 * l1
        catastrophic_2p0 = l2 <= CATASTROPHIC_JUMP_2P0 * l1
    gate["iter1_loss_total"] = iter1_metrics.get("loss_total") if iter1_metrics else None
    gate["iter2_loss_total"] = iter2_metrics.get("loss_total") if iter2_metrics else None
    gate["catastrophic_jump_1p5"] = catastrophic_1p5
    gate["catastrophic_jump_2p0"] = catastrophic_2p0

    # 3. PSNR
    psnr = fm.get("val_psnr", 0.0)
    psnr_pass = psnr >= PSNR_THRESHOLD
    gate["val_psnr"] = psnr
    gate["psnr_pass"] = psnr_pass
    gate["psnr_threshold"] = PSNR_THRESHOLD

    # 4. depth RMSE
    depth_rmse = fm.get("val_depth_rmse_m_raw", float("inf"))
    depth_rmse_pass = depth_rmse <= DEPTH_RMSE_THRESHOLD
    gate["val_depth_rmse_m_raw"] = depth_rmse
    gate["depth_rmse_pass"] = depth_rmse_pass
    gate["depth_rmse_threshold"] = DEPTH_RMSE_THRESHOLD

    # 5. clip_active_ratio reported
    clip_active_ratio = fm.get("clip_active_ratio")
    clip_active_ratio_reported = clip_active_ratio is not None
    gate["clip_active_ratio"] = clip_active_ratio
    gate["clip_active_ratio_reported"] = clip_active_ratio_reported

    # 6. warmup_iters
    warmup_iters = fm.get("warmup_iters", 0)
    gate["warmup_iters"] = warmup_iters

    # gate status
    blocking = []
    if not loss_decreased:
        blocking.append("loss_decreased=False, expected True")
    if not catastrophic_1p5:
        blocking.append(f"iter2 catastrophic jump: {gate['iter2_loss_total']:.4f} > 1.5 * {gate['iter1_loss_total']:.4f}")
    if not psnr_pass:
        blocking.append(f"val_psnr {psnr:.2f} < {PSNR_THRESHOLD} dB")
    if not depth_rmse_pass:
        blocking.append(f"val_depth_rmse_m_raw {depth_rmse:.4f} > {DEPTH_RMSE_THRESHOLD} m")
    if not clip_active_ratio_reported:
        blocking.append("clip_active_ratio not reported in final_metrics")

    if len(blocking) == 0:
        gate["status"] = "PASS"
        gate["label"] = "M4-A2-0 Smoke PASS"
        gate["recommendation"] = "M4-A2-0 passed. Proceed to M4-A2-1 densification run."
    else:
        gate["status"] = "FAIL"
        gate["label"] = "M4-A2-0 Smoke FAIL"
        gate["blocking_reasons"] = blocking
        gate["recommendation"] = (
            "M4-A2-0 did not pass smoke gate. "
            "Do NOT proceed to M4-A2-1. "
            "Review warmup/optimizer configuration or pivot discussion."
        )

    output_dir = Path(args.output_dir) if args.output_dir else run_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    gate_path = output_dir / "m4_a2_0_gate.json"
    gate_path.write_text(json.dumps(gate, indent=2))

    print("=== M4-A2-0 Smoke Gate ===")
    print(f"  Status: {gate['status']}")
    print(f"  Label: {gate['label']}")
    print(f"  loss_decreased: {loss_decreased}")
    print(f"  catastrophic_jump_1p5: {catastrophic_1p5} (iter1={gate.get('iter1_loss_total', 'N/A'):.4f}, iter2={gate.get('iter2_loss_total', 'N/A'):.4f})")
    print(f"  val_PSNR: {psnr:.2f} dB (threshold {PSNR_THRESHOLD} dB)")
    print(f"  val_depth_RMSE: {depth_rmse:.4f} m (threshold {DEPTH_RMSE_THRESHOLD} m)")
    print(f"  clip_active_ratio: {clip_active_ratio}")
    print(f"  warmup_iters: {warmup_iters}")
    if blocking:
        for reason in blocking:
            print(f"    X {reason}")
    print(f"\n  Recommendation: {gate.get('recommendation', 'N/A')}")
    print(f"  Gate artifact: {gate_path.resolve()}")


if __name__ == "__main__":
    main()
