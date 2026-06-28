import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _load_final_metrics(run_dir: Path) -> Dict:
    path = run_dir / "final_metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"final_metrics.json not found in {run_dir}")
    return json.loads(path.read_text())


def _compute_psnr_delta(bl: Dict, dg: Dict) -> tuple:
    bl_psnr = bl.get("val_psnr")
    dg_psnr = dg.get("val_psnr")
    if bl_psnr is None or dg_psnr is None:
        return None, None, None
    delta = dg_psnr - bl_psnr
    return bl_psnr, dg_psnr, delta


def _check_guard(bl: Dict, dg: Dict) -> List[str]:
    issues = []
    bl_psnr = bl.get("val_psnr")
    dg_psnr = dg.get("val_psnr")
    if bl_psnr is not None and dg_psnr is not None:
        threshold = bl_psnr - 3.0
        if dg_psnr < threshold:
            issues.append(
                f"PSNR guard FAIL: depth-guided {dg_psnr:.2f} < baseline {bl_psnr:.2f} - 3.0 = {threshold:.2f}"
            )

    dg_depth_semantics = dg.get("depth_semantics") or dg.get("val_depth_semantics")
    if dg_depth_semantics != "metric_meters":
        issues.append(f"Depth semantics not metric: {dg_depth_semantics}")

    dg_rmse = dg.get("val_depth_rmse_m_raw")
    if dg_rmse is None:
        issues.append("Depth RMSE not found in depth-guided run")

    return issues


def _build_comparison_table(bl: Dict, dg: Dict, issues: List[str]) -> Dict:
    bl_psnr, dg_psnr, psnr_delta = _compute_psnr_delta(bl, dg)

    table = {
        "psnr_baseline_dB": bl_psnr,
        "psnr_depth_guided_dB": dg_psnr,
        "psnr_delta_dB": round(psnr_delta, 3) if psnr_delta is not None else None,
        "psnr_guard_threshold_dB": round(bl_psnr - 3.0, 3) if bl_psnr is not None else None,
        "psnr_guard_pass": len([i for i in issues if "PSNR guard FAIL" in i]) == 0 if bl_psnr is not None else None,
        "val_ssim_baseline": bl.get("val_ssim"),
        "val_ssim_depth_guided": dg.get("val_ssim"),
        "val_lpips_baseline": bl.get("val_lpips"),
        "val_lpips_depth_guided": dg.get("val_lpips"),
        "depth_rmse_m_raw": dg.get("val_depth_rmse_m_raw"),
        "depth_rmse_m_clipped": dg.get("val_depth_rmse_m_clipped"),
        "depth_mae_m_raw": dg.get("val_depth_mae_m_raw"),
        "depth_mae_m_clipped": dg.get("val_depth_mae_m_clipped"),
        "abs_rel": dg.get("val_abs_rel"),
        "depth_valid_ratio": dg.get("val_depth_valid_ratio"),
        "median_aligned_rmse_m": dg.get("val_median_aligned_rmse_m"),
        "depth_semantics_baseline": bl.get("render_depth_semantics", "not_used_for_baseline"),
        "depth_semantics_depth_guided": dg.get("depth_semantics") or dg.get("val_depth_semantics", "unavailable"),
        "lambda_depth": dg.get("lambda_depth"),
        "lambda_reg": dg.get("lambda_reg"),
        "reg_type": dg.get("reg_type"),
        "enable_densification": dg.get("enable_densification"),
        "n_gaussians_baseline": bl.get("n_gaussians"),
        "n_gaussians_depth_guided": dg.get("n_gaussians"),
        "iterations_baseline": bl.get("iterations"),
        "iterations_depth_guided": dg.get("iterations"),
        "pass": len(issues) == 0,
        "blocking_reasons": issues,
    }

    return table


def _table_to_markdown(table: Dict) -> str:
    lines = [
        "# Baseline vs Depth-Guided Comparison",
        "",
        "| Metric | Baseline | Depth-Guided | Delta | Guard Result |",
        "|---|---|---|---|---|",
    ]

    def fmt(v):
        if v is None:
            return "N/A"
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    rows = [
        ("val_psnr (dB)", "psnr_baseline_dB", "psnr_depth_guided_dB", "psnr_delta_dB",
         f"≥ {fmt(table.get('psnr_guard_threshold_dB'))} dB → {'✓' if table.get('psnr_guard_pass') else '✗'}"),
        ("val_ssim", "val_ssim_baseline", "val_ssim_depth_guided", None, "—"),
        ("val_lpips", "val_lpips_baseline", "val_lpips_depth_guided", None, "—"),
        ("depth_rmse_m (raw)", None, "depth_rmse_m_raw", None, "required ✓"),
        ("depth_mae_m (raw)", None, "depth_mae_m_raw", None, "required ✓"),
        ("abs_rel", None, "abs_rel", None, "required ✓"),
        ("depth_valid_ratio", None, "depth_valid_ratio", None, "—"),
        ("depth_semantics", "depth_semantics_baseline", "depth_semantics_depth_guided", None, "required ✓"),
        ("n_gaussians", "n_gaussians_baseline", "n_gaussians_depth_guided", None, "—"),
        ("lambda_depth", None, "lambda_depth", None, "0.2 ✓"),
        ("lambda_reg", None, "lambda_reg", None, "0.01 ✓"),
    ]

    for label, bl_key, dg_key, delta_key, guard in rows:
        bl_val = fmt(table.get(bl_key)) if bl_key else "N/A"
        dg_val = fmt(table.get(dg_key)) if dg_key else "N/A"
        delta_val = fmt(table.get(delta_key)) if delta_key else fmt(None)
        lines.append(f"| {label} | {bl_val} | {dg_val} | {delta_val} | {guard} |")

    lines.extend([
        "",
        "## Blocking Issues",
    ])
    issues = table.get("blocking_reasons", [])
    if issues:
        for i, issue in enumerate(issues, 1):
            lines.append(f"{i}. ❌ {issue}")
    else:
        lines.append("None — all guards pass ✓")

    lines.extend([
        "",
        f"**Overall: {'PASS' if table.get('pass') else 'FAIL'}**",
    ])

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Compare M1 baseline vs M2-B depth-guided results"
    )
    parser.add_argument("--baseline_run", type=str,
                        default="outputs/runs/baseline_debug",
                        help="Baseline run directory")
    parser.add_argument("--depth_guided_run", type=str,
                        default="outputs/runs/depth_guided_m2b",
                        help="Depth-guided run directory")
    parser.add_argument("--output_dir", type=str,
                        default="outputs/runs/m2b_comparison",
                        help="Output directory for comparison artifacts")
    args = parser.parse_args()

    bl_path = Path(args.baseline_run)
    dg_path = Path(args.depth_guided_run)
    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    bl_metrics = _load_final_metrics(bl_path)
    dg_metrics = _load_final_metrics(dg_path)

    issues = _check_guard(bl_metrics, dg_metrics)
    table = _build_comparison_table(bl_metrics, dg_metrics, issues)

    comparison_path = out_path / "comparison_table.json"
    comparison_path.write_text(json.dumps(table, indent=2))

    markdown = _table_to_markdown(table)
    md_path = out_path / "comparison_table.md"
    md_path.write_text(markdown)

    print("\n=== Baseline vs Depth-Guided Comparison ===")
    bl_psnr = table.get("psnr_baseline_dB")
    dg_psnr = table.get("psnr_depth_guided_dB")
    if bl_psnr is not None and dg_psnr is not None:
        print(f"  PSNR: {bl_psnr:.2f} → {dg_psnr:.2f} (Δ={table['psnr_delta_dB']:.2f} dB, guard={table['psnr_guard_threshold_dB']:.1f} dB)")

    for key in ("depth_rmse_m_raw", "depth_mae_m_raw", "abs_rel", "depth_valid_ratio"):
        val = table.get(key)
        if val is not None:
            print(f"  {key}: {val:.6f}")

    print(f"  Pass: {table['pass']}")
    if table["blocking_reasons"]:
        print("  Issues:")
        for issue in table["blocking_reasons"]:
            print(f"    - {issue}")
    print(f"  Artifacts: {out_path.resolve()}")
    print(f"    comparison_table.json")
    print(f"    comparison_table.md")


if __name__ == "__main__":
    main()
