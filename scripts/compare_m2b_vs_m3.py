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


def _fmt(val, precision: int = 4) -> str:
    if val is None:
        return "N/A"
    if isinstance(val, float):
        return f"{val:.{precision}f}"
    if isinstance(val, bool):
        return str(val)
    return str(val)


def _extract_row(run_id: str, metrics: Dict) -> Dict:
    return {
        "run_id": run_id,
        "val_psnr": metrics.get("val_psnr"),
        "val_ssim": metrics.get("val_ssim"),
        "val_lpips": metrics.get("val_lpips"),
        "val_depth_rmse_m_raw": metrics.get("val_depth_rmse_m_raw"),
        "val_depth_mae_m_raw": metrics.get("val_depth_mae_m_raw"),
        "val_abs_rel": metrics.get("val_abs_rel"),
        "val_depth_valid_ratio": metrics.get("val_depth_valid_ratio"),
        "val_w_photo_mean": metrics.get("val_w_photo_mean") or metrics.get("w_photo_mean"),
        "val_w_photo_min": metrics.get("val_w_photo_min") or metrics.get("w_photo_min"),
        "val_w_photo_max": metrics.get("val_w_photo_max") or metrics.get("w_photo_max"),
        "val_w_photo_p10": metrics.get("val_w_photo_p10") or metrics.get("w_photo_p10"),
        "val_w_photo_p50": metrics.get("val_w_photo_p50") or metrics.get("w_photo_p50"),
        "val_w_photo_p90": metrics.get("val_w_photo_p90") or metrics.get("w_photo_p90"),
        "val_fraction_w_photo_at_min": metrics.get("val_fraction_w_photo_at_min") or metrics.get("fraction_w_photo_at_min"),
        "val_fraction_w_photo_at_one": metrics.get("val_fraction_w_photo_at_one") or metrics.get("fraction_w_photo_at_one"),
        "val_w_photo_p90_minus_p10": metrics.get("val_w_photo_p90_minus_p10") or metrics.get("w_photo_p90_minus_p10"),
        "normalization_mode": metrics.get("normalization_mode", "N/A"),
        "lambda_depth": metrics.get("lambda_depth"),
        "lambda_reg": metrics.get("lambda_reg"),
        "alpha": metrics.get("alpha"),
        "w_photo_min": metrics.get("w_photo_min"),
        "variant": metrics.get("variant"),
        "mask_used": metrics.get("mask_used", False),
        "mask_coverage": metrics.get("mask_coverage"),
        "mask_effective": metrics.get("mask_effective"),
        "n_gaussians": metrics.get("n_gaussians"),
        "iterations": metrics.get("iterations"),
        "depth_semantics": metrics.get("depth_semantics") or metrics.get("val_depth_semantics", "N/A"),
        "enable_densification": metrics.get("enable_densification"),
    }


def _per_variant_checks(rows: List[Dict]) -> Dict:
    checks = {}
    for row in rows:
        rid = row["run_id"]
        if not rid.startswith("M3"):
            continue
        w_mean = row.get("val_w_photo_mean")
        checks[f"{rid}_w_photo_mean_in_range"] = (
            w_mean is not None and 0.15 <= w_mean <= 0.95
            if w_mean is not None else None
        )
        frac_min = row.get("val_fraction_w_photo_at_min")
        checks[f"{rid}_fraction_at_min_ok"] = (
            frac_min is not None and frac_min < 0.90
            if frac_min is not None else None
        )
        frac_one = row.get("val_fraction_w_photo_at_one")
        checks[f"{rid}_fraction_at_one_ok"] = (
            frac_one is not None and frac_one < 0.90
            if frac_one is not None else None
        )
        spread = row.get("val_w_photo_p90_minus_p10")
        checks[f"{rid}_spread_ok"] = (
            spread is not None and spread >= 0.05
            if spread is not None else None
        )
        norm_mode = row.get("normalization_mode", "N/A")
        checks[f"{rid}_normalization_p95_detached"] = norm_mode == "p95_detached"
        if rid in ("M3-H2", "M3-H3"):
            me = row.get("mask_effective")
            checks[f"{rid}_mask_effective"] = (
                me if me is not None else False
            )
    return checks


def _compute_gate_decisions(table: List[Dict], m1: Dict, m2b_r1: Dict) -> Dict:
    m1_psnr = m1.get("val_psnr") if m1 else None
    m2b_r1_psnr = m2b_r1.get("val_psnr") if m2b_r1 else None

    if not m1 or m1_psnr is None:
        return {
            "status": "INVALID_FOR_GATE",
            "reason": "m1_missing",
            "minimum_pass": False,
            "psnr_guard_pass": False,
            "depth_rmse_guard_pass": False,
            "blocking_reasons": ["M1 baseline metrics missing or val_psnr is None; cannot evaluate gate."],
            "m1_val_psnr": None, "m2b_r1_val_psnr": m2b_r1_psnr,
            "best_m3_run_id": None, "best_m3_val_psnr": None,
            "best_m3_val_depth_rmse_m_raw": None, "best_m3_val_w_photo_mean": None,
            "best_h12_run_id": None, "best_h12_val_psnr": None,
            "best_h12_val_depth_rmse_m_raw": None,
            "h3_was_run_when_needed": None, "h3_should_run": False,
            "h3_trigger_reason": None,
            "num_m3_variants": len([r for r in table if r["run_id"].startswith("M3")]),
            "collapsed_variants": [],
        }
    if not m2b_r1 or m2b_r1_psnr is None:
        return {
            "status": "INVALID_FOR_GATE",
            "reason": "m2b_r1_missing",
            "minimum_pass": False,
            "psnr_guard_pass": False,
            "depth_rmse_guard_pass": False,
            "blocking_reasons": ["M2-B R1 metrics missing or val_psnr is None; cannot evaluate gate."],
            "m1_val_psnr": m1_psnr, "m2b_r1_val_psnr": None,
            "best_m3_run_id": None, "best_m3_val_psnr": None,
            "best_m3_val_depth_rmse_m_raw": None, "best_m3_val_w_photo_mean": None,
            "best_h12_run_id": None, "best_h12_val_psnr": None,
            "best_h12_val_depth_rmse_m_raw": None,
            "h3_was_run_when_needed": None, "h3_should_run": False,
            "h3_trigger_reason": None,
            "num_m3_variants": len([r for r in table if r["run_id"].startswith("M3")]),
            "collapsed_variants": [],
        }

    best_m3 = None
    best_psnr = -1e9
    for row in table:
        if row["run_id"].startswith("M3"):
            p = row.get("val_psnr")
            if p is not None and p > best_psnr:
                best_psnr = p
                best_m3 = row

    m3_rows = [r for r in table if r["run_id"].startswith("M3")]
    h3_present = any(r["run_id"] == "M3-H3" for r in m3_rows)
    best_h12 = None
    best_h12_psnr = -1e9
    for r in m3_rows:
        if r["run_id"] in ("M3-H1", "M3-H2"):
            p = r.get("val_psnr")
            if p is not None and p > best_h12_psnr:
                best_h12_psnr = p
                best_h12 = r

    threshold = m1_psnr - 3.0  # dynamic baseline-derived threshold (D8)
    guard_issues = []
    if best_m3 is None:
        guard_issues.append("No M3 variant runs found in table.")
    elif best_m3.get("val_psnr") is None:
        guard_issues.append("Best M3 has val_psnr=None.")
    else:
        if best_m3["val_psnr"] < threshold:
            guard_issues.append(
                f"PSNR guard FAIL: best M3 {best_m3['val_psnr']:.2f} < {threshold:.2f} "
                f"(baseline {m1_psnr:.2f} - 3.0)"
            )
        if best_m3.get("val_depth_rmse_m_raw") is not None and best_m3["val_depth_rmse_m_raw"] > 0.030:
            guard_issues.append(
                f"Depth RMSE FAIL: {best_m3['val_depth_rmse_m_raw']:.4f} > 0.030 m"
            )

    h3_should_run = False
    h3_trigger_reason = None
    if best_h12 and best_h12.get("val_psnr") is not None:
        psnr_condition = best_h12["val_psnr"] < threshold
        rmse = best_h12.get("val_depth_rmse_m_raw")
        rmse_condition = rmse is not None and rmse <= 0.030
        h3_should_run = psnr_condition and rmse_condition
        if h3_should_run:
            h3_trigger_reason = (
                f"best(H1,H2).val_psnr={best_h12['val_psnr']:.2f} < baseline_threshold={threshold:.2f} "
                f"AND best(H1,H2).val_depth_rmse_m_raw={rmse:.4f} <= 0.030"
            )

    gate = {
        "status": "OK",
        "reason": None,
        "m1_val_psnr": m1_psnr,
        "m2b_r1_val_psnr": m2b_r1_psnr,
        "best_m3_run_id": best_m3["run_id"] if best_m3 else None,
        "best_m3_val_psnr": best_m3.get("val_psnr") if best_m3 else None,
        "best_m3_val_depth_rmse_m_raw": best_m3.get("val_depth_rmse_m_raw") if best_m3 else None,
        "best_m3_val_w_photo_mean": best_m3.get("val_w_photo_mean") if best_m3 else None,
        "psnr_guard_pass": len([i for i in guard_issues if "PSNR guard FAIL" in i]) == 0,
        "depth_rmse_guard_pass": len([i for i in guard_issues if "Depth RMSE FAIL" in i]) == 0,
        "minimum_pass": len(guard_issues) == 0,
        "blocking_reasons": guard_issues,
        "best_h12_run_id": best_h12["run_id"] if best_h12 else None,
        "best_h12_val_psnr": best_h12.get("val_psnr") if best_h12 else None,
        "best_h12_val_depth_rmse_m_raw": best_h12.get("val_depth_rmse_m_raw") if best_h12 else None,
        "h3_was_run_when_needed": (
            None if not best_h12 else
            (h3_present if h3_should_run else not h3_present)
        ),
        "h3_should_run": h3_should_run,
        "h3_trigger_reason": h3_trigger_reason,
        "num_m3_variants": len(m3_rows),
        "collapsed_variants": [],
    }

    if best_m3:
        w_mean = best_m3.get("val_w_photo_mean")
        gate["w_photo_mean_in_range"] = w_mean is not None and 0.15 <= w_mean <= 0.95 if w_mean is not None else None
        frac_min = best_m3.get("val_fraction_w_photo_at_min")
        gate["fraction_at_min_ok"] = frac_min is not None and frac_min < 0.90 if frac_min is not None else None
        frac_one = best_m3.get("val_fraction_w_photo_at_one")
        gate["fraction_at_one_ok"] = frac_one is not None and frac_one < 0.90 if frac_one is not None else None
        spread = best_m3.get("val_w_photo_p90_minus_p10")
        gate["w_photo_spread_ok"] = spread is not None and spread >= 0.05 if spread is not None else None

    gate["per_variant"] = _per_variant_checks(table)

    collapsed = []
    for rid, key in gate.get("per_variant", {}).items():
        if rid.endswith("_fraction_at_min_ok") and key is False:
            v = rid.replace("_fraction_at_min_ok", "")
            collapsed.append(v)
        if rid.endswith("_fraction_at_one_ok") and key is False:
            v = rid.replace("_fraction_at_one_ok", "")
            if v not in collapsed:
                collapsed.append(v)
        if rid.endswith("_spread_ok") and key is False:
            v = rid.replace("_spread_ok", "")
            if v not in collapsed:
                collapsed.append(v)
    gate["collapsed_variants"] = collapsed

    return gate


def _build_table_markdown(table: List[Dict], gate: Dict) -> str:
    lines = [
        "# M3 Comparison Table: M1 / M2-B / M3",
        "",
        "| Run ID | PSNR | SSIM | Depth RMSE | Depth MAE | AbsRel | Mean w | λ_depth |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for row in table:
        lines.append(
            f"| {row['run_id']} "
            f"| {_fmt(row.get('val_psnr'))} "
            f"| {_fmt(row.get('val_ssim'))} "
            f"| {_fmt(row.get('val_depth_rmse_m_raw'), 6)} "
            f"| {_fmt(row.get('val_depth_mae_m_raw'), 6)} "
            f"| {_fmt(row.get('val_abs_rel'), 6)} "
            f"| {_fmt(row.get('val_w_photo_mean'))} "
            f"| {_fmt(row.get('lambda_depth'))} |"
        )

    lines.extend([
        "",
        "## Gate Decision",
        f"**Status: {gate.get('status', 'UNKNOWN')}**",
    ])
    status = gate.get('status')
    if status == 'INVALID_FOR_GATE':
        lines.append(f"**Reason: {gate.get('reason', 'unspecified')}**")
        lines.append("")
        lines.append("*Comparison skipped — reference run(s) missing.*")
    elif status == 'OK':
        lines.append(f"**Minimum PASS: {'✓' if gate.get('minimum_pass') else '✗'}**")
        lines.append(f"- PSNR guard (≥ baseline - 3.0): {'✓' if gate.get('psnr_guard_pass') else '✗'}")
        lines.append(f"- Depth RMSE guard (≤ 0.030 m): {'✓' if gate.get('depth_rmse_guard_pass') else '✗'}")
        lines.append(f"- Mean w_photo in [0.15, 0.95]: {_fmt(gate.get('w_photo_mean_in_range'))}")
        lines.append(f"- fraction_w_photo_at_min < 0.90: {_fmt(gate.get('fraction_at_min_ok'))}")
        lines.append(f"- fraction_w_photo_at_one < 0.90: {_fmt(gate.get('fraction_at_one_ok'))}")
        lines.append(f"- w_photo spread ≥ 0.05: {_fmt(gate.get('w_photo_spread_ok'))}")
        if gate.get("h3_should_run"):
            lines.append(f"- **H3 should run: YES** — {gate.get('h3_trigger_reason')}")
        else:
            lines.append("- H3 should run: NO")
        lines.append("")
        lines.append("### Per-Variant Collapse Detection")
    if gate.get("blocking_reasons"):
        for issue in gate["blocking_reasons"]:
            lines.append(f"- ❌ {issue}")
    for rid, conds in sorted(gate.get("per_variant", {}).items()):
        parts = rid.split("_", 1)
        variant_prefix = parts[0] if len(parts) == 2 else rid
        check_name = parts[1] if len(parts) == 2 else ""
        lines.append(f"- {rid}: {'✓' if conds else '✗'}")
    collapsed = gate.get("collapsed_variants", [])
    if collapsed:
        lines.append(f"- **Collapsed variants**: {', '.join(collapsed)}")
    else:
        lines.append("- **Collapsed variants**: none")
    lines.append(f"- H3 run when needed: {_fmt(gate.get('h3_was_run_when_needed'))}")

    lines.extend([
        "",
        "### Best H1/H2 (pre-H3 gate)",
        f"- Best: {gate.get('best_h12_run_id')} @ {_fmt(gate.get('best_h12_val_psnr'))} dB",
        f"- Depth RMSE: {_fmt(gate.get('best_h12_val_depth_rmse_m_raw'), 6)} m",
        "",
        "## Key Reference Points",
        f"- M1 Baseline PSNR: {_fmt(gate.get('m1_val_psnr'))} dB",
        f"- M2-B R1 (best depth recovery) PSNR: {_fmt(gate.get('m2b_r1_val_psnr'))} dB",
        f"- Best M3: {gate.get('best_m3_run_id')} at {_fmt(gate.get('best_m3_val_psnr'))} dB",
        f"- Best M3 depth RMSE: {_fmt(gate.get('best_m3_val_depth_rmse_m_raw'), 6)} m",
        f"- Best M3 mean w_photo: {_fmt(gate.get('best_m3_val_w_photo_mean'))}",
    ])

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Compare M1 / M2-B / M3 results"
    )
    parser.add_argument("--baseline", type=str,
                        default="outputs/runs/baseline_debug",
                        help="M1 baseline run directory")
    parser.add_argument("--m2b_orig", type=str, default=None,
                        help="M2-B original run directory")
    parser.add_argument("--m2b_r1", type=str, default=None,
                        help="M2-B R1 recovery run directory")
    parser.add_argument("--m2b_r3", type=str, default=None,
                        help="M2-B R3 recovery run directory")
    parser.add_argument("--m3_h1", type=str, default=None,
                        help="M3-H1 run directory")
    parser.add_argument("--m3_h2", type=str, default=None,
                        help="M3-H2 run directory")
    parser.add_argument("--m3_h3", type=str, default=None,
                        help="M3-H3 run directory (optional)")
    parser.add_argument("--output_dir", type=str,
                        default="outputs/runs/m3_comparison",
                        help="Output directory")
    args = parser.parse_args()

    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    run_paths = [
        ("M1 baseline", args.baseline),
        ("M2-B original", args.m2b_orig),
        ("M2-B R1", args.m2b_r1),
        ("M2-B R3", args.m2b_r3),
        ("M3-H1", args.m3_h1),
        ("M3-H2", args.m3_h2),
        ("M3-H3", args.m3_h3),
    ]

    table = []
    m1_metrics = None
    m2b_r1_metrics = None

    for run_id, path_str in run_paths:
        if path_str is None:
            continue
        p = Path(path_str)
        if not p.exists():
            print(f"WARNING: {run_id} directory not found: {p}")
            continue
        try:
            metrics = _load_final_metrics(p)
        except FileNotFoundError:
            print(f"WARNING: {run_id} final_metrics.json not found in {p}")
            continue
        row = _extract_row(run_id, metrics)
        table.append(row)

        if run_id == "M1 baseline":
            m1_metrics = metrics
        elif run_id == "M2-B R1":
            m2b_r1_metrics = metrics

    if not table:
        print("ERROR: No valid run directories found.")
        sys.exit(1)

    if m1_metrics is None:
        print("FATAL: M1 baseline final_metrics.json not found. Cannot run comparison.")
        sys.exit(1)
    if m2b_r1_metrics is None:
        print("FATAL: M2-B R1 final_metrics.json not found. Cannot run comparison.")
        sys.exit(1)

    gate = _compute_gate_decisions(
        table,
        m1=m1_metrics,
        m2b_r1=m2b_r1_metrics,
    )

    comparison_path = out_path / "comparison_table.json"
    comparison_path.write_text(json.dumps({
        "runs": table,
        "gate_decision": gate,
    }, indent=2))

    markdown = _build_table_markdown(table, gate)
    md_path = out_path / "comparison_table.md"
    md_path.write_text(markdown)

    print("\n=== M3 Comparison ===")
    print(f"  Status: {gate.get('status', 'UNKNOWN')}")
    if gate.get('reason'):
        print(f"  Reason: {gate['reason']}")
    print(f"  Runs compared: {', '.join(r['run_id'] for r in table)}")
    status = gate.get('status')
    if status == 'OK':
        print(f"  Best M3: {gate.get('best_m3_run_id')} @ {_fmt(gate.get('best_m3_val_psnr'))} dB")
        print(f"  Depth RMSE: {_fmt(gate.get('best_m3_val_depth_rmse_m_raw'), 6)} m")
        print(f"  Minimum PASS: {'✓' if gate.get('minimum_pass') else '✗'}")
        if gate.get("h3_should_run"):
            print(f"  H3 should run: YES — {gate.get('h3_trigger_reason')}")
    if gate.get("blocking_reasons"):
        for issue in gate["blocking_reasons"]:
            print(f"    ✗ {issue}")
    print(f"  Artifacts: {out_path.resolve()}")


if __name__ == "__main__":
    main()
