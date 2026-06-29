import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


M1_BASELINE_PSNR = 20.15
PSNR_THRESHOLD = M1_BASELINE_PSNR - 3.0  # 17.15 dB
DEPTH_RMSE_THRESHOLD = 0.030  # m
M3_H1_BEST_DEPTH = 0.0364  # m reference

VRAM_GREEN = 16.0
VRAM_ACCEPTABLE = 20.0
VRAM_WARNING = 21.5
VRAM_FAIL = 21.5


def _load_final_metrics(run_dir: Path) -> Dict:
    path = run_dir / "final_metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"final_metrics.json not found in {run_dir}")
    return json.loads(path.read_text())


def _load_vram_stats(run_dir: Path) -> Optional[Dict]:
    path = run_dir / "metrics.jsonl"
    if not path.exists():
        return None
    vram_vals = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if "vram_gb" in entry:
                    vram_vals.append(entry["vram_gb"])
            except (json.JSONDecodeError, ValueError):
                continue
    if not vram_vals:
        return None
    return {
        "max_vram_gb": max(vram_vals),
        "mean_vram_gb": sum(vram_vals) / len(vram_vals),
        "n_samples": len(vram_vals),
    }


def _fmt(val, precision: int = 4) -> str:
    if val is None:
        return "N/A"
    if isinstance(val, float):
        return f"{val:.{precision}f}"
    if isinstance(val, bool):
        return str(val)
    return str(val)


def _delta(base, val, higher_is_better: bool = True) -> str:
    if base is None or val is None:
        return ""
    diff = val - base
    pct = (diff / abs(base)) * 100 if base != 0 else 0.0
    arrow = "▲" if ((higher_is_better and diff > 0) or (not higher_is_better and diff < 0)) else "▼"
    return f" {arrow} {diff:+.4f} ({pct:+.1f}%)"


def _vram_tier(vram_gb: Optional[float]) -> str:
    if vram_gb is None:
        return "N/A"
    if vram_gb <= VRAM_GREEN:
        return "green"
    elif vram_gb <= VRAM_ACCEPTABLE:
        return "acceptable"
    elif vram_gb < VRAM_WARNING:
        return "warning"
    else:
        return "fail"


def _build_comparison_table(m3_row: Dict, m4_row: Dict, vram: Optional[Dict]) -> List[Dict]:
    metrics_keys = [
        ("val_psnr", "PSNR (dB)", True),
        ("val_ssim", "SSIM", True),
        ("val_lpips", "LPIPS", False),
        ("val_depth_rmse_m_raw", "Depth RMSE (m)", False),
        ("val_depth_mae_m_raw", "Depth MAE (m)", False),
        ("val_abs_rel", "Abs Rel", False),
        ("n_gaussians", "N Gaussians", None),
    ]
    table = []
    for key, label, higher_better in metrics_keys:
        m3v = m3_row.get(key)
        m4v = m4_row.get(key)
        row = {
            "metric": label,
            "key": key,
            "m3_h1": m3v,
            "m4_a1": m4v,
            "delta": _delta(m3v, m4v, higher_better) if higher_better is not None else _fmt(m4v - m3v if m4v is not None and m3v is not None else None),
        }
        table.append(row)

    w_photo_keys = [
        ("val_w_photo_mean", "w_photo mean"),
        ("val_w_photo_min", "w_photo min"),
        ("val_w_photo_max", "w_photo max"),
        ("val_w_photo_p10", "w_photo p10"),
        ("val_w_photo_p50", "w_photo p50"),
        ("val_w_photo_p90", "w_photo p90"),
        ("val_fraction_w_photo_at_min", "frac at w_min"),
        ("val_fraction_w_photo_at_one", "frac at 1.0"),
    ]
    for key, label in w_photo_keys:
        m3v = m3_row.get(key)
        m4v = m4_row.get(key)
        table.append({
            "metric": label,
            "key": key,
            "m3_h1": m3v,
            "m4_a1": m4v,
            "delta": _delta(m3v, m4v, higher_better=None) if m4v is not None and m3v is not None else "",
        })

    if vram:
        table.append({
            "metric": "VRAM peak (GB)",
            "key": "vram_gb",
            "m3_h1": None,
            "m4_a1": vram["max_vram_gb"],
            "delta": f"tier={_vram_tier(vram['max_vram_gb'])}",
        })

    return table


def _compute_gate(m3_row: Dict, m4_row: Dict, vram: Optional[Dict]) -> Dict:
    m4_psnr = m4_row.get("val_psnr")
    m4_rmse = m4_row.get("val_depth_rmse_m_raw")
    m4_n_gaussians = m4_row.get("n_gaussians")
    m3_rmse = m3_row.get("val_depth_rmse_m_raw")

    results = {}
    blocking = []

    # PSNR gate
    psnr_pass = m4_psnr is not None and m4_psnr >= PSNR_THRESHOLD
    results["psnr_pass"] = psnr_pass
    results["psnr_value"] = m4_psnr
    results["psnr_threshold"] = PSNR_THRESHOLD
    if not psnr_pass:
        blocking.append(f"PSNR {_fmt(m4_psnr)} < {_fmt(PSNR_THRESHOLD)} dB")

    # Depth RMSE gate
    depth_pass = m4_rmse is not None and m4_rmse <= DEPTH_RMSE_THRESHOLD
    results["depth_rmse_pass"] = depth_pass
    results["depth_rmse_value"] = m4_rmse
    results["depth_rmse_threshold"] = DEPTH_RMSE_THRESHOLD
    if not depth_pass and m4_rmse is not None:
        blocking.append(f"Depth RMSE {_fmt(m4_rmse, 6)} > {_fmt(DEPTH_RMSE_THRESHOLD, 3)} m")

    # Partial positive check
    improved = m4_rmse is not None and m3_rmse is not None and m4_rmse < m3_rmse
    partial = improved and not depth_pass
    results["improved_over_m3_h1"] = improved
    results["m3_h1_depth_rmse"] = m3_rmse
    if improved:
        rel_improvement = (m3_rmse - m4_rmse) / m3_rmse * 100
        results["relative_improvement_pct"] = round(rel_improvement, 2)
        results["partial_positive"] = partial
    else:
        results["relative_improvement_pct"] = None
        results["partial_positive"] = False

    # M4-A1b trigger (depth_RMSE ≤ 0.0335 or ≥8% relative improvement)
    trigger_a1b = False
    if m4_rmse is not None:
        if m4_rmse <= 0.0335:
            trigger_a1b = True
        elif improved and results["relative_improvement_pct"] is not None and results["relative_improvement_pct"] >= 8.0:
            trigger_a1b = True
    results["m4_a1b_trigger"] = trigger_a1b

    # Gaussian count
    n_gauss_ok = m4_n_gaussians is not None and m4_n_gaussians == 50000
    results["n_gaussians_ok"] = n_gauss_ok
    results["n_gaussians"] = m4_n_gaussians
    if not n_gauss_ok and m4_n_gaussians is not None:
        blocking.append(f"n_gaussians={m4_n_gaussians}, expected 50000")

    # VRAM gate
    vram_tier = "N/A"
    if vram is not None:
        vram_max = vram["max_vram_gb"]
        vram_tier = _vram_tier(vram_max)
        if vram_tier == "fail":
            blocking.append(f"VRAM {_fmt(vram_max, 1)} GB >= {VRAM_FAIL} GB (FAIL)")
        elif vram_tier == "warning":
            blocking.append(f"VRAM {_fmt(vram_max, 1)} GB in warning range ({VRAM_ACCEPTABLE}-{VRAM_WARNING} GB)")
    results["vram_tier"] = vram_tier
    results["vram_info"] = vram

    # Final status
    if psnr_pass and depth_pass and n_gauss_ok:
        if vram_tier in ("green", "acceptable"):
            results["status"] = "FULL PASS"
            results["label"] = "M4-A1 Full PASS"
        elif vram_tier == "warning":
            results["status"] = "FULL PASS (VRAM WARNING)"
            results["label"] = "M4-A1 Full PASS with VRAM concern"
        else:
            results["status"] = "VRAM FAIL"
            results["label"] = "M4-A1 VRAM FAIL"
    elif psnr_pass and partial:
        results["status"] = "PARTIAL POSITIVE"
        results["label"] = "M4-A1 Partial Positive"
    elif psnr_pass and not depth_pass and not improved:
        results["status"] = "NEGATIVE (no improvement)"
        results["label"] = "M4-A1 Negative"
    else:
        results["status"] = "FAIL"
        results["label"] = "M4-A1 FAIL"

    results["blocking_reasons"] = blocking
    results["vram_tier"] = vram_tier

    results["recommendation"] = _recommendation(results)

    return results


def _recommendation(gate: Dict) -> str:
    status = gate.get("status", "")

    if status == "FULL PASS" or status == "FULL PASS (VRAM WARNING)":
        return "Proceed to M4-A2 (multi-criteria density control)."
    if status == "PARTIAL POSITIVE":
        if gate.get("m4_a1b_trigger"):
            return "Run M4-A1b (100K) — 50K shows meaningful improvement."
        return "Partial improvement but below A1b trigger threshold. Proceed to M4-A2 low-expectation."
    if "VRAM FAIL" in status:
        return "VRAM exceeded safe limit. Investigate memory usage before continuing."
    if "NEGATIVE" in status:
        return "50K did not improve depth. Skip 100K. Consider M4-A2 low-expectation or pivot discussion."
    return "Gate failed. Review blocking reasons before proceeding."


def _format_table_markdown(comparison: List[Dict], gate: Dict, m3_run_id: str, m4_run_id: str) -> str:
    lines = [
        f"# M3-H1 vs M4-A1 Comparison",
        f"",
        f"**Run IDs:** {m3_run_id} (M3-H1 baseline) vs {m4_run_id} (M4-A1 50K)",
        f"**M4 Gate Status:** {gate.get('status', 'UNKNOWN')}",
        f"**Label:** {gate.get('label', 'N/A')}",
        f"",
        f"## Metrics",
        f"",
        f"| Metric | M3-H1 | M4-A1 | Δ |",
        f"|--------|-------|-------|---|",
    ]

    for row in comparison:
        m3_val = _fmt(row["m3_h1"])
        m4_val = _fmt(row["m4_a1"])
        delta_val = row.get("delta", "")
        lines.append(f"| {row['metric']} | {m3_val} | {m4_val} | {delta_val} |")

    lines.extend([
        f"",
        f"## Gate Decision",
        f"",
        f"| Check | Result | Details |",
        f"|-------|--------|---------|",
        f"| PSNR ≥ {_fmt(PSNR_THRESHOLD)} dB | {'✅' if gate.get('psnr_pass') else '❌'} | {_fmt(gate.get('psnr_value'))} dB |",
        f"| Depth RMSE ≤ {_fmt(DEPTH_RMSE_THRESHOLD, 3)} m | {'✅' if gate.get('depth_rmse_pass') else '❌'} | {_fmt(gate.get('depth_rmse_value'), 6)} m |",
        f"| n_gaussians == 50000 | {'✅' if gate.get('n_gaussians_ok') else '❌'} | {gate.get('n_gaussians', 'N/A')} |",
        f"| Improved over M3-H1 | {'✅' if gate.get('improved_over_m3_h1') else '❌'} | baseline {_fmt(gate.get('m3_h1_depth_rmse'), 6)} m |",
        f"| VRAM tier | {gate.get('vram_tier', 'N/A')} | {_fmt(gate.get('vram_info', {}).get('max_vram_gb')) if gate.get('vram_info') else 'N/A'} GB peak |",
    ])

    if gate.get("blocking_reasons"):
        lines.extend([
            f"",
            f"### Blocking Issues",
        ])
        for reason in gate["blocking_reasons"]:
            lines.append(f"- ❌ {reason}")

    lines.extend([
        f"",
        f"**Recommendation:** {gate.get('recommendation', 'N/A')}",
        f"",
        f"### M4-A1b Trigger",
        f"- depth_RMSE ≤ 0.0335m or ≥8% relative improvement: **{'YES' if gate.get('m4_a1b_trigger') else 'NO'}**",
        f"- Relative improvement: {_fmt(gate.get('relative_improvement_pct', 0), 2)}%",
    ])

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(
        description="Compare M3-H1 vs M4-A1 results"
    )
    parser.add_argument("--m3_h1", type=str, required=True,
                        help="M3-H1 run directory (baseline)")
    parser.add_argument("--m4_a1", type=str, required=True,
                        help="M4-A1 run directory")
    parser.add_argument("--output_dir", type=str,
                        default="outputs/runs/m4_a1_comparison",
                        help="Output directory")
    args = parser.parse_args()

    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    m3_path = Path(args.m3_h1)
    m4_path = Path(args.m4_a1)

    if not m3_path.exists():
        print(f"ERROR: M3-H1 directory not found: {m3_path}")
        sys.exit(1)
    if not m4_path.exists():
        print(f"ERROR: M4-A1 directory not found: {m4_path}")
        sys.exit(1)

    m3_metrics = _load_final_metrics(m3_path)
    m4_metrics = _load_final_metrics(m4_path)
    vram_stats = _load_vram_stats(m4_path)

    m3_row = {**m3_metrics, "run_id": "M3-H1"}
    m4_row = {**m4_metrics, "run_id": "M4-A1"}

    comparison = _build_comparison_table(m3_row, m4_row, vram_stats)
    gate = _compute_gate(m3_row, m4_row, vram_stats)

    comparison_path = out_path / "comparison_table.json"
    comparison_path.write_text(json.dumps({
        "m3_h1_run": str(m3_path),
        "m4_a1_run": str(m4_path),
        "comparison": comparison,
        "gate": gate,
        "vram": vram_stats,
    }, indent=2))

    markdown = _format_table_markdown(comparison, gate, str(m3_path), str(m4_path))
    md_path = out_path / "comparison_table.md"
    md_path.write_text(markdown)

    print(f"\n=== M4-A1 Comparison ===")
    print(f"  Status: {gate.get('status', 'UNKNOWN')}")
    print(f"  Label: {gate.get('label', 'N/A')}")
    if gate.get("blocking_reasons"):
        for reason in gate["blocking_reasons"]:
            print(f"    ✗ {reason}")
    print(f"  PSNR: {_fmt(gate.get('psnr_value'))} dB (threshold {_fmt(PSNR_THRESHOLD)} dB)")
    print(f"  Depth RMSE: {_fmt(gate.get('depth_rmse_value'), 6)} m (threshold {_fmt(DEPTH_RMSE_THRESHOLD, 3)} m)")
    print(f"  A1b trigger: {'YES' if gate.get('m4_a1b_trigger') else 'NO'}")
    print(f"\n  Recommendation: {gate.get('recommendation', 'N/A')}")
    print(f"  Artifacts: {out_path.resolve()}")


if __name__ == "__main__":
    main()
