import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from surgtwin.data.manifest import load_manifest, filter_by_split, validate_manifest_entry
from surgtwin.gaussian.backend_gsplat import GsplatBackend
from surgtwin.training.manifest_snapshot import write_manifest_snapshot
from surgtwin.training.uncertainty_config import UncertaintyConfig
from surgtwin.training.uncertainty_trainer import UncertaintyTrainer
from surgtwin.training.logging_utils import write_json, collect_environment


def _preflight_m2a_artifact(artifact_path: str, allow_mock: bool) -> None:
    if allow_mock:
        print("WARNING: --allow_mock_m2a set; bypassing M2-A gate check (NOT for production).")
        return

    if not artifact_path:
        raise SystemExit(
            "FATAL: --depth_semantics_artifact is required in prod mode. "
            "Point to outputs/runs/m2a_gate/final_gate_decision.json produced by "
            "scripts/verify_render_depth_semantics.py."
        )

    p = Path(artifact_path)
    if not p.exists():
        raise SystemExit(
            f"FATAL: M2-A artifact not found at '{p}'. "
            f"Production training will not proceed with a missing gate artifact."
        )

    try:
        data = json.loads(p.read_text())
    except json.JSONDecodeError as e:
        raise SystemExit(f"FATAL: M2-A artifact at {p} is not valid JSON: {e}")

    if not data.get("depth_semantics_verified", False):
        raise SystemExit(
            f"FATAL: M2-A gate not PASS: {p}. "
            f"depth_semantics_verified=false. "
            f"Production training will not proceed without verified depth semantics."
        )

    if data.get("m2a_gate") != "PASS":
        raise SystemExit(
            f"FATAL: m2a_gate='{data.get('m2a_gate')}' (expected 'PASS'). "
            f"Production training will not proceed."
        )

    print(f"M2-A gate PASS verified: {p}")


def _resolve_output_dir(output_dir: Path, allow_mock: bool) -> Path:
    if allow_mock:
        debug_dir = output_dir.parent / f"_debug_{output_dir.name}"
        debug_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Debug run output: {debug_dir}")
        print(f"  This run will NOT be eligible for gate comparison.")
        return debug_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="M3 Uncertainty-Weighted Gaussian Splatting Training"
    )
    parser.add_argument("--manifest", type=str, required=True,
                        help="Path to manifest JSONL")
    parser.add_argument("--output_dir", type=str,
                        default="outputs/runs/uncertainty_m3",
                        help="Output directory")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--init_num_points", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_every", type=int, default=100)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--ckpt_every", type=int, default=500)

    parser.add_argument("--variant", type=str, default="h1",
                        choices=["h1", "h2", "h3"],
                        help="M3 variant: h1=residual only, h2=mask-aware, h3=low depth")
    parser.add_argument("--lambda_depth", type=float, default=0.2,
                        help="Weight for depth loss")
    parser.add_argument("--lambda_reg", type=float, default=0.0,
                        help="Weight for regularization (locked to 0.0 for M3)")
    parser.add_argument("--alpha", type=float, default=2.0,
                        help="Alpha for w_photo = exp(-alpha * u_photo)")
    parser.add_argument("--w_photo_min", type=float, default=0.15,
                        help="Minimum photometric weight")
    parser.add_argument("--mask_boost", type=float, default=0.5,
                        help="Boost added to u_photo in masked regions (H2)")
    parser.add_argument("--mask_dir", type=str, default=None,
                        help="Directory with precomputed specular masks (H2)")
    parser.add_argument("--depth_semantics_artifact", type=str,
                        default="outputs/runs/m2a_gate/final_gate_decision.json",
                        help="Path to M2-A gate decision JSON")
    parser.add_argument("--prod_mode", action="store_true", default=True,
                        help="Production mode (DEFAULT): hard-fail if M2-A artifact "
                             "missing OR depth_semantics_verified=false.")
    parser.add_argument("--allow_mock_m2a", action="store_true",
                        help="Debug/smoke-test mode: skip M2-A gate check. "
                             "Setting this with --prod_mode is a hard error.")
    parser.add_argument("--no_clip_grad", action="store_true",
                        help="Disable gradient clipping")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Max gradient norm for clipping")

    args = parser.parse_args()

    # --- M2-A pre-flight gate (Directives D2 + D6) ---
    if args.allow_mock_m2a and args.prod_mode:
        raise SystemExit(
            "FATAL: --allow_mock_m2a is incompatible with prod_mode. "
            "Production runs MUST NOT accept mock M2-A artifacts."
        )
    _preflight_m2a_artifact(args.depth_semantics_artifact, args.allow_mock_m2a)

    # Resolve output dir (debug mode relocates to _debug_ prefix)
    output_dir = _resolve_output_dir(Path(args.output_dir), args.allow_mock_m2a)

    entries = load_manifest(Path(args.manifest))
    for entry in entries:
        validate_manifest_entry(entry)

    train = filter_by_split(entries, "train")
    val = filter_by_split(entries, "val")

    if not train:
        raise ValueError(f"No training entries found (split='train') in manifest.")
    if not val:
        raise ValueError(f"No validation entries found (split='val') in manifest.")

    run_mode = "debug" if args.allow_mock_m2a else "production"
    config = UncertaintyConfig(
        iterations=args.iterations,
        init_num_points=args.init_num_points,
        seed=args.seed,
        val_every=args.val_every,
        log_every=args.log_every,
        ckpt_every=args.ckpt_every,
        variant=args.variant,
        lambda_depth=args.lambda_depth,
        lambda_reg=args.lambda_reg,
        alpha=args.alpha,
        w_photo_min=args.w_photo_min,
        mask_boost=args.mask_boost,
        mask_dir=args.mask_dir,
        depth_semantics_artifact_path=args.depth_semantics_artifact,
        clip_grad_norm=not args.no_clip_grad,
        max_grad_norm=args.max_grad_norm,
    )

    backend = GsplatBackend()

    config_dict = asdict(config)
    config_dict["run_mode"] = run_mode
    config_dict["gate_eligible"] = run_mode == "production"
    write_json(output_dir / "config.json", config_dict)
    write_json(output_dir / "environment.json", collect_environment(config.backend))

    trainer = UncertaintyTrainer(
        train_entries=train,
        val_entries=val,
        backend=backend,
        config=config,
        output_dir=output_dir,
    )

    final_metrics = trainer.fit()
    trainer.save_side_by_side_panels()

    write_manifest_snapshot(
        output_dir=output_dir,
        manifest_path=Path(args.manifest),
        entries=entries,
        train_entries=train,
        val_entries=val,
        extra={"split_strategy": trainer.split_strategy,
               "variant": args.variant,
               "alpha": args.alpha,
               "lambda_depth": args.lambda_depth,
               "run_mode": run_mode},
    )

    variant_label = f"M3-{args.variant.upper()}"
    print(f"\n=== {variant_label} Training Complete ===")
    print(f"  initial_loss_total: {final_metrics['initial_loss_total']:.6f}")
    print(f"  final_loss_total:   {final_metrics['final_loss_total']:.6f}")
    print(f"  loss_decreased: {final_metrics['loss_decreased']}")
    print(f"  val_psnr: {final_metrics['val_psnr']:.4f}")
    print(f"  val_ssim: {final_metrics['val_ssim']:.4f}")
    if final_metrics.get("val_lpips") is not None:
        print(f"  val_lpips: {final_metrics['val_lpips']:.4f}")
    if "val_depth_rmse_m_raw" in final_metrics:
        print(f"  depth_rmse_m_raw: {final_metrics['val_depth_rmse_m_raw']:.6f}")
        print(f"  depth_mae_m_raw:  {final_metrics['val_depth_mae_m_raw']:.6f}")
    if "val_w_photo_mean" in final_metrics:
        print(f"  val_w_photo_mean: {final_metrics['val_w_photo_mean']:.4f}")
    print(f"  outputs: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
