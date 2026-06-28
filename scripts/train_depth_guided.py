import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from surgtwin.data.manifest import load_manifest, filter_by_split, validate_manifest_entry
from surgtwin.gaussian.backend_gsplat import GsplatBackend
from surgtwin.training.depth_guided_config import DepthGuidedConfig
from surgtwin.training.depth_guided_trainer import DepthGuidedTrainer
from surgtwin.training.logging_utils import write_json, collect_environment


def main():
    parser = argparse.ArgumentParser(
        description="M2-B Depth-Guided Gaussian Splatting Training"
    )
    parser.add_argument("--manifest", type=str, required=True,
                        help="Path to manifest JSONL")
    parser.add_argument("--output_dir", type=str, default="outputs/runs/depth_guided_m2b",
                        help="Output directory")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--init_num_points", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr_means", type=float, default=1e-3)
    parser.add_argument("--lr_opacities", type=float, default=5e-2)
    parser.add_argument("--val_every", type=int, default=100)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--ckpt_every", type=int, default=500)

    parser.add_argument("--lambda_depth", type=float, default=0.2,
                        help="Weight for depth loss (default: 0.2)")
    parser.add_argument("--lambda_reg", type=float, default=0.01,
                        help="Weight for regularization (default: 0.01)")
    parser.add_argument("--reg_type", type=str, default="scale_drift",
                        choices=["scale_drift"],
                        help="Regularization type (default: scale_drift)")
    parser.add_argument("--depth_near_m", type=float, default=0.02,
                        help="Near depth range in meters (default: 0.02)")
    parser.add_argument("--depth_far_m", type=float, default=0.30,
                        help="Far depth range in meters (default: 0.30)")
    parser.add_argument("--depth_semantics_artifact", type=str,
                        default="outputs/runs/depth_semantics_m2a/final_gate_decision.json",
                        help="Path to M2-A gate decision JSON")
    parser.add_argument("--no_clip_grad", action="store_true",
                        help="Disable gradient clipping")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Max gradient norm for clipping (default: 1.0)")

    args = parser.parse_args()

    entries = load_manifest(Path(args.manifest))
    for entry in entries:
        validate_manifest_entry(entry)

    train = filter_by_split(entries, "train")
    val = filter_by_split(entries, "val")

    if not train:
        raise ValueError(f"No training entries found (split='train') in manifest. Got {len(entries)} total entries.")
    if not val:
        raise ValueError(f"No validation entries found (split='val') in manifest.")

    config = DepthGuidedConfig(
        iterations=args.iterations,
        init_num_points=args.init_num_points,
        seed=args.seed,
        lr_means=args.lr_means,
        lr_opacities=args.lr_opacities,
        val_every=args.val_every,
        log_every=args.log_every,
        ckpt_every=args.ckpt_every,
        lambda_depth=args.lambda_depth,
        lambda_reg=args.lambda_reg,
        reg_type=args.reg_type,
        depth_near_m=args.depth_near_m,
        depth_far_m=args.depth_far_m,
        depth_semantics_artifact_path=args.depth_semantics_artifact,
        clip_grad_norm=not args.no_clip_grad,
        max_grad_norm=args.max_grad_norm,
    )

    backend = GsplatBackend()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    write_json(output_dir / "config.json", config.__dict__)
    write_json(output_dir / "environment.json", collect_environment(config.backend))

    trainer = DepthGuidedTrainer(
        train_entries=train,
        val_entries=val,
        backend=backend,
        config=config,
        output_dir=output_dir,
    )

    final_metrics = trainer.fit()
    trainer.save_side_by_side_panels()

    print("\n=== M2-B Depth-Guided Training Complete ===")
    print(f"  initial_loss_total: {final_metrics['initial_loss_total']:.6f}")
    print(f"  final_loss_total:   {final_metrics['final_loss_total']:.6f}")
    print(f"  loss_decreased: {final_metrics['loss_decreased']}")
    print(f"  val_psnr: {final_metrics['val_psnr']:.4f}")
    print(f"  val_ssim: {final_metrics['val_ssim']:.4f}")
    if final_metrics.get("val_lpips") is not None:
        print(f"  val_lpips: {final_metrics['val_lpips']:.4f}")
    else:
        print(f"  val_lpips: unavailable ({final_metrics.get('val_lpips_unavailable_reason', 'unknown reason')})")
    if "val_depth_rmse_m_raw" in final_metrics:
        print(f"  depth_rmse_m_raw: {final_metrics['val_depth_rmse_m_raw']:.6f}")
        print(f"  depth_mae_m_raw:  {final_metrics['val_depth_mae_m_raw']:.6f}")
    print(f"  outputs: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
