import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from surgtwin.data.manifest import load_manifest, filter_by_split, validate_manifest_entry
from surgtwin.gaussian.backend_gsplat import GsplatBackend
from surgtwin.training.config import BaselineConfig
from surgtwin.training.trainer import BaselineTrainer


def main():
    parser = argparse.ArgumentParser(description="Milestone 1 baseline training (RGB-only L1).")
    parser.add_argument("--manifest", type=str, required=True, help="Path to manifest JSONL")
    parser.add_argument("--output_dir", type=str, default="outputs/runs/baseline_debug", help="Output directory")
    parser.add_argument("--iterations", type=int, default=1000, help="Number of training iterations")
    parser.add_argument("--init_num_points", type=int, default=20000, help="Initial Gaussian count")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--lr_means", type=float, default=1e-3, help="Learning rate for means")
    parser.add_argument("--lr_opacities", type=float, default=5e-2, help="Learning rate for opacities")
    parser.add_argument("--val_every", type=int, default=100, help="Validation interval")
    parser.add_argument("--log_every", type=int, default=10, help="Logging interval")
    parser.add_argument("--ckpt_every", type=int, default=500, help="Checkpoint interval")
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

    config = BaselineConfig(
        iterations=args.iterations,
        init_num_points=args.init_num_points,
        seed=args.seed,
        lr_means=args.lr_means,
        lr_opacities=args.lr_opacities,
        val_every=args.val_every,
        log_every=args.log_every,
        ckpt_every=args.ckpt_every,
    )

    backend = GsplatBackend()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trainer = BaselineTrainer(
        train_entries=train,
        val_entries=val,
        backend=backend,
        config=config,
        output_dir=output_dir,
    )

    final_metrics = trainer.fit()

    print("\n=== Milestone 1 Complete ===")
    print(f"  initial_loss: {final_metrics['initial_loss']:.6f}")
    print(f"  final_loss:   {final_metrics['final_loss']:.6f}")
    print(f"  loss_decreased: {final_metrics['loss_decreased']}")
    print(f"  val_psnr: {final_metrics['val_psnr']:.4f}")
    print(f"  val_ssim: {final_metrics['val_ssim']:.4f}")
    if final_metrics.get("val_lpips") is not None:
        print(f"  val_lpips: {final_metrics['val_lpips']:.4f}")
    else:
        print(f"  val_lpips: unavailable ({final_metrics.get('val_lpips_unavailable_reason', 'unknown reason')})")
    print(f"  outputs: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
