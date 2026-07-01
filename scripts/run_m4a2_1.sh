#!/usr/bin/env bash
set -euo pipefail
export PATH=/usr/bin:/usr/local/cuda/bin:/home/omers/miniconda3/envs/surgtwin/bin:${PATH}
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
cd /home/omers/SurgTwin-GS

RUN_DIR="/home/omers/SurgTwin-GS/outputs/runs/m4_a2_1"

exec nohup /home/omers/miniconda3/envs/surgtwin/bin/python \
    /home/omers/SurgTwin-GS/scripts/train_uncertainty.py \
    --manifest "/home/omers/SurgTwin-GS/data/processed/manifests/servct_manifest.jsonl" \
    --output_dir "${RUN_DIR}" \
    --init_num_points 20000 \
    --iterations 500 \
    --warmup_iters 200 \
    --lr_opacities 1e-2 \
    --max_grad_norm 1.5 \
    --variant h1 \
    --lambda_depth 0.2 \
    --lambda_reg 0.0 \
    --depth_semantics_artifact "/home/omers/SurgTwin-GS/outputs/runs/m2a_gate/final_gate_decision.json" \
    --log_every 10 \
    --val_every 50 \
    --enable_densification \
    --densify_from_iter 200 \
    --densify_every 100 \
    --densify_until_iter 450 \
    --densify_depth_residual_threshold 0.02 \
    --densify_w_photo_threshold 0.3 \
    --densify_max_clone_per_step 5000 \
    --densify_max_clone_fraction 0.15 \
    --densify_max_gaussians 50000 \
    --prune_min_opacity 0.005 \
    --max_prune_fraction_per_step 0.05 \
    --clone_offset_scale_factor 0.25 \
    2>&1 | tee "${RUN_DIR}/train.log"
echo "Training completed (exit code $?). PID file not written (foreground run)."

# Evaluate gate
/home/omers/miniconda3/envs/surgtwin/bin/python \
    /home/omers/SurgTwin-GS/scripts/evaluate_m4_a2_1.py \
    --run_dir "${RUN_DIR}" \
    || echo "Gate evaluation completed (non-zero exit is expected for non-FULL_SUCCESS)."

echo "=== M4-A2-1 run complete ==="
echo "  Train output: ${RUN_DIR}"
echo "  Train log: ${RUN_DIR}/train.log"
echo "  Gate: ${RUN_DIR}/m4_a2_1_gate.json"
