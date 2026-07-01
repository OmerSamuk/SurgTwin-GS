#!/usr/bin/env bash
set -euo pipefail
export PATH=/usr/bin:/usr/local/cuda/bin:/home/omers/miniconda3/envs/surgtwin/bin:${PATH}
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
cd /home/omers/SurgTwin-GS
rm -rf outputs/runs/m4_a2_0_20k
exec nohup /home/omers/miniconda3/envs/surgtwin/bin/python /home/omers/SurgTwin-GS/scripts/train_uncertainty.py \
    --manifest /home/omers/SurgTwin-GS/data/processed/manifests/servct_manifest.jsonl \
    --output_dir /home/omers/SurgTwin-GS/outputs/runs/m4_a2_0_20k \
    --init_num_points 20000 \
    --iterations 200 \
    --warmup_iters 200 \
    --lr_opacities 1e-2 \
    --max_grad_norm 1.5 \
    --variant h1 \
    --lambda_depth 0.2 \
    --lambda_reg 0.0 \
    --depth_semantics_artifact /home/omers/SurgTwin-GS/outputs/runs/m2a_gate/final_gate_decision.json \
    --log_every 10 \
    --val_every 50 \
    > /home/omers/SurgTwin-GS/outputs/runs/m4_a2_0_20k_train.log 2>&1 &
echo $! > /home/omers/SurgTwin-GS/outputs/runs/m4_a2_0_20k_train.pid
echo "Launched M4-A2-0 20K smoke training PID=$!"
echo "NOTE: First run — no --overwrite. If rerun needed, add --overwrite explicitly."
