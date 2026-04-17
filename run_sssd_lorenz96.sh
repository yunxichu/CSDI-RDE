#!/bin/bash
# Lorenz96 SSSD v2 训练（mask 修复版）- 与 Lorenz63 sssd_v2 保持一致参数
set -e
GPU=${GPU:-3}
DEV="cuda:${GPU}"
LOG_DIR="./experiments_v2/logs"
OUT_DIR="./experiments_v2/lorenz96/sssd_v2"

mkdir -p ${LOG_DIR} ${OUT_DIR}
LOG_FILE="${LOG_DIR}/lorenz96_sssd_v2_$(date +%Y%m%d_%H%M%S).log"

echo "============================================" | tee -a ${LOG_FILE}
echo "  Lorenz96 SSSD v2 训练 - GPU ${GPU}" | tee -a ${LOG_FILE}
echo "  开始时间: $(date)" | tee -a ${LOG_FILE}
echo "  对齐基线: trainlength=60, horizon_steps=40" | tee -a ${LOG_FILE}
echo "============================================" | tee -a ${LOG_FILE}

python3 baselines/sssd_forecast.py \
    --dataset lorenz96 \
    --ground_path ./lorenz96_rde_delay/results/gt_100_20260323_192045.csv \
    --data_path ./lorenz96_rde_delay/results/imputed_100_20260323_192045.csv \
    --trainlength 60 --horizon_steps 40 \
    --d_model 64 --n_layers 4 --diffusion_steps 100 \
    --window_size 20 --epochs 500 --batch_size 16 --lr 1e-4 --seed 42 \
    --patience 100 \
    --device ${DEV} --out_dir ${OUT_DIR} 2>&1 | tee -a ${LOG_FILE}

echo "" | tee -a ${LOG_FILE}
echo "============================================" | tee -a ${LOG_FILE}
echo "  完成时间: $(date)" | tee -a ${LOG_FILE}
echo "============================================" | tee -a ${LOG_FILE}
