#!/bin/bash
# EEG SSSD 训练脚本 - GPU 3
# 增加训练轮数和patience确保收敛到极限

set -e
GPU=3
DEV="cuda:${GPU}"
LOG_DIR="./experiments_v2/logs"
OUT_DIR="./experiments_v2/eeg/sssd_v2"

mkdir -p ${LOG_DIR} ${OUT_DIR}

LOG_FILE="${LOG_DIR}/eeg_sssd_v2_$(date +%Y%m%d_%H%M%S).log"

echo "============================================" | tee -a ${LOG_FILE}
echo "  EEG SSSD 训练 - GPU ${GPU}" | tee -a ${LOG_FILE}
echo "  开始时间: $(date)" | tee -a ${LOG_FILE}
echo "============================================" | tee -a ${LOG_FILE}

# 关键改进：
# 1. epochs: 100 -> 500 (充分训练)
# 2. patience: 15 -> 100 (给模型更多机会)
# 3. diffusion_steps: 50 -> 100 (更精细的扩散过程)

python3 baselines/sssd_forecast.py \
    --dataset eeg \
    --imputed_path ./save/eeg_csdi_imputed/eeg_full.npy \
    --ground_path ./save/eeg_csdi_imputed/eeg_full.npy \
    --history_timesteps 976 --horizon_steps 24 \
    --target_dims 0,1,2 \
    --d_model 64 --n_layers 4 --diffusion_steps 100 \
    --window_size 48 --epochs 500 --batch_size 16 --lr 1e-4 --seed 42 \
    --patience 100 \
    --device ${DEV} --out_dir ${OUT_DIR} 2>&1 | tee -a ${LOG_FILE}

echo "" | tee -a ${LOG_FILE}
echo "============================================" | tee -a ${LOG_FILE}
echo "  完成时间: $(date)" | tee -a ${LOG_FILE}
echo "  日志: ${LOG_FILE}" | tee -a ${LOG_FILE}
echo "============================================" | tee -a ${LOG_FILE}
