#!/bin/bash
# PM2.5 SSSD 训练脚本 - GPU 7
# 增加训练轮数和patience确保收敛到极限

set -e
GPU=7
DEV="cuda:${GPU}"
LOG_DIR="./experiments_v2/logs"
OUT_DIR="./experiments_v2/pm25/sssd_v2"

mkdir -p ${LOG_DIR} ${OUT_DIR}

LOG_FILE="${LOG_DIR}/pm25_sssd_v2_$(date +%Y%m%d_%H%M%S).log"

echo "============================================" | tee -a ${LOG_FILE}
echo "  PM2.5 SSSD 训练 - GPU ${GPU}" | tee -a ${LOG_FILE}
echo "  开始时间: $(date)" | tee -a ${LOG_FILE}
echo "============================================" | tee -a ${LOG_FILE}

# 关键改进：
# 1. epochs: 100 -> 500 (充分训练)
# 2. patience: 15 -> 100 (给模型更多机会)
# 3. diffusion_steps: 50 -> 100 (更精细的扩散过程)

python3 baselines/sssd_forecast.py \
    --dataset pm25 \
    --imputed_history_path ./save/pm25_history_imputed_split0.5_seed42_20260324_154400/history_imputed.csv \
    --ground_path ./data/pm25/Code/STMVL/SampleData/pm25_ground.txt \
    --split_ratio 0.5 --horizon_days 1 \
    --d_model 64 --n_layers 4 --diffusion_steps 100 \
    --window_size 48 --epochs 500 --batch_size 16 --lr 1e-4 --seed 42 \
    --patience 100 \
    --device ${DEV} --out_dir ${OUT_DIR} 2>&1 | tee -a ${LOG_FILE}

echo "" | tee -a ${LOG_FILE}
echo "============================================" | tee -a ${LOG_FILE}
echo "  完成时间: $(date)" | tee -a ${LOG_FILE}
echo "  日志: ${LOG_FILE}" | tee -a ${LOG_FILE}
echo "============================================" | tee -a ${LOG_FILE}
