#!/bin/bash
# EEG RDE-GPR 对齐基线实验
# 基线设置: history_timesteps=976, horizon_steps=24, target_dims=0,1,2, 前馈
# 默认 trainlength=500 作为 GP 合理上限，可用 TL= 环境变量覆盖
set -e
TL=${TL:-500}
NJ=${NJ:-4}
MAXD=${MAXD:-20}
OUT_DIR=/home/rhl/Github/experiments_v2/eeg/rdegpr_tl${TL}
LOG_DIR=/home/rhl/Github/experiments_v2/logs
mkdir -p ${OUT_DIR} ${LOG_DIR}

TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/eeg_rdegpr_aligned_tl${TL}_${TS}.log"

echo "============================================" | tee -a ${LOG_FILE}
echo "  EEG RDE-GPR 对齐基线" | tee -a ${LOG_FILE}
echo "  history_timesteps=976, horizon_steps=24, target_dims=0,1,2" | tee -a ${LOG_FILE}
echo "  trainlength=${TL}, max_delay=${MAXD}" | tee -a ${LOG_FILE}
echo "  开始: $(date)" | tee -a ${LOG_FILE}
echo "============================================" | tee -a ${LOG_FILE}

python3 /home/rhl/Github/rde_gpr/eeg_CSDIimpute_after-RDEgpr.py \
  --imputed_path /home/rhl/Github/save/eeg_csdi_imputed/eeg_full.npy \
  --ground_path /home/rhl/Github/save/eeg_csdi_imputed/eeg_full.npy \
  --history_timesteps 976 --horizon_steps 24 \
  --target_indices "0,1,2" \
  --L 4 --s 50 --trainlength ${TL} --n_jobs ${NJ} \
  --use_delay_embedding --max_delay ${MAXD} \
  --out_dir ${OUT_DIR} 2>&1 | tee -a ${LOG_FILE} | tail -30

echo "" | tee -a ${LOG_FILE}
echo "  完成: $(date)" | tee -a ${LOG_FILE}
