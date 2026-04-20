#!/bin/bash
# PM2.5 RDE-GPR 对齐基线实验
# 基线设置: history=4379 (split_ratio=0.5), horizon_steps=24, target=全 36 站
# GP O(n^3) 无法对 trainlength=4379 直接硬跑，合理上限 ~500-1000
# 选 trainlength=500 作为首选 (GP 可接受的计算量)
set -e
TL=${TL:-500}
NJ=${NJ:-4}
OUT_DIR=/home/rhl/Github/experiments_v2/pm25/rdegpr_tl${TL}
LOG_DIR=/home/rhl/Github/experiments_v2/logs
mkdir -p ${OUT_DIR} ${LOG_DIR}

TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/pm25_rdegpr_aligned_tl${TL}_${TS}.log"

echo "============================================" | tee -a ${LOG_FILE}
echo "  PM2.5 RDE-GPR 对齐基线" | tee -a ${LOG_FILE}
echo "  trainlength=${TL}, horizon_steps=24, 全 36 站" | tee -a ${LOG_FILE}
echo "  开始: $(date)" | tee -a ${LOG_FILE}
echo "============================================" | tee -a ${LOG_FILE}

python3 /home/rhl/Github/rde_gpr/pm25_CSDIimpute_after-RDEgpr.py \
  --imputed_history_path /home/rhl/Github/save/pm25_history_imputed_split0.5_seed42_20260324_154400/history_imputed.csv \
  --ground_path /home/rhl/Github/data/pm25/Code/STMVL/SampleData/pm25_ground.txt \
  --split_ratio 0.5 --horizon_steps 24 \
  --L 4 --s 50 --trainlength ${TL} --n_jobs ${NJ} \
  --target_indices "" \
  --out_dir ${OUT_DIR} 2>&1 | tee -a ${LOG_FILE} | tail -30

echo "" | tee -a ${LOG_FILE}
echo "  完成: $(date)" | tee -a ${LOG_FILE}
