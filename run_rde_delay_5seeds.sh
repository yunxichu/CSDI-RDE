#!/bin/bash
# 5 组 seed 循环跑 Lorenz63 / Lorenz96 的 RDE-Delay (CSDI补值后预测, horizon=40)
# 结果输出到 experiments_v2/<dataset>/rde_delay/run_<timestamp>_seed<N>/metrics.json
#
# 用法:
#   bash run_rde_delay_5seeds.sh lorenz63    # 跑 Lorenz63 5 组
#   bash run_rde_delay_5seeds.sh lorenz96    # 跑 Lorenz96 5 组

set -e
DATASET=${1:-lorenz63}
SEEDS="42 43 44 45 46"
N_JOBS=4
LOG_DIR=/home/rhl/Github/experiments_v2/logs

# 限制 BLAS 内部线程，避免 n_jobs=4 × 每进程 N 线程 变成数十核并发
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p ${LOG_DIR}

if [ "${DATASET}" = "lorenz63" ]; then
  SCRIPT_DIR=/home/rhl/Github/lorenz_rde_delay
  OUT_ROOT=/home/rhl/Github/experiments_v2/lorenz63/rde_delay
elif [ "${DATASET}" = "lorenz96" ]; then
  SCRIPT_DIR=/home/rhl/Github/lorenz96_rde_delay
  OUT_ROOT=/home/rhl/Github/experiments_v2/lorenz96/rde_delay
else
  echo "Unknown dataset: ${DATASET}"; exit 1
fi

mkdir -p ${OUT_ROOT}

for S in ${SEEDS}; do
  TS=$(date +%Y%m%d_%H%M%S)
  LOG="${LOG_DIR}/${DATASET}_rde_delay_seed${S}_${TS}.log"
  echo "==== ${DATASET} RDE-Delay seed=${S} → ${LOG}"
  cd ${SCRIPT_DIR}
  python3 -u inference/eval_aligned.py \
    --seed ${S} --out_root ${OUT_ROOT} --n_jobs ${N_JOBS} 2>&1 | tee ${LOG}
  echo ""
done

echo "All 5 seeds done for ${DATASET}. Results in: ${OUT_ROOT}"
