#!/bin/bash
# 跑 Lorenz63 剩余 4 组 (43-46) + Lorenz96 全 5 组 (42-46)
# 两个循环串行，但 Lorenz63 和 Lorenz96 之间并行（通过 & 分叉）
# BLAS 线程 = 1，每循环 n_jobs=4 → 共 ~8 核
set -u
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

LOG_DIR=/home/rhl/Github/experiments_v2/logs
mkdir -p ${LOG_DIR}

run_seeds() {
  local DATASET=$1; shift
  local SEEDS="$@"
  local SCRIPT_DIR OUT_ROOT
  if [ "${DATASET}" = "lorenz63" ]; then
    SCRIPT_DIR=/home/rhl/Github/lorenz_rde_delay
    OUT_ROOT=/home/rhl/Github/experiments_v2/lorenz63/rde_delay
  else
    SCRIPT_DIR=/home/rhl/Github/lorenz96_rde_delay
    OUT_ROOT=/home/rhl/Github/experiments_v2/lorenz96/rde_delay
  fi
  mkdir -p ${OUT_ROOT}
  for S in ${SEEDS}; do
    TS=$(date +%Y%m%d_%H%M%S)
    LOG="${LOG_DIR}/${DATASET}_rde_delay_seed${S}_${TS}.log"
    echo "==== [${DATASET}] seed=${S} -> ${LOG}"
    cd ${SCRIPT_DIR}
    python3 -u inference/eval_aligned.py --seed ${S} --out_root ${OUT_ROOT} --n_jobs 4 > ${LOG} 2>&1
    echo "==== [${DATASET}] seed=${S} done"
  done
}

run_seeds lorenz63 43 44 45 46 &
L63_PID=$!
run_seeds lorenz96 42 43 44 45 46 &
L96_PID=$!

wait ${L63_PID}
echo "Lorenz63 (43-46) all done."
wait ${L96_PID}
echo "Lorenz96 (42-46) all done."
echo "All 9 remaining runs complete."
