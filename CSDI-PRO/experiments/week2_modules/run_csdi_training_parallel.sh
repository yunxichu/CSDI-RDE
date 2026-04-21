#!/bin/bash
# Launch 4 CSDI variants in parallel on GPU 0-3.
# Logs go to results/csdi_run_<variant>_<tag>.log
#
# Usage: bash experiments/week2_modules/run_csdi_training_parallel.sh [TAG] [EPOCHS] [N_SAMPLES]

set -euo pipefail
TAG="${1:-v3_big}"
EPOCHS="${2:-60}"
N_SAMPLES="${3:-32000}"
BATCH="${4:-128}"
CHANNELS="${5:-128}"
LAYERS="${6:-8}"

CACHE="experiments/week2_modules/data/lorenz63_clean_64k_L128.npz"
LOGDIR="experiments/week2_modules/results"
mkdir -p "$LOGDIR"

echo "=== CSDI parallel training ==="
echo "  tag=${TAG}  epochs=${EPOCHS}  n_samples=${N_SAMPLES}  batch=${BATCH}  ch=${CHANNELS} layers=${LAYERS}"
echo "  cache=${CACHE}"
echo "  logs -> ${LOGDIR}/csdi_run_{variant}_${TAG}.log"

launch() {
    local gpu=$1 variant=$2
    local tag="${variant}_${TAG}"
    local logfile="${LOGDIR}/csdi_run_${tag}.log"
    echo "[launch] GPU ${gpu}  variant=${variant}  -> ${logfile}"
    CUDA_VISIBLE_DEVICES=${gpu} nohup python -u -m experiments.week2_modules.train_dynamics_csdi \
        --variant ${variant} \
        --epochs ${EPOCHS} \
        --n_samples ${N_SAMPLES} \
        --batch_size ${BATCH} \
        --seq_len 128 \
        --channels ${CHANNELS} \
        --n_layers ${LAYERS} \
        --cache_path ${CACHE} \
        --tag ${tag} \
        > "${logfile}" 2>&1 &
    echo "  pid=$!"
}

launch 0 full
launch 1 no_noise
launch 2 no_mask
launch 3 vanilla

echo
echo "Wait with: wait"
echo "Tail all logs: tail -f ${LOGDIR}/csdi_run_*_${TAG}.log"
wait
echo "=== all 4 variants finished ==="
