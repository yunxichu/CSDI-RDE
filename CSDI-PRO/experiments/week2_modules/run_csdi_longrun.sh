#!/bin/bash
# Long-run CSDI training: 512K data × 200 epochs × 4 variants on GPU 0-3.
# Target: ~400K gradient steps (vs 15K in v3) to reach useful DDPM quality.
#
# Usage: bash experiments/week2_modules/run_csdi_longrun.sh [TAG] [EPOCHS] [BATCH]
#   TAG    default: v5_long
#   EPOCHS default: 200
#   BATCH  default: 256

set -euo pipefail
TAG="${1:-v5_long}"
EPOCHS="${2:-200}"
BATCH="${3:-256}"
LR="${4:-5e-4}"
CHANNELS="${5:-128}"
LAYERS="${6:-8}"
SAVE_EVERY="${7:-50}"

CACHE="experiments/week2_modules/data/lorenz63_clean_512k_L128.npz"
LOGDIR="experiments/week2_modules/results"
mkdir -p "$LOGDIR"

# Verify dataset exists
if [ ! -f "$CACHE" ]; then
    echo "[ERROR] Dataset not found: $CACHE"
    echo "  Run first: python -m experiments.week2_modules.make_lorenz_dataset --n_samples 512000 --n_workers 8 --tag 512k_L128"
    exit 1
fi

CACHE_SIZE=$(du -sh "$CACHE" | cut -f1)
echo "=== CSDI long-run training ==="
echo "  tag=${TAG}  epochs=${EPOCHS}  batch=${BATCH}  lr=${LR}  ch=${CHANNELS}  layers=${LAYERS}"
echo "  save_every=${SAVE_EVERY} epochs"
echo "  cache=${CACHE}  (${CACHE_SIZE})"
echo "  target steps = $(python -c "print(f'{512000//${BATCH}*${EPOCHS}:,}')")"
echo "  logs -> ${LOGDIR}/csdi_longrun_{variant}_${TAG}.log"

launch() {
    local gpu=$1 variant=$2 seed=$3
    local tag="${variant}_${TAG}"
    local logfile="${LOGDIR}/csdi_longrun_${tag}.log"
    echo "[launch] GPU ${gpu}  variant=${variant}  seed=${seed}  -> ${logfile}"
    CUDA_VISIBLE_DEVICES=${gpu} nohup python -u -m experiments.week2_modules.train_dynamics_csdi \
        --variant ${variant} \
        --epochs ${EPOCHS} \
        --n_samples 512000 \
        --batch_size ${BATCH} \
        --seq_len 128 \
        --channels ${CHANNELS} \
        --n_layers ${LAYERS} \
        --lr ${LR} \
        --cache_path ${CACHE} \
        --save_every ${SAVE_EVERY} \
        --seed ${seed} \
        --tag ${tag} \
        > "${logfile}" 2>&1 &
    echo "  pid=$!"
}

launch 0 full     42
launch 1 no_mask  42
launch 2 no_noise 42
launch 3 vanilla  42

echo
echo "Monitor: tail -f ${LOGDIR}/csdi_longrun_*_${TAG}.log"
echo "Wait with: wait"
wait
echo "=== all 4 variants finished ==="
