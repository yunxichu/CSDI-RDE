#!/bin/bash
# L96 CSDI long-run training: 1M data × 25-50 epochs × 4 variants + extra seed on GPU 0-4.
#
# L96 N=20 F=8 attractor_std = 3.639
# Expected total steps: ~100K-200K (vs L63 CSDI's 40K @ best ep20)
#
# Usage: bash experiments/week2_modules/run_csdi_l96_longrun.sh [TAG]
#   TAG default: l96_v1

set -euo pipefail
TAG="${1:-l96_v1}"
EPOCHS_SHORT="${2:-25}"
EPOCHS_LONG="${3:-50}"
BATCH="${4:-256}"
LR="${5:-5e-4}"
CHANNELS="${6:-128}"
LAYERS="${7:-8}"
SAVE_EVERY="${8:-5}"

CACHE="experiments/week2_modules/data/lorenz96_clean_1M_L128_N20.npz"
LOGDIR="experiments/week2_modules/results"
mkdir -p "$LOGDIR"

if [ ! -f "$CACHE" ]; then
    echo "[ERROR] Dataset not found: $CACHE"
    echo "  Run first: python -m experiments.week2_modules.make_lorenz96_dataset --n_samples 1000000 --N 20 --tag 1M_L128_N20 --n_workers 24"
    exit 1
fi

CACHE_SIZE=$(du -sh "$CACHE" | cut -f1)
echo "=== CSDI-L96 long-run training (5 GPUs) ==="
echo "  tag=${TAG}  epochs=${EPOCHS_SHORT}/${EPOCHS_LONG}  batch=${BATCH}  lr=${LR}  ch=${CHANNELS}  layers=${LAYERS}"
echo "  save_every=${SAVE_EVERY} epochs"
echo "  cache=${CACHE}  (${CACHE_SIZE})"
echo "  target steps @ 25ep = $(python -c "print(f'{1000000//${BATCH}*${EPOCHS_SHORT}:,}')")"
echo "  target steps @ 50ep = $(python -c "print(f'{1000000//${BATCH}*${EPOCHS_LONG}:,}')")"
echo "  logs -> ${LOGDIR}/csdi_l96_{variant}_{tag}.log"

launch() {
    local gpu=$1 variant=$2 seed=$3 epochs=$4 extra_tag=$5
    local tag="l96_${variant}${extra_tag}_${TAG}"
    local logfile="${LOGDIR}/csdi_l96_${tag}.log"
    echo "[launch] GPU ${gpu}  variant=${variant}  seed=${seed}  epochs=${epochs}  -> ${logfile}"
    CUDA_VISIBLE_DEVICES=${gpu} nohup python -u -m experiments.week2_modules.train_dynamics_csdi \
        --variant ${variant} \
        --epochs ${epochs} \
        --n_samples 1000000 \
        --batch_size ${BATCH} \
        --seq_len 128 \
        --channels ${CHANNELS} \
        --n_layers ${LAYERS} \
        --lr ${LR} \
        --cache_path ${CACHE} \
        --save_every ${SAVE_EVERY} \
        --seed ${seed} \
        --tag ${tag} \
        --data_dim 20 \
        --attractor_std 3.639 \
        --system lorenz96 \
        --eval_N 20 \
        --eval_F 8.0 \
        --eval_dt 0.05 \
        > "${logfile}" 2>&1 &
    echo "  pid=$!"
}

launch 0 full     42   ${EPOCHS_SHORT} ""           # main ckpt for downstream L96 PT eval
launch 1 no_mask  42   ${EPOCHS_SHORT} ""           # ablation: no delay-attention
launch 2 no_noise 42   ${EPOCHS_SHORT} ""           # ablation: no noise conditioning
launch 3 vanilla  42   ${EPOCHS_SHORT} ""           # ablation: plain CSDI baseline
launch 4 full     1337 ${EPOCHS_LONG}  "_s1337"     # extra seed + longer for ceiling ckpt

echo
echo "Monitor: tail -f ${LOGDIR}/csdi_l96_*_${TAG}.log"
echo "GPU util: watch -n 5 nvidia-smi"
echo "Wait with: wait"
wait
echo "=== all 5 L96 training jobs finished ==="
