#!/bin/bash
# V2: PM2.5 基线实验
# 3个基线: NeuralCDE, GRU-ODE-Bayes, SSSD
# 使用CSDI补值后的历史数据

set -e

SEED=42
EPOCHS=100
BATCH=128
LR=1e-3
WINDOW=48
HIDDEN=64
DEVICE0="cuda:0"
DEVICE1="cuda:1"

GROUND_PATH="./data/pm25/Code/STMVL/SampleData/pm25_ground.txt"
SPLIT_RATIO=0.5
HORIZON_DAYS=1

IMPUTED_DIR=$(ls -d ./save/pm25_history_imputed_split${SPLIT_RATIO}_seed${SEED}_* 2>/dev/null | head -1)
if [ -z "${IMPUTED_DIR}" ]; then
    echo "❌ 找不到补值结果"
    exit 1
fi
IMPUTED_HISTORY_PATH="${IMPUTED_DIR}/history_imputed.csv"
echo "✅ 补值结果: ${IMPUTED_DIR}"

OUT_BASE="./experiments_v1/pm25"
mkdir -p ${OUT_BASE}/{neuralcde,gruodebayes,sssd}

echo "================================================================"
echo "  V2: PM2.5 基线实验"
echo "================================================================"

echo ""
echo "=== PM2.5: NeuralCDE (cuda:0) ==="
python baselines/neuralcde_forecast.py \
    --dataset pm25 \
    --imputed_history_path "${IMPUTED_HISTORY_PATH}" \
    --ground_path "${GROUND_PATH}" \
    --split_ratio ${SPLIT_RATIO} --horizon_days ${HORIZON_DAYS} \
    --window_size ${WINDOW} --hidden_channels ${HIDDEN} --num_layers 3 \
    --epochs ${EPOCHS} --batch_size ${BATCH} --lr ${LR} --seed ${SEED} \
    --device ${DEVICE0} \
    --out_dir ${OUT_BASE}/neuralcde

echo ""
echo "=== PM2.5: GRU-ODE-Bayes (cuda:0) ==="
python baselines/gruodebayes_forecast.py \
    --dataset pm25 \
    --imputed_history_path "${IMPUTED_HISTORY_PATH}" \
    --ground_path "${GROUND_PATH}" \
    --split_ratio ${SPLIT_RATIO} --horizon_days ${HORIZON_DAYS} \
    --window_size ${WINDOW} --hidden_size ${HIDDEN} \
    --p_hidden 32 --prep_hidden 32 \
    --delta_t 0.1 --time_scale 0.02 --solver euler \
    --epochs ${EPOCHS} --batch_size ${BATCH} --lr ${LR} --seed ${SEED} \
    --device ${DEVICE0} \
    --out_dir ${OUT_BASE}/gruodebayes

echo ""
echo "=== PM2.5: SSSD (cuda:1) ==="
python baselines/sssd_forecast.py \
    --dataset pm25 \
    --imputed_history_path "${IMPUTED_HISTORY_PATH}" \
    --ground_path "${GROUND_PATH}" \
    --split_ratio ${SPLIT_RATIO} --horizon_days ${HORIZON_DAYS} \
    --d_model ${HIDDEN} --n_layers 4 --diffusion_steps 100 \
    --window_size ${WINDOW} \
    --epochs ${EPOCHS} --batch_size 16 --lr 1e-4 --seed ${SEED} \
    --device ${DEVICE1} \
    --out_dir ${OUT_BASE}/sssd

echo ""
echo "================================================================"
echo "  🎉 V2 完成！结果在 ${OUT_BASE}/"
echo "================================================================"
