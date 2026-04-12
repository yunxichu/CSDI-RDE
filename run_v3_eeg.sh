#!/bin/bash
# V3: EEG 基线实验
# 前置条件：需要先运行CSDI补值生成 imputed 数据
# 3个基线: NeuralCDE, GRU-ODE-Bayes, SSSD
# target_dims: 前3个通道 (0,1,2)

set -e

SEED=42
EPOCHS=100
BATCH=128
LR=1e-3
WINDOW=48
HIDDEN=64
DEVICE0="cuda:0"
DEVICE1="cuda:1"

GROUND_PATH="./data/eeg/eeg_ground.npy"
IMPUTED_PATH="./save/eeg_csdi_imputed.npy"
HISTORY_TIMESTEPS=100
HORIZON_STEPS=24
TARGET_DIMS="0,1,2"

# 检查补值数据是否存在
if [ ! -f "${IMPUTED_PATH}" ]; then
    echo "⚠️  EEG补值数据不存在，尝试使用线性填充数据"
    IMPUTED_PATH="./data/eeg/eeg_linear_filled.npy"
    if [ ! -f "${IMPUTED_PATH}" ]; then
        echo "❌ 找不到EEG补值数据，请先运行CSDI补值"
        echo "运行: python experiments/exe_eeg.py --device cuda:0"
        exit 1
    fi
fi

OUT_BASE="./experiments_v1/eeg"
mkdir -p ${OUT_BASE}/{neuralcde,gruodebayes,sssd}

echo "================================================================"
echo "  V3: EEG 基线实验"
echo "================================================================"
echo "  Ground: ${GROUND_PATH}"
echo "  Imputed: ${IMPUTED_PATH}"
echo "  Target dims: ${TARGET_DIMS}"
echo "================================================================"

echo ""
echo "=== EEG: NeuralCDE (cuda:0) ==="
python baselines/neuralcde_forecast.py \
    --dataset eeg \
    --imputed_path "${IMPUTED_PATH}" \
    --ground_path "${GROUND_PATH}" \
    --history_timesteps ${HISTORY_TIMESTEPS} --horizon_steps ${HORIZON_STEPS} \
    --target_dims ${TARGET_DIMS} \
    --window_size ${WINDOW} --hidden_channels ${HIDDEN} --num_layers 3 \
    --epochs ${EPOCHS} --batch_size ${BATCH} --lr ${LR} --seed ${SEED} \
    --device ${DEVICE0} \
    --out_dir ${OUT_BASE}/neuralcde

echo ""
echo "=== EEG: GRU-ODE-Bayes (cuda:0) ==="
python baselines/gruodebayes_forecast.py \
    --dataset eeg \
    --imputed_path "${IMPUTED_PATH}" \
    --ground_path "${GROUND_PATH}" \
    --history_timesteps ${HISTORY_TIMESTEPS} --horizon_steps ${HORIZON_STEPS} \
    --target_dims ${TARGET_DIMS} \
    --window_size ${WINDOW} --hidden_size ${HIDDEN} \
    --p_hidden 32 --prep_hidden 32 \
    --delta_t 0.1 --time_scale 0.02 --solver euler \
    --epochs ${EPOCHS} --batch_size ${BATCH} --lr ${LR} --seed ${SEED} \
    --device ${DEVICE0} \
    --out_dir ${OUT_BASE}/gruodebayes

echo ""
echo "=== EEG: SSSD (cuda:1) ==="
python baselines/sssd_forecast.py \
    --dataset eeg \
    --imputed_path "${IMPUTED_PATH}" \
    --ground_path "${GROUND_PATH}" \
    --history_timesteps ${HISTORY_TIMESTEPS} --horizon_steps ${HORIZON_STEPS} \
    --target_dims ${TARGET_DIMS} \
    --d_model ${HIDDEN} --n_layers 4 --diffusion_steps 50 \
    --window_size ${WINDOW} \
    --epochs ${EPOCHS} --batch_size 16 --lr 1e-4 --seed ${SEED} \
    --device ${DEVICE1} \
    --out_dir ${OUT_BASE}/sssd

echo ""
echo "================================================================"
echo "  🎉 V3 完成！结果在 ${OUT_BASE}/"
echo "================================================================"
