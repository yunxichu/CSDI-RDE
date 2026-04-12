#!/bin/bash
# PM2.5 全基线对比一键运行脚本
# 包含：RDE-GPR (全维度) + GRU + LSTM + NeuralCDE + GRU-ODE-Bayes
# 每个方法运行后自动整理结果到 ./save/pm25_comparison/

set -e

GROUND_PATH="./data/pm25/Code/STMVL/SampleData/pm25_ground.txt"
SPLIT_RATIO=0.5
SEED=42
HORIZON_DAYS=1
DEVICE="cuda:0"

# 找到补值结果
IMPUTED_DIR=$(ls -d ./save/pm25_history_imputed_split${SPLIT_RATIO}_seed${SEED}_* 2>/dev/null | head -1)
if [ -z "${IMPUTED_DIR}" ]; then
    echo "❌ 找不到补值结果，请先运行CSDI补值"
    echo "运行: python imputation/pm25_CSDIimpute_3.py ..."
    exit 1
fi
IMPUTED_HISTORY_PATH="${IMPUTED_DIR}/history_imputed.csv"
echo "✅ 找到补值结果: ${IMPUTED_DIR}"

# 创建对比目录
CMP_DIR="./save/pm25_comparison"
mkdir -p ${CMP_DIR}/{rdegpr,gru,lstm,neuralcde,gruodebayes}

echo ""
echo "================================================================"
echo "  PM2.5 全基线对比实验"
echo "================================================================"
echo "  Ground: ${GROUND_PATH}"
echo "  History: ${IMPUTED_HISTORY_PATH}"
echo "  Split: ${SPLIT_RATIO}, Horizon: ${HORIZON_DAYS}天, Seed: ${SEED}"
echo "================================================================"

# ---- Step 1: RDE-GPR 全维度 ----
echo ""
echo "=== [1/5] RDE-GPR 全维度预测 ==="
python rde_gpr/pm25_CSDIimpute_after-RDEgpr.py \
    --imputed_history_path "${IMPUTED_HISTORY_PATH}" \
    --ground_path "${GROUND_PATH}" \
    --split_ratio ${SPLIT_RATIO} \
    --horizon_days ${HORIZON_DAYS} \
    --L 4 --s 50 --trainlength 4000 --n_jobs 8 \
    --target_indices "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35" \
    --history_timesteps 72 \
    --seed ${SEED}

RDE_DIR=$(ls -td ./save/pm25_test_plot_with_history_* 2>/dev/null | head -1)
if [ -n "${RDE_DIR}" ] && [ -f "${RDE_DIR}/future_pred.csv" ]; then
    cp "${RDE_DIR}/future_pred.csv" ${CMP_DIR}/rdegpr/
    cp "${RDE_DIR}/future_pred_std.csv" ${CMP_DIR}/rdegpr/ 2>/dev/null || true
    echo "✅ RDE-GPR 结果已复制"
else
    echo "⚠️  RDE-GPR 结果未找到"
fi

# ---- Step 2: GRU ----
echo ""
echo "=== [2/5] GRU 基线 ==="
python baselines/gru_lstm_forecast.py \
    --dataset pm25 --model gru \
    --imputed_history_path "${IMPUTED_HISTORY_PATH}" \
    --ground_path "${GROUND_PATH}" \
    --split_ratio ${SPLIT_RATIO} \
    --horizon_days ${HORIZON_DAYS} \
    --window_size 48 --hidden_size 64 --num_layers 2 \
    --epochs 100 --batch_size 128 --lr 1e-3 --seed ${SEED} \
    --device ${DEVICE}

GRU_DIR=$(ls -td ./save/pm25_gru_* 2>/dev/null | head -1)
if [ -n "${GRU_DIR}" ] && [ -f "${GRU_DIR}/future_pred.csv" ]; then
    cp "${GRU_DIR}/future_pred.csv" ${CMP_DIR}/gru/
    echo "✅ GRU 结果已复制"
else
    echo "⚠️  GRU 结果未找到"
fi

# ---- Step 3: LSTM ----
echo ""
echo "=== [3/5] LSTM 基线 ==="
python baselines/gru_lstm_forecast.py \
    --dataset pm25 --model lstm \
    --imputed_history_path "${IMPUTED_HISTORY_PATH}" \
    --ground_path "${GROUND_PATH}" \
    --split_ratio ${SPLIT_RATIO} \
    --horizon_days ${HORIZON_DAYS} \
    --window_size 48 --hidden_size 64 --num_layers 2 \
    --epochs 100 --batch_size 128 --lr 1e-3 --seed ${SEED} \
    --device ${DEVICE}

LSTM_DIR=$(ls -td ./save/pm25_lstm_* 2>/dev/null | head -1)
if [ -n "${LSTM_DIR}" ] && [ -f "${LSTM_DIR}/future_pred.csv" ]; then
    cp "${LSTM_DIR}/future_pred.csv" ${CMP_DIR}/lstm/
    echo "✅ LSTM 结果已复制"
else
    echo "⚠️  LSTM 结果未找到"
fi

# ---- Step 4: NeuralCDE (修复版) ----
echo ""
echo "=== [4/5] NeuralCDE 基线 (修复版) ==="
python baselines/neuralcde_forecast.py \
    --dataset pm25 \
    --imputed_history_path "${IMPUTED_HISTORY_PATH}" \
    --ground_path "${GROUND_PATH}" \
    --split_ratio ${SPLIT_RATIO} \
    --horizon_days ${HORIZON_DAYS} \
    --window_size 48 --hidden_channels 64 --num_layers 3 \
    --epochs 100 --batch_size 128 --lr 1e-3 --seed ${SEED} \
    --device ${DEVICE}

NC_DIR=$(ls -td ./save/pm25_neuralcde_* 2>/dev/null | head -1)
if [ -n "${NC_DIR}" ] && [ -f "${NC_DIR}/future_pred.csv" ]; then
    cp "${NC_DIR}/future_pred.csv" ${CMP_DIR}/neuralcde/
    echo "✅ NeuralCDE 结果已复制"
else
    echo "⚠️  NeuralCDE 结果未找到"
fi

# ---- Step 5: GRU-ODE-Bayes ----
echo ""
echo "=== [5/5] GRU-ODE-Bayes 基线 ==="
python baselines/pm25_gruodebayes_forecast.py \
    --imputed_history_path "${IMPUTED_HISTORY_PATH}" \
    --ground_path "${GROUND_PATH}" \
    --split_ratio ${SPLIT_RATIO} \
    --horizon_days ${HORIZON_DAYS} \
    --hidden_size 64 --p_hidden 32 --prep_hidden 32 \
    --window_size 48 --delta_t 0.1 --time_scale 0.02 --solver euler \
    --epochs 100 --batch_size 128 --lr 1e-3 --seed ${SEED} \
    --device ${DEVICE}

GB_DIR=$(ls -td ./save/pm25_gruodebayes_split* 2>/dev/null | head -1)
if [ -n "${GB_DIR}" ] && [ -f "${GB_DIR}/future_pred.csv" ]; then
    cp "${GB_DIR}/future_pred.csv" ${CMP_DIR}/gruodebayes/
    echo "✅ GRU-ODE-Bayes 结果已复制"
else
    echo "⚠️  GRU-ODE-Bayes 结果未找到"
fi

# ---- 生成对比图 ----
echo ""
echo "================================================================"
echo "  生成对比可视化"
echo "================================================================"

python visualization/pm25_full_comparison.py \
    --results_dir ${CMP_DIR} \
    --out_dir ./save/visualization_results/pm25_full \
    --ground_path "${GROUND_PATH}" \
    --split_ratio ${SPLIT_RATIO} \
    --horizon_steps 24

echo ""
echo "================================================================"
echo "  🎉 所有步骤完成！"
echo "  结果目录: ${CMP_DIR}/"
echo "  可视化: ./save/visualization_results/pm25_full/"
echo "================================================================"
