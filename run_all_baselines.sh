#!/bin/bash
# ==============================================================================
# 完整基线实验运行脚本
# ==============================================================================
# 使用方法:
#   1. 确保在 /home/rhl/Github 目录
#   2. 给脚本执行权限: chmod +x run_all_baselines.sh
#   3. 在screen中运行: screen -S baselines; ./run_all_baselines.sh
#   4. 分离screen: Ctrl+A, D
#   5. 重新连接: screen -r baselines
#
# GPU分配:
#   GPU 0: NeuralCDE + GRU-ODE-Bayes (串行)
#   GPU 2: SSSD (独立运行)
# ==============================================================================

set -e
cd /home/rhl/Github

SEED=42
EPOCHS=100
OUT_BASE="./experiments_v1"
LOG_DIR="${OUT_BASE}/logs"
mkdir -p ${LOG_DIR}

echo "================================================================"
echo "  开始运行所有基线实验"
echo "  时间: $(date)"
echo "  输出: ${OUT_BASE}/"
echo "================================================================"

# ==============================================================================
# V1: Lorenz63 + Lorenz96
# ==============================================================================

L96_IMPUTED=$(ls -t ./lorenz96_rde_delay/results/imputed_100_*.csv | head -1)
L96_GT=$(ls -t ./lorenz96_rde_delay/results/gt_100_*.csv | head -1)
L63_IMPUTED=$(ls -t ./lorenz_rde_delay/results/imputed_100_*.csv | head -1)
L63_GT=$(ls -t ./lorenz_rde_delay/results/gt_100_*.csv | head -1)

echo ""
echo "=== V1: Lorenz63 + Lorenz96 ==="
echo "  L96: ${L96_IMPUTED}"
echo "  L63: ${L63_IMPUTED}"

# --- Lorenz96: NeuralCDE (GPU 0) ---
echo "[L96-NeuralCDE] 开始 $(date)"
CUDA_VISIBLE_DEVICES=0 python3 baselines/neuralcde_forecast.py \
    --dataset lorenz96 \
    --data_path "${L96_IMPUTED}" --ground_path "${L96_GT}" \
    --trainlength 60 --horizon_steps 40 \
    --window_size 20 --hidden_channels 64 --num_layers 3 \
    --epochs ${EPOCHS} --batch_size 128 --lr 1e-3 --seed ${SEED} \
    --device cuda:0 \
    --out_dir ${OUT_BASE}/lorenz96/neuralcde \
    2>&1 | tee ${LOG_DIR}/l96_neuralcde.log
echo "[L96-NeuralCDE] 完成 $(date)"

# --- Lorenz96: GRU-ODE-Bayes (GPU 0) ---
echo "[L96-GRUODEBayes] 开始 $(date)"
CUDA_VISIBLE_DEVICES=0 python3 baselines/gruodebayes_forecast.py \
    --dataset lorenz96 \
    --data_path "${L96_IMPUTED}" --ground_path "${L96_GT}" \
    --trainlength 60 --horizon_steps 40 \
    --window_size 20 --hidden_size 64 \
    --p_hidden 32 --prep_hidden 32 \
    --delta_t 0.1 --time_scale 0.02 --solver euler \
    --epochs ${EPOCHS} --batch_size 128 --lr 1e-3 --seed ${SEED} \
    --device cuda:0 \
    --out_dir ${OUT_BASE}/lorenz96/gruodebayes \
    2>&1 | tee ${LOG_DIR}/l96_gruodebayes.log
echo "[L96-GRUODEBayes] 完成 $(date)"

# --- Lorenz96: SSSD (GPU 2) ---
echo "[L96-SSSD] 开始 $(date)"
CUDA_VISIBLE_DEVICES=2 python3 baselines/sssd_forecast.py \
    --dataset lorenz96 \
    --data_path "${L96_IMPUTED}" --ground_path "${L96_GT}" \
    --trainlength 60 --horizon_steps 40 \
    --d_model 64 --n_layers 4 --diffusion_steps 50 \
    --window_size 20 \
    --epochs ${EPOCHS} --batch_size 16 --lr 1e-4 --seed ${SEED} \
    --device cuda:0 \
    --out_dir ${OUT_BASE}/lorenz96/sssd \
    2>&1 | tee ${LOG_DIR}/l96_sssd.log
echo "[L96-SSSD] 完成 $(date)"

# --- Lorenz63: NeuralCDE (GPU 0) ---
echo "[L63-NeuralCDE] 开始 $(date)"
CUDA_VISIBLE_DEVICES=0 python3 baselines/neuralcde_forecast.py \
    --dataset lorenz63 \
    --data_path "${L63_IMPUTED}" --ground_path "${L63_GT}" \
    --trainlength 60 --horizon_steps 40 \
    --window_size 20 --hidden_channels 64 --num_layers 3 \
    --epochs ${EPOCHS} --batch_size 128 --lr 1e-3 --seed ${SEED} \
    --device cuda:0 \
    --out_dir ${OUT_BASE}/lorenz63/neuralcde \
    2>&1 | tee ${LOG_DIR}/l63_neuralcde.log
echo "[L63-NeuralCDE] 完成 $(date)"

# --- Lorenz63: GRU-ODE-Bayes (GPU 0) ---
echo "[L63-GRUODEBayes] 开始 $(date)"
CUDA_VISIBLE_DEVICES=0 python3 baselines/gruodebayes_forecast.py \
    --dataset lorenz63 \
    --data_path "${L63_IMPUTED}" --ground_path "${L63_GT}" \
    --trainlength 60 --horizon_steps 40 \
    --window_size 20 --hidden_size 64 \
    --p_hidden 32 --prep_hidden 32 \
    --delta_t 0.1 --time_scale 0.02 --solver euler \
    --epochs ${EPOCHS} --batch_size 128 --lr 1e-3 --seed ${SEED} \
    --device cuda:0 \
    --out_dir ${OUT_BASE}/lorenz63/gruodebayes \
    2>&1 | tee ${LOG_DIR}/l63_gruodebayes.log
echo "[L63-GRUODEBayes] 完成 $(date)"

# --- Lorenz63: SSSD (GPU 2) ---
echo "[L63-SSSD] 开始 $(date)"
CUDA_VISIBLE_DEVICES=2 python3 baselines/sssd_forecast.py \
    --dataset lorenz63 \
    --data_path "${L63_IMPUTED}" --ground_path "${L63_GT}" \
    --trainlength 60 --horizon_steps 40 \
    --d_model 64 --n_layers 4 --diffusion_steps 50 \
    --window_size 20 \
    --epochs ${EPOCHS} --batch_size 16 --lr 1e-4 --seed ${SEED} \
    --device cuda:0 \
    --out_dir ${OUT_BASE}/lorenz63/sssd \
    2>&1 | tee ${LOG_DIR}/l63_sssd.log
echo "[L63-SSSD] 完成 $(date)"

# ==============================================================================
# V2: PM2.5
# ==============================================================================

GROUND_PATH="./data/pm25/Code/STMVL/SampleData/pm25_ground.txt"
SPLIT_RATIO=0.5
HORIZON_DAYS=1

IMPUTED_DIR=$(ls -d ./save/pm25_history_imputed_split${SPLIT_RATIO}_seed${SEED}_* 2>/dev/null | head -1)
if [ -z "${IMPUTED_DIR}" ]; then
    echo "⚠️  找不到PM2.5补值结果，跳过V2"
else
    IMPUTED_HISTORY_PATH="${IMPUTED_DIR}/history_imputed.csv"
    echo ""
    echo "=== V2: PM2.5 ==="
    echo "  补值: ${IMPUTED_DIR}"

    echo "[PM25-NeuralCDE] 开始 $(date)"
    CUDA_VISIBLE_DEVICES=0 python3 baselines/neuralcde_forecast.py \
        --dataset pm25 \
        --imputed_history_path "${IMPUTED_HISTORY_PATH}" \
        --ground_path "${GROUND_PATH}" \
        --split_ratio ${SPLIT_RATIO} --horizon_days ${HORIZON_DAYS} \
        --window_size 48 --hidden_channels 64 --num_layers 3 \
        --epochs ${EPOCHS} --batch_size 128 --lr 1e-3 --seed ${SEED} \
        --device cuda:0 \
        --out_dir ${OUT_BASE}/pm25/neuralcde \
        2>&1 | tee ${LOG_DIR}/pm25_neuralcde.log
    echo "[PM25-NeuralCDE] 完成 $(date)"

    echo "[PM25-GRUODEBayes] 开始 $(date)"
    CUDA_VISIBLE_DEVICES=0 python3 baselines/gruodebayes_forecast.py \
        --dataset pm25 \
        --imputed_history_path "${IMPUTED_HISTORY_PATH}" \
        --ground_path "${GROUND_PATH}" \
        --split_ratio ${SPLIT_RATIO} --horizon_days ${HORIZON_DAYS} \
        --window_size 48 --hidden_size 64 \
        --p_hidden 32 --prep_hidden 32 \
        --delta_t 0.1 --time_scale 0.02 --solver euler \
        --epochs ${EPOCHS} --batch_size 128 --lr 1e-3 --seed ${SEED} \
        --device cuda:0 \
        --out_dir ${OUT_BASE}/pm25/gruodebayes \
        2>&1 | tee ${LOG_DIR}/pm25_gruodebayes.log
    echo "[PM25-GRUODEBayes] 完成 $(date)"

    echo "[PM25-SSSD] 开始 $(date)"
    CUDA_VISIBLE_DEVICES=2 python3 baselines/sssd_forecast.py \
        --dataset pm25 \
        --imputed_history_path "${IMPUTED_HISTORY_PATH}" \
        --ground_path "${GROUND_PATH}" \
        --split_ratio ${SPLIT_RATIO} --horizon_days ${HORIZON_DAYS} \
        --d_model 64 --n_layers 4 --diffusion_steps 100 \
        --window_size 48 \
        --epochs ${EPOCHS} --batch_size 16 --lr 1e-4 --seed ${SEED} \
        --device cuda:0 \
        --out_dir ${OUT_BASE}/pm25/sssd \
        2>&1 | tee ${LOG_DIR}/pm25_sssd.log
    echo "[PM25-SSSD] 完成 $(date)"
fi

# ==============================================================================
# V3: EEG
# ==============================================================================

EEG_GROUND="./data/eeg/eeg_ground.npy"
EEG_IMPUTED="./save/eeg_csdi_imputed.npy"

if [ ! -f "${EEG_IMPUTED}" ]; then
    EEG_IMPUTED="./data/eeg/eeg_linear_filled.npy"
fi

if [ -f "${EEG_GROUND}" ] && [ -f "${EEG_IMPUTED}" ]; then
    echo ""
    echo "=== V3: EEG ==="
    echo "  Ground: ${EEG_GROUND}"
    echo "  Imputed: ${EEG_IMPUTED}"

    echo "[EEG-NeuralCDE] 开始 $(date)"
    CUDA_VISIBLE_DEVICES=0 python3 baselines/neuralcde_forecast.py \
        --dataset eeg \
        --imputed_path "${EEG_IMPUTED}" \
        --ground_path "${EEG_GROUND}" \
        --history_timesteps 100 --horizon_steps 24 \
        --target_dims 0,1,2 \
        --window_size 48 --hidden_channels 64 --num_layers 3 \
        --epochs ${EPOCHS} --batch_size 128 --lr 1e-3 --seed ${SEED} \
        --device cuda:0 \
        --out_dir ${OUT_BASE}/eeg/neuralcde \
        2>&1 | tee ${LOG_DIR}/eeg_neuralcde.log
    echo "[EEG-NeuralCDE] 完成 $(date)"

    echo "[EEG-GRUODEBayes] 开始 $(date)"
    CUDA_VISIBLE_DEVICES=0 python3 baselines/gruodebayes_forecast.py \
        --dataset eeg \
        --imputed_path "${EEG_IMPUTED}" \
        --ground_path "${EEG_GROUND}" \
        --history_timesteps 100 --horizon_steps 24 \
        --target_dims 0,1,2 \
        --window_size 48 --hidden_size 64 \
        --p_hidden 32 --prep_hidden 32 \
        --delta_t 0.1 --time_scale 0.02 --solver euler \
        --epochs ${EPOCHS} --batch_size 128 --lr 1e-3 --seed ${SEED} \
        --device cuda:0 \
        --out_dir ${OUT_BASE}/eeg/gruodebayes \
        2>&1 | tee ${LOG_DIR}/eeg_gruodebayes.log
    echo "[EEG-GRUODEBayes] 完成 $(date)"

    echo "[EEG-SSSD] 开始 $(date)"
    CUDA_VISIBLE_DEVICES=2 python3 baselines/sssd_forecast.py \
        --dataset eeg \
        --imputed_path "${EEG_IMPUTED}" \
        --ground_path "${EEG_GROUND}" \
        --history_timesteps 100 --horizon_steps 24 \
        --target_dims 0,1,2 \
        --d_model 64 --n_layers 4 --diffusion_steps 50 \
        --window_size 48 \
        --epochs ${EPOCHS} --batch_size 16 --lr 1e-4 --seed ${SEED} \
        --device cuda:0 \
        --out_dir ${OUT_BASE}/eeg/sssd \
        2>&1 | tee ${LOG_DIR}/eeg_sssd.log
    echo "[EEG-SSSD] 完成 $(date)"
else
    echo "⚠️  EEG数据不完整，跳过V3"
    echo "  需要先运行CSDI补值: python experiments/exe_eeg.py --device cuda:0"
fi

# ==============================================================================
# 汇总结果
# ==============================================================================

echo ""
echo "================================================================"
echo "  所有实验完成！汇总结果："
echo "================================================================"

for ds in lorenz63 lorenz96 pm25 eeg; do
    for method in neuralcde gruodebayes sssd; do
        metrics_file="${OUT_BASE}/${ds}/${method}/metrics.json"
        if [ -f "${metrics_file}" ]; then
            rmse=$(python3 -c "import json; d=json.load(open('${metrics_file}')); print(f\"{d['overall']['rmse']:.4f}\")")
            mae=$(python3 -c "import json; d=json.load(open('${metrics_file}')); print(f\"{d['overall']['mae']:.4f}\")")
            echo "  ${ds}/${method}: RMSE=${rmse}, MAE=${mae}"
        fi
    done
done

echo ""
echo "完成时间: $(date)"
echo "================================================================"
