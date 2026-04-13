#!/bin/bash
# 修复版基线实验一键运行脚本
# 修复内容：
#   1. SSSD mask语义修正（mask=1表示缺失/需预测）
#   2. PM25 NaN前向填充处理
# 使用方法：bash run_fixed_baselines.sh [GPU_ID]
# 示例：bash run_fixed_baselines.sh 0

set -e
GPU=${1:-0}
DEV="cuda:${GPU}"

if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "GPU不可用，使用CPU"
    DEV="cpu"
fi

OUT_BASE="./experiments_v2"
mkdir -p ${OUT_BASE}/{lorenz63,lorenz96,pm25,eeg}/{neuralcde,gruodebayes,sssd}

echo "============================================"
echo "  修复版基线实验"
echo "  输出目录: ${OUT_BASE}"
echo "  设备: ${DEV}"
echo "============================================"

# ===== Lorenz63 =====
L63_GT="./lorenz_rde_delay/results/gt_100_20260320_110418.csv"
L63_DATA="./lorenz_rde_delay/results/imputed_100_20260320_110418.csv"

echo ""
echo "[1/12] Lorenz63 - NeuralCDE"
python3 baselines/neuralcde_forecast.py \
    --dataset lorenz63 \
    --ground_path ${L63_GT} --data_path ${L63_DATA} \
    --trainlength 60 --horizon_steps 40 \
    --window_size 20 --hidden_channels 64 --num_layers 3 \
    --epochs 100 --batch_size 128 --lr 1e-3 --seed 42 \
    --device ${DEV} --out_dir ${OUT_BASE}/lorenz63/neuralcde

echo ""
echo "[2/12] Lorenz63 - GRU-ODE-Bayes"
python3 baselines/gruodebayes_forecast.py \
    --dataset lorenz63 \
    --ground_path ${L63_GT} --data_path ${L63_DATA} \
    --trainlength 60 --horizon_steps 40 \
    --window_size 20 --hidden_size 64 \
    --epochs 100 --batch_size 128 --lr 1e-3 --seed 42 \
    --device ${DEV} --out_dir ${OUT_BASE}/lorenz63/gruodebayes

echo ""
echo "[3/12] Lorenz63 - SSSD"
python3 baselines/sssd_forecast.py \
    --dataset lorenz63 \
    --ground_path ${L63_GT} --data_path ${L63_DATA} \
    --trainlength 60 --horizon_steps 40 \
    --d_model 64 --n_layers 4 --diffusion_steps 50 \
    --window_size 20 --epochs 100 --batch_size 16 --lr 1e-4 --seed 42 \
    --device ${DEV} --out_dir ${OUT_BASE}/lorenz63/sssd

# ===== Lorenz96 =====
L96_GT="./lorenz96_rde_delay/results/gt_100_20260323_192045.csv"
L96_DATA="./lorenz96_rde_delay/results/imputed_100_20260323_192045.csv"

echo ""
echo "[4/12] Lorenz96 - NeuralCDE"
python3 baselines/neuralcde_forecast.py \
    --dataset lorenz96 \
    --ground_path ${L96_GT} --data_path ${L96_DATA} \
    --trainlength 60 --horizon_steps 40 \
    --window_size 20 --hidden_channels 64 --num_layers 3 \
    --epochs 100 --batch_size 128 --lr 1e-3 --seed 42 \
    --device ${DEV} --out_dir ${OUT_BASE}/lorenz96/neuralcde

echo ""
echo "[5/12] Lorenz96 - GRU-ODE-Bayes"
python3 baselines/gruodebayes_forecast.py \
    --dataset lorenz96 \
    --ground_path ${L96_GT} --data_path ${L96_DATA} \
    --trainlength 60 --horizon_steps 40 \
    --window_size 20 --hidden_size 64 \
    --epochs 100 --batch_size 128 --lr 1e-3 --seed 42 \
    --device ${DEV} --out_dir ${OUT_BASE}/lorenz96/gruodebayes

echo ""
echo "[6/12] Lorenz96 - SSSD"
python3 baselines/sssd_forecast.py \
    --dataset lorenz96 \
    --ground_path ${L96_GT} --data_path ${L96_DATA} \
    --trainlength 60 --horizon_steps 40 \
    --d_model 64 --n_layers 4 --diffusion_steps 50 \
    --window_size 20 --epochs 100 --batch_size 16 --lr 1e-4 --seed 42 \
    --device ${DEV} --out_dir ${OUT_BASE}/lorenz96/sssd

# ===== PM2.5 =====
PM25_HIST="./save/pm25_history_imputed_split0.5_seed42_20260324_154400/history_imputed.csv"
PM25_GT="./data/pm25/Code/STMVL/SampleData/pm25_ground.txt"

echo ""
echo "[7/12] PM2.5 - NeuralCDE"
python3 baselines/neuralcde_forecast.py \
    --dataset pm25 \
    --imputed_history_path ${PM25_HIST} --ground_path ${PM25_GT} \
    --split_ratio 0.5 --horizon_days 1 \
    --window_size 48 --hidden_channels 64 --num_layers 3 \
    --epochs 100 --batch_size 128 --lr 1e-3 --seed 42 \
    --device ${DEV} --out_dir ${OUT_BASE}/pm25/neuralcde

echo ""
echo "[8/12] PM2.5 - GRU-ODE-Bayes"
python3 baselines/gruodebayes_forecast.py \
    --dataset pm25 \
    --imputed_history_path ${PM25_HIST} --ground_path ${PM25_GT} \
    --split_ratio 0.5 --horizon_days 1 \
    --window_size 48 --hidden_size 64 \
    --epochs 100 --batch_size 128 --lr 1e-3 --seed 42 \
    --max_train_samples 500 \
    --device ${DEV} --out_dir ${OUT_BASE}/pm25/gruodebayes

echo ""
echo "[9/12] PM2.5 - SSSD"
python3 baselines/sssd_forecast.py \
    --dataset pm25 \
    --imputed_history_path ${PM25_HIST} --ground_path ${PM25_GT} \
    --split_ratio 0.5 --horizon_days 1 \
    --d_model 64 --n_layers 4 --diffusion_steps 50 \
    --window_size 48 --epochs 100 --batch_size 16 --lr 1e-4 --seed 42 \
    --device ${DEV} --out_dir ${OUT_BASE}/pm25/sssd

# ===== EEG =====
EEG_DATA="./save/eeg_csdi_imputed/eeg_full.npy"

echo ""
echo "[10/12] EEG - NeuralCDE"
python3 baselines/neuralcde_forecast.py \
    --dataset eeg \
    --imputed_path ${EEG_DATA} --ground_path ${EEG_DATA} \
    --history_timesteps 976 --horizon_steps 24 \
    --target_dims 0,1,2 \
    --window_size 48 --hidden_channels 64 --num_layers 3 \
    --epochs 100 --batch_size 128 --lr 1e-3 --seed 42 \
    --device ${DEV} --out_dir ${OUT_BASE}/eeg/neuralcde

echo ""
echo "[11/12] EEG - GRU-ODE-Bayes"
python3 baselines/gruodebayes_forecast.py \
    --dataset eeg \
    --imputed_path ${EEG_DATA} --ground_path ${EEG_DATA} \
    --history_timesteps 976 --horizon_steps 24 \
    --target_dims 0,1,2 \
    --window_size 48 --hidden_size 64 \
    --epochs 100 --batch_size 128 --lr 1e-3 --seed 42 \
    --device ${DEV} --out_dir ${OUT_BASE}/eeg/gruodebayes

echo ""
echo "[12/12] EEG - SSSD"
python3 baselines/sssd_forecast.py \
    --dataset eeg \
    --imputed_path ${EEG_DATA} --ground_path ${EEG_DATA} \
    --history_timesteps 976 --horizon_steps 24 \
    --target_dims 0,1,2 \
    --d_model 64 --n_layers 4 --diffusion_steps 50 \
    --window_size 48 --epochs 100 --batch_size 16 --lr 1e-4 --seed 42 \
    --device ${DEV} --out_dir ${OUT_BASE}/eeg/sssd

echo ""
echo "============================================"
echo "  所有实验完成！"
echo "============================================"
