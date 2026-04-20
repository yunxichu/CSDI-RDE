#!/bin/bash
# GPU专用修复版基线实验一键运行脚本
# 使用2张GPU并行运行，每张GPU跑6个实验
# 修复内容：
#   1. SSSD mask语义修正（mask=1表示缺失/需预测）
#   2. PM25 NaN前向填充处理
# 使用方法：bash run_gpu_fixed_baselines.sh

set -e

echo "============================================"
echo "  GPU修复版基线实验"
echo "  输出目录: ./experiments_v2"
echo "  GPU: 2张卡并行"
echo "============================================"

# 检查GPU是否可用
if ! python3 -c "import torch; assert torch.cuda.is_available() and torch.cuda.device_count() >= 2" 2>/dev/null; then
    echo "错误：需要至少2张GPU"
    exit 1
fi

OUT_BASE="./experiments_v2"
mkdir -p ${OUT_BASE}/{lorenz63,lorenz96,pm25,eeg}/{neuralcde,gruodebayes,sssd}
mkdir -p ${OUT_BASE}/logs

# 数据路径
L63_GT="./lorenz_rde_delay/results/gt_100_20260320_110418.csv"
L63_DATA="./lorenz_rde_delay/results/imputed_100_20260320_110418.csv"
L96_GT="./lorenz96_rde_delay/results/gt_100_20260323_192045.csv"
L96_DATA="./lorenz96_rde_delay/results/imputed_100_20260323_192045.csv"
PM25_HIST="./save/pm25_history_imputed_split0.5_seed42_20260324_154400/history_imputed.csv"
PM25_GT="./data/pm25/Code/STMVL/SampleData/pm25_ground.txt"
EEG_IMPUTED="./save/eeg_csdi_imputed/eeg_imputed.npy"
EEG_GT="./save/eeg_csdi_imputed/eeg_full.npy"

# GPU 0 实验（6个）
echo ""
echo "[GPU 0] 启动6个实验..."

# Lorenz63
python3 baselines/neuralcde_forecast.py \
    --dataset lorenz63 \
    --ground_path ${L63_GT} --data_path ${L63_DATA} \
    --trainlength 60 --horizon_steps 40 \
    --window_size 20 --hidden_channels 64 --num_layers 3 \
    --epochs 100 --batch_size 128 --lr 1e-3 --seed 42 \
    --device cuda:0 --out_dir ${OUT_BASE}/lorenz63/neuralcde 2>&1 | tee ${OUT_BASE}/logs/lorenz63_neuralcde.log &

python3 baselines/gruodebayes_forecast.py \
    --dataset lorenz63 \
    --ground_path ${L63_GT} --data_path ${L63_DATA} \
    --trainlength 60 --horizon_steps 40 \
    --window_size 20 --hidden_size 64 \
    --epochs 100 --batch_size 128 --lr 1e-3 --seed 42 \
    --device cuda:0 --out_dir ${OUT_BASE}/lorenz63/gruodebayes 2>&1 | tee ${OUT_BASE}/logs/lorenz63_gruodebayes.log &

python3 baselines/sssd_forecast.py \
    --dataset lorenz63 \
    --ground_path ${L63_GT} --data_path ${L63_DATA} \
    --trainlength 60 --horizon_steps 40 \
    --d_model 64 --n_layers 4 --diffusion_steps 50 \
    --window_size 20 --epochs 100 --batch_size 16 --lr 1e-4 --seed 42 \
    --device cuda:0 --out_dir ${OUT_BASE}/lorenz63/sssd 2>&1 | tee ${OUT_BASE}/logs/lorenz63_sssd.log &

# Lorenz96
python3 baselines/neuralcde_forecast.py \
    --dataset lorenz96 \
    --ground_path ${L96_GT} --data_path ${L96_DATA} \
    --trainlength 60 --horizon_steps 40 \
    --window_size 20 --hidden_channels 64 --num_layers 3 \
    --epochs 100 --batch_size 128 --lr 1e-3 --seed 42 \
    --device cuda:0 --out_dir ${OUT_BASE}/lorenz96/neuralcde 2>&1 | tee ${OUT_BASE}/logs/lorenz96_neuralcde.log &

python3 baselines/gruodebayes_forecast.py \
    --dataset lorenz96 \
    --ground_path ${L96_GT} --data_path ${L96_DATA} \
    --trainlength 60 --horizon_steps 40 \
    --window_size 20 --hidden_size 64 \
    --epochs 100 --batch_size 128 --lr 1e-3 --seed 42 \
    --device cuda:0 --out_dir ${OUT_BASE}/lorenz96/gruodebayes 2>&1 | tee ${OUT_BASE}/logs/lorenz96_gruodebayes.log &

python3 baselines/sssd_forecast.py \
    --dataset lorenz96 \
    --ground_path ${L96_GT} --data_path ${L96_DATA} \
    --trainlength 60 --horizon_steps 40 \
    --d_model 64 --n_layers 4 --diffusion_steps 50 \
    --window_size 20 --epochs 100 --batch_size 16 --lr 1e-4 --seed 42 \
    --device cuda:0 --out_dir ${OUT_BASE}/lorenz96/sssd 2>&1 | tee ${OUT_BASE}/logs/lorenz96_sssd.log &

# GPU 1 实验（6个）
echo ""
echo "[GPU 1] 启动6个实验..."

# PM2.5
python3 baselines/neuralcde_forecast.py \
    --dataset pm25 \
    --imputed_history_path ${PM25_HIST} --ground_path ${PM25_GT} \
    --split_ratio 0.5 --horizon_days 1 \
    --window_size 48 --hidden_channels 64 --num_layers 3 \
    --epochs 100 --batch_size 128 --lr 1e-3 --seed 42 \
    --device cuda:1 --out_dir ${OUT_BASE}/pm25/neuralcde 2>&1 | tee ${OUT_BASE}/logs/pm25_neuralcde.log &

python3 baselines/gruodebayes_forecast.py \
    --dataset pm25 \
    --imputed_history_path ${PM25_HIST} --ground_path ${PM25_GT} \
    --split_ratio 0.5 --horizon_days 1 \
    --window_size 48 --hidden_size 64 \
    --epochs 100 --batch_size 128 --lr 1e-3 --seed 42 \
    --max_train_samples 500 \
    --device cuda:1 --out_dir ${OUT_BASE}/pm25/gruodebayes 2>&1 | tee ${OUT_BASE}/logs/pm25_gruodebayes.log &

python3 baselines/sssd_forecast.py \
    --dataset pm25 \
    --imputed_history_path ${PM25_HIST} --ground_path ${PM25_GT} \
    --split_ratio 0.5 --horizon_days 1 \
    --d_model 64 --n_layers 4 --diffusion_steps 50 \
    --window_size 48 --epochs 100 --batch_size 16 --lr 1e-4 --seed 42 \
    --device cuda:1 --out_dir ${OUT_BASE}/pm25/sssd 2>&1 | tee ${OUT_BASE}/logs/pm25_sssd.log &

# EEG
python3 baselines/neuralcde_forecast.py \
    --dataset eeg \
    --imputed_path ${EEG_IMPUTED} --ground_path ${EEG_GT} \
    --history_timesteps 976 --horizon_steps 24 \
    --target_dims 0,1,2 \
    --window_size 48 --hidden_channels 64 --num_layers 3 \
    --epochs 100 --batch_size 128 --lr 1e-3 --seed 42 \
    --device cuda:1 --out_dir ${OUT_BASE}/eeg/neuralcde 2>&1 | tee ${OUT_BASE}/logs/eeg_neuralcde.log &

python3 baselines/gruodebayes_forecast.py \
    --dataset eeg \
    --imputed_path ${EEG_IMPUTED} --ground_path ${EEG_GT} \
    --history_timesteps 976 --horizon_steps 24 \
    --target_dims 0,1,2 \
    --window_size 48 --hidden_size 64 \
    --epochs 100 --batch_size 128 --lr 1e-3 --seed 42 \
    --device cuda:1 --out_dir ${OUT_BASE}/eeg/gruodebayes 2>&1 | tee ${OUT_BASE}/logs/eeg_gruodebayes.log &

python3 baselines/sssd_forecast.py \
    --dataset eeg \
    --imputed_path ${EEG_IMPUTED} --ground_path ${EEG_GT} \
    --history_timesteps 976 --horizon_steps 24 \
    --target_dims 0,1,2 \
    --d_model 64 --n_layers 4 --diffusion_steps 50 \
    --window_size 48 --epochs 100 --batch_size 16 --lr 1e-4 --seed 42 \
    --device cuda:1 --out_dir ${OUT_BASE}/eeg/sssd 2>&1 | tee ${OUT_BASE}/logs/eeg_sssd.log &

echo ""
echo "============================================"
echo "  所有12个实验已在2张GPU上启动！"
echo "  日志目录: ${OUT_BASE}/logs/"
echo "  输出目录: ${OUT_BASE}/"
echo ""
echo "  监控命令:"
echo "  - 查看GPU使用: watch -n 1 nvidia-smi"
echo "  - 查看日志: tail -f ${OUT_BASE}/logs/*.log"
echo "  - 查看进度: ls -la ${OUT_BASE}/**/*.npy"
echo "============================================"
