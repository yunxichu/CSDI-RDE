#!/bin/bash
# v2 基线实验 - 顺序执行版（单 GPU，低 CPU 负载）
# 每次运行 2 个实验（同一数据集，不同方法），等待完成后继续
# 适合单卡或双卡服务器，避免 OOM 和 CPU 过载
#
# 用法：
#   bash run_v2_sequential.sh              # 默认使用 cuda:0
#   bash run_v2_sequential.sh --gpu 1      # 使用 cuda:1
#   bash run_v2_sequential.sh --dataset lorenz96  # 只跑某个数据集

set -e

GPU=0
DATASET=""

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu) GPU="$2"; shift 2 ;;
        --dataset) DATASET="$2"; shift 2 ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

DEVICE="cuda:${GPU}"
OUT_BASE="./experiments_v2"
LOG_DIR="${OUT_BASE}/logs"
mkdir -p ${LOG_DIR}

# 数据路径
L63_GT="./lorenz_rde_delay/results/gt_100_20260320_110418.csv"
L63_DATA="./lorenz_rde_delay/results/imputed_100_20260320_110418.csv"
L96_GT="./lorenz96_rde_delay/results/gt_100_20260323_192045.csv"
L96_DATA="./lorenz96_rde_delay/results/imputed_100_20260323_192045.csv"
PM25_HIST="./save/pm25_history_imputed_split0.5_seed42_20260324_154400/history_imputed.csv"
PM25_GT="./data/pm25/Code/STMVL/SampleData/pm25_ground.txt"
EEG_IMPUTED="./save/eeg_csdi_imputed/eeg_imputed.npy"
EEG_GT="./save/eeg_csdi_imputed/eeg_full.npy"

echo "============================================"
echo "  v2 顺序执行基线实验"
echo "  GPU: ${DEVICE}"
echo "  输出目录: ${OUT_BASE}"
if [ -n "${DATASET}" ]; then
    echo "  只跑数据集: ${DATASET}"
fi
echo "============================================"

run_dataset_lorenz63() {
    echo ""
    echo ">>> [Lorenz63] 开始..."
    mkdir -p ${OUT_BASE}/lorenz63/{neuralcde,gruodebayes,sssd}

    # NeuralCDE + GRU-ODE-Bayes 并行（同一 GPU，内存占用低）
    python3 baselines/neuralcde_forecast.py \
        --dataset lorenz63 \
        --ground_path ${L63_GT} --data_path ${L63_DATA} \
        --trainlength 60 --horizon_steps 40 \
        --window_size 20 --hidden_channels 64 --num_layers 3 \
        --epochs 100 --batch_size 128 --lr 1e-3 --seed 42 \
        --device ${DEVICE} --out_dir ${OUT_BASE}/lorenz63/neuralcde \
        2>&1 | tee ${LOG_DIR}/lorenz63_neuralcde.log &

    python3 baselines/gruodebayes_forecast.py \
        --dataset lorenz63 \
        --ground_path ${L63_GT} --data_path ${L63_DATA} \
        --trainlength 60 --horizon_steps 40 \
        --window_size 20 --hidden_size 64 \
        --epochs 100 --batch_size 128 --lr 1e-3 --seed 42 \
        --device ${DEVICE} --out_dir ${OUT_BASE}/lorenz63/gruodebayes \
        2>&1 | tee ${LOG_DIR}/lorenz63_gruodebayes.log &

    wait
    echo ">>> [Lorenz63] NeuralCDE + GRU-ODE-Bayes 完成"

    # SSSD 单独跑（内存较大）
    python3 baselines/sssd_forecast.py \
        --dataset lorenz63 \
        --ground_path ${L63_GT} --data_path ${L63_DATA} \
        --trainlength 60 --horizon_steps 40 \
        --d_model 64 --n_layers 4 --diffusion_steps 50 \
        --window_size 20 --epochs 100 --batch_size 16 --lr 1e-4 --seed 42 \
        --device ${DEVICE} --out_dir ${OUT_BASE}/lorenz63/sssd \
        2>&1 | tee ${LOG_DIR}/lorenz63_sssd.log

    echo ">>> [Lorenz63] 全部完成"
}

run_dataset_lorenz96() {
    echo ""
    echo ">>> [Lorenz96] 开始..."
    mkdir -p ${OUT_BASE}/lorenz96/{neuralcde,gruodebayes,sssd}

    python3 baselines/neuralcde_forecast.py \
        --dataset lorenz96 \
        --ground_path ${L96_GT} --data_path ${L96_DATA} \
        --trainlength 60 --horizon_steps 40 \
        --window_size 20 --hidden_channels 64 --num_layers 3 \
        --epochs 100 --batch_size 128 --lr 1e-3 --seed 42 \
        --device ${DEVICE} --out_dir ${OUT_BASE}/lorenz96/neuralcde \
        2>&1 | tee ${LOG_DIR}/lorenz96_neuralcde.log &

    python3 baselines/gruodebayes_forecast.py \
        --dataset lorenz96 \
        --ground_path ${L96_GT} --data_path ${L96_DATA} \
        --trainlength 60 --horizon_steps 40 \
        --window_size 20 --hidden_size 64 \
        --epochs 100 --batch_size 128 --lr 1e-3 --seed 42 \
        --device ${DEVICE} --out_dir ${OUT_BASE}/lorenz96/gruodebayes \
        2>&1 | tee ${LOG_DIR}/lorenz96_gruodebayes.log &

    wait
    echo ">>> [Lorenz96] NeuralCDE + GRU-ODE-Bayes 完成"

    python3 baselines/sssd_forecast.py \
        --dataset lorenz96 \
        --ground_path ${L96_GT} --data_path ${L96_DATA} \
        --trainlength 60 --horizon_steps 40 \
        --d_model 64 --n_layers 4 --diffusion_steps 50 \
        --window_size 20 --epochs 100 --batch_size 16 --lr 1e-4 --seed 42 \
        --device ${DEVICE} --out_dir ${OUT_BASE}/lorenz96/sssd \
        2>&1 | tee ${LOG_DIR}/lorenz96_sssd.log

    echo ">>> [Lorenz96] 全部完成"
}

run_dataset_pm25() {
    echo ""
    echo ">>> [PM2.5] 开始..."
    mkdir -p ${OUT_BASE}/pm25/{neuralcde,gruodebayes,sssd}

    python3 baselines/neuralcde_forecast.py \
        --dataset pm25 \
        --imputed_history_path ${PM25_HIST} --ground_path ${PM25_GT} \
        --split_ratio 0.5 --horizon_days 1 \
        --window_size 48 --hidden_channels 64 --num_layers 3 \
        --epochs 100 --batch_size 128 --lr 1e-3 --seed 42 \
        --device ${DEVICE} --out_dir ${OUT_BASE}/pm25/neuralcde \
        2>&1 | tee ${LOG_DIR}/pm25_neuralcde.log &

    python3 baselines/gruodebayes_forecast.py \
        --dataset pm25 \
        --imputed_history_path ${PM25_HIST} --ground_path ${PM25_GT} \
        --split_ratio 0.5 --horizon_days 1 \
        --window_size 48 --hidden_size 64 \
        --epochs 100 --batch_size 128 --lr 1e-3 --seed 42 \
        --max_train_samples 500 \
        --device ${DEVICE} --out_dir ${OUT_BASE}/pm25/gruodebayes \
        2>&1 | tee ${LOG_DIR}/pm25_gruodebayes.log &

    wait
    echo ">>> [PM2.5] NeuralCDE + GRU-ODE-Bayes 完成"

    python3 baselines/sssd_forecast.py \
        --dataset pm25 \
        --imputed_history_path ${PM25_HIST} --ground_path ${PM25_GT} \
        --split_ratio 0.5 --horizon_days 1 \
        --d_model 64 --n_layers 4 --diffusion_steps 50 \
        --window_size 48 --epochs 100 --batch_size 16 --lr 1e-4 --seed 42 \
        --device ${DEVICE} --out_dir ${OUT_BASE}/pm25/sssd \
        2>&1 | tee ${LOG_DIR}/pm25_sssd.log

    echo ">>> [PM2.5] 全部完成"
}

run_dataset_eeg() {
    echo ""
    echo ">>> [EEG] 开始..."
    mkdir -p ${OUT_BASE}/eeg/{neuralcde,gruodebayes,sssd}

    python3 baselines/neuralcde_forecast.py \
        --dataset eeg \
        --imputed_path ${EEG_IMPUTED} --ground_path ${EEG_GT} \
        --history_timesteps 976 --horizon_steps 24 \
        --target_dims 0,1,2 \
        --window_size 48 --hidden_channels 64 --num_layers 3 \
        --epochs 100 --batch_size 128 --lr 1e-3 --seed 42 \
        --device ${DEVICE} --out_dir ${OUT_BASE}/eeg/neuralcde \
        2>&1 | tee ${LOG_DIR}/eeg_neuralcde.log &

    python3 baselines/gruodebayes_forecast.py \
        --dataset eeg \
        --imputed_path ${EEG_IMPUTED} --ground_path ${EEG_GT} \
        --history_timesteps 976 --horizon_steps 24 \
        --target_dims 0,1,2 \
        --window_size 48 --hidden_size 64 \
        --epochs 100 --batch_size 128 --lr 1e-3 --seed 42 \
        --device ${DEVICE} --out_dir ${OUT_BASE}/eeg/gruodebayes \
        2>&1 | tee ${LOG_DIR}/eeg_gruodebayes.log &

    wait
    echo ">>> [EEG] NeuralCDE + GRU-ODE-Bayes 完成"

    python3 baselines/sssd_forecast.py \
        --dataset eeg \
        --imputed_path ${EEG_IMPUTED} --ground_path ${EEG_GT} \
        --history_timesteps 976 --horizon_steps 24 \
        --target_dims 0,1,2 \
        --d_model 64 --n_layers 4 --diffusion_steps 50 \
        --window_size 48 --epochs 100 --batch_size 16 --lr 1e-4 --seed 42 \
        --device ${DEVICE} --out_dir ${OUT_BASE}/eeg/sssd \
        2>&1 | tee ${LOG_DIR}/eeg_sssd.log

    echo ">>> [EEG] 全部完成"
}

# 执行（按指定数据集或全部）
if [ -z "${DATASET}" ]; then
    run_dataset_lorenz96
    run_dataset_pm25
    run_dataset_eeg
    run_dataset_lorenz63
else
    case ${DATASET} in
        lorenz63) run_dataset_lorenz63 ;;
        lorenz96) run_dataset_lorenz96 ;;
        pm25)     run_dataset_pm25 ;;
        eeg)      run_dataset_eeg ;;
        *) echo "未知数据集: ${DATASET}. 可选: lorenz63, lorenz96, pm25, eeg"; exit 1 ;;
    esac
fi

echo ""
echo "============================================"
echo "  全部实验完成！"
echo "  检查结果: ls experiments_v2/**/*.json"
echo "============================================"
