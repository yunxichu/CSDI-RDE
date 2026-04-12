#!/bin/bash
# V1: Lorenz63 + Lorenz96 基线实验
# 3个基线: NeuralCDE, GRU-ODE-Bayes, SSSD
# GPU分配: NeuralCDE+GRU-ODE-Bayes用cuda:0, SSSD用cuda:1
# 注意: 两个实验可以并行跑在不同GPU上

set -e

SEED=42
EPOCHS=100
BATCH=128
LR=1e-3
WINDOW=20
HIDDEN=64

# 最新数据文件
L96_IMPUTED=$(ls -t ./lorenz96_rde_delay/results/imputed_100_*.csv | head -1)
L96_GT=$(ls -t ./lorenz96_rde_delay/results/gt_100_*.csv | head -1)
L63_IMPUTED=$(ls -t ./lorenz_rde_delay/results/imputed_100_*.csv | head -1)
L63_GT=$(ls -t ./lorenz_rde_delay/results/gt_100_*.csv | head -1)

OUT_BASE="./experiments_v1"
mkdir -p ${OUT_BASE}/{lorenz63,lorenz96}/{neuralcde,gruodebayes,sssd}

echo "================================================================"
echo "  V1: Lorenz63 + Lorenz96 基线实验"
echo "================================================================"
echo "  Lorenz96: ${L96_IMPUTED}"
echo "  Lorenz63: ${L63_IMPUTED}"
echo "  输出: ${OUT_BASE}/"
echo "================================================================"

# ===== Lorenz96 =====
echo ""
echo "=== Lorenz96: NeuralCDE (cuda:0) ==="
python baselines/neuralcde_forecast.py \
    --dataset lorenz96 \
    --data_path "${L96_IMPUTED}" \
    --ground_path "${L96_GT}" \
    --trainlength 60 --horizon_steps 40 \
    --window_size ${WINDOW} --hidden_channels ${HIDDEN} --num_layers 3 \
    --epochs ${EPOCHS} --batch_size ${BATCH} --lr ${LR} --seed ${SEED} \
    --device cuda:0 \
    --out_dir ${OUT_BASE}/lorenz96/neuralcde

echo ""
echo "=== Lorenz96: GRU-ODE-Bayes (cuda:0) ==="
python baselines/gruodebayes_forecast.py \
    --dataset lorenz96 \
    --data_path "${L96_IMPUTED}" \
    --ground_path "${L96_GT}" \
    --trainlength 60 --horizon_steps 40 \
    --window_size ${WINDOW} --hidden_size ${HIDDEN} \
    --p_hidden 32 --prep_hidden 32 \
    --delta_t 0.1 --time_scale 0.02 --solver euler \
    --epochs ${EPOCHS} --batch_size ${BATCH} --lr ${LR} --seed ${SEED} \
    --device cuda:0 \
    --out_dir ${OUT_BASE}/lorenz96/gruodebayes

echo ""
echo "=== Lorenz96: SSSD (cuda:1) ==="
python baselines/sssd_forecast.py \
    --dataset lorenz96 \
    --data_path "${L96_IMPUTED}" \
    --ground_path "${L96_GT}" \
    --trainlength 60 --horizon_steps 40 \
    --d_model ${HIDDEN} --n_layers 4 --diffusion_steps 50 \
    --window_size ${WINDOW} \
    --epochs ${EPOCHS} --batch_size 16 --lr 1e-4 --seed ${SEED} \
    --device cuda:1 \
    --out_dir ${OUT_BASE}/lorenz96/sssd

# ===== Lorenz63 =====
echo ""
echo "=== Lorenz63: NeuralCDE (cuda:0) ==="
python baselines/neuralcde_forecast.py \
    --dataset lorenz63 \
    --data_path "${L63_IMPUTED}" \
    --ground_path "${L63_GT}" \
    --trainlength 60 --horizon_steps 40 \
    --window_size ${WINDOW} --hidden_channels ${HIDDEN} --num_layers 3 \
    --epochs ${EPOCHS} --batch_size ${BATCH} --lr ${LR} --seed ${SEED} \
    --device cuda:0 \
    --out_dir ${OUT_BASE}/lorenz63/neuralcde

echo ""
echo "=== Lorenz63: GRU-ODE-Bayes (cuda:0) ==="
python baselines/gruodebayes_forecast.py \
    --dataset lorenz63 \
    --data_path "${L63_IMPUTED}" \
    --ground_path "${L63_GT}" \
    --trainlength 60 --horizon_steps 40 \
    --window_size ${WINDOW} --hidden_size ${HIDDEN} \
    --p_hidden 32 --prep_hidden 32 \
    --delta_t 0.1 --time_scale 0.02 --solver euler \
    --epochs ${EPOCHS} --batch_size ${BATCH} --lr ${LR} --seed ${SEED} \
    --device cuda:0 \
    --out_dir ${OUT_BASE}/lorenz63/gruodebayes

echo ""
echo "=== Lorenz63: SSSD (cuda:1) ==="
python baselines/sssd_forecast.py \
    --dataset lorenz63 \
    --data_path "${L63_IMPUTED}" \
    --ground_path "${L63_GT}" \
    --trainlength 60 --horizon_steps 40 \
    --d_model ${HIDDEN} --n_layers 4 --diffusion_steps 50 \
    --window_size ${WINDOW} \
    --epochs ${EPOCHS} --batch_size 16 --lr 1e-4 --seed ${SEED} \
    --device cuda:1 \
    --out_dir ${OUT_BASE}/lorenz63/sssd

echo ""
echo "================================================================"
echo "  🎉 V1 完成！结果在 ${OUT_BASE}/"
echo "================================================================"
