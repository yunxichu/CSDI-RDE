#!/bin/bash
# CPU模式运行基线实验（GPU不可用时的回退方案）
# 注意：SSSD在CPU上非常慢，建议在GPU上运行

set -e
cd /home/rhl/Github

SEED=42
EPOCHS=100
OUT_BASE="./experiments_v1"
LOG_DIR="${OUT_BASE}/logs"
mkdir -p ${LOG_DIR}

L96_IMPUTED=$(ls -t ./lorenz96_rde_delay/results/imputed_100_*.csv | head -1)
L96_GT=$(ls -t ./lorenz96_rde_delay/results/gt_100_*.csv | head -1)
L63_IMPUTED=$(ls -t ./lorenz_rde_delay/results/imputed_100_*.csv | head -1)
L63_GT=$(ls -t ./lorenz_rde_delay/results/gt_100_*.csv | head -1)

echo "================================================================"
echo "  CPU模式基线实验"
echo "  L96: ${L96_IMPUTED}"
echo "  L63: ${L63_IMPUTED}"
echo "================================================================"

# ===== Lorenz63 (15维，最快) =====
echo "[L63-NeuralCDE] 开始 $(date)"
python3 baselines/neuralcde_forecast.py \
    --dataset lorenz63 \
    --data_path "${L63_IMPUTED}" --ground_path "${L63_GT}" \
    --trainlength 60 --horizon_steps 40 \
    --window_size 20 --hidden_channels 64 --num_layers 3 \
    --epochs ${EPOCHS} --batch_size 128 --lr 1e-3 --seed ${SEED} \
    --device auto \
    --out_dir ${OUT_BASE}/lorenz63/neuralcde \
    2>&1 | tee ${LOG_DIR}/l63_neuralcde.log
echo "[L63-NeuralCDE] 完成 $(date)"

echo "[L63-GRUODEBayes] 开始 $(date)"
python3 baselines/gruodebayes_forecast.py \
    --dataset lorenz63 \
    --data_path "${L63_IMPUTED}" --ground_path "${L63_GT}" \
    --trainlength 60 --horizon_steps 40 \
    --window_size 20 --hidden_size 64 \
    --p_hidden 32 --prep_hidden 32 \
    --delta_t 0.1 --time_scale 0.02 --solver euler \
    --epochs ${EPOCHS} --batch_size 128 --lr 1e-3 --seed ${SEED} \
    --device auto \
    --out_dir ${OUT_BASE}/lorenz63/gruodebayes \
    2>&1 | tee ${LOG_DIR}/l63_gruodebayes.log
echo "[L63-GRUODEBayes] 完成 $(date)"

# ===== Lorenz96 (100维) =====
echo "[L96-NeuralCDE] 开始 $(date)"
python3 baselines/neuralcde_forecast.py \
    --dataset lorenz96 \
    --data_path "${L96_IMPUTED}" --ground_path "${L96_GT}" \
    --trainlength 60 --horizon_steps 40 \
    --window_size 20 --hidden_channels 64 --num_layers 3 \
    --epochs ${EPOCHS} --batch_size 128 --lr 1e-3 --seed ${SEED} \
    --device auto \
    --out_dir ${OUT_BASE}/lorenz96/neuralcde \
    2>&1 | tee ${LOG_DIR}/l96_neuralcde.log
echo "[L96-NeuralCDE] 完成 $(date)"

echo "[L96-GRUODEBayes] 开始 $(date)"
python3 baselines/gruodebayes_forecast.py \
    --dataset lorenz96 \
    --data_path "${L96_IMPUTED}" --ground_path "${L96_GT}" \
    --trainlength 60 --horizon_steps 40 \
    --window_size 20 --hidden_size 64 \
    --p_hidden 32 --prep_hidden 32 \
    --delta_t 0.1 --time_scale 0.02 --solver euler \
    --epochs ${EPOCHS} --batch_size 128 --lr 1e-3 --seed ${SEED} \
    --device auto \
    --out_dir ${OUT_BASE}/lorenz96/gruodebayes \
    2>&1 | tee ${LOG_DIR}/l96_gruodebayes.log
echo "[L96-GRUODEBayes] 完成 $(date)"

echo "================================================================"
echo "  NeuralCDE + GRU-ODE-Bayes 完成！"
echo "  SSSD需要GPU，请用 run_all_baselines.sh 在GPU服务器上运行"
echo "================================================================"
