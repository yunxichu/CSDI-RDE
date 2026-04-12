#!/bin/bash
# PM2.5 RDE-GPR 快速测试脚本
# 只跑前4个站点，减少计算量

set -e

GROUND_PATH="./data/pm25/Code/STMVL/SampleData/pm25_ground.txt"
SPLIT_RATIO=0.5
SEED=42
HORIZON_DAYS=1

IMPUTED_DIR=$(ls -d ./save/pm25_history_imputed_split${SPLIT_RATIO}_seed${SEED}_* 2>/dev/null | head -1)
if [ -z "${IMPUTED_DIR}" ]; then
    echo "❌ 找不到补值结果，请先运行CSDI补值"
    exit 1
fi
IMPUTED_HISTORY_PATH="${IMPUTED_DIR}/history_imputed.csv"
echo "✅ 找到补值结果: ${IMPUTED_DIR}"

CMP_DIR="./save/pm25_comparison_rde_fast"
mkdir -p ${CMP_DIR}/rdegpr

echo ""
echo "================================================================"
echo "  PM2.5 RDE-GPR 快速测试 (前4个站点)"
echo "================================================================"
echo "  Ground: ${GROUND_PATH}"
echo "  History: ${IMPUTED_HISTORY_PATH}"
echo "  Split: ${SPLIT_RATIO}, Horizon: ${HORIZON_DAYS}天, Seed: ${SEED}"
echo "  站点: 0,1,2,3 (前4个)"
echo "  参数: L=4, s=20, trainlength=1000, n_jobs=4"
echo "================================================================"

# 运行 RDE-GPR 前4个站点
echo ""
echo "=== 运行 RDE-GPR ==="
python rde_gpr/pm25_CSDIimpute_after-RDEgpr.py \
    --imputed_history_path "${IMPUTED_HISTORY_PATH}" \
    --ground_path "${GROUND_PATH}" \
    --split_ratio ${SPLIT_RATIO} \
    --horizon_days ${HORIZON_DAYS} \
    --L 4 --s 20 --trainlength 1000 --n_jobs 4 \
    --target_indices "0,1,2,3" \
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

# 生成对比图
echo ""
echo "================================================================"
echo "  生成对比可视化"
echo "================================================================"

python visualization/pm25_full_comparison.py \
    --results_dir ${CMP_DIR} \
    --out_dir ./save/visualization_results/pm25_rde_fast \
    --ground_path "${GROUND_PATH}" \
    --split_ratio ${SPLIT_RATIO} \
    --horizon_steps 24

echo ""
echo "================================================================"
echo "  🎉 RDE-GPR 快速测试完成！"
echo "  结果目录: ${CMP_DIR}/"
echo "  可视化: ./save/visualization_results/pm25_rde_fast/"
echo "================================================================"
