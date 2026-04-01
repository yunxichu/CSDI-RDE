#!/bin/bash
# 批量测试不同L值的效果

IMPUTED_PATH="./save/physio_history_imputed_split0.5_seed1_20260325_001513/history_imputed.npy"
S=100
TRAINLENGTH=36
N_JOBS=2
TARGET_INDICES="0,1,2,3"
MAX_SAMPLES=100

echo "========================================"
echo "批量测试L值 (8-15)"
echo "========================================"

for L in {8..15}; do
    echo ""
    echo "=== 测试 L=$L ==="
    
    python rde_gpr/physio_CSDIimpute_after-RDEgpr.py \
        --imputed_history_path $IMPUTED_PATH \
        --history_timesteps 36 \
        --horizon_timesteps 12 \
        --use_ground_truth_sliding \
        --L $L \
        --s $S \
        --trainlength $TRAINLENGTH \
        --n_jobs $N_JOBS \
        --target_indices $TARGET_INDICES \
        --max_samples $MAX_SAMPLES \
        --out_dir ./save/physio_test_L${L}/
    
    echo "完成 L=$L"
done

echo ""
echo "========================================"
echo "所有测试完成！"
echo "========================================"

# 汇总结果
echo ""
echo "=== 结果汇总 ==="
for L in {8..15}; do
    if [ -f "./save/physio_test_L${L}/metrics.json" ]; then
        echo ""
        echo "L=$L:"
        cat ./save/physio_test_L${L}/metrics.json
    fi
done
