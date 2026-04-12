#!/bin/bash
# 一键运行所有可视化脚本，生成高质量对比图

set -e  # 遇到错误立即退出

echo "=========================================="
echo "CSDI-RDE-GPR 实验可视化一键运行脚本"
echo "=========================================="
echo ""

# 创建输出目录
mkdir -p ./save/visualization_results

# 1. EEG 对比图可视化
echo "[1/3] 生成 EEG 高质量对比图..."
if [ -d "./save/eeg_comparison" ]; then
    python visualization/eeg_full_comparison.py \
        --results_dir ./save/eeg_comparison \
        --out_dir ./save/visualization_results/eeg \
        --ground_path ./data/eeg/eeg_ground.npy
    echo "✓ EEG 对比图生成完成"
else
    echo "✗ 未找到 EEG 对比结果目录: ./save/eeg_comparison"
    echo "  请先运行: python baselines/eeg_forecast_comparison.py"
fi
echo ""

# 2. PM2.5 对比图可视化
echo "[2/3] 生成 PM2.5 高质量对比图..."
if [ -d "./save/pm25_comparison" ]; then
    python visualization/pm25_full_comparison.py \
        --results_dir ./save/pm25_comparison \
        --out_dir ./save/visualization_results/pm25
    echo "✓ PM2.5 对比图生成完成"
else
    echo "✗ 未找到 PM2.5 对比结果目录: ./save/pm25_comparison"
    echo "  请先运行 PM2.5 预测实验"
fi
echo ""

# 3. 复制 Lorenz 系统的高质量对比图
echo "[3/3] 复制 Lorenz 系统对比图..."
mkdir -p ./save/visualization_results/lorenz96
mkdir -p ./save/visualization_results/lorenz63

# Lorenz96
if [ -f "./lorenz96_rde_delay/results/full_comparison_20260323_192045.png" ]; then
    cp ./lorenz96_rde_delay/results/full_comparison_20260323_192045.png \
       ./save/visualization_results/lorenz96/full_comparison.png
    cp ./lorenz96_rde_delay/results/imputation_quality_20260323_192045.png \
       ./save/visualization_results/lorenz96/imputation_quality.png
    echo "✓ Lorenz96 对比图复制完成"
else
    echo "✗ 未找到 Lorenz96 对比图"
fi

# Lorenz63
if [ -f "./lorenz_rde_delay/results/full_comparison_20260320_110418.png" ]; then
    cp ./lorenz_rde_delay/results/full_comparison_20260320_110418.png \
       ./save/visualization_results/lorenz63/full_comparison.png
    cp ./lorenz_rde_delay/results/imputation_quality_20260320_105649.png \
       ./save/visualization_results/lorenz63/imputation_quality.png
    echo "✓ Lorenz63 对比图复制完成"
else
    echo "✗ 未找到 Lorenz63 对比图"
fi
echo ""

# 4. 汇总结果
echo "=========================================="
echo "可视化结果汇总"
echo "=========================================="
echo ""
echo "所有对比图已保存到: ./save/visualization_results/"
echo ""
echo "目录结构:"
find ./save/visualization_results -type f -name "*.png" 2>/dev/null | head -20 || echo "  (暂无结果)"
echo ""
echo "=========================================="
echo "完成!"
echo "=========================================="
