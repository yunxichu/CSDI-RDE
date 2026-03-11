# PM2.5 预测网格搜索使用说明

## 功能概述

该脚本对 RDE-GPR 预测模型的两个关键参数进行网格搜索：
- **L**（随机嵌入维度）: 5, 8, 11, 14, 17, 20（间隔3）
- **trainlength**（训练窗口长度）: 200, 400, 600, ..., 2000（间隔200）

共计 **6 × 10 = 60** 种参数组合。

## 使用方法

### 基本命令

```bash
python pm25_grid_search.py \
  --imputed_history_path /home/rhl/CSDI-main_test/save/pm25_history_imputed_split0.5_seed42_20260128_101132/history_imputed.csv \
  --ground_path /home/rhl/CSDI-main_test/data/pm25/Code/STMVL/SampleData/pm25_ground.txt \
  --split_ratio 0.5 \
  --horizon_days 1 \
  --s 50 \
  --n_jobs 8 \
  --target_indices 0,1,2
```

### 参数说明

#### 必需参数
- `--imputed_history_path`: 已补值的历史数据（CSV格式）
- `--ground_path`: 原始真值数据（用于时间索引和评估）

#### 数据划分
- `--split_ratio`: 训练集比例（默认0.5）

#### 预测范围
- `--horizon_days`: 预测天数（优先级高）
- `--horizon_steps`: 预测步数（当horizon_days=0时生效）

#### RDE-GPR参数
- `--s`: 每步抽样组合数（默认50）
- `--steps_ahead`: 前瞻步数（默认1）
- `--n_jobs`: 并行进程数（默认8）
- `--noise_strength`: 训练数据加噪强度（默认0.0）
- `--no_optimize_hyp`: 是否关闭超参数优化

#### 其他
- `--target_indices`: 预测维度（如"0,1,2"，空=全维）
- `--out_dir`: 输出目录（默认自动生成）
- `--seed`: 随机种子（默认42）

## 输出结构

运行完成后会生成如下目录结构：

```
save/pm25_grid_search_YYYYMMDD_HHMMSS/
├── args.json                          # 运行参数
├── grid_search_results.csv            # 所有结果汇总表
├── summary.json                       # 最优参数摘要
├── heatmap_rmse.png                   # RMSE热力图
├── heatmap_mae.png                    # MAE热力图
├── trend_plots.png                    # 参数趋势图
└── L{L}_trainlen{trainlength}/        # 每组参数的详细结果
    ├── future_pred.csv                # 预测值
    ├── future_pred_std.csv            # 预测标准差
    ├── metrics_per_dim.csv            # 各维度指标
    ├── result.json                    # 该组结果摘要
    ├── forecast_dim0.png              # 维度0预测对比图
    ├── forecast_dim1.png              # 维度1预测对比图
    ├── forecast_dim2.png              # 维度2预测对比图
    └── rmse_per_dim.png               # 各维度RMSE柱状图
```

## 结果解读

### 1. grid_search_results.csv

包含所有参数组合的性能指标：

| L | trainlength | rmse | mae | valid_points | elapsed_time | status |
|---|-------------|------|-----|--------------|--------------|--------|
| 5 | 200 | 12.34 | 8.56 | 24 | 45.2 | success |
| 5 | 400 | 11.89 | 8.12 | 24 | 52.1 | success |
| ... | ... | ... | ... | ... | ... | ... |

### 2. summary.json

包含最优参数信息：

```json
{
  "total_combinations": 60,
  "successful_runs": 58,
  "failed_runs": 2,
  "best_rmse": {
    "L": 11,
    "trainlength": 800,
    "rmse": 10.23,
    "mae": 7.45,
    "output_dir": "..."
  },
  "best_mae": {
    "L": 14,
    "trainlength": 600,
    "rmse": 10.56,
    "mae": 7.12,
    "output_dir": "..."
  }
}
```

### 3. 可视化图片

#### heatmap_rmse.png / heatmap_mae.png
- 横轴：L值
- 纵轴：trainlength值
- 颜色：RMSE/MAE大小（颜色越深=性能越好）
- 用于快速识别最优参数区域

#### trend_plots.png
- 左图：RMSE vs L（不同trainlength）
- 右图：RMSE vs trainlength（不同L）
- 用于分析参数对性能的影响趋势

#### forecast_dim*.png
- 展示真实值 vs 预测值的时序对比
- 每个维度一张图

#### rmse_per_dim.png
- 各维度的RMSE柱状图
- 用于识别哪些维度预测效果较差

## 性能优化建议

### 加速运行
1. 增加 `--n_jobs`（根据CPU核心数）
2. 减小 `--s`（降低每步的组合采样数）
3. 使用 `--no_optimize_hyp`（关闭超参数优化）

### 提高精度
1. 增大 `--s`（更多组合采样）
2. 开启超参数优化（默认已开启）
3. 添加轻微噪声 `--noise_strength 1e-4`

## 常见问题

### Q1: 部分参数组合失败怎么办？
A: 检查 `result.json` 中的 `error` 字段，常见原因：
- L值过大（超过数据维度）
- trainlength超过历史数据长度
- 数据中存在异常值

### Q2: 预测结果全为0或NaN？
A: 可能原因：
- 历史数据最后一行为0
- target_indices设置错误（空列表）
- GPR模型大量失败

### Q3: 运行时间过长？
A: 估算：60组合 × 每组5-10分钟 ≈ 5-10小时
建议：
- 先用小规模测试（如只测试几个L值）
- 使用多核并行（增大n_jobs）
- 在服务器后台运行

## 下一步建议

1. **查看summary.json**：快速找到最优参数
2. **检查热力图**：理解参数空间的性能分布
3. **分析趋势图**：判断是否需要扩展搜索范围
4. **检查最优组的详细结果**：
   - 查看预测曲线是否合理
   - 检查各维度的预测质量
   - 确认是否存在过拟合

## 示例：完整运行流程

```bash
# 1. 运行网格搜索
python pm25_grid_search.py \
  --imputed_history_path ./data/history_imputed.csv \
  --ground_path ./data/pm25_ground.txt \
  --split_ratio 0.5 \
  --horizon_days 1 \
  --s 50 \
  --n_jobs 8 \
  --target_indices 0,1,2

# 2. 运行完成后查看结果
cd save/pm25_grid_search_YYYYMMDD_HHMMSS/

# 3. 查看最优参数
cat summary.json

# 4. 查看可视化
open heatmap_rmse.png
open trend_plots.png

# 5. 查看最优组的详细结果
cd L11_trainlen800/
open forecast_dim0.png
```

## 公式说明

### RMSE（均方根误差）

\[
\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}
\]

其中：
- \(y_i\) 是真实值
- \(\hat{y}_i\) 是预测值
- \(n\) 是样本数量

RMSE对大误差更敏感，值越小表示预测越准确。

### MAE（平均绝对误差）

\[
\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|
\]

MAE对异常值不如RMSE敏感，更稳健。

## 技术细节

### RDE-GPR算法
- **RDE**: Random Embedding（随机嵌入）
- **GPR**: Gaussian Process Regression（高斯过程回归）
- 核心思想：从 \(D\) 维特征中随机选择 \(L\) 维组合，训练多个GPR模型，通过KDE融合预测

### 参数影响
- **L**: 控制模型复杂度
  - L太小：欠拟合
  - L太大：过拟合，计算慢
- **trainlength**: 控制历史信息量
  - 太小：信息不足
  - 太大：引入噪声，计算慢

---

**作者**: Claude
**版本**: 1.0
**更新日期**: 2026-01-29
