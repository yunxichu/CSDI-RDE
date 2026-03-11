# PM2.5 预测网格搜索工具包

## 📋 目录

1. [概述](#概述)
2. [文件说明](#文件说明)
3. [快速开始](#快速开始)
4. [详细使用指南](#详细使用指南)
5. [输出结果说明](#输出结果说明)
6. [常见问题](#常见问题)

## 概述

本工具包实现了对PM2.5预测模型（RDE-GPR）的自动化网格搜索功能，用于寻找最优的超参数组合。

### 网格搜索参数

- **L（随机嵌入维度）**: 5, 8, 11, 14, 17, 20（间隔3）
- **trainlength（训练窗口长度）**: 200, 400, 600, ..., 2000（间隔200）
- **总组合数**: 60种

### 核心功能

✅ 自动化参数搜索  
✅ 并行计算加速  
✅ 性能指标评估（RMSE、MAE）  
✅ 丰富的可视化输出  
✅ 详细的结果分析  

## 文件说明

### 核心脚本

| 文件名 | 说明 | 用途 |
|--------|------|------|
| `pm25_grid_search.py` | 主网格搜索脚本 | 执行完整的60组参数搜索 |
| `analyze_grid_results.py` | 结果分析脚本 | 深度分析网格搜索结果 |
| `quick_test.sh` | 快速测试脚本 | 小规模测试（4组参数） |
| `run_grid_search.sh` | 完整运行脚本 | 便捷运行完整网格搜索 |

### 文档

| 文件名 | 说明 |
|--------|------|
| `README.md` | 本文件，主要说明文档 |
| `README_grid_search.md` | 详细使用指南 |

## 快速开始

### 第一步：快速测试（推荐）

在运行完整网格搜索前，先用小规模测试验证环境：

```bash
./quick_test.sh /path/to/history_imputed.csv /path/to/pm25_ground.txt
```

这会测试4组参数组合，预计运行时间：10-20分钟。

### 第二步：运行完整网格搜索

如果快速测试成功，运行完整搜索：

```bash
./run_grid_search.sh /path/to/history_imputed.csv /path/to/pm25_ground.txt
```

或者使用Python脚本（更多控制）：

```bash
python pm25_grid_search.py \
  --imputed_history_path /path/to/history_imputed.csv \
  --ground_path /path/to/pm25_ground.txt \
  --split_ratio 0.5 \
  --horizon_days 1 \
  --s 50 \
  --n_jobs 8 \
  --target_indices 0,1,2
```

预计运行时间：5-10小时（取决于硬件）

### 第三步：分析结果

```bash
python analyze_grid_results.py ./save/pm25_grid_search_YYYYMMDD_HHMMSS/
```

## 详细使用指南

### 参数说明

#### 必需参数

- `--imputed_history_path`: 已补值的历史数据（CSV格式）
- `--ground_path`: 原始真值数据（用于时间索引和评估）

#### 可选参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--split_ratio` | 0.5 | 训练/测试集划分比例 |
| `--horizon_days` | 0.0 | 预测天数（>0时优先使用） |
| `--horizon_steps` | 0 | 预测步数 |
| `--s` | 50 | 每步抽样组合数 |
| `--n_jobs` | 8 | 并行进程数 |
| `--steps_ahead` | 1 | 前瞻步数 |
| `--target_indices` | "" | 预测维度（如"0,1,2"） |
| `--noise_strength` | 0.0 | 训练数据加噪强度 |
| `--seed` | 42 | 随机种子 |
| `--out_dir` | "" | 输出目录（默认自动生成） |

### 输出目录结构

```
save/pm25_grid_search_YYYYMMDD_HHMMSS/
├── args.json                          # 运行参数记录
├── grid_search_results.csv            # 所有结果汇总
├── summary.json                       # 最优参数摘要
├── heatmap_rmse.png                   # RMSE热力图
├── heatmap_mae.png                    # MAE热力图
├── trend_plots.png                    # 参数趋势图
├── L{L}_trainlen{trainlength}/        # 每组参数的详细结果
│   ├── future_pred.csv                # 预测值
│   ├── future_pred_std.csv            # 预测标准差
│   ├── metrics_per_dim.csv            # 各维度指标
│   ├── result.json                    # 结果摘要
│   ├── forecast_dim*.png              # 预测对比图
│   └── rmse_per_dim.png               # 各维度RMSE
└── analysis_*.{csv,png,json}          # 结果分析文件
```

## 输出结果说明

### 主要输出文件

#### 1. grid_search_results.csv

包含所有参数组合的性能指标，字段包括：
- `L`: 随机嵌入维度
- `trainlength`: 训练窗口长度
- `rmse`: 均方根误差
- `mae`: 平均绝对误差
- `valid_points`: 有效预测点数
- `elapsed_time`: 计算时间（秒）
- `status`: 运行状态（success/failed）

#### 2. summary.json

包含最优参数信息：
```json
{
  "best_rmse": {
    "L": 11,
    "trainlength": 800,
    "rmse": 10.23,
    "mae": 7.45
  },
  "best_mae": {
    "L": 14,
    "trainlength": 600,
    "rmse": 10.56,
    "mae": 7.12
  }
}
```

#### 3. 可视化图表

- **heatmap_rmse.png**: RMSE热力图，直观显示参数空间的性能分布
- **heatmap_mae.png**: MAE热力图
- **trend_plots.png**: 参数趋势图，显示RMSE随参数变化的趋势
- **forecast_dim*.png**: 各维度的预测vs真值对比图
- **rmse_per_dim.png**: 各维度的RMSE柱状图

### 结果分析输出

运行 `analyze_grid_results.py` 后会生成：

- **analysis_parameter_boxplots.png**: 参数分布箱线图
- **analysis_interaction_heatmaps.png**: 参数交互效应热力图
- **analysis_performance_distribution.png**: 性能分布图
- **analysis_pareto_frontier.png**: 帕累托前沿（性能vs速度权衡）
- **analysis_sensitivity.png**: 敏感性分析
- **analysis_recommendations.json**: 参数选择建议

## 性能指标说明

### RMSE（均方根误差）

\[
\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}
\]

- **特点**: 对大误差更敏感
- **解释**: 值越小表示预测越准确
- **适用**: 当需要惩罚大误差时

### MAE（平均绝对误差）

\[
\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|
\]

- **特点**: 对异常值不敏感
- **解释**: 值越小表示预测越准确
- **适用**: 当需要稳健性度量时

## 常见问题

### Q1: 如何加速网格搜索？

**方法1**: 增加并行进程数
```bash
--n_jobs 16  # 根据CPU核心数调整
```

**方法2**: 减小每步抽样数
```bash
--s 30  # 默认50，可以减小到30
```

**方法3**: 关闭超参数优化
```bash
--no_optimize_hyp
```

### Q2: 部分参数组合失败怎么办？

1. 检查失败组合的 `result.json` 中的错误信息
2. 常见原因：
   - L值超过数据维度
   - trainlength超过历史数据长度
   - 数据中存在异常值

### Q3: 预测结果全为0或NaN？

可能原因：
1. 历史数据最后一行为0（持久性预测导致）
2. `target_indices` 设置错误（空列表）
3. GPR模型大量失败

**解决方法**:
1. 检查 `history_imputed.csv` 最后几行
2. 开启 `--debug` 模式查看详细信息
3. 添加轻微噪声 `--noise_strength 1e-4`

### Q4: 如何选择最优参数？

参考 `analysis_recommendations.json` 中的建议：

1. **最佳性能**: 追求最低RMSE
2. **最快速度**: 在保证良好性能下计算最快
3. **平衡选择**: 性能和速度的最佳平衡
4. **稳健选择**: 中等性能，较为稳健

### Q5: 运行时间太长怎么办？

预计时间：60组合 × 5-10分钟/组 ≈ 5-10小时

**建议**:
1. 先运行 `quick_test.sh` 估算单组时间
2. 在服务器后台运行：
   ```bash
   nohup ./run_grid_search.sh ... > grid_search.log 2>&1 &
   ```
3. 使用多核服务器增大 `--n_jobs`

## 进阶使用

### 自定义网格范围

修改 `pm25_grid_search.py` 中的以下代码：

```python
# 第565-566行
L_values = list(range(5, 21, 3))  # [5, 8, 11, 14, 17, 20]
trainlength_values = list(range(200, 2001, 200))  # [200, 400, ..., 2000]
```

例如，只测试较大的trainlength：
```python
trainlength_values = list(range(800, 2001, 200))  # [800, 1000, ..., 2000]
```

### 添加新的评估指标

在 `compute_metrics` 函数中添加：

```python
def compute_metrics(y_true, y_pred):
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if mask.sum() == 0:
        return {"rmse": np.nan, "mae": np.nan, "mape": np.nan}
    
    diff = y_true[mask] - y_pred[mask]
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(np.abs(diff)))
    
    # 添加MAPE
    mape = float(np.mean(np.abs(diff / y_true[mask])) * 100)
    
    return {"rmse": rmse, "mae": mae, "mape": mape}
```

## 技术支持

如有问题，请：
1. 查看日志文件（`grid_search.log`）
2. 检查 `debug_report.json`（如果使用了 `--debug`）
3. 参考 `README_grid_search.md` 获取更详细的说明

## 更新日志

### v1.0 (2026-01-29)
- ✅ 初始版本发布
- ✅ 实现L和trainlength网格搜索
- ✅ 添加性能指标评估
- ✅ 添加可视化功能
- ✅ 添加结果分析工具

---

**作者**: Claude  
**版本**: 1.0  
**更新日期**: 2026-01-29
