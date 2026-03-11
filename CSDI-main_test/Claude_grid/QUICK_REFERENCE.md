# PM2.5 网格搜索 - 快速参考卡

## 🚀 快速命令

### 快速测试（4组参数，10-20分钟）
```bash
./quick_test.sh history_imputed.csv pm25_ground.txt
```

### 完整搜索（60组参数，5-10小时）
```bash
./run_grid_search.sh history_imputed.csv pm25_ground.txt
```

### 分析结果
```bash
python analyze_grid_results.py ./save/pm25_grid_search_YYYYMMDD_HHMMSS/
```

## 📊 网格参数

| 参数 | 值 | 数量 |
|------|-------|------|
| L | 5, 8, 11, 14, 17, 20 | 6 |
| trainlength | 200, 400, ..., 2000 | 10 |
| **总组合** | - | **60** |

## ⚙️ 常用参数

```bash
--split_ratio 0.5         # 训练集比例
--horizon_days 1          # 预测1天
--s 50                    # 每步50个组合
--n_jobs 8                # 8个并行进程
--target_indices 0,1,2    # 预测维度0,1,2
--noise_strength 1e-4     # 添加噪声
--no_optimize_hyp         # 关闭超参优化（加速）
```

## 📁 重要输出文件

| 文件 | 说明 |
|------|------|
| `summary.json` | 最优参数 |
| `grid_search_results.csv` | 所有结果 |
| `heatmap_rmse.png` | 性能热力图 |
| `analysis_recommendations.json` | 参数建议 |

## 🔍 查看结果

### 最优参数
```bash
cat save/pm25_grid_search_*/summary.json | grep -A 5 best_rmse
```

### 性能排名
```bash
sort -t',' -k3 -n save/pm25_grid_search_*/grid_search_results.csv | head -10
```

## 💡 性能优化

| 目标 | 方法 | 示例 |
|------|------|------|
| 加速 | 增加进程 | `--n_jobs 16` |
| 加速 | 减少抽样 | `--s 30` |
| 加速 | 关闭优化 | `--no_optimize_hyp` |
| 提高精度 | 增加抽样 | `--s 100` |
| 提高精度 | 添加噪声 | `--noise_strength 1e-4` |

## 🐛 常见错误

| 错误 | 原因 | 解决 |
|------|------|------|
| 预测全0 | 历史末行为0 | 检查history_imputed.csv |
| 部分失败 | L超过维度 | 检查result.json |
| NaN输出 | GPR失败 | 添加 --noise_strength 1e-4 |

## 📈 指标解释

**RMSE**: 均方根误差，对大误差敏感  
**MAE**: 平均绝对误差，更稳健

\[
\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}
\]

\[
\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|
\]

## 🎯 推荐工作流

1. **快速测试** → 验证环境（10-20分钟）
2. **完整搜索** → 找最优参数（5-10小时）
3. **结果分析** → 深入理解（5分钟）
4. **选择参数** → 查看recommendations.json
5. **验证结果** → 检查最优组的可视化

## 📞 帮助

```bash
python pm25_grid_search.py --help
python analyze_grid_results.py --help
```

详细文档：`README.md` 和 `README_grid_search.md`
