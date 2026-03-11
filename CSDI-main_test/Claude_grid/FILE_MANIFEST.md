# PM2.5 网格搜索工具包 - 文件清单

## 📦 工具包内容（共9个文件）

### 🔧 核心脚本（4个）

1. **pm25_grid_search.py** (30KB)
   - 主要网格搜索脚本
   - 对L和trainlength进行60组参数搜索
   - 自动生成性能指标和可视化
   - 用法：`python pm25_grid_search.py --imputed_history_path ... --ground_path ...`

2. **analyze_grid_results.py** (18KB)
   - 结果深度分析脚本
   - 生成参数重要性、交互效应、敏感性分析
   - 生成帕累托前沿和参数推荐
   - 用法：`python analyze_grid_results.py <grid_search_output_dir>`

3. **quick_test.sh** (9.5KB, 可执行)
   - 快速测试脚本（4组参数）
   - 用于验证环境和估算运行时间
   - 预计时间：10-20分钟
   - 用法：`./quick_test.sh <history_imputed.csv> <ground.txt>`

4. **run_grid_search.sh** (4.8KB, 可执行)
   - 完整网格搜索运行脚本（60组参数）
   - 支持后台运行和日志记录
   - 预计时间：5-10小时
   - 用法：`./run_grid_search.sh <history_imputed.csv> <ground.txt> [可选参数]`

### 📚 文档（4个）

5. **README.md** (8.6KB)
   - 主要说明文档
   - 包含概述、快速开始、详细指南
   - 包含常见问题和技术支持

6. **README_grid_search.md** (6.5KB)
   - 详细使用指南
   - 包含参数说明、结果解读、公式说明
   - 包含示例和最佳实践

7. **QUICK_REFERENCE.md** (2.8KB)
   - 快速参考卡片
   - 常用命令速查
   - 参数配置速查表

8. **config_example.txt** (1.8KB)
   - 示例配置文件
   - 可作为模板复制修改
   - 包含参数说明和调优建议

### 📋 本文件

9. **FILE_MANIFEST.md**
   - 本文件，文件清单
   - 列出所有文件及其用途

## 🎯 使用流程

### 第一次使用

```bash
# 1. 阅读文档
cat README.md

# 2. 快速测试
./quick_test.sh your_history.csv your_ground.txt

# 3. 如果测试成功，运行完整搜索
./run_grid_search.sh your_history.csv your_ground.txt

# 4. 分析结果
python analyze_grid_results.py ./save/pm25_grid_search_YYYYMMDD_HHMMSS/
```

### 日常使用

```bash
# 使用配置文件运行
cp config_example.txt my_config.txt
# 编辑 my_config.txt 修改参数
python pm25_grid_search.py @my_config.txt

# 快速查看结果
cat save/*/summary.json

# 查看可视化
open save/*/heatmap_rmse.png
```

## 📊 输出说明

### 主要输出文件

运行网格搜索后，会在 `save/pm25_grid_search_YYYYMMDD_HHMMSS/` 生成：

**汇总文件**:
- `args.json` - 运行参数记录
- `grid_search_results.csv` - 所有60组结果
- `summary.json` - 最优参数摘要

**可视化文件**:
- `heatmap_rmse.png` - RMSE热力图
- `heatmap_mae.png` - MAE热力图
- `trend_plots.png` - 参数趋势图

**详细结果** (每组参数一个子目录):
- `L{L}_trainlen{trainlength}/`
  - `future_pred.csv` - 预测值
  - `future_pred_std.csv` - 预测标准差
  - `metrics_per_dim.csv` - 各维度指标
  - `result.json` - 结果摘要
  - `forecast_dim*.png` - 预测对比图
  - `rmse_per_dim.png` - 各维度RMSE

### 分析结果文件

运行 `analyze_grid_results.py` 后额外生成：

**统计文件**:
- `analysis_L_statistics.csv` - L参数统计
- `analysis_trainlen_statistics.csv` - trainlength参数统计
- `analysis_L_sensitivity.csv` - L敏感性分析
- `analysis_trainlen_sensitivity.csv` - trainlength敏感性分析
- `analysis_pareto_optimal.csv` - 帕累托最优解
- `analysis_recommendations.json` - 参数选择建议

**可视化文件**:
- `analysis_parameter_boxplots.png` - 参数分布箱线图
- `analysis_interaction_heatmaps.png` - 参数交互效应
- `analysis_performance_distribution.png` - 性能分布图
- `analysis_pareto_frontier.png` - 帕累托前沿
- `analysis_sensitivity.png` - 敏感性分析图

## 💻 系统要求

### Python依赖

```bash
numpy
pandas
matplotlib
seaborn
scipy
scikit-learn
tqdm
```

安装命令：
```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn tqdm
```

### 硬件建议

- **CPU**: 8核或更多（用于并行计算）
- **内存**: 8GB+
- **磁盘**: 1GB+（用于存储结果）
- **运行时间**: 5-10小时（完整搜索）

## 🔄 版本信息

- **版本**: 1.0
- **更新日期**: 2026-01-29
- **作者**: Claude
- **Python版本**: 3.7+

## 📞 支持

如遇问题：
1. 查看 `README.md` 的"常见问题"部分
2. 检查运行日志 `grid_search.log`
3. 查看失败组合的 `result.json`

## 📄 许可

本工具包为学术研究使用。使用时请遵守相关数据使用规定。

---

**完整工具包大小**: 约82KB（压缩前）
**建议Python版本**: 3.8+
**测试环境**: Ubuntu 24, Python 3.11
