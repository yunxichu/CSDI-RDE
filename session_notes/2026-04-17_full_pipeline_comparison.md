# 2026-04-17 Pivotal Correction: Full-Pipeline vs Preprocessing-Aligned 对比

## 关键澄清（用户 2026-04-17 下午对话）

**之前的实验设计问题**：
- experiments_v2 里所有基线（NeuralCDE / GRU-ODE-Bayes / SSSD）用的 **data_path 都是 CSDI 补值后的数据**（`imputed_100.csv` / `history_imputed.csv` / `eeg_imputed.npy`）
- 这等于把 CSDI 补值的好处**也给了基线**，没有体现 "CSDI 补值 → RDE-GPR" 整套 pipeline 的价值
- 我的方法 CSDI-RDE-GPR 本意应该是"**处理含稀疏/缺失的原始数据**"，对应的基线应该**直接在缺失数据上跑**（这些基线天然支持不规则/缺失时间序列）

## 两种对比维度（都保留）

### Track-A 预处理对齐对比（现有 experiments_v2 全部数据）
- **目的**：在"已经 CSDI 补值的相同数据"上对比"哪种预测模型更好"
- **基线输入**：CSDI 补值后的完整数据
- **我的方法输入**：同样的 CSDI 补值后数据 → RDE-GPR
- **对比意义**：方法本身的预测能力差异（不包含补值贡献）
- **现有数据**：全部已跑 / 跑着（见 inventory v2）

### Track-B 完整 pipeline 对比（新增补跑）
- **目的**：展示 "CSDI-RDE-GPR 整套方法" vs "基线直接处理稀疏数据"
- **基线输入**：**原始稀疏/缺失数据**（不做任何补值，或用朴素的 naive fill 作 naive baseline）
- **我的方法输入**：原始稀疏 → CSDI 补值 → RDE-GPR
- **对比意义**：展示 CSDI 补值 + RDE-GPR 的端到端价值
- **现有数据**：无，**需要重跑全部基线**

## Track-B 补跑实验清单

| 数据集 | 基线新输入 | 基线脚本改造 | 我的方法（已有） |
|--------|-----------|--------------|------------------|
| Lorenz63 | sparse_50.csv（50 点每 8 步）trainlength=30 horizon=20 | 改 data_path + trainlength 参数即可 | `summary_*.txt` 里已有 Sparse→Imputed 对比 |
| Lorenz96 | 同上 | 同上 | 同上 |
| PM2.5 | 原始含缺失 hist (pm25_ground 前半段 + 真实缺失 mask) | 基线通用脚本需确认能接受 NaN | 需重新从头跑整个 CSDI-RDE-GPR pipeline |
| EEG | 原始 50% 缺失 EEG (用 missing_positions 生成) | 基线需接受缺失输入 | 需重新从头跑整个 pipeline |

## 预期效果

Track-B 下 CSDI-RDE-GPR 应该比"基线直接处理稀疏"有显著提升，因为：
1. CSDI 补值填充缺失位置 → 数据完整
2. RDE-GPR 在完整数据上做精细滚动预测 → 数据能力被发挥

Track-A 下 CSDI-RDE-GPR 可能和基线差距没那么大（都用了补值数据）。

## 执行顺序

1. 🟢 **Lorenz63/96 Track-B**：最简单，sparse_50 数据已可从 gt_100 提取，基线脚本不需要大改。GPU 30-60 min/数据集。
2. 🟡 **PM2.5 Track-B**：需确认基线通用脚本对 NaN 的处理。1-2h。
3. 🟡 **EEG Track-B**：需从 `missing_positions.csv` + `eeg_full.npy` 生成真实缺失数据。1-2h。

先做 Lorenz63/96，再做 PM25 / EEG。

## 报告文档要更新

- [ ] README.md：明确 "CSDI-RDE-GPR 是完整 pipeline"，对比基线也应对齐
- [ ] 结题报告.md：Table 7 主对比应该是 Track-B，Track-A 作为"方法本身预测能力"补充
- [ ] comparison_v3/build_comparison.py：扩展生成 A 和 B 两张对比表
- [ ] experiments_v2/comparison_v3/summary.md：增加 Track A/B 的说明
