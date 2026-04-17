# 2026-04-17 Claude 对话快照（当天实验补跑全过程）

本文档是当天 Claude Code 协助补跑 CSDI-RDE-GPR vs 基线实验的关键对话摘要，记录决策轨迹和认识迭代。

## 时间线

### 上午：实验审计
1. 用户问 SSSD 实验跑完没 → 检查发现 Lorenz63 EEG 已完成，PM25 仍在跑（GPU 7）
2. 用户要求做对比评估 → 发现 [visualization/baseline_comparison.py](../visualization/baseline_comparison.py) 里 Lorenz63 RDE-Delay 的 0.22 其实是 CSDI 补值 RMSE 而不是预测 RMSE（来自 `25experiments.csv` 第 0 列误读）
3. 深度审计所有数据源：Lorenz96 的 0.34 是单次实验不是均值，PM25 的 11.42 是 cherry-picked 单站点，EEG 的 7.53 是单步滚动模式与前馈基线不可比

### 中午：补跑计划与执行
4. 用户要求补齐实验，从紧迫的开始
5. 执行：
   - Lorenz96 SSSD v2（GPU 3，~15 min 完，RMSE=6.66）
   - Lorenz63 RDE-Delay 5 seeds（CPU 4 核，RMSE=1.40±0.41）
   - Lorenz96 RDE-Delay 5 seeds（CPU 4 核，RMSE=0.26±0.11）
   - EEG RDE-GPR Mode A direct（91.06 — 后来发现跑偏）
   - EEG GRU-ODE-Bayes Mode B h=976（重复 experiments_v2 的数据验证）

### 下午早：模式对齐的重大修正
6. 用户询问 PM25 大家都做 teacher-forcing 是否可行、做过没
7. 调研发现 EEG 上 `eeg_forecast_comparison.py` 已做过 Mode B 统一对齐（结题报告 Table 5）
8. **重大修正**：核查所有基线脚本源码，发现
   - 通用 `baselines/neuralcde_forecast.py`、`gruodebayes_forecast.py`、`sssd_forecast.py` **默认都是 Mode B 滚动 teacher-forcing**
   - experiments_v2 全部基线默认就是 Mode B → 和 RDE-Delay 已模式对齐
   - 我之前 "EEG RDE-GPR Mode A direct=91" 是跑偏方向
9. 重新跑 EEG RDE-GPR Mode B 默认滚动 → RMSE=12.30（正确对齐版）

### 下午：方法命名修正
10. 用户指出 Lorenz63/96 的 "RDE" 就是 RDE-GPR
11. 修正 comparison_v3 的表格列名为 RDE-GPR（空间集成）和 RDE-Delay-GPR（延迟嵌入）

### 下午晚：发现 EEG 输入错误
12. 用户问"试试 EEG 没 delay 版"，同时指出"对比应该是基线和 CSDI 补值后的 RDE-GPR"
13. **发现 bug**：我之前 EEG RDE-GPR 的 `--imputed_path` 用了 `eeg_full.npy`（真值），而不是 `eeg_imputed.npy`（CSDI 补值版）
14. 重跑 EEG RDE-GPR：
    - 空间集成版（新）
    - RDE-Delay-GPR 重跑（用正确 imputed）→ RMSE=12.13（稍微好过 12.30）

### 关键澄清（用户 2026-04-17 下午 17:30 前后）
15. 用户澄清**方法定位**：
    > "我的方法是 CSDI-RDE-GPR（包含有无 delay），所以意思是我本身应该是稀疏数据集上面去做，对应的我对比的基线方法是不能补值的，就必须是有缺失的情况下直接跑基线，然后再来对比我的 CSDI-RDE-GPR"
16. **最终实验设计分两 track**：
    - Track-A（保留）：都用 CSDI 补值数据 → 比较预测模型能力
    - Track-B（新增）：基线直接处理稀疏/缺失数据 → 比较 CSDI+RDE-GPR 整套 pipeline vs 基线端到端

## 认识迭代记录

| 阶段 | 我的认识 | 实际情况 |
|------|----------|----------|
| 初始 | 基线是 Mode A 前馈 | 其实是 Mode B 滚动 teacher-forcing |
| 修正1 | EEG RDE-GPR Mode B 输入用 eeg_full.npy 即可 | 应该用 eeg_imputed.npy 才对齐基线 |
| 修正2 | 对比两边都用 CSDI 补值数据就是公平 | 这样没体现 CSDI 补值的价值；基线应该用稀疏/缺失原始数据 |

## 当天产出（commit 轨迹）

1. `audit: 实验数据审计 + 修正 baseline_comparison 脚本`
2. `exp: Lorenz63/96 RDE-Delay 5-seed aligned eval + Lorenz96 SSSD v2`
3. `exp: EEG RDE-GPR Mode A 前馈`
4. `docs: 新增完整实验清单 (已做/未做/运行中)`
5. `exp: EEG GRU-ODE-Bayes Mode B h=976`
6. `exp: EEG RDE-GPR Mode B h=976 (对齐 experiments_v2 基线)`
7. `docs: 彻底修正实验清单 — experiments_v2 基线全部是 Mode B 滚动 teacher-forcing`
8. `docs: 新增 experiments_v2/comparison_v3/ 对比汇总文件夹`
9. `docs: 修正方法命名 — RDE-GPR (空间集成) / RDE-Delay-GPR (延迟嵌入)`
10. 即将：`docs: 分 Track-A 预处理对齐 / Track-B 完整 pipeline 两个对比维度 + 补跑 Lorenz Sparse 基线`

## 关键文件索引

- [session_notes/2026-04-17_experiment_audit_and_plan.md](2026-04-17_experiment_audit_and_plan.md)
- [session_notes/2026-04-17_experiment_inventory.md](2026-04-17_experiment_inventory.md)
- [session_notes/2026-04-17_full_pipeline_comparison.md](2026-04-17_full_pipeline_comparison.md)
- [experiments_v2/comparison_v3/](../experiments_v2/comparison_v3/)
- [experiments_v2/logs/](../experiments_v2/logs/)
