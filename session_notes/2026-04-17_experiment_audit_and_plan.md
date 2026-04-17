# 2026-04-17 实验审计与补跑计划

## 项目定位

- **核心方法**：CSDI 补值 → RDE / RDE-Delay → GPR 预测，用于**高缺失率 / 高维**时序数据
- **基线选择原则**：所有基线都是能处理缺失值的时序预测方法（NeuralCDE / GRU-ODE-Bayes / SSSD）
- **对比目的**：体现 RDE-Delay 在高缺失 + 高维场景下优于端到端深度学习基线

## 一、当前实验状态

### 已完成 / 仍在跑（基线 experiments_v2）

| 数据集 | NeuralCDE | GRU-ODE-Bayes | SSSD v1 (mask错误) | SSSD v2 (mask修复) |
|--------|-----------|---------------|---------------------|---------------------|
| Lorenz63 | ✅ 6.05 | ✅ 5.97 | 18.80 | ✅ 15.21 |
| Lorenz96 | ✅ 9.94 | ✅ 4.10 | 5.59 | ❌ **缺** |
| PM2.5 | ✅ 15.06 | ✅ 20.99 | 105.21 (NaN多) | 🔄 **~29% 还在跑** (GPU 7, PID 909979, ≈30h剩) |
| EEG | ✅ 17.04 | ✅ 6.24 | 87.57 | ✅ 64.06 |

基线统一设置：**前馈一次性预测**，Adam lr=1e-3, batch 128
- Lorenz63/96: trainlength=60, horizon=40
- PM25: history=4379, horizon=24, target=全 36 站
- EEG: history=976, horizon=24, target=0,1,2

### RDE-Delay 实验数据的严重问题

#### 问题 1：Lorenz63 结题报告 0.22 是错的
- `lorenz_rde_delay/results/25experiments.csv` 是 20×5 矩阵
- 列含义（[experiment_summary.md](../lorenz_rde_delay/experiment_summary.md)）：
  - col 0: **CSDI 补值 RMSE** (均值 0.2161)
  - col 1: RDE(稀疏) RMSE (2.04)
  - col 2: RDE-Delay(稀疏) RMSE (3.53)
  - col 3: RDE(补值) RMSE (1.23)
  - col 4: **RDE-Delay(补值) RMSE (1.43)** ← 真实值
- 结题报告 Table 7 把 **0.22** 当成 RDE-Delay，实为 CSDI 补值 RMSE
- 原 [visualization/baseline_comparison.py](../visualization/baseline_comparison.py) 读 col[0,1] 也是错的

#### 问题 2：Lorenz96 结题报告 0.34 是单次结果不是均值
- `lorenz96_rde_delay/results/` 下 8 次 summary_*.txt RDE-Delay(Imputed→20) RMSE：
  ```
  2.6927, 4.6987 (失败: CSDI RMSE=2.87/1.x)
  0.3402 (20260323_183409) ← 结题报告用的单次
  0.3336, 0.1798, 0.2918, 0.3848, 0.2317
  ```
- 去失败 6 次均值 = 0.294，中位数 0.34
- **此外还存在 horizon 不对齐**：RDE 评估的是"前 20 步"，基线评估的是"40 步"

#### 问题 3：PM2.5 11.42 是 cherry-picked 单站点
- `best_record/pm25_test_plot_with_history_v3/`
  - 整体（3 站）：RMSE=13.90, MAE=9.27
  - 站点 001001：RMSE=11.42（结题报告用这个）
  - 站点 001002：RMSE=16.75
  - 站点 001003：RMSE=9.44
- `best_record/pm25_rc_rde_0.5_42_20260317_122531/` 全 36 站：RMSE=15.39
- 基线是全 36 站点平均 → **公平对比应用 15.39 或 13.90，不是 11.42**

#### 问题 4：EEG 7.53 是单步滚动模式，与前馈基线不可比
- `save/eeg_rdegpr_h100_horizon24_20260331_013232/`：history=100, horizon=24, **前馈** → RMSE=63.93
- `结题报告素材/data/comparison_summary.csv`: history=100, **单步滚动**(每步用GT更新) → RMSE=7.53
- 基线 experiments_v2: history=**976**, horizon=24, **前馈** → GRU-ODE-Bayes 6.24
- **三者都不可比**

## 二、可比性状态速查

| 数据集 | 基线 vs 现有 RDE-Delay | 对齐差异 |
|--------|------------------------|----------|
| Lorenz63 | ⚠️ 部分 | trainlength=60/horizon=40 基本同，但 RDE 硬编码"compare 20"不是全 40 步 |
| Lorenz96 | ⚠️ 部分 | 同上 |
| PM2.5 | ❌ 不可比 | 站点数（1/3 vs 36）+ trainlength（5-100 vs 4379） |
| EEG | ❌ 不可比 | history（100 vs 976）+ 模式（滚动 vs 前馈） |

## 三、补跑计划（按优先级）

### P1: Lorenz96 SSSD v2（mask 修复版）— 快
**原因**：其他数据集已有 sssd_v2，Lorenz96 还是旧的 mask 错误版
**命令**：仿 [run_sssd_lorenz63.sh](../run_sssd_lorenz63.sh) 改成 lorenz96
**预估**：~20 分钟（参考 Lorenz63 的速度）
**GPU**：任一空闲（目前 GPU 0/2/3/4 空闲）

### P2: Lorenz63 RDE-Delay（对齐 horizon=40 全 40 步）
**当前**：[lorenz_rde_delay/inference/test_comb_rde.py](../lorenz_rde_delay/inference/test_comb_rde.py) 写死了 "predict 40 compare 20"
**需要**：评估全 40 步 RMSE（或明确说清楚"前 20 步"vs"全 40 步"两个指标）
**方案**：改脚本或加参数，输出两个指标
**预估**：数分钟/组，多组取均值

### P3: Lorenz96 RDE-Delay（同 P2）
**方案**：同 P2

### P4: PM2.5 RDE-Delay 全 36 站 + 长 trainlength
**脚本**：[rde_gpr/pm25_CSDIimpute_after-RDEgpr.py](../rde_gpr/pm25_CSDIimpute_after-RDEgpr.py)
**参数**：
- `--target_indices ""`（全 36 站）
- `--horizon_steps 24`
- `--trainlength 4379`（对齐基线）但 GP 可能跑不动
- 或 `--trainlength 100-500` 作为合理上限
- 用 `--split_ratio 0.5` 对齐
**预估**：全 36 站 × 24 步 GP 可能要几小时
**CPU 约束**：n_jobs ≤ 8

### P5: EEG RDE-GPR（对齐 history=976, horizon=24, 前馈）
**脚本**：[rde_gpr/eeg_CSDIimpute_after-RDEgpr.py](../rde_gpr/eeg_CSDIimpute_after-RDEgpr.py)
**参数**：
- `--history_timesteps 976 --horizon_steps 24`
- `--target_indices 0,1,2`
- `--trainlength 976`（对齐 history）
- `--use_delay_embedding --max_delay 20`
- **前馈模式**（不是单步滚动）

### P6: 等 PM25 SSSD v2 完成，然后整合所有结果

## 四、交付物

- 修正版 [visualization/baseline_comparison.py](../visualization/baseline_comparison.py)（RDE-Delay 数据源用对齐后的新实验）
- 完整版 summary_table.csv
- 标注"可比"/"参考"的分层对比表
- 每个实验单独 git commit 保存结果
