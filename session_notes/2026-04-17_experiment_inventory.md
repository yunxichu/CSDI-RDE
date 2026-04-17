# 2026-04-17 实验清单（已做 / 未做 / 运行中）

> 目的：系统梳理项目所有实验，覆盖 **四数据集 × 所有方法 × 两种预测模式**，标明每格状态与下一步优先级。
> RDE-Delay / RDE-GPR 是本项目主方法，NeuralCDE / GRU-ODE-Bayes / SSSD 是缺失值处理领域三大深度学习基线。

## 两种预测模式的定义

| 模式 | 说明 | 典型方法 |
|------|------|---------|
| **Mode A 前馈 (feed-forward)** | 模型一次输出完整 horizon 步，中间步不看 GT | 基线天然做法；RDE-GPR 需 `--multi_step --multi_step_mode direct`；RDE-Delay 的 `eval_aligned.py` 默认即为此 |
| **Mode B 单步滚动 + teacher-forcing** | 每步只预测 1 步，下一步输入用**真值** `future_truth` 替换，训练窗口随之滑动 | RDE-GPR 的 `*_CSDIimpute_after-RDEgpr.py` 默认；基线 `eeg_*_forecast.py` 加 `--use_teacher_forcing` 亦可 |

两种模式的 RMSE 不可直接比较：**Mode B 误差不累积 → 同方法 RMSE 会低一个数量级**。

---

## 一、基线实验 (experiments_v2/) — Mode A 前馈

所有基线用 `--lr 1e-3 --batch 128 --epochs 100/500`，前馈一次性预测。

### 对齐设置（权威）

| 数据集 | history / trainlength | horizon | target_dims | ground_path |
|--------|------------------------|---------|-------------|-------------|
| Lorenz63 | 60 | 40 | 全 3 维 | [gt_100_20260320_110418.csv](../lorenz_rde_delay/results/gt_100_20260320_110418.csv) |
| Lorenz96 | 60 | 40 | 全 100 维 | [gt_100_20260323_192045.csv](../lorenz96_rde_delay/results/gt_100_20260323_192045.csv) |
| PM2.5 | 4379 (split=0.5) | 24 | 全 36 站 | pm25_ground.txt |
| EEG | 976 | 24 | dims 0,1,2 | eeg_full.npy |

### 状态矩阵（overall RMSE / MAE）

| 方法 | Lorenz63 | Lorenz96 | PM2.5 | EEG |
|------|----------|----------|-------|-----|
| NeuralCDE | ✅ 6.05 / 4.22 | ✅ 9.94 / 7.16 | ✅ 15.06 / 10.44 | ✅ 17.04 / 12.27 |
| GRU-ODE-Bayes | ✅ 5.97 / 4.03 | ✅ 4.10 / 3.26 | ✅ 20.99 / 15.56 | ✅ 6.24 / 5.19 |
| SSSD v1 (mask错误版) | 18.80 / 15.20 | 5.59 / 4.43 | 105.21 / 95.32 | 87.57 / 73.15 |
| **SSSD v2** (mask 修复) | ✅ 15.21 / 12.02 | ✅ 6.66 / 5.18 | 🔄 **运行中** (GPU 7, ~30%) | ✅ 64.06 / 56.48 |

### 基线 dim 0 提取（和 RDE-Delay 只跑 dim0 公平对比）

| 方法 | Lorenz63 dim0 RMSE | Lorenz96 dim0 RMSE |
|------|---------------------|----------------------|
| NeuralCDE | 2.75 | 7.06 |
| GRU-ODE-Bayes | 6.78 | 2.70 |
| SSSD v2 | 15.19 | 3.54 |

---

## 二、RDE-Delay / RDE-GPR (我的方法) — 两种模式

### Mode A 前馈（对齐基线）

| 数据集 | 方法 | 状态 | RMSE / MAE | 备注 |
|--------|------|------|------------|------|
| Lorenz63 | RDE (5 seeds) | ✅ 完成 | **0.57 ± 0.14** | dim0, trainlength=60, horizon=40, 5 seeds 均值 |
| Lorenz63 | RDE-Delay (5 seeds) | ✅ 完成 | **1.40 ± 0.41** | 同上 |
| Lorenz96 | RDE (5 seeds) | ✅ 完成 | **0.28 ± 0.10** | dim0, trainlength=60, horizon=40, 5 seeds 均值 |
| Lorenz96 | RDE-Delay (5 seeds) | ✅ 完成 | **0.26 ± 0.11** | 同上 |
| PM2.5 | RDE-GPR | ❌ **未做** | — | 脚本 [pm25_CSDIimpute_after-RDEgpr.py](../rde_gpr/pm25_CSDIimpute_after-RDEgpr.py) 无 `--multi_step` 选项，默认是 Mode B |
| EEG | RDE-GPR | 🔄 **运行中** | — | `history=976, horizon=24, multi_step direct`, 约 50 分钟 |

### Mode B 单步滚动 + teacher-forcing（GT 泄露到滑窗）

| 数据集 | 方法 | 状态 | RMSE / MAE | 来源 |
|--------|------|------|------------|------|
| Lorenz63 | RDE-GPR | ❌ **未做** | — | 脚本缺 (Lorenz 用 eval_aligned.py) |
| Lorenz96 | RDE-GPR | ❌ **未做** | — | 同上 |
| PM2.5 | RDE-GPR 全 36 站 | ❌ 严格未做（有旧跑） | 旧跑: 15.39 / 10.01 | [best_record/pm25_rc_rde_0.5_42_20260317_122531](../best_record/pm25_rc_rde_0.5_42_20260317_122531/) trainlength=4, L=10 (参数差) |
| PM2.5 | RDE-GPR 3 站 | 旧跑 | 13.90 / 9.27 (3站) / 11.42 (站001001) | [best_record/pm25_test_plot_with_history_v3](../best_record/pm25_test_plot_with_history_v3/) |
| EEG | RDE-GPR, h=100 | ✅ **已做** | **7.53 / 6.23** | [结题报告素材/data/comparison_summary.csv](../结题报告素材/data/comparison_summary.csv)，[eeg_rdegpr_h100_horizon24_20260331_013232](../save/eeg_rdegpr_h100_horizon24_20260331_013232/) |
| EEG | RDE-GPR, h=976 | ❌ 未做严格 Mode B | — | 当前跑的是 Mode A direct |

---

## 三、基线的 Mode B（让大家都 teacher-forcing）

结题报告 Table 5 的"公平对比"：**只在 EEG 上做过**，只跑 GRU / LSTM / GRU-ODE-Bayes / NeuralCDE，**history=100**。

### 已做

| 数据集 | 方法 | 状态 | RMSE / MAE (Mode B) | 来源 |
|--------|------|------|------------------|------|
| EEG (h=100) | **RDE-GPR** | ✅ | **7.53 / 6.23** | comparison_summary.csv |
| EEG (h=100) | GRU | ✅ | 11.25 / 9.30 | 同上 |
| EEG (h=100) | LSTM | ✅ | 11.21 / 9.40 | 同上 |
| EEG (h=100) | GRU-ODE-Bayes | ✅ | 9.62 / 8.08 | 同上 |
| EEG (h=100) | NeuralCDE | ⚠️ 脚本存在但 CSV 里无结果 | — | [baselines/eeg_neuralcde_forecast.py](../baselines/eeg_neuralcde_forecast.py) 已支持 `--use_teacher_forcing` |

### 未做（Mode B）

| 数据集 | 方法 | 状态 | 脚本是否已支持 `--use_teacher_forcing` | 代价 |
|--------|------|------|----------------------------------------|------|
| EEG (h=976 对齐) | NeuralCDE | ❌ | ✅ eeg_neuralcde_forecast.py 支持 | 低（调参即可） |
| EEG (h=976) | GRU-ODE-Bayes | ❌ | ✅ eeg_gruodebayes_forecast.py 支持 | 低 |
| EEG (h=976) | SSSD | ❌ | ❌ sssd_forecast.py 无此参数 | 高（改代码 + 慢，扩散模型每步重新 denoise 100 步） |
| Lorenz63 | NeuralCDE/GRU-ODE-Bayes/SSSD | ❌ | ❌ baselines/*_forecast.py 无此参数 | 中（需改基线代码） |
| Lorenz96 | NeuralCDE/GRU-ODE-Bayes/SSSD | ❌ | ❌ 同上 | 中 |
| PM2.5 | NeuralCDE/GRU-ODE-Bayes/SSSD | ❌ | ❌ 同上 | 中 |

---

## 四、总结：所有缺口

### 🔴 P0 紧急，不做就不能发布最终对比
- [x] Lorenz96 SSSD v2（已完成，RMSE=6.66）
- [ ] PM2.5 SSSD v2（🔄 运行中，GPU 7，~30%）
- [ ] EEG RDE-GPR Mode A 前馈 h=976（🔄 运行中，~50 min）

### 🟡 P1 重要，严重影响"严谨对齐"
- [ ] **PM2.5 RDE-GPR 严格 Mode A 前馈全 36 站**
  - 脚本 [pm25_CSDIimpute_after-RDEgpr.py](../rde_gpr/pm25_CSDIimpute_after-RDEgpr.py) 需加 `--multi_step --multi_step_mode direct` 选项
  - 工作量：中（参考 [eeg_CSDIimpute_after-RDEgpr.py](../rde_gpr/eeg_CSDIimpute_after-RDEgpr.py) 已有的实现）
- [ ] **EEG 基线 Mode B teacher-forcing，h=976**（与 Mode A 的 h=976 基线双赛道对比）
  - NeuralCDE / GRU-ODE-Bayes 脚本已支持，调参重跑即可
  - 工作量：低（2 次运行）

### 🟢 P2 可选，若要做"所有数据集的统一 teacher-forcing 对齐"路线
- [ ] Lorenz63/96 所有基线的 Mode B
  - NeuralCDE / GRU-ODE-Bayes 基线脚本需要加 `--use_teacher_forcing` 支持（参考 EEG 脚本）
  - 工作量：高（改 3-6 个脚本，每个跑 3 个数据集）
- [ ] PM2.5 所有基线的 Mode B
  - 同上
- [ ] Lorenz63/96/PM2.5 RDE-GPR 的严格 Mode B 评估（已经是 teacher-forcing 默认，只需补齐 metrics 到 experiments_v2 格式）

### 🟤 P3 不建议做（代价过高）
- [ ] SSSD 在任何数据集的 Mode B teacher-forcing（扩散模型逐步 denoise 24 次太慢，且意义不大）
- [ ] Lorenz63/96 基线的**全 100/3 维** RDE-Delay（GP 逐维跑代价爆炸）

---

## 五、建议的下一步顺序

**当前状态**：
1. PM2.5 SSSD v2 在 GPU 7（~30% 完成，~30h 剩余）
2. EEG RDE-GPR Mode A direct 在 CPU（~80% dim0 完成，~40 min 剩余）

**接下来按优先级**：
1. ⏳ **等 EEG RDE-GPR 跑完**（~40 min）→ 记录 Mode A h=976 前馈 RMSE
2. **启动 PM2.5 RDE-GPR Mode B 全 36 站**（默认 teacher-forcing，就是现有脚本） → 作为 "PM25 当前可用的对齐对比"
3. **改 PM25 RDE-GPR 脚本加 `--multi_step`** → 严格 Mode A 前馈对齐 SSSD/NeuralCDE
4. **跑 EEG 基线 Mode B h=976**（NeuralCDE / GRU-ODE-Bayes）→ EEG 双赛道
5. 可选：如要做全数据集 Mode B 对齐，再考虑改 Lorenz63/96/PM25 基线

**估算时间**（不含神经网络重训）：
- PM2.5 RDE-GPR Mode B (OMP=1, NJ=2, TL=500): **1-2 小时**
- EEG 基线 Mode B h=976: **每个 30-60 min**（已训练过，只需推理）
- 改 PM25 加 --multi_step：开发 ~30 min + 跑 1-2 小时
