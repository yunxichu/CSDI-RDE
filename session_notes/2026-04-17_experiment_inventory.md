# 2026-04-17 实验清单（修正版 v3 — Track-A / Track-B 双对比维度）

> **v3 更新（2026-04-17 下午晚）**：用户明确 CSDI-RDE-GPR 是完整 pipeline，方法定位是处理稀疏/缺失数据。
> 故分为两个对比 Track：
> - **Track-A 预处理对齐对比**（现有 experiments_v2 全部数据，两边都用 CSDI 补值数据）
> - **Track-B 完整 pipeline 对比**（新增：基线直接用稀疏/缺失数据，我方法用 CSDI 补值 + RDE-GPR）
>
> 详见 [2026-04-17_full_pipeline_comparison.md](2026-04-17_full_pipeline_comparison.md)
> 详细对话时间线：[2026-04-17_conversation_snapshot.md](2026-04-17_conversation_snapshot.md)

---

# 原 v2 清单（Track-A 预处理对齐）

> **重大修正**：初次整理时误以为 `experiments_v2/` 里所有基线是"前馈模式"。实际核查后发现**通用基线脚本默认就是 Mode B 单步滚动 + teacher-forcing**。
>
> 这意味着 RDE-Delay 和基线在 `experiments_v2/` 里**已经模式对齐**，无需改写基线代码。只需补齐缺失的 RDE-GPR 对齐实验和仍在跑的 SSSD v2。

---

## 0.0 方法命名规范（与结题报告一致）

| 名称 | 特征构造 | 对应脚本标签 |
|------|----------|--------------|
| **RDE-GPR** | **空间集成**：同一时刻 L 个不同维度 `[x_{d1}(t), ..., x_{dL}(t)]` | Lorenz 的 "RDE"；PM25 默认 |
| **RDE-Delay-GPR** | **延迟嵌入**：不同维度×不同时间延迟 `[x_{d1}(t-τ1), ..., x_{dM}(t-τM)]` | Lorenz 的 "RDE-Delay"；EEG 加 `--use_delay_embedding` |

Lorenz63/96 的 `eval_aligned.py` **同时跑**了 RDE-GPR 和 RDE-Delay-GPR 两个 variant。

## 0. 术语与核心区分

| 模式 | 定义 | 代表 |
|------|------|------|
| **Mode A 纯前馈** | 模型一次输出 horizon 步，中间步不看 GT；OR 自回归：用自己前一步预测作为下一步输入 | 基线训练目标；RDE-GPR 加 `--multi_step` |
| **Mode B 滚动 + teacher-forcing** | 每预测一步后，下一步窗口引入 `future_truth[t]` 真值 | 所有 experiments_v2 通用基线默认；RDE-GPR 默认 |

**模式 B 下误差不累积 → 同方法 RMSE 会低 1 个数量级**。仅同模式间可比。

---

## 1. 每个脚本的默认预测模式（已核查）

### 通用基线脚本（experiments_v2 全部用这些）

| 脚本 | 默认模式 | 关键代码位置 |
|------|----------|--------------|
| [baselines/neuralcde_forecast.py](../baselines/neuralcde_forecast.py) | **Mode B 滚动** | Line 221: `full = [history, fut_clean]; windows = [full[T_hist - W + i : T_hist + i] for i in range(horizon)]` |
| [baselines/gruodebayes_forecast.py](../baselines/gruodebayes_forecast.py) | **Mode B 滚动** | Line 398: `win = full[len(history) - W + step : len(history) + step]` |
| [baselines/sssd_forecast.py](../baselines/sssd_forecast.py) | **Mode B 滚动** | Line 351: `cur = np.vstack([cur[1:], fut_true_scaled_clean[i][np.newaxis]])` |

### PM25 专用脚本（未在 experiments_v2 使用）

| 脚本 | 默认模式 | 备注 |
|------|----------|------|
| [baselines/pm25_neuralcde_forecast.py](../baselines/pm25_neuralcde_forecast.py) | Mode B 滚动 | Line 320: `full = [history, fut_true]` |
| [baselines/pm25_gruodebayes_forecast.py](../baselines/pm25_gruodebayes_forecast.py) | **Mode A 自回归** | Line 525: `use_autoregressive=True` 默认 → 用预测值作为下一步输入（不看真值）|
| [baselines/pm25_sssd_forecast.py](../baselines/pm25_sssd_forecast.py) | Mode B 滚动 | Line 214: `cur = np.vstack([cur[1:], true_val])` |

### EEG 专用脚本（未在 experiments_v2 使用，仅用于 `eeg_forecast_comparison.py`）

| 脚本 | 默认模式 | 开关 |
|------|----------|------|
| [baselines/eeg_neuralcde_forecast.py](../baselines/eeg_neuralcde_forecast.py) | **Mode A 前馈** | `--use_teacher_forcing` 切 Mode B |
| [baselines/eeg_gruodebayes_forecast.py](../baselines/eeg_gruodebayes_forecast.py) | **Mode A 前馈** | `--use_teacher_forcing` 切 Mode B |

### RDE 方法脚本

| 脚本 | 默认模式 | 备注 |
|------|----------|------|
| [rde_gpr/pm25_CSDIimpute_after-RDEgpr.py](../rde_gpr/pm25_CSDIimpute_after-RDEgpr.py) | **Mode B 滚动** | `seq_true = vstack([history, future_truth])`，滑窗包含真值 |
| [rde_gpr/eeg_CSDIimpute_after-RDEgpr.py](../rde_gpr/eeg_CSDIimpute_after-RDEgpr.py) | **Mode B 滚动** | `--multi_step --multi_step_mode direct` 切 Mode A |
| [lorenz_rde_delay/inference/eval_aligned.py](../lorenz_rde_delay/inference/eval_aligned.py) | **Mode B 滚动** | 每步滑窗从 `imputed_100` 抽（imputed 含 GT） |
| [lorenz96_rde_delay/inference/eval_aligned.py](../lorenz96_rde_delay/inference/eval_aligned.py) | **Mode B 滚动** | 同上 |

---

## 2. experiments_v2/ 实际已做的实验（**全部 Mode B 滚动 teacher-forcing**）

所有基线都用**通用**脚本（通过对比 `args.json` 里的 `num_layers / p_hidden / delta_t / solver` 确认）。

| 数据集 | 方法 | 模式 | RMSE | MAE |
|--------|------|------|------|-----|
| Lorenz63 | NeuralCDE | Mode B | 6.05 | 4.22 |
| Lorenz63 | GRU-ODE-Bayes | Mode B | 5.97 | 4.03 |
| Lorenz63 | SSSD v1 (mask 错误) | Mode B | 18.80 | 15.20 |
| Lorenz63 | **SSSD v2** | Mode B | **15.21** | 12.02 |
| Lorenz63 | **RDE-GPR (5 seeds, 空间集成)** | Mode B | **0.57 ± 0.14** | — |
| Lorenz63 | **RDE-Delay-GPR (5 seeds, 延迟嵌入)** | Mode B | **1.40 ± 0.41** | — |
| Lorenz96 | NeuralCDE | Mode B | 9.94 | 7.16 |
| Lorenz96 | GRU-ODE-Bayes | Mode B | 4.10 | 3.26 |
| Lorenz96 | SSSD v1 | Mode B | 5.59 | 4.43 |
| Lorenz96 | **SSSD v2** | Mode B | **6.66** | 5.18 |
| Lorenz96 | **RDE-GPR (5 seeds, 空间集成)** | Mode B | **0.28 ± 0.10** | — |
| Lorenz96 | **RDE-Delay-GPR (5 seeds, 延迟嵌入)** | Mode B | **0.26 ± 0.11** | — |
| PM25 | NeuralCDE | Mode B | 15.06 | 10.44 |
| PM25 | GRU-ODE-Bayes | Mode B | 20.99 | 15.56 |
| PM25 | SSSD v1 | Mode B | 105.21 | 95.32 |
| PM25 | **SSSD v2** | Mode B | 🔄 运行中（~30%, GPU 7） | — |
| EEG | NeuralCDE | Mode B | 17.04 | 12.27 |
| EEG | GRU-ODE-Bayes | Mode B | 6.24 | 5.19 |
| EEG | SSSD v1 | Mode B | 87.57 | 73.15 |
| EEG | **SSSD v2** | Mode B | **64.06** | 56.48 |

**补充验证**（今天新做）：
| 数据集 | 方法 | 模式 | RMSE | 说明 |
|--------|------|------|------|------|
| EEG | GRU-ODE-Bayes (EEG 专用脚本 + `--use_teacher_forcing`) | Mode B | 6.22 | 几乎等于上表 6.24 → 证实 experiments_v2 默认是 Mode B |

---

## 3. 关键修正：EEG RDE-GPR 之前跑偏方向

| 实验 | 模式 | RMSE | 对齐? |
|------|------|------|-------|
| EEG RDE-GPR `multi_step direct` (我之前跑) | **Mode A 前馈** | 91.06 | ❌ experiments_v2 基线没跑 Mode A |
| EEG RDE-GPR 默认滚动（现在 🔄 运行中） | **Mode B 滚动** | 🔄 | ✅ 对齐 experiments_v2 17.04/6.24/64.06 |

**Mode A 的 91.06 不是"RDE-GPR 表现差"的证据 — 它只是说 GP 不擅长长 horizon 纯前馈**，而基线的 17.04/6.24 也都是 Mode B，不是前馈。当前新跑的 Mode B 才是真正可比的数据。

---

## 4. 当前缺口（剩余补跑清单）

### 🔴 P0 必须完成（直接影响最终对比表）

| # | 任务 | 状态 | 预估时间 | 备注 |
|---|------|------|----------|------|
| 1 | EEG RDE-GPR Mode B h=976 (默认滚动) | 🔄 运行中 | ~30 min | CPU 2 核，trainlength=300 |
| 2 | PM25 SSSD v2 (experiments_v2/pm25/sssd_v2) | 🔄 GPU 7 ~30% | ~24h 剩 | 不打断 |
| 3 | PM25 RDE-GPR Mode B 全 36 站对齐 | ⏳ 排队（等 EEG 完） | 1-2h | CPU 2 核，trainlength=500, OMP=1 |

### 🟡 P1 可选（Mode A 对照赛道）

如果要在报告里加一组"大家都前馈"的对照：

| # | 任务 | 代价 |
|---|------|------|
| 4 | Lorenz63/96/PM25 基线 Mode A 严格前馈 | 改通用脚本加 `--no_teacher_forcing`，~2-3h 开发 + 4-6 次推理 |
| 5 | Lorenz63/96/PM25 RDE-Delay Mode A 前馈 | 改 eval_aligned.py 加 `--feed_forward` mode |

### 🟤 P2 不建议做

| # | 任务 | 原因 |
|---|------|------|
| 6 | SSSD 的 Mode A 自回归版 | 扩散模型每步 denoise 代价极高 |
| 7 | Lorenz63/96 全 100/3 维 RDE-Delay | GP 逐维 × 24 步代价爆炸 |

---

## 5. 有疑问但已废弃的旧结果（参考用，非对齐主基线）

| 实验 | 数据 | 问题 |
|------|------|------|
| `结题报告素材/data/comparison_summary.csv` EEG | RDE-GPR=7.53, GRU-ODE-Bayes=9.62 | history=100，与 experiments_v2 h=976 不一致，仅作参考 |
| `save/eeg_rdegpr_h100_horizon24_20260331_013232` | RMSE=63.93 | h=100, 前馈模式，既不对齐 h 也不对齐模式 |
| `best_record/pm25_test_plot_with_history_v3` | 3 站 RMSE=13.90 | 只 3 站点，与基线 36 站不对齐 |
| `best_record/pm25_rc_rde_0.5_42_20260317_122531` | 36 站 RMSE=15.39 | L=10, trainlength=4，与标准参数 L=4, trainlength=500 不对齐 |
| `experiments_v2/eeg/rdegpr_aligned` | RMSE=91.06 | Mode A 前馈，不对齐基线 Mode B |
| `experiments_v2/eeg/gruodebayes_modeB` | RMSE=6.22 | 用 EEG 专用脚本 + teacher_forcing，和通用脚本 6.24 近似，仅作复核 |

---

## 6. 最终交付物（等 P0 完成后生成）

1. **对齐后对比表 CSV**：`experiments_v2/figures/summary_table_modeB.csv`
2. **柱状对比图**：`experiments_v2/figures/rmse_modeB.png` / `mae_modeB.png`
3. **轨迹对比图**（dim 0）：各数据集 `{dataset}_trajectory_comparison_modeB.png`
4. **方法说明**：在 README 或 session notes 中明确标注"所有对比均在 Mode B 单步滚动 teacher-forcing 模式下进行"

---

## 7. 运行中/已启动的后台任务

| PID/TaskID | 说明 | GPU/CPU | 开始时间 | 预估完成 |
|------------|------|---------|----------|----------|
| 909979 | PM25 SSSD v2 Mode B | GPU 7 | 2026-04-16 18:46 | 2026-04-18 晚 |
| task beqb8twda | EEG RDE-GPR Mode B h=976 | CPU 2核 | 今天 15:30 | 今天 16:00 |

（Monitor 任务已挂在 EEG RDE-GPR 上，完成自动通知）
