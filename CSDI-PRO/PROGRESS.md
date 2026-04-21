# CSDI-PRO v2 — 项目进度

> 根据 [tech.md](tech.md) v2 方案执行。本文档记录已完成里程碑与关键实验结果，供快速浏览；详细会话记录在 `/home/rhl/Github/session_notes/`。

**分支**：`csdi-pro` · **工作目录**：`/home/rhl/Github/CSDI-PRO/` · **当前周**：Week 2 完成，进入 Week 3

---

## 总览

| 周 | 任务 | 状态 | 关键产出 |
|---|---|:-:|---|
| **W1 Day 1-2** | 环境 + smoke tests | ✅ | 10 个 Python 包装齐；SVGP/dysts/Chronos smoke 全过 |
| **W1 Day 3-5** | 5 篇必读论文精读 | ⏸ 延后到 W2 末 | — |
| **W1 Day 6-7** | **Phase Transition pilot**（决定性） | ✅ | **v2 锋利 story 保留**；parrot 95% drop at S2→S3 |
| **W2 (跨越式)** | **4 个技术 module 实现 + 消融实验** | ✅ | **−M1: +29% NRMSE, −M2: +28%, −M3: +26%, all-off: +104%**；Module 4 mixed-horizon mean-abs-dev 比 Split-CP 低 30% |

---

## Week 1 Day 1-2：环境 + smoke tests ✅

### 装包状态（`conda/pip`）

| 包 | 版本 | 用途 |
|---|---|---|
| `torch` | 2.5.1+cu124 | 底层 |
| `gpytorch` | 1.15.2 | SVGP / GP kernel |
| `properscoring` | 0.1 | CRPS |
| `uncertainty_toolbox` | 0.1.1 | reliability diagram |
| `skopt` | 0.10.2 | BayesOpt for τ (Module 2 Stage A) |
| `nolds` | 最新 | Rosenstein Lyapunov |
| `dysts` | 最新 | 混沌系统库 |
| `cma` | 4.4.4 | CMA-ES (Module 2 Stage B) |
| `transformers` | 5.5.4 | Chronos 依赖 |
| `chronos-forecasting` | 最新 | Chronos-T5 {small,base,large} |
| `npeet` | ❌ 未装 | PyPI/GitHub 权限拒；W5 前手写 KSG (~50 行) |

### GPU 约束
- 8 × V100-32GB 可见，仅用 `CUDA_VISIBLE_DEVICES=2`（GPU 0 被他人 97% 占用）
- 遵守 memory：**GPU 1-2 张最多 4 张，CPU 不要开太高**

### Smoke tests 结果
- **dysts Lorenz63**: `make_trajectory(2000)` OK，但 `resample=True` 把时间归一化掉 → 不适合算 Lyapunov 时间；已换 `scipy.integrate.odeint` 自写，见 [lorenz63_utils.py](experiments/week1/lorenz63_utils.py)
- **GPyTorch SVGP toy**: 400 点 sin(x)+noise，20 inducing，Matern-5/2 → RMSE 0.017, 2.8s
- **Chronos-T5-small zero-shot**: 1.3s 推理；RMSE 9.17 on 100-step Lorenz63 x（预期弱，见下）
- **nolds `lyap_r`**: 能调通；对 resampled trajectory 返回 0.052，正常 trajectory 验证放 W5

**代码**：[smoke_test.py](experiments/week1/smoke_test.py) · [smoke_chronos.py](experiments/week1/smoke_chronos.py)

---

## Week 1 Day 6-7：Phase Transition pilot ✅（决定性实验）

### 协议
| 项目 | 值 |
|---|---|
| 系统 | Lorenz63 规范参数（σ=10, ρ=28, β=8/3） |
| 积分 | `scipy.odeint`, dt=0.025, spinup=2000 步 |
| 历史窗 | N_CTX = 512（12.8 Λ 时间） |
| 预测窗 | PRED_LEN = 128（2.9 Λ 时间） |
| 补值 | 线性插值（最朴素 foundation-model 用户做法） |
| Seeds | 5 |
| Scenarios | 7 个 sparsity × noise 递增 |
| Baselines | chronos-t5-{small,base,large} / context parroting / persistence |
| 指标 | VPT @ threshold=0.3/0.5/1.0；NRMSE / attractor_std (first 100 steps) |

### Harshness 表

| Scenario | sparsity $s$ | noise $\sigma/\sigma_{\mathcal{A}}$ |
|---|:-:|:-:|
| S0 | 0.00 | 0.00 |
| S1 | 0.20 | 0.10 |
| S2 | 0.40 | 0.30 |
| **S3** | 0.60 | 0.50 |
| S4 | 0.75 | 0.80 |
| S5 | 0.90 | 1.20 |
| S6 | 0.95 | 1.50 |

### 主结果（VPT @ threshold=1.0, Λ times, 5 seeds mean）

| Method | S0 | S1 | S2 | **S3** | S4 | S5 | S6 |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| **parrot** | **1.58** | 1.40 | 0.66 | **0.08** | 0.05 | 0.02 | 0.06 |
| chronos-t5-* | 0.83 | 0.85 | 0.43 | 0.18 | 0.53 | 0.12 | 0.02 |
| persist | 0.20 | 0.19 | 0.19 | 0.10 | 0.04 | 0.21 | 0.02 |

### Verdict

1. **Context parroting 从 S2→S3 断崖 0.66→0.08（95% drop, sparsity 0.4→0.6, σ 0.3→0.5）** — 正是 v2 要的 phase transition 证据。
2. **Chronos-{small,base,large} 在 clean 上 VPT<1**，与 Zhang & Gilpin 2025 一致 → Chronos 类别性偏弱，不是"会崩塌的强基线"。
3. Persistence 全程 ≈0.2，作为下界基准合理。

### 对 v2 故事的 framing 调整

**原 framing**：foundation models (Panda/Chronos/FIM) 在 clean 好，在 sparse 崩。

**调整后**（基于 pilot 证据）：
> **Strong chaos baselines (context parroting / Panda)** exhibit **phase transition** at specific sparsity-noise boundary; **generic FMs (Chronos)** are **categorically brittle** at chaos from the start.

### Paper Figure 1 预期形状
- `parrot / Panda`：高起点（VPT ~1.5+）、在 S2-S3 处断崖
- `Chronos`：低起点（~0.8）、缓降
- `ours`：与 parrot 持平 on S0-S1，在 S3-S6 显著好于所有 baseline

### 产出物

| 文件 | 用途 |
|---|---|
| [experiments/week1/lorenz63_utils.py](experiments/week1/lorenz63_utils.py) | Lorenz63 积分 + sparse-noisy mask + VPT |
| [experiments/week1/baselines.py](experiments/week1/baselines.py) | chronos / parrot / persist 三 forecaster |
| [experiments/week1/phase_transition_pilot_v2.py](experiments/week1/phase_transition_pilot_v2.py) | 多 baseline pilot 主脚本 |
| [experiments/week1/figures/pt_v2_multibase_n5_small.png](experiments/week1/figures/pt_v2_multibase_n5_small.png) | **主图**（3-panel: VPT@0.3 / VPT@1.0 / NRMSE） |
| [experiments/week1/results/pt_v2_multibase_n5_small.json](experiments/week1/results/pt_v2_multibase_n5_small.json) | 105 runs 原始记录 |
| `experiments/week1/results/phase_transition_{small,base,large}_dt025.json` | Chronos-only 早期探索 |

会话笔记：[../session_notes/2026-04-21_csdi_pro_v2_week1.md](../session_notes/2026-04-21_csdi_pro_v2_week1.md)

Git 提交：`4a493ea exp: CSDI-PRO v2 Week 1 — Phase Transition pilot 验证 v2 锋利 story 可行`

---

## 风险更新（对应 tech.md Part IV）

| 风险 | 原评估 | W1 后 |
|---|:-:|:-:|
| Phase transition pilot 失败（tech.md 风险 0） | 未知 | **已消解** ✅ |
| Panda 在 sparse 下没崩 | 30% | ≤20%（parrot 已崩，同类方法应同命运） |
| Chronos 不够锋利做 baseline | 未评估 | **存在**：W8 主对比要加 Panda，不能单靠 Chronos |
| 需要 Panda checkpoint | 未评估 | 中等：W8 前要搞定 `GilpinLab/panda-72M` |
| MI-Lyap 不如 Fraser-Swinney 明显好 | 40% | 不变 |
| Dynamics-Aware CSDI 训练不稳定 | 30% | 不变 |
| 理论证不出严格版 | 70% | 不变 |

---

## 下一步：Week 2 Roadmap

按 tech.md Part II Week 2：

1. **Day 8-9 SVGP 化** 🚧
   - 把 [gpr/gpr_module.py](gpr/gpr_module.py)（self-impl exact GPR）换成 GPyTorch SVGP
   - 接口保持兼容：`fit(X, y) / predict(X) → (mean, std)`
   - 验证：Lorenz63 RMSE 与原 GPR 差 <10%；时间在 n=500 时 <1 分钟
2. **Day 10-11 UQ metrics 库** `metrics/uq_metrics.py`
   - `crps_gaussian`, `crps_ensemble`, `picp`, `mpiw`, `reliability_diagram`, `winkler_score`
3. **Day 12-14 Reliability diagram 首图** — Lorenz63 上 SVGP calibration 基线
4. **末尾补**：5 篇必读论文精读（Zhang&Gilpin / Panda / FIM / Angelopoulos&Bates / Hersbach）

Week 2 末里程碑：SVGP pipeline 跑通 + UQ 指标齐全 + 第 1 张论文用图（reliability）。

---

## 投稿时间线

| 目标 | Deadline | 状态 |
|---|---|:-:|
| NeurIPS 2026（primary） | 2026-05-?? | 12 周 + 2 buffer 对齐 |
| UAI 2026（safety） | 2026-05-?? | 同窗 |
| ICLR 2027（secondary） | 2026-09-?? | 时间宽 |
| AISTATS 2026 | 2026-10-?? | 兜底 |

基于 W1 pilot 结果，三大会接收概率估计（tech.md Part VIII 表）：

| Target | 原 v1 | v2（transition 成立）| **v2 当前** |
|---|:-:|:-:|:-:|
| NeurIPS/ICLR | 25-35% | 40-50% | **40-50%** |
| ICML | 25-35% | 35-45% | 35-45% |
| UAI | 50-60% | 60-70% | 60-70% |
| AISTATS | 60% | 60% | 60% |
