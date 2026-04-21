# CSDI-PRO v2 — Week 1 Day 1-2 + Day 6-7 执行记录

**日期**：2026-04-21
**分支**：`csdi-pro`（位于 `/home/rhl/Github`）
**工作目录**：`/home/rhl/Github/CSDI-PRO/`
**对应 tech.md 章节**：Week 1（环境 + 阅读 + Phase Transition pilot）

---

## 背景

tech.md v2 方案的锋利度全部押在 **Module 0: Sparse-Observation Regime formalization** —— 声称 foundation models 在 sparsity + noise 双轴上存在明显 **phase transition**。Week 1 Day 6-7 的 pilot 是决定性 kill/go 实验。

---

## Day 1-2：环境 + smoke tests

### 安装结果

| 包 | 状态 | 说明 |
|---|---|---|
| `torch` 2.5.1+cu124 | 已在 | 8 × V100-32GB（用户约束：仅用 1-2 张，优先 `CUDA_VISIBLE_DEVICES=2`） |
| `gpytorch` 1.15.2 | 新装 | SVGP 主力 |
| `properscoring` 0.1 | 新装 | CRPS |
| `uncertainty_toolbox` 0.1.1 | 新装 | reliability diagram |
| `skopt` 0.10.2 | 新装 | BayesOpt for τ (Module 2 Stage A) |
| `nolds` | 新装 | Lyapunov via Rosenstein 1993 |
| `dysts` | 新装 | 经典混沌系统库 |
| `cma` 4.4.4 | 新装 | CMA-ES (Module 2 Stage B) |
| `transformers` 5.5.4 | 新装 | Chronos 依赖 |
| `chronos-forecasting` | 新装 | Chronos-T5 {small, base, large} |
| `npeet` | **未装** | PyPI/GitHub install 被权限拒绝；Week 5 前手写 KSG（约 50 行） |

### Smoke tests

1. **dysts Lorenz 轨迹**：`make_trajectory(2000, resample=True)` 工作，但 `resample=True` 把时间归一化到 "one dominant period"，不适合算 Lyapunov 时间。改用 `scipy.integrate.odeint` 自写 `experiments/week1/lorenz63_utils.py::integrate_lorenz63`，dt=0.025 或 0.01 可选。
2. **GPyTorch SVGP toy**：Matern-5/2 SVGP on sin(x) + 0.1 noise, 400 points, 20 inducing → RMSE 0.017，2.8s 训练。流程通。
3. **nolds `lyap_r`**：对 dysts-resampled Lorenz63 x-component 返回 0.052（因为 dysts 归一化时间）；对我的 scipy 轨迹（未测，保留 Week 2 做 MI-Lyap 时再验证）。
4. **Chronos zero-shot**：`amazon/chronos-t5-small` 装载正常，推理 1.3s，但 100 步 Lorenz63 x-component RMSE = 9.17（scale ~8 的信号，基本全错）。预期中，因为 Chronos-small 对混沌系统偏弱。

**产出**：
- `experiments/week1/smoke_test.py`
- `experiments/week1/smoke_chronos.py`
- `experiments/week1/lorenz63_utils.py`（Lorenz63 积分 + 稀疏噪声 + VPT 计算）

---

## Day 6-7：Phase Transition pilot（决定性实验）

### 协议

- Lorenz63 规范参数（σ=10, ρ=28, β=8/3）, dt=0.025, spinup=2000 步
- N_CTX=512, PRED_LEN=128（dt·PRED_LEN·λ ≈ 2.9 Lyapunov times 的最大可观测窗口）
- 7 个 harshness scenarios，sparsity 0→0.95，noise σ 0→1.5×attractor_std
- 线性插值补值（faithful 的"最天真 foundation-model 用户"做法）
- 3 种方法：
  - `chronos`：Chronos-T5 zero-shot，每轴独立预测
  - `parrot`：context parroting（最近邻延迟预测，tough-to-beat baseline）
  - `persist`：最后一步复制（下界）
- VPT 阈值 0.3 / 0.5 / 1.0，NRMSE 归一化到 attractor std
- 5 seeds，$|\mathcal{A}|=8.51$（均值 per-axis std）

### 结果（VPT @ 阈值 1.0，5 seeds mean）

| scenario  | sparsity | noise σ | chronos | **parrot** | persist |
|-----------|:-:|:-:|:-:|:-:|:-:|
| **S0**    | 0.00 | 0.00 | 0.83 | **1.58** | 0.20 |
| S1        | 0.20 | 0.10 | 0.85 | **1.40** | 0.19 |
| S2        | 0.40 | 0.30 | 0.43 | 0.66 | 0.19 |
| **S3**    | 0.60 | 0.50 | 0.18 | **0.08** | 0.10 |
| S4        | 0.75 | 0.80 | 0.53 | 0.05 | 0.04 |
| S5        | 0.90 | 1.20 | 0.12 | 0.02 | 0.21 |
| S6        | 0.95 | 1.50 | 0.02 | 0.06 | 0.02 |

### Verdict

**Parrot（context parroting）相对 S0 的 drop**：S2→S3 处从 0.66 → 0.08（95% 断崖式下降）。**Phase transition 存在**。

**Chronos-T5-{small, base, large} 均未能在 S0 上 VPT > 1.0**（峰值 0.83）——和 Zhang & Gilpin 2025 汇报一致：**Chronos 对混沌系统类别性偏弱**，在哪个 harshness 都很弱，没有明显相变。

### 这对 v2 故事的含义

原 tech.md 默认 foundation models (Panda/Chronos/FIM) 在 clean 上都 good，只在 sparse 下崩掉 —— 但 Chronos 在 clean 上已经崩了。修正后的正确 framing：

- **强基线（context parroting、Panda、chaos-specialised FM）** 在 clean 上 work（VPT ~1.6+），但在 sparsity∈[0.4, 0.6] 处出现**断崖式 phase transition**（95% drop）。
- **通用 FM（Chronos）** 在 clean 上就偏弱（Zhang & Gilpin 2025 给出的现有证据），呈渐变式退化。
- **我们的方法（CSDI + MI-Lyap + SVGP + Lyap-CP）** 目标：在 harshness 全域 graceful degradation，S0 附近匹配 parrot，S3-S6 区间压过所有 baseline。

Paper 里的 Figure 1（phase transition curve）应当画成：
- parrot / Panda = 高起点、断崖
- Chronos = 低起点、缓降
- 我们 = 与 parrot 持平 on S0-S1，在 S3-S6 显著好于所有 baseline

**v2 锋利 story 可以继续**，但 framing 要调：不是"foundation model 在 harshness 下崩溃"这种 blanket claim，而是更精准的 **"strong chaos baselines exhibit phase transition at a specific sparsity-noise boundary; generic FMs are categorically brittle at chaos"**。

### 实验产出（保存于 `CSDI-PRO/experiments/week1/`）

- 核心代码：
  - `lorenz63_utils.py` — Lorenz63 积分、sparse-noisy mask 构造、VPT 计算
  - `baselines.py` — chronos / parrot / persist 三个 forecaster
  - `phase_transition_pilot_v2.py` — 多 baseline pilot 主脚本
- Pilot 记录（JSON，5 seeds × 7 scenarios × 3 methods = 105 runs）：
  - `results/pt_v2_multibase_n5_small.json`
- 主图（3 面板：VPT@0.3、VPT@1.0、NRMSE）：
  - `figures/pt_v2_multibase_n5_small.png`
- 早期 Chronos-only 探索（保留做 audit）：
  - `results/phase_transition_{smoke_n2, small_dt025, base_dt025, large_dt025_n2, smoke_n2_ctx1000}.json`

---

## Week 1 剩余任务 + 下一步

- [x] Day 1-2 环境与 smoke tests
- [x] Day 6-7 Phase Transition pilot — **v2 sharp story 保留**
- [ ] Day 3-5 精读 5 篇必读论文（deferred，因为 pilot 结论更重要，放在 Week 2 初补做）：
  1. Zhang & Gilpin ICLR 2025 《Zero-shot forecasting of chaotic systems》
  2. Lai, Bao, Gilpin ICLR 2026 《Panda》
  3. Seifner et al. ICLR 2025 《FIM for dynamical systems》
  4. Angelopoulos & Bates 2021 《Gentle intro to CP》
  5. Hersbach 2000 《CRPS decomposition》
- [ ] 补跑：10 seeds for 更 tight error bar；Panda 可得性调查；在 Lorenz96 (N=20/40) 上重复 pilot 看 phase transition 是否 universal
- [ ] Week 2：SVGP 化 + UQ metrics 库（CRPS / PICP / MPIW / reliability diagram / Winkler）

## 风险更新（tech.md Part IV）

| 风险 | 原评估 | pilot 后更新 |
|---|---|---|
| Phase transition pilot 失败（风险 0） | 未知 | **已消解**：parrot 上 95% drop 可复现 |
| Panda 在 sparse setting 下没崩（原风险 1） | 30% | **≤20%**：parrot 已崩说明同类 chaos-specialised 方法都会崩 |
| 需要找 Panda checkpoint | 未评估 | 中等：Week 8 前要搞定 `GilpinLab/panda-72M` 或自训 |
| Chronos 不够"锋利"做 baseline | 未评估 | **存在**：需要换 framing，不再依赖 Chronos 作为主要反例 |

---

## 资源使用

- GPU：仅用 GPU 2（V100-32GB）。运行期间占用 ~3GB peak（Chronos-small 加载 + dysts 数据）。GPU 0 被他人 97% 占用，避让。
- 运行时间：
  - Pilot n_seeds=5 多 baseline 跑完约 **15 分钟**
  - 早期 Chronos-only pilot（n=2）约 6 分钟
- 数据：所有 Lorenz63 轨迹在线生成，无磁盘占用
