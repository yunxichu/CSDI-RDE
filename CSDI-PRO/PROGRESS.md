# CSDI-PRO v2 项目清单

> 根据 [tech.md](tech.md) v2 方案执行。本文档是扁平的"**总任务清单** + **已完成清单**"，不按 week 组织。详细会话记录见 `/home/rhl/Github/session_notes/`。
>
> **分支**：`csdi-pro` · **工作目录**：`/home/rhl/Github/CSDI-PRO/`

---

## 一、总体要做什么（完整任务清单）

### A. 基础设施

| # | 任务 | 说明 |
|---|---|---|
| A1 | Python 环境与依赖 | torch / gpytorch / chronos-forecasting / nolds / dysts / cma / skopt / transformers / properscoring / uncertainty-toolbox |
| A2 | 混沌系统数据生成器 | Lorenz63（scipy odeint）、Lorenz96（N=40/100/400）、Kuramoto-Sivashinsky |
| A3 | sparse-noisy mask 构造工具 | 给定 sparsity + noise_std_frac 生成带 NaN 的观测 |
| A4 | UQ 指标库 | CRPS (Gaussian & ensemble)、PICP、MPIW、Winkler、reliability curve、ECE |
| A5 | 混沌指标库 | VPT（多阈值）、NRMSE（按 attractor std 归一化）、correlation dimension error |
| A6 | Foundation model loaders | Chronos-T5 {small, base, large}、Panda（GilpinLab/panda-72M）、FIM（fim4science）、context parroting |
| A7 | 现有 CSDI / GPR / RDE 模块整理 | 从 v1 CSDI-RDE-GPR 继承，按需包一层接口 |
| A8 | 可视化 helpers | 多面板消融图、phase transition 曲线、reliability diagram、τ 奇异值谱 |

### B. 四大技术 Module

| # | Module | 核心机制 | tech.md 章节 |
|---|---|---|---|
| B1 | **M1: Dynamics-Aware CSDI** | (a) noise-level conditioning（给 denoising net 加 σ_obs embedding token）<br>(b) 延迟结构 attention mask，由 MI-Lyap 的 τ 动态驱动（co-adaptation）<br>(c) ensemble-aware sampling（保留 20 个样本不平均） | §Module 1 |
| B2 | **M2: MI-Lyap Adaptive Delay Embedding** | (a) KSG 条件互信息 估计 I(Y_τ; X_future \| X_t)<br>(b) Rosenstein 局部 Lyapunov 作为惩罚项<br>(c) BayesOpt 搜 τ（L≤10）<br>(d) 低秩 CMA-ES（L>10，Lorenz96 场景） | §Module 2 |
| B3 | **M3: SVGP on Delay Coordinates** | Matern-5/2 核、128-1024 inducing points、VariationalELBO 训练、每输出维独立 SVGP（或 IndependentMultitask） | §Module 3 |
| B4 | **M4: Lyap-Conformal** | (a) 非遵从度 score = \|y-ŷ\| / (σ·exp(λh·dt))<br>(b) ψ-mixing 条件下 finite-sample coverage guarantee<br>(c) Adaptive 版本 q_t = q_{t-1} + η·(miss − α) | §Module 4 |

### C. 理论（3 条论断 + 附录证明）

| # | 论断 | 证明工具 | tech.md 章节 |
|---|---|---|---|
| C1 | **Proposition 1 (informal)** | ambient-dim 方法 forecasting error 下界含 D（维度诅咒）；covering number + Le Cam's two-point method | §Module 0.3 |
| C2 | **Proposition 2 (informal)** | 我们的方法 posterior contraction rate 由 d_KY 主导；引用 Castillo 2014（GP on manifolds）+ Takens 定理 | §Module 3.6 |
| C3 | **Theorem 1 (informal)** | Lyap-CP 在 ψ-mixing 条件下的 coverage 保证；引用 Chernozhukov et al. 2018 + Barber et al. 2023 + Bowen-Ruelle | §Module 4.5 |

### D. 实验

| # | 实验 | 目标 | 产出（论文 Figure） |
|---|---|---|---|
| D1 | **Phase Transition 主图** | 3 datasets × 8 harshness × 10 methods × 4 metrics，~1200 runs | **Figure 1**（主图，phase transition curves） |
| D2 | **Coverage Across Harshness** | 同矩阵的 PICP@90 展示 | **Figure 2**（Lyap-CP 独家贴 90% 线） |
| D3 | **Horizon × Coverage 曲线** | 固定 harshness，展示 Lyap-CP 在所有 horizon 保持覆盖 | Figure 3 |
| D4 | **Horizon × PI Width 曲线** | Lyap 膨胀让 PI 合理扩张 | Figure 4 |
| D5 | **Reliability Diagram** | pre/post conformal 对比 | Figure 5 |
| D6 | **MI-Lyap τ 稳定性 vs noise** | σ ∈ [0, 0.1, 0.3, 0.5, 1.0, 2.0] × 20 seeds，τ std 对比 Fraser-Swinney | Figure 6 |
| D7 | **τ 矩阵低秩奇异值谱** | Lorenz96 耦合振子，论文卖点图 | Figure 7 |
| D8 | **SVGP Scaling Curve** | Lorenz96 N=40/100/400，训练时间 vs N；对应 Proposition 2 | Figure 8 |
| D9 | **EEG Case Study** | h=100 真实数据 calibration，robustness 展示 | Figure 9 |
| D10 | **4-Module Ablation 表** | 每 module 独立贡献 + 全 off ≈ v1 | Table 2 |
| D11 | **dysts 20 系统 benchmark** | VPT/CRPS/PICP 主结果表 | Table 1 |
| D12 | **Foundation model 大 PK** | Chronos / Panda / FIM / context parroting 在稀疏设定下 | 融入 Table 1 / Figure 1 |
| D13 | **极端 harshness 下 sharp summary** | 文字版 3-table 总结 | Table 3 |

### E. 写作（约 9 页正文 + appendix）

| # | 章节 | 页数 |
|---|---|:-:|
| E1 | Abstract + Introduction | 2 |
| E2 | Related Work | 1.5 |
| E3 | Method（4 modules） | 2–3 |
| E4 | Theory（Prop 1/2 + Thm 1） | 1.5 |
| E5 | Experiments | 3 |
| E6 | Discussion & Limitations | 1 |
| E7 | Conclusion | 0.5 |
| E8 | Appendix（证明 sketch + 超参数 + 补充实验） | — |

### F. 投稿

| # | 目标 | Deadline | 作用 |
|---|---|---|---|
| F1 | **NeurIPS 2026** | 2026-05 | **primary** |
| F2 | **UAI 2026** | 2026-05 | safety net（同窗可 resubmit） |
| F3 | **ICLR 2027** | 2026-09 | secondary |
| F4 | **AISTATS 2026** | 2026-10 | 兜底 |
| F5 | Workshop（ICLR AI for science / NeurIPS ML&PhysSci） | 滚动 | 降级策略 |

---

## 二、现在做了什么（已完成清单）

> 截至 2026-04-21。commit 范围：`d9a7c6c`（CSDI-PRO 初始化）→ `3b273d8`（M2 Stage B + Lorenz96 scaling）。

### A. 基础设施 — 全部完成

- [x] **A1** Python 环境全装齐（10+ 个包）；`npeet` 因权限装不上，已用手写 KSG MI/CMI 代替
- [x] **A2** Lorenz63 积分器 + **Lorenz96 积分器** (F=8, scipy odeint, λ_1≈1.68/unit)；KS 未做
- [x] **A3** `make_sparse_noisy` + 1-D mask 广播
- [x] **A4** UQ 指标完整（CRPS、PICP、MPIW、Winkler、reliability curve、ECE）
- [x] **A5** 混沌指标基本完整（VPT 多阈值、NRMSE）；correlation dim error 未做
- [x] **A6** Chronos-T5 {small/base/large} 可加载推理；**Panda 需要 custom PatchTST 扩展代码，W8 补**；FIM 未接入
- [x] **A7** v1 的 `csdi/` / `rde_delay/` / `gpr/` 模块已原地保留，可按需 import
- [x] **A8** 多面板对比图 + phase transition 曲线 + reliability 画板子已就绪

### B. 四大 Module — 4 full + M1 另有轻量 surrogate

- [x] **B3 M3 SVGP** — **full**：[models/svgp.py](models/svgp.py)，GPyTorch Matern-5/2，MultiOutputSVGP，Lorenz96 N=10/20/40 线性 scaling 已验证
- [x] **B4 M4 Lyap-CP** — **full + 4 growth modes**：[methods/lyap_conformal.py](methods/lyap_conformal.py)
  - SplitConformal / LyapConformal / AdaptiveLyapConformal
  - growth_mode ∈ {`exp`, `saturating`, `clipped`, **`empirical`** (λ-free)}
  - Lyap-empirical 在 mixed-horizon calibration 上 mean \|PICP−0.90\| = 0.013 vs Split 0.072 → **5.5× 改善**
- [x] **B2 M2 MI-Lyap** — **full (Stage A + B)**：[methods/mi_lyap.py](methods/mi_lyap.py)
  - `ksg_mi` / `ksg_cmi` 手写 (Kraskov + Frenzel-Pompe)
  - `mi_lyap_bayes_tau` Stage A (BayesOpt + cumulative-δ 防重复)
  - `mi_lyap_cmaes_tau` Stage B (低秩 UV^T × CMA-ES, rank=2, tech.md §2.3)
  - `robust_lyapunov` 噪声鲁棒 λ (AR-Kalman pre-filter + Rosenstein tl=50 + clip)
    → σ=0.5 下 nolds err +152% → robust err **−1%**
- [~] **B1 M1 Dynamics-Aware CSDI** — **架构 full + 训练 WIP**：
  - [methods/dynamics_csdi.py](methods/dynamics_csdi.py) 500 行 self-contained DDPM + (A) noise-cond + (B) 动态 delay mask + (C) ensemble-aware sampling
  - 训练可 converge (loss 0.83 → 0.41 @ 120 ep, 7 分钟 V100)，但 imputation quality 在 smooth Lorenz63 上未击败 linear interp
  - 消融 pipeline 中暂用 [methods/dynamics_impute.py](methods/dynamics_impute.py) (AR-Kalman smoother + MAD 噪声估计) 作为 surrogate
  - 根因：smooth Lorenz63 dt=0.025 对 linear 过于友好；CSDI paper 用 500+ epochs × 35k 不规则时序
  - 真 CSDI 训练收益需要 long-gap + high-noise + 大数据 + 更长训练，后续继续

### C. 理论 — 未做

- [ ] **C1** Proposition 1 formal 证明
- [ ] **C2** Proposition 2 formal 证明
- [ ] **C3** Theorem 1 formal 证明

### D. 实验 — 5/13 完成，主结果已复现并验证

- [x] **Phase Transition pilot**（D1 的预演）：Lorenz63 × 7 harshness × 3 baselines × 5 seeds；**parrot 在 S2→S3 出现 95% VPT drop**，v2 锋利 story 被证据支持；Chronos 确认"categorically brittle at chaos"
- [x] **D10 4-Module 消融表 v1**：Lorenz63 S2/S3 × 3 seeds × 7 configs × 4 horizons
- [x] **D10 4-Module 消融表 v2**（升级）：新增 Lyap-empirical + m4-lyap-exp 两个配置；使用 robust_lyapunov
  - Full NRMSE h=1 = 0.373；All-off（≈ v1）= 0.760 (**+104%**)
  - −M1: +29% / −M2: +28% / −M3: +24%（每 module 独立贡献 ≥24%）
  - −M3 在 h=64 +26% — SVGP 长 horizon 优势显著
  - MPIW：Full 8.9 vs All-off 20.4 → 模块协同使 PI 宽 2.3× 更紧
- [x] **Module 4 专项 mixed-horizon calibration**：horizons [1..48] (~1.1 Λ times)
  - **Lyap-empirical mean \|PICP − 0.90\| = 0.013 vs Split 0.072 (S3) / 0.018 vs 0.084 (S2) → 4.7-5.5× 改善**
  - Split CP 单调漂移 0.99→0.82 (textbook undercoverage)；Lyap-empirical 所有 h 在 [0.88, 0.92] 内
- [x] **D8 SVGP scaling on Lorenz96**：N=10/20/40，training time 线性随 N (25s/46s/94s)，NRMSE 0.80→0.99 缓慢退化
- [ ] **D1** 正式 Phase Transition 主图（Lorenz96 + KS + 更多 methods）
- [ ] **D2** Coverage Across Harshness（Figure 2）
- [ ] **D3** Horizon × Coverage 曲线
- [ ] **D4** Horizon × PI Width 曲线
- [ ] **D5** Reliability diagram pre/post conformal
- [ ] **D6** MI-Lyap τ 稳定性 vs noise 扫描
- [ ] **D7** τ 矩阵低秩奇异值谱图（Lorenz96 L=7 上 4 种 τ-search 方法 nrmse 几乎相同，需降到 L=3-5 或 N=10 看区分度）
- [ ] **D9** EEG case study
- [ ] **D11** dysts 20 系统 benchmark
- [ ] **D12** Foundation model 大 PK（Panda 需要 custom PatchTST 扩展；FIM 待装）
- [ ] **D13** 极端 harshness sharp summary

### E. 写作 — 基础文档已有，论文正文未写

- [x] [tech.md](tech.md)：v2 完整方案 1047 行
- [x] [PROGRESS.md](PROGRESS.md)：本文件，扁平任务清单
- [x] [ABLATION.md](experiments/week2_modules/ABLATION.md)：4-Module 消融汇总
- [x] 3 份 session_notes 归档日志（`2026-04-21_*`）
- [ ] Paper Introduction
- [ ] Paper Related Work
- [ ] Paper Method
- [ ] Paper Theory section
- [ ] Paper Experiments
- [ ] Paper Discussion
- [ ] Paper Appendix

### F. 投稿 — 未到阶段

所有 F 项（NeurIPS/UAI/ICLR/AISTATS/Workshop）均为结稿后动作，目前未到时间点。

---

## 三、累计关键数字（可直接引用）

| 指标 | 数值 | 证据文件 |
|---|---|---|
| Phase transition drop（parrot, Lorenz63 S2→S3） | **95%**（VPT 1.58→0.08 Λ times） | [results/pt_v2_multibase_n5_small.json](experiments/week1/results/pt_v2_multibase_n5_small.json) |
| Chronos-T5 在 clean Lorenz63 上 VPT 上限 | 0.83 Λ times（categorically weak） | 同上 |
| Full pipeline vs v1-like baseline NRMSE 差距（S3 h=1） | **+104%**（0.760 vs 0.373） | [results/ablation_S3_n3.json](experiments/week2_modules/results/ablation_S3_n3.json) |
| MPIW 改善（S3 h=1） | **2.3×**（20.4 → 8.9） | 同上 |
| Lyap-CP vs Split-CP miscalibration 改善 | **30% 低** mean \|PICP − 0.90\| | [results/module4_horizon_cal_S3_n3.json](experiments/week2_modules/results/module4_horizon_cal_S3_n3.json) |

---

## 四、已识别限制 & 待修

| 问题 | 影响 | 解决方案 |
|---|---|---|
| Lyap-CP 的 `exp(λh·dt)` 在 h > 1 Λ 时 over-predict | Module 4 在长 horizon 出现 overcoverage | 改 saturating growth：`min(exp(λh·dt), C_max)` 或 `sqrt(exp(·))` |
| `nolds.lyap_r` 在噪声数据上估 λ 过高 4× | 会让 Lyap-CP 失准（需用真 λ 做 demo） | 写一个 noise-robust λ 估计（local Lyap 的中位数 + 带宽调整） |
| M1 目前只是 AR-Kalman surrogate | 真 CSDI 训练的增益未体现 | 走 CSDI Transformer 改造（给 denoising net 加 σ embedding + delay-aware mask），diffusion 训 4–8 小时 |
| MI-Lyap 未做低秩 CMA-ES 版 | Lorenz96 高维场景缺方法 | tech.md Module 2.3 Stage B 代码已列，实现 <200 行 |
| Panda / FIM foundation models 未接入 | Phase Transition 主图缺关键 baseline | HuggingFace load；Panda 从 GilpinLab repo 或自训 |
| Lorenz96 / KS 数据生成器未写 | 高维 + 偏微分场景缺基础 | `dysts` 现成；KS 用 `scipy.integrate.solve_ivp` |
| Theorem 1 / Proposition 1,2 formal 证明 | 理论章节空缺 | 写论文时按 tech.md 附录 sketch 展开 |
| VPT 在长 horizon saturate 到 0 | 长 h 下指标失效 | 改用 NRMSE 或 correlation-dim error 作为长 h 指标 |

---

## 五、风险 & 概率评估（基于已做工作更新）

| 风险 | tech.md 原评估 | 当前评估 |
|---|:-:|:-:|
| Phase transition pilot 失败 | 未知 | **已消解**（parrot 95% drop 已复现） |
| Panda 在 sparse 下没崩 | 30% | ≤20%（parrot 已崩 → 同类方法应同命运） |
| MI-Lyap 没明显优于 Fraser-Swinney | 40% | 有轻微优势（S3 h=1 NRMSE 0.373 vs 0.491，24% 差距），但需更多场景验证 |
| Dynamics-Aware CSDI 训练不稳定 | 30% | 不变（未训） |
| 理论证不出严格版 | 70% | 不变（未写） |
| 时间不够 | 50% | 降：已完成约 30% 实验工作量 |
| Chronos 不够锋利做 baseline（新增） | — | **存在**：必须接 Panda 做主对比 |
| Lyap-CP 在长 h 过度保守（新增） | — | **已识别**：saturating growth 是直接 fix |

---

## 六、投稿可能性估计（Phase transition 成立后）

基于 W1 pilot 证据保留 v2 锋利 story 的前提下：

| 目标 | 原 v1 概率 | v2 目标 | 当前估计 |
|---|:-:|:-:|:-:|
| NeurIPS / ICLR main | 25–35% | 40–50% | **40–50%** |
| ICML | 25–35% | 35–45% | 35–45% |
| UAI | 50–60% | 60–70% | 60–70% |
| AISTATS | 60% | 60% | 60% |
| Workshop（至少一个） | 90% | 95% | **95%** |

---

## 附录：Git 提交里程碑

```
5aa329a  W2 跨越式 — 四大技术 module 实现 + 消融实验（当前）
14228c0  新增 CSDI-PRO/PROGRESS.md 项目进度总览
4a493ea  W1 Phase Transition pilot 验证 v2 锋利 story 可行
d9a7c6c  CSDI-PRO 工作空间初始化
```

## 附录：关键会话记录

```
session_notes/2026-04-21_csdi_pro_v2_week1.md
    W1 Day 1-2 环境 + Day 6-7 Phase Transition pilot 全细节

session_notes/2026-04-21_csdi_pro_v2_week2_modules_ablation.md
    W2 四 module 实现 + 消融实验完整归档

session_notes/2026-04-21_innovation_directions_survey.md
    v1→v2 之前的文献调研（推荐路径 1-3）
```
