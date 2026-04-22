# TODO — tech.md 技术对照表 + 剩余工作清单

> **目的**：按 tech.md 的 checklist 逐条核对实际完成情况；**未完成项带复现入口**，供下次开机后立即上手。
>
> 源文档：[tech.md](tech.md)（1046 行 v2 技术规范）
> 配套：[COMPLETE_WORK_LOG_zh.md](COMPLETE_WORK_LOG_zh.md) + [EXPERIMENTS_REPORT_zh.md](EXPERIMENTS_REPORT_zh.md)
>
> **最后更新**：2026-04-23

---

## 一、总体完成度（一眼望去）

| 维度 | 计划 | 已完成 | 未完成 |
|---|:-:|:-:|:-:|
| **图（主图 + D 系列）** | 9 + 4 = 13 张 | **12** ✅ | D9 EEG（1 张）❌ |
| **表** | 3 张 | 1 张 ✅ | Table 1 dysts / Table 3 summary ⚠️ |
| **数据集覆盖** | Lorenz63 / Lorenz96 / KS / dysts 20 / EEG | **Lorenz63 (全)、Lorenz96 (部分)** | KS / dysts 20 / EEG ❌ |
| **Foundation model PK** | Chronos / Panda / FIM / Parrot | Chronos / Panda / Parrot ✅ | **FIM** ❌ |
| **理论 3 命题** | Proposition 1/2 + Theorem 1 | Informal sketch ✅ | **Formal proofs** ❌ |
| **论文写作 12 章** | 12 节 + appendix | 首版中文 + 英文草稿 v0 ✅ | 多轮 refine、LaTeX 化、submit-ready ⚠️ |
| **核心 claim** | 5 条 | Claim 1 / 5 ✅ 完整；Claim 3 empirical ✅ | **Claim 2 / 4 formal**（依赖理论证明） |

**总估算完成度 ≈ 75%。** 剩下的 25% 集中在 3 块：高维/真实数据、理论证明、论文多轮迭代。

---

## 二、图 × 计划 vs 实际（tech.md "Deliverables Checklist"）

| # | tech.md 图 | 实际状态 | 备注 |
|:-:|---|:-:|---|
| Fig 1 | Phase Transition (3 datasets × harshness → VPT) | ⚠️ **只做 Lorenz63**，n=5 × 5 methods 完整 | **Lorenz96 / KS 未做** |
| Fig 2 | Coverage Across Harshness (3 datasets × harshness → PICP@90) | ⚠️ 只做 Lorenz63 (D2)，7 scenarios × AR-K + CSDI 双版 | **L96 / KS 未做** |
| Fig 3 | Horizon vs coverage curve (Lyap-CP calibrated) | ✅ 做了（D3 + Fig 5，AR-K + CSDI 双版） | 完成 |
| Fig 4 | Horizon vs PI width | ✅ 做了（D4，AR-K + CSDI 双版） | 完成 |
| Fig 5 | Reliability diagram (pre/post conformal) | ✅ 做了（D5，AR-K + CSDI 双版） | 完成 |
| Fig 6 | MI-Lyap τ 稳定性 vs Fraser vs noise | ✅ 做了（D6） | 完成 |
| Fig 7 | **Low-rank structure of optimal τ** | ✅ 做了（D7 v2，L=3/5/7 覆盖） | 完成 |
| Fig 8 | SVGP scaling + theoretical rate consistency | ✅ 做了（Fig 6 in our numbering） | 完成 |
| Fig 9 | **EEG case study** | ❌ **未做** | **需要 EEG 数据集** |
| — | Fig 1b（Phase Transition CSDI 升级）🆕 | ✅ 额外贡献 n=5 × 7 scenarios | 超额完成 |
| — | Fig 4b（dual-M1 ablation bars）🆕 | ✅ 额外贡献 | 超额完成 |
| — | Fig 2 Trajectory overlay | ✅ AR-K + CSDI 双版 | 完成 |
| — | Fig 3 Separatrix ensemble | ✅ AR-K + CSDI 双版 | 完成 |

---

## 三、表 × 计划 vs 实际

| # | tech.md 表 | 实际状态 | 备注 |
|:-:|---|:-:|---|
| Table 1 | **主结果（dysts 20 systems × 全 methods × VPT/CRPS/PICP）** | ❌ **未做** | 大任务，需 17 小时 foundation model 推理 |
| Table 2 | 消融表 | ✅ **超额完成** — dual-M1 × 9 configs × S2+S3 × 3 horizons | [`ablation_final_dualM1_merged.md`](experiments/week2_modules/results/ablation_final_dualM1_merged.md) |
| Table 3 | 极端 harshness 下的 sharp summary | ⚠️ 数据全在（Fig 1），只需整理文字版 | 1 小时工作 |

---

## 四、Claim × Evidence 矩阵

| # | Claim | Evidence | 状态 |
|:-:|---|---|:-:|
| **C1** | foundation models 在 sparse-obs regime 相变 | Fig 1 (5 methods × 7 scenarios × 5 seeds)；Panda −85%, Parrot −92% | ✅ |
| C2 | Ambient-dim 方法失败是 **fundamental** 而非 engineering | **Proposition 1 formal proof** | ❌ **需要写** |
| C3 | 我们的方法 sample complexity 由 $d_\text{KY}$ 主导 | Proposition 2 informal + Fig 6 empirical（Lorenz96 N-linear） | ⚠️ **半** — empirical 有，formal 缺 |
| **C4** | First distribution-free calibrated coverage for chaotic forecasting | **Theorem 1 formal proof** + Fig 2/3/5 | ⚠️ **半** — empirical 有，formal 缺 |
| C5 | 每个 module 独立贡献 | Table 2 dual-M1 ablation, every −M flip 退化 ≥ 24% | ✅ |

---

## 五、Week-by-Week 计划对照

| Week | tech.md 计划 | 实际 | Gap |
|:-:|---|:-:|---|
| W1 | 环境 + 阅读 + Phase Transition pilot | ✅ **超额**（n=5 pilot 全 5 methods，Panda 接入） | — |
| W2 | SVGP + UQ metrics | ✅ | — |
| W3 | Vanilla Conformal | ✅ | — |
| W4 | Lyap-Conformal（4 growth modes） | ✅ **超额**（empirical 模式 5.5× 改善） | — |
| W5 | MI-Lyap BayesOpt | ✅ | — |
| W6 | 低秩 CMA-ES + 噪声鲁棒性 | ✅（D6 τ-stability + D7 τ-spectrum） | — |
| W7 | Dynamics-Aware CSDI 动态 delay mask | ✅ **超额**（三 bug 诊断成为独立 methodological 贡献） | — |
| **W8** | **高维 L96 + Foundation model 大 PK** | ⚠️ **部分** | **Lorenz96 Phase Transition 未跑**、**FIM 未接**、**dysts 20 未跑** |
| **W9** | **主图 Phase Transition（3 datasets）** | ⚠️ **部分**（只 Lorenz63） | **L96 / KS 未做** |
| W10 | 消融 + 理论章节 | ⚠️ **部分**（消融 ✅；理论 informal） | **Prop 1/2 + Thm 1 formal proofs** |
| **W11** | **EEG case study + 写作 Push 1** | ⚠️ **部分**（写作开始 ✅；EEG ❌） | **EEG 数据 + 跑 pipeline** |
| W12 | 写作 Push 2 + review + 润色 + submit | ❌ **未开始** | Experiments / Discussion / Appendix / LaTeX |

---

## 六、未完成项 — 按优先级 + 动手难度排序

### 🔥 高价值 × 中等难度（**下次开机 3-7 天可完成**）

#### T1. Table 3：极端 harshness sharp summary
- **工作量**：1 小时
- **数据源**：`experiments/week1/results/pt_v2_with_panda_n5_small.json`
- **做法**：从 Fig 1 原始 JSON 抽 S4/S5/S6 的 VPT@1.0 数字，整理成 3 行 × 5 方法的精炼表；配上 4-5 句文字评论（"Ours 是唯一在 S5 有非零 VPT 的方法"等）
- **可一次写在 paper_draft_zh.md §5.2 末尾或 §5.8**

#### T2. Proposition 1/2 + Theorem 1 的 formal proofs
- **工作量**：每条 1-2 天（阅读相关论文 + 证明 sketch 展开 + LaTeX 化）
- **基础**：tech.md §0.3 (Prop 1), §3.6 (Prop 2), §4.5 (Thm 1) 已有 informal sketch
- **参考文献**：
  - Prop 1：Tsybakov 2009 *Introduction to Nonparametric Estimation* Chap 2（Le Cam）
  - Prop 2：Castillo 2014 *Thick & slab rates under GP priors* + Takens 定理
  - Thm 1：Chernozhukov-Wüthrich-Zhu 2018（CP under weak dependence）+ Bowen-Ruelle ψ-mixing
- **放位置**：paper_draft_zh.md Appendix A.1 / A.2 / A.3

### 🔥🔥 高价值 × 高难度（**需重训/新实验，3-5 天**）

#### T3. Lorenz96 Phase Transition（Fig 1 扩展）
- **工作量**：2-3 天
- **障碍**：现 CSDI M1 只训了 Lorenz63 (D=3)，L96 是 D=40，需要重训 CSDI
- **两种办法**：
  - **(a)** 在 Lorenz96 上重训 CSDI M1（~8 GPU 小时）：改 `methods/dynamics_csdi.py` 的 data_dim 参数 → 生成 L96 cache（~1 小时）→ 跑 4 变种 × 200 epochs（~8 小时）
  - **(b)** Lorenz96 不用 CSDI 版 ours，就用 AR-Kalman M1 跑 Phase Transition（降级 claim 但省事，~1 天）
- **具体入口**：
  1. 仿照 `experiments/week2_modules/make_lorenz_dataset.py` 写 L96 版（参考 `experiments/week1/lorenz96_utils.py`）
  2. 仿照 `experiments/week2_modules/run_csdi_longrun.sh`
  3. 复用 `experiments/week1/phase_transition_pilot_v2.py`（需调整 D=40 维度）

#### T4. KS（Kuramoto-Sivashinsky）PDE 场景
- **工作量**：3-5 天
- **障碍**：目前**没有 KS integrator**
- **入口**：
  1. 写 `experiments/week1/ks_utils.py`（`scipy.integrate.solve_ivp` + 周期性 boundary, ETDRK4 推进）
  2. 确定 KS 的 attractor std 和 Lyapunov 指数
  3. 跑 Phase Transition 同一模板

#### T5. dysts 20 systems benchmark（Table 1）
- **工作量**：1-2 天（主要是等）
- **计算**：20 systems × 7 scenarios × 3 seeds × 5 methods = **2100 runs**，约 17 GPU 小时
- **入口**：
  1. `pip install dysts`（dependency 简单）
  2. 用 `dysts.base.make_trajectory(system_name, n=512, pts_per_period=...)` 生成轨迹
  3. 复用 `phase_transition_pilot_v2.py` 的 scenario 和 method 框架
  4. 输出 Table 1：systems × methods 的 VPT/CRPS/PICP 主结果

### 🟡 中价值 × 低难度（**低优先，但完成度加分**）

#### T6. FIM Foundation Model 接入
- **工作量**：半天
- **入口**：
  1. 从 https://fim4science.github.io 下载权重
  2. 写 `baselines/fim_adapter.py`（仿照 `baselines/panda_adapter.py`）
  3. 加到 `phase_transition_pilot_v2.py` 的 method 分支
- **价值**：让 foundation model PK 更完整，审稿人会问"为什么没 FIM？"

### 🟡🟡 中价值 × 中难度（**需要数据源**）

#### T7. EEG Case Study（Fig 9）
- **工作量**：2-3 天
- **障碍**：**需要 EEG 数据集**
- **候选数据**：CHB-MIT Scalp EEG Database (PhysioNet 公开)、TUSZ (Temple University Hospital)
- **入口**：
  1. 下载 EEG，选 1 个典型 subject
  2. 跑 pipeline with h=100，记录 PI 的 calibration
  3. 画 reliability diagram on real data
- **价值**：真实数据 case study 是 paper quality 的关键加分项

### 🟢 低价值 × 低难度（**润色**）

#### T8. 论文 LaTeX 化（NeurIPS template）
- **工作量**：半天
- **入口**：
  1. 从 NeurIPS 2026 官网下载 template
  2. 把 `paper_draft_zh.md` / `paper_draft.md` 的 markdown 转 LaTeX
  3. figures 用 `.pdf` 取代 `.png`（matplotlib 可直接存 PDF）
  4. 补 BibTeX `refs.bib`

#### T9. 论文多轮 refine
- **工作量**：2-3 天
- **入口**：
  1. 重读 Abstract / Introduction，去掉冗余
  2. Related Work 扩 1 页（目前只有 1 页）
  3. Discussion / Limitations 扩（现 0.5 页，可到 1 页）
  4. 摸清 NeurIPS 2026 每章节的 page budget

---

## 七、推荐的下次开机动作（3 级选择）

### Level 1：最短路径到 paper submission（~1 周）
1. T1 Table 3（1 hr）
2. T2 Prop 1/2 + Thm 1 formal proofs（5 天）
3. T8 LaTeX 化（半天）
4. T9 refine（1 天）
→ **可以在无新实验的情况下投稿**（放弃 Lorenz96 / KS / dysts / EEG，strategize 作为 "Scope" 写在 §6 Limitations）

### Level 2：加强实验证据（~2 周）
- Level 1 全部 + T3 Lorenz96 Phase Transition（用 fallback 的 AR-Kalman M1 版本，2 天）+ T6 FIM 接入（半天）
→ paper Fig 1 升级为 "Lorenz63 + Lorenz96 两个 panels"，证据链更强

### Level 3：完整 tech.md checklist（~4 周）
- Level 2 全部 + T4 KS + T5 dysts 20 + T7 EEG
→ tech.md 100% 完成，paper 可投 NeurIPS/ICLR main track 且数据超过 rebuttal 常见 ask

---

## 八、每项 TODO 的入口文件（便于直接打开）

| TODO | 主要改的文件 | 运行命令 / 脚手架 |
|:-:|---|---|
| T1 | `paper_draft_zh.md §5.8` + `EXPERIMENTS_REPORT_zh.md §2.1` 衍生 | python 脚本从 `pt_v2_with_panda_n5_small.json` 读数生成 |
| T2 | `paper_draft_zh.md Appendix A.1/A.2/A.3` | 纯写作，参考 `tech.md §0.3 §3.6 §4.5` |
| T3 | 新增 `experiments/week1/ks_utils.py` 的对偶 L96 版本 | `CUDA_VISIBLE_DEVICES=0 python -m experiments.week1.phase_transition_pilot_v2 --dataset lorenz96 ...` |
| T4 | 新增 `experiments/week1/ks_utils.py` | `CUDA_VISIBLE_DEVICES=0 python -m experiments.week1.phase_transition_pilot_v2 --dataset ks ...` |
| T5 | 新增 `experiments/week1/run_dysts_benchmark.py` | `python -m experiments.week1.run_dysts_benchmark --systems 20 --n_seeds 3` |
| T6 | 新增 `baselines/fim_adapter.py` | `phase_transition_pilot_v2.py --methods ... fim` |
| T7 | 新增 `experiments/week1/eeg_case_study.py` | 需要本地 EEG 数据 |
| T8 | 新增 `paper/neurips2026.tex` + `paper/refs.bib` | 复制 `paper_draft.md` 后转 LaTeX |
| T9 | `paper_draft.md` 多轮修改 | 无 |

---

## 九、tech.md 里**不打算做**的项（诚实标注）

| tech.md 项 | 为什么放弃 | 替代计划 |
|---|---|---|
| Lorenz96 N=400 scaling | 计算量过大（单次 SVGP 训练数小时），paper 用 N=40 已足够 | 在 §6 Limitations 提"Lorenz96 up to N=40" |
| FIM + Panda 同时跑 PK | Panda + Chronos + Parrot 已够 "foundation model family" 叙事 | FIM 可作 rebuttal material |
| 12-week 严格时间表 | 12 周计划是 tech.md v2 时的估算，实际我们 Week 1-10 的工作压缩在 2-3 周完成 | 时间线已远超 tech.md 预期 |

---

**End of TODO. 此文档应作为下次开机的第一份阅读材料。**
