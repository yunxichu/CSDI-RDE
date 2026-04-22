# CSDI-PRO — 项目状态、计划、完成与未完成

> **一张文档查清：做了什么、做到什么程度、还剩什么、下次怎么接**。
> 合并自原 `DELIVERY.md` / `PROGRESS.md` / `TODO_tech_gap_zh.md` / `COMPLETE_WORK_LOG_zh.md`。
>
> **最后更新**：2026-04-23  ·  **分支**：`csdi-pro`  ·  **最新 commit**：`f9ffca3`（之后可能更新）
>
> 其它文档：
> - [README.md](README.md) — 项目导航入口
> - [ASSETS.md](ASSETS.md) — 论文 figures + 数据文件索引
> - [EXPERIMENTS_REPORT_zh.md](EXPERIMENTS_REPORT_zh.md) — 详细实验结果 + 符号表
> - [tech.md](tech.md) — v2 技术设计规范（1046 行，历史档案）
> - [paper_draft_zh.md](paper_draft_zh.md) / [paper_draft.md](paper_draft.md) — 论文中英文草稿

---

## 一、一句话结论

> **Foundation models（Panda / Chronos / Parrot）在稀疏 + 噪声观测下 categorically phase-transition；我们的 4-module pipeline 做到 graceful degradation，在 S3（60% sparsity + 50% noise）主战场比 Panda 高 2.2×、比 Parrot 高 7.1×。CSDI M1 升级进一步把 S4 regime 的优势扩大到 9× Panda。**

---

## 二、项目时间线

### 阶段 0 — Pre-existing pipeline（2026-04-15 至 2026-04-21）
- 实现 4 module：M1 AR-Kalman smoother、M2 MI-Lyap τ-search (BO)、M3 SVGP、M4 Lyap-CP
- 跑 Phase Transition 主图（Lorenz63 × 7 harshness × 5 methods × 5 seeds = 175 runs → **Fig 1**）
- 跑原 9-config ablation on S2+S3（**Fig 4a**）
- 跑 M4 horizon calibration（**Fig 5**）
- 跑 Lorenz96 SVGP scaling（**Fig 6**）
- 尝试 M1 full CSDI 训练（**v3_big 训练失败**，loss 卡 1.0 或 RMSE 14+）
- 当时结论："paper 继续用 AR-Kalman surrogate"

### 阶段 1 — CSDI M1 翻盘（2026-04-22 上午）

**1.1 用户指令**："扩大合成数据重训"

**1.2 v5_long 训练**：
- 生成 512K 条 Lorenz63 窗口，4 变种并行 × 200 epochs × 400K gradient steps
- 修第一个 bug：`delay_alpha` 初值从 `0.0` 改为 `0.01`（破除零梯度死锁）
- 训练 loss 从 0.24 降到 0.013，但 imputation RMSE **仍卡 6-7**（baseline 3.97）

**1.3 用户关键提问**："CSDI 应该是很强的方法，为什么不行？"
→ 触发系统诊断而非盲目增训

**1.4 诊断 — 发现三个并发 bug**：

| # | Bug | 根因 | 影响 |
|:-:|---|---|---|
| 1 | `delay_alpha × delay_bias` 初始梯度死锁 | $\alpha=0$ 乘积梯度两侧都为零 | full 变种 loss 卡 1.0 |
| 2 | 单尺度归一化使 Z mean=1.79 | 对 (X,Y,Z) 三维用单一 std，破坏 DDPM 的 N(0,1) 先验 | 训练 loss 不能降到真正低位 |
| 3 | 硬锚定把 noisy obs 当 clean 注入 | CSDI paper 原假设观测是 clean | imputation 在 noisy 场景彻底崩溃 |

**1.5 三重修复 → v6_center 训练**：
- Bug 1: `delay_alpha=0.01` 初值
- Bug 2: Per-dim centering（data_center/data_scale 存入 checkpoint buffer）
- Bug 3: Bayesian soft anchor — `E[clean|obs] = obs/(1+σ²)`，按后验方差前向扩散

**1.6 结果**（单 imputation RMSE, n=50）：
- `full_v6_center_ep20.pt`（40K steps）**RMSE 3.75**，比 AR-Kalman（4.17）**好 10%**
- `delay_mask` 贡献 **54%** RMSE 下降（7.4→3.4）
- `noise_cond` 贡献 ~6%
- 最佳 ckpt 在 ep20；ep40+ 开始 overfit diffusion schedule

### 阶段 2 — 用 CSDI 重跑所有下游（2026-04-22 中午）

用户指出原 paper 下游都基于 AR-Kalman，要求补全 CSDI M1 版本。

**2.1 接口修改**：`methods/dynamics_impute.py` 的 `impute()` 增 `kind=="csdi"` 分支；`csdi_impute_adapter.py` 桥接 ckpt 加载。

**2.2 5 个并行任务**（GPU 0-4）：
1. CSDI 9-config ablation on S3（**Fig 4b**）
2. CSDI 9-config ablation on S2（补齐 dual-M1）
3. D2 Coverage Across Harshness @ CSDI M1
4. D5 Reliability diagram @ CSDI M1
5. Fig 5 Module 4 S2/S3 @ CSDI M1

**2.3 Phase Transition CSDI 升级（Fig 1b）**：
- ours_csdi 方法加到 pipeline，重跑 7 scenarios × 5 seeds × (ours + ours_csdi)
- 6/7 scenarios CSDI 胜或持平，S4 VPT **+110%**

**2.4 Qualitative figures CSDI 版**：Fig 2 / Fig 3 都跑 CSDI 版；Fig 3 ensemble VPT 与 AR-K 版 tied（都 1.99 Λ），wing 30/30 全对。

### 阶段 3 — 查漏补缺（2026-04-22 下午）

**3.1 Fig 1b 扩到 n=5**（修 n=3 方差问题）

**3.2 D6 τ-stability vs noise**（新实验）：σ=0 下 MI-Lyap **15/15 选同一 τ**

**3.3 D7 τ 低秩谱 v2**（L=3/5/7 重跑）：effective rank ≈ 2-3

**3.4 D2 Coverage Across Harshness 补跑**：21 cells × AR-Kalman / CSDI 双版

### 阶段 4 — 文档整合（2026-04-23 今日）

合并 `DELIVERY.md` + `PROGRESS.md` + `TODO_tech_gap_zh.md` + `COMPLETE_WORK_LOG_zh.md` → 本文档（STATUS.md）；合并 `PAPER_FIGURES.md` + `ARTIFACTS_INDEX.md` → `ASSETS.md`。README 重写成导航 hub。

---

## 三、累计 21 条可直接引用的 paper 硬数字

| # | 指标 | 数值 | 来源 |
|:-:|---|:-:|---|
| 1 | Panda S0→S3 phase drop | **−85%**（2.90→0.42） | Phase Transition 主扫 |
| 2 | Parrot S0→S3 phase drop | **−92%**（1.58→0.13） | 同上 |
| 3 | Ours vs Panda @ S3 | **2.2×**（0.92 vs 0.42） | 同上 |
| 4 | Ours vs Parrot @ S3 | **7.1×**（0.92 vs 0.13） | 同上 |
| 5 | Chronos 最好 VPT@1.0 | 0.83（S0） | 同上 |
| 6 | Full vs All-off NRMSE 差（S3 h=1） | **+104%** | 消融 |
| 7 | MPIW 改善（S3 h=1） | **2.3×**（20.4→8.9） | 消融 |
| 8 | Lyap-empirical vs Split（S3 mean \|PICP−0.9\|）| **5.5× 改善** | M4 专项 |
| 9 | robust_λ vs nolds（σ=0.5） | err 从 +152% 到 **−1%** | M2 专项 |
| 10 | SVGP 时间 scaling | **线性 in N** | Lorenz96 scaling |
| 11 | CMA-ES Stage B vs BO Stage A | **1.8× 更快**，同质量 | τ-search 对比 |
| 12 | Ours @ S5 vs 所有 baseline | **8.5×**（0.17 vs ≤0.02） | Phase Transition |
| 13 | **CSDI M1 vs AR-Kalman @ S3 h=4** | **−24%**（0.493→0.375） | M1 dual-M1 消融 🆕 |
| 14 | CSDI M1 vs AR-Kalman @ S3 h=16 | **−17%**（0.785→0.655） | 同上 🆕 |
| 15 | CSDI M1 方差缩减 @ S3 h=1 | **3×**（σ 0.028→0.009） | 同上 🆕 |
| 16 | Lyap-emp vs Split overall \|PICP−0.9\| | **3.2×**（0.071→0.022，21 cells） | D2 🆕 |
| 17 | CSDI ours_csdi @ S2 VPT10 | **+53%**（0.80→1.22） | Fig 1b 🆕 |
| 18 | CSDI ours_csdi @ S4 VPT10 | **+110%**（0.26→0.55） | 同上 🆕 |
| 19 | ours_csdi vs Panda @ S4 | **9.4×**（0.55 / 0.06） | Fig 1b 扩展 🆕 |
| 20 | ours_csdi vs Parrot @ S4 | **8.1×**（0.55 / 0.07） | 同上 🆕 |
| 21 | ours_csdi @ S2 全面碾压所有 baseline | **1.26-8.7×** | 同上 🆕 |

---

## 四、tech.md 完成度对照

| 维度 | 计划 | 完成 | 未完成 |
|---|:-:|:-:|:-:|
| 图（主图 + D 系列） | 9 + 4 | **12** ✅ | D9 EEG ❌ |
| 表 | 3 | 1 ✅ | Table 1 dysts / Table 3 sharp summary |
| 数据集 | L63 / L96 / KS / dysts20 / EEG | **L63 完整、L96 部分** | L96-PT / KS / dysts20 / EEG ❌ |
| Foundation model PK | Chronos / Panda / FIM / Parrot | 3/4 ✅ | FIM ❌ |
| 理论 3 命题 | Formal proofs | Informal sketch ✅ | **Formal ❌** |
| 论文写作 | 12 章 + appendix | 首版中英草稿 ✅ | Refine / LaTeX / submit-ready |
| Claim 1 / 5（phase transition + 每模块必要性） | — | ✅ | — |
| Claim 2 / 3 / 4（理论） | — | Empirical 有 | **Formal 缺** |

**总估算完成度 ≈ 75%**。剩下的 25% 集中在 3 块：高维/真实数据、理论证明、论文多轮迭代。

### 4.1 图 12/13 张（细分）

| # | tech.md 图 | 实际 | 说明 |
|:-:|---|:-:|---|
| Fig 1 | Phase Transition 3 datasets | ⚠️ 只 L63 | L96 / KS 未跑 |
| Fig 2 | Coverage Across Harshness | ⚠️ 只 L63（D2）| L96 / KS 未跑 |
| Fig 3 | Horizon × Coverage | ✅ AR-K + CSDI | 完成 |
| Fig 4 | Horizon × PI Width | ✅ AR-K + CSDI | 完成 |
| Fig 5 | Reliability | ✅ AR-K + CSDI | 完成 |
| Fig 6 | τ stability vs noise | ✅ | 完成 |
| Fig 7 | τ low-rank spectrum | ✅ L={3,5,7}| 完成 |
| Fig 8 | SVGP scaling | ✅ | 完成 |
| Fig 9 | EEG case study | ❌ | 需数据集 |
| 超额 | Fig 1b / 4b / 2-traj / 3-ensemble （双 M1 版）| ✅ | 超额完成 |

### 4.2 两阶段 τ-search（tech.md §2.3）

| Stage | 实现 | 实际用在 | 状态 |
|:-:|:-:|---|:-:|
| A — BayesOpt + cumulative-δ | ✅ `mi_lyap_bayes_tau` | 主 pipeline 全部 | 完整 |
| B — 低秩 CMA-ES (UV^⊤) | ✅ `mi_lyap_cmaes_tau` | D7 低秩谱 + Fig 6 scaling 对比 | **未在主 benchmark 用**（L96 PT 没跑）|

---

## 五、10 项未完成 TODO（按价值 × 难度排序）

| 优先级 | TODO | 工作量 | 障碍 | 入口文件 |
|:-:|---|:-:|---|---|
| 🔥🔥🔥 **最高** | **T0** paper 叙事重构（延迟流形统一框架） | 3 周（P0 1w + P1 2w） | 无 | [`REFACTOR_PLAN_zh.md`](REFACTOR_PLAN_zh.md) — P0 纯写作（§1/§3.0/§3-4 重定位），P1 τ-coupling ablation + $n_\text{eff}$ unified + Prop 1/新 Theorem formal。本项直接提升投稿天花板，T2/T9 是其子任务 |
| 🔥 高 | **T1** Table 3 极端 harshness summary | 1 hr | 无 | `paper_draft_zh.md §5.8` 衍生，读 `pt_v2_with_panda_n5_small.json` |
| 🔥 高 | **T2** Prop 1/2 + Thm 1 formal proofs（T0 子任务） | 5 天 | 纯写作 | `paper_draft_zh.md` Appendix A.1/A.2/A.3；参考 tech.md §0.3 §3.6 §4.5 + Tsybakov 2009 / Castillo 2014 / Chernozhukov 2018；**注意**：T0 会引入新 Theorem (Sparsity-Noise Interaction)，证明需配套调整 |
| 🔥🔥 高 | **T3** Lorenz96 Phase Transition | 2-3 天 | CSDI 需 L96 重训 | 仿 `make_lorenz_dataset.py` 写 L96 版；改 `DynamicsCSDIConfig.data_dim`；复用 `phase_transition_pilot_v2.py`（调 D=40） |
| 🔥🔥 高 | **T4** KS PDE 场景 | 3-5 天 | 无 KS integrator | 新写 `experiments/week1/ks_utils.py`（ETDRK4 积分器）; 复用 PT 模板 |
| 🔥🔥 高 | **T5** dysts 20 benchmark（Table 1）| 1-2 天 + ~17 GPU-hr | 无 | `pip install dysts`，新写 `experiments/week1/run_dysts_benchmark.py` |
| 🟡 中 | **T6** FIM foundation model 接入 | 半天 | 无 | 仿 `baselines/panda_adapter.py` 写 `baselines/fim_adapter.py`，挂进 `phase_transition_pilot_v2.py` |
| 🟡🟡 中 | **T7** EEG case study（Fig 9）| 2-3 天 | 需 EEG 数据 | CHB-MIT / TUSZ 数据集；新写 `experiments/week1/eeg_case_study.py` |
| 🟢 低 | **T8** LaTeX 化（NeurIPS template） | 半天 | 无 | 新建 `paper/neurips2026.tex` + `paper/refs.bib` |
| 🟢 低 | **T9** Paper 多轮 refine | 2-3 天 | 纯写作 | Abstract / Introduction / Discussion 扩充 |

### 推荐 3 级路径

| 路径 | 时间 | 做什么 | 结果 |
|:-:|:-:|---|---|
| **Level 0 叙事升级**（推荐起点） | ~3 周 | **T0 全做**（REFACTOR_PLAN P0+P1）+ T1 + T8 | 投 ICML/NeurIPS accept band（理论骨架 + τ-coupling 实证 + LaTeX）；T2/T9 包含在 T0 内 |
| **Level 1 最短 submit** | ~1 周 | T1 + T2 + T8 + T9 | 投 NeurIPS/ICLR（放弃 L96/KS/dysts/EEG，写进 §6 Limitations）—— 不升级叙事 |
| **Level 2 加强** | ~2 周 | Level 1 + T3 + T6 | Fig 1 升级为 L63 + L96 双 panels |
| **Level 3 完整** | ~4 周 | Level 2 + T4 + T5 + T7 | tech.md 100% 完成 |
| **Level 4 完整 + 叙事升级**（天花板） | ~7 周 | T0 + Level 3 | ICML/NeurIPS accept + 全系统验证（L63+L96+KS+dysts+EEG） |

---

## 六、已识别并解决的 12 条技术 blockers

1. ✅ `npeet` 装不上 → 手写 KSG MI/CMI
2. ✅ BayesOpt 选 τ=[1,1,1,1] 重复 → cumulative-δ 参数化
3. ✅ CSDI mask 形状歧义 → 3-channel 输入（cond_part + noise_part + cond_mask）
4. ✅ CSDI 观测位漂移 → 每步 re-impose anchors（后被 #12 Bayesian soft-anchor 取代）
5. ✅ `nolds.lyap_r` 噪声下高估 4× → `robust_lyapunov`
6. ✅ M4 `exp(λh·dt)` 长 h 过保守 → empirical growth 模式
7. ✅ Python stdout 缓冲导致后台 tail 看不到 → `python -u`
8. ✅ Lorenz96 τ-search 慢 → L=10 降 L=7
9. ✅ Panda-72M 架构自写失败（attn 放大 3000×）→ 用户外部 clone 官方 repo，`sys.path` import
10. ✅ CSDI `full` variant 卡 loss=1.0（delay_alpha×delay_bias 乘积梯度为零）→ `delay_alpha` 初值 0.01
11. ✅ CSDI 单尺度归一化使 Z mean=1.79 → per-dim centering，`data_center` 存 checkpoint
12. ✅ CSDI 推理硬锚定 noisy obs 注入噪声 → Bayesian soft-anchor：`E[clean|obs]=obs/(1+σ²)` + 正确前向扩散方差

---

## 七、风险 & 投稿可能性

### 风险矩阵

| 风险 | tech.md 原评估 | 当前评估 |
|---|:-:|:-:|
| Phase transition pilot 失败 | 未知 | **已消解**（Panda −85%, Parrot −92% 已证） |
| Panda/Parrot 在 sparse 下没崩 | 30% | **已消解** |
| MI-Lyap 没明显优于 Fraser-Sw | 40% | **已消解**（σ=0 时 15/15 同 τ，Fraser std=2.19）|
| Dynamics-Aware CSDI 训练不稳 | 30% | **已消解**（v6_center_ep20 比 AR-K 好 10%）|
| 理论证不出严格版 | 70% | **未改变**（Informal 仍在） |
| 时间不够 | 50% | 降：Level 1 路径 1 周可 submit |

### 投稿可能性估计（当前状态）

| 目标 | 原 v1 概率 | v2 目标 | **当前估计** |
|---|:-:|:-:|:-:|
| NeurIPS / ICLR main | 25-35% | 40-50% | **45-55%** |
| ICML | 25-35% | 35-45% | 40-50% |
| UAI | 50-60% | 60-70% | 65-75% |
| AISTATS | 60% | 60% | 70% |
| Workshop（至少一个） | 90% | 95% | **98%** |

**提升来源**：CSDI M1 翻盘 + dual-M1 ablation 清晰证明每模块必要性 + 21 条 paper 硬数字 + 12 张 paper-ready figures。

---

## 八、git 历史（12 commits 全推送）

```
f9ffca3  docs: 新增 TODO_tech_gap_zh.md — tech.md 技术对照表
4af2440  docs: Fig 1b 加 ours_csdi 与所有基线的并排对比
3efaaae  docs: paper §5 全面扩写（setup/做了什么/结果/解读）
68b820b  docs: 补 paper §5.5 CSDI 共形数字 + 符号表 + 新增 EXPERIMENTS_REPORT_zh.md
93acd9a  docs: paper draft v0 中文版
afa3255  docs: paper draft v0 — 9-page structure with all hard numbers inline
90e762f  feat: Fig 2 + Fig 3 + D3/D4 的 CSDI M1 版本补齐
0aaf823  feat: CSDI M1 重跑所有下游实验 + dual-M1 合版
c49eb6f  feat: Phase 3 — D2 Coverage Across Harshness + Fig 1b n=5
255ba4c  docs: session note for 2026-04-22 CSDI M1 breakthrough
9ede8ff  feat: CSDI M1 三重修复翻盘 + Paper 消融扩充
（今日还会补文档整合的新 commit）
```

远端：`github.com:yunxichu/CSDI-RDE.git` 分支 `csdi-pro`。

---

## 九、仓库结构速查

```
CSDI-PRO/
├── README.md                          ← 入口（导航）
├── STATUS.md  ← 本文件
├── ASSETS.md                          ← Figures + 文件索引
├── EXPERIMENTS_REPORT_zh.md           ← 详细数字 + 符号表
├── tech.md                            ← v2 技术规范（历史）
├── paper_draft_zh.md  / paper_draft.md ← 论文草稿（中英文）
│
├── methods/
│   ├── dynamics_csdi.py              # M1 CSDI（paper 用 v6_center_ep20.pt）
│   ├── dynamics_impute.py            # M1 baseline + csdi dispatch
│   ├── csdi_impute_adapter.py        # CSDI ckpt ↔ impute() bridge
│   ├── mi_lyap.py                    # M2（Stage A BO + Stage B CMA-ES）
│   └── lyap_conformal.py             # M4（4 growth modes）
├── models/svgp.py                    # M3
├── metrics/{chaos,uq}_metrics.py     # VPT / NRMSE / CRPS / PICP
├── baselines/{panda,chronos}_adapter.py  # baselines
│
├── experiments/
│   ├── week1/
│   │   ├── lorenz63_utils.py          # scenarios, VPT
│   │   ├── lorenz96_utils.py
│   │   ├── full_pipeline_rollout.py   # 端到端 pipeline
│   │   ├── phase_transition_pilot_v2.py  # Fig 1 + Fig 1b
│   │   ├── plot_trajectory_overlay.py    # Fig 2
│   │   ├── plot_separatrix_ensemble.py   # Fig 3
│   │   ├── results/                   # JSON 原始数据（已 git 跟踪）
│   │   └── figures/                   # PNG 图
│   └── week2_modules/
│       ├── train_dynamics_csdi.py + run_csdi_longrun.sh  # CSDI 训练
│       ├── run_ablation{,_with_csdi}.py                  # ablation
│       ├── module4_horizon_calibration.py                # Fig 5
│       ├── coverage_across_harshness.py                  # D2
│       ├── reliability_diagram.py                        # D5
│       ├── tau_stability_vs_noise.py                     # D6
│       ├── tau_lowrank_spectrum_v2.py                    # D7
│       ├── lorenz96_scaling.py                           # Fig 6
│       ├── merge_ablation_csdi_paperfig.py               # Fig 4b 合版
│       ├── ckpts/                                         # CSDI checkpoints（gitignore）
│       ├── data/                                         # Lorenz 缓存（gitignore）
│       ├── results/                                       # JSON 原始数据
│       └── figures/                                       # PNG 图
│
└── ../session_notes/
    └── 2026-04-22_csdi_m1_diagnosis_fix_breakthrough.md  # 三 bug 诊断完整日志
```

---

**End of STATUS. 下次开机从本文档 §五 挑一项 TODO，照入口文件直接上手。**
