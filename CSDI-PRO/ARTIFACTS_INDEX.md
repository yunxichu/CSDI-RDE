# CSDI-PRO Artifacts Index — 全仓库文件归档索引

> **目的**：一张表查清每个产物（figure / data / checkpoint / script / note）**在哪、做什么、支撑哪条 paper claim 或 ablation**。
>
> 配套文档：
> - [DELIVERY.md](DELIVERY.md) — 交付主文档（数字总结、故事线）
> - [PAPER_FIGURES.md](PAPER_FIGURES.md) — 每张 paper figure 的详细描述 + 复现命令（图为中心）
> - [PROGRESS.md](PROGRESS.md) — 扁平任务清单
>
> **最后更新**：2026-04-22（新增 CSDI M1 v6_center + D3/D4/D5 + Figure 4b 后）

---

## 1. Paper Figures（按 paper 里的出现顺序）

| # | 文件 | 呈现内容 | 支撑的 paper claim / ablation |
|:-:|---|---|---|
| **Fig 1** | [figures/pt_v2_with_panda_n5_small_paperfig.png](experiments/week1/figures/pt_v2_with_panda_n5_small_paperfig.png) | Phase Transition 主图，L63 × 7 harshness × 5 methods × 5 seeds，VPT@1.0 / VPT@0.3 / NRMSE 三面板 | **核心卖点**：S3 ours 比 Panda 高 2.2×、比 Parrot 高 7.1×；Panda/Parrot phase-drop −85/−92%，ours 平滑 −47% |
| **Fig 2** | [figures/trajectory_overlay_seed3_S0_S2_S3_S5.png](experiments/week1/figures/trajectory_overlay_seed3_S0_S2_S3_S5.png) | 观测空间轨迹叠加，4 scenarios × 3 channels，5 方法的点预测 + VPT 崩断点 | **Fig 1 的直观可视化版**，展示 ours 稳跟真值、其他偏离 |
| **Fig 3** ⭐ | [figures/separatrix_ensemble_seed4_S0_K30_ic05.png](experiments/week1/figures/separatrix_ensemble_seed4_S0_K30_ic05.png) | SVGP ensemble rollout，6 面板 = 3 channels + butterfly 相位图 + wing 直方图 + VPT 直方图 + std 曲线 | **paper 核心 novelty 图**："模型知道自己何时不确定" —— ensemble std 在 separatrix h=60/104 放大 45×/100×；30/30 正确终态 wing |
| Fig 4a | [figures/ablation_S3.png](experiments/week2_modules/figures/ablation_S3.png) | 原 9-config × 4 horizons × 4 metrics，AR-Kalman M1 | Paper Table 2 原版；每 module 独立贡献 ≥ 24% |
| **Fig 4b** 🆕 | [figures/ablation_final_s3_paperfig.png](experiments/week2_modules/figures/ablation_final_s3_paperfig.png) | **dual-M1 消融**：3 panels (h=1/4/16) × 9 configs × (AR-Kalman vs CSDI paired bars) | **CSDI M1 在 7/8 configs 上 h=4 一致带来 −18% 到 −24%**；优势随 horizon 放大（h=1: 3% → h=4: 21%） |
| Fig 5 | [figures/module4_horizon_cal_S3.png](experiments/week2_modules/figures/module4_horizon_cal_S3.png) | Module 4 mixed-horizon calibration，4 growth modes + Split CP | **Lyap-empirical vs Split mean \|PICP−0.90\| 5.5× 改善**（0.013 vs 0.072） |
| Fig 6 | [figures/lorenz96_svgp_scaling.png](experiments/week2_modules/figures/lorenz96_svgp_scaling.png) | SVGP Lorenz96 N={10,20,40} 训练时间 vs N，NRMSE | **Proposition 2 实证**：训练时间线性 in N（exact GPR N=40 会 OOM） |
| Fig 7 ⚠️ | [figures/tau_low_rank_spectrum.png](experiments/week2_modules/figures/tau_low_rank_spectrum.png) | τ 矩阵 UV^T 奇异值谱 | Stage B 低秩 τ-search 有效（CMA-ES rank=2 1.8× 快于 BO）；L=7 区分度不足，需 L=3-5 重跑 |
| **D3** 🆕 | [figures/horizon_coverage_paperfig.png](experiments/week2_modules/figures/horizon_coverage_paperfig.png) | Horizon × Coverage 独立图（paper 版），2 panels × 5 CP methods | **Lyap-empirical 全 horizon 稳在 [0.88, 0.92]**，Split CP 漂到 0.80 |
| **D4** 🆕 | [figures/horizon_piwidth_paperfig.png](experiments/week2_modules/figures/horizon_piwidth_paperfig.png) | Horizon × PI Width，同设置 | **Lyap-growth 让 PI 合理扩张**；Lyap-empirical 既不过窄也不过宽 |
| **D2** 🆕 | [figures/coverage_across_harshness_paperfig.png](experiments/week2_modules/figures/coverage_across_harshness_paperfig.png) | Coverage Across Harshness, S0-S6 × 3 horizons × 3 seeds | **Lyap-emp 3.2× 更准校准**（\|PICP−0.9\| 0.071→0.022，18/21 cells 胜） |
| **D5** 🆕 | [figures/reliability_diagram_paperfig.png](experiments/week2_modules/figures/reliability_diagram_paperfig.png) | Reliability diagram, α∈{0.01..0.5}, Raw Gaussian vs Split CP | **Raw GP 严重过覆盖**（α=0.3 → PICP 0.98 vs nominal 0.70）；**Split CP 完美贴 y=x**，CP 校准必要性铁证 |
| **D6** 🆕 | [figures/tau_stability_paperfig.png](experiments/week2_modules/figures/tau_stability_paperfig.png) | MI-Lyap τ-stability vs noise, 6 σ × 15 seeds × 3 methods | **MI-Lyap 在 σ=0 时 15/15 选同一 τ（std=0）**；σ=0.5 下比 Fraser std 小 47% |
| **D7 v2** 🆕 | [figures/tau_lowrank_spectrum_paperfig.png](experiments/week2_modules/figures/tau_lowrank_spectrum_paperfig.png) | τ 矩阵奇异值谱，L ∈ {3,5,7} × 5 seeds | **有效 rank ≈ 2-3**（明显小于 full rank L-1），验证 §2.3 low-rank ansatz |
| **Fig 1b** 🆕 | [figures/pt_v2_csdi_upgrade_n3.png](experiments/week1/figures/pt_v2_csdi_upgrade_n3.png) | Phase Transition ours vs ours_csdi, 7 scenarios × 3 seeds | **S6 noise-floor VPT 0.02→0.25 (10×)**，RMSE 7/7 全胜 |

**待补**：D2 Coverage Across Harshness / D9 EEG / D11 dysts

---

## 2. 实验数据 JSON + 日志

### 2.1 Phase Transition（主图数据）

| 文件 | 包含 | 用途 |
|---|---|---|
| [experiments/week1/results/pt_v2_with_panda_n5_small.json](experiments/week1/results/pt_v2_with_panda_n5_small.json) | 175 runs (L63 × 7 × 5 × 5 seeds) 的 VPT03/05/10 + NRMSE + infer_time | **Fig 1 的原始数据**；paper Table 1 数字来源 |
| [experiments/week1/results/pt_v2_with_panda_n5_small.md](experiments/week1/results/pt_v2_with_panda_n5_small.md) | 同数据的 markdown table | 复制到 paper 附录 |
| [experiments/week1/results/separatrix_ensemble_seed4_S0_K30.json](experiments/week1/results/separatrix_ensemble_seed4_S0_K30.json) | Ensemble 30 paths 的 VPT/PICP/separatrix std/wing counts | **Fig 3 的数字来源** |
| [experiments/week1/results/separatrix_ensemble_seed4_S0_K30.npz](experiments/week1/results/separatrix_ensemble_seed4_S0_K30.npz) | 30 条 path + truth + quantile bands（64KB） | **Fig 3 的可复现性**（未来重绘不用重跑） |

### 2.2 Module 消融

| 文件 | 包含 | 用途 |
|---|---|---|
| [experiments/week2_modules/results/ablation_S3_n3_v2.json](experiments/week2_modules/results/ablation_S3_n3_v2.json) | S3 × 9 configs × 3 seeds × 4 horizons（AR-Kalman M1） | **Paper Table 2 原版** |
| [experiments/week2_modules/results/ablation_S2_n3_v2.json](experiments/week2_modules/results/ablation_S2_n3_v2.json) | S2 × 9 configs | 补充 S2 数据 |
| [experiments/week2_modules/results/ablation_with_csdi_v6_ep20.json](experiments/week2_modules/results/ablation_with_csdi_v6_ep20.json) | **CSDI M1 × 5 configs × S2+S3**（2026-04-22 batch 1） | **Fig 4b 输入** |
| [experiments/week2_modules/results/ablation_with_csdi_v6_ep20_9cfg_S3.json](experiments/week2_modules/results/ablation_with_csdi_v6_ep20_9cfg_S3.json) | **CSDI × 5 新 configs × S3**（batch 2：csdi-m2a/m2b/m3/m4splitcp/m4lyap-exp） | **Fig 4b 输入** |
| **[experiments/week2_modules/results/ablation_final_s3_merged.json](experiments/week2_modules/results/ablation_final_s3_merged.json)** | **合并的 dual-M1 table**（AR-Kalman vs CSDI 并排，9 configs × 3 horizons） | **paper Table 2 最终版数据** |
| **[experiments/week2_modules/results/ablation_final_s3_merged.md](experiments/week2_modules/results/ablation_final_s3_merged.md)** | 同上，markdown 版 | paper 里贴的表 |
| [experiments/week2_modules/ABLATION.md](experiments/week2_modules/ABLATION.md) | 原 ablation 汇总说明 | 历史档案 |

### 2.3 Module 4 专项（CP 校准）

| 文件 | 包含 | 用途 |
|---|---|---|
| [experiments/week2_modules/results/module4_horizon_cal_S3_n3.json](experiments/week2_modules/results/module4_horizon_cal_S3_n3.json) | S3 × 4 growth modes × 8 horizons × 3 seeds PICP/MPIW | **Fig 5 / D3 / D4 输入** |
| [experiments/week2_modules/results/module4_horizon_cal_S2_n3.json](experiments/week2_modules/results/module4_horizon_cal_S2_n3.json) | S2 同 | 同上 |
| [experiments/week2_modules/results/reliability_diagram_n3_v1.json](experiments/week2_modules/results/reliability_diagram_n3_v1.json) | α ∈ {0.01..0.5} × S2+S3 × 3 seeds 的 Raw/Split PICP | **D5 输入** |
| [experiments/week2_modules/results/tau_stability_n15_v1.json](experiments/week2_modules/results/tau_stability_n15_v1.json) | 6 σ × 15 seeds × 3 方法的 τ vector | **D6 输入** |
| [experiments/week2_modules/results/tau_spectrum_v2.json](experiments/week2_modules/results/tau_spectrum_v2.json) | L ∈ {3,5,7} × 5 seeds 的 CMA-ES UV^T 奇异值 | **D7 输入** |
| [experiments/week1/results/pt_v2_csdi_upgrade_n3.json](experiments/week1/results/pt_v2_csdi_upgrade_n3.json) | Phase Transition ours vs ours_csdi, 7 scenarios × 3 seeds（先导，已废） | 历史 |
| **[experiments/week1/results/pt_v2_csdi_upgrade_n5.json](experiments/week1/results/pt_v2_csdi_upgrade_n5.json)** | **n=5 主数字版**：6/7 场景 CSDI 胜，S4 VPT 翻倍 | **Fig 1b 输入（主）** |
| [experiments/week2_modules/results/coverage_across_harshness_n3_v1.json](experiments/week2_modules/results/coverage_across_harshness_n3_v1.json) | 7 scenarios × 3 horizons × 3 seeds, Split + Lyap-emp PICP/MPIW | **D2 输入** |

### 2.4 SVGP scaling

| 文件 | 包含 |
|---|---|
| [experiments/week2_modules/results/lorenz96_scaling_N10_20_40.json](experiments/week2_modules/results/lorenz96_scaling_N10_20_40.json) | N={10,20,40} × 2 seeds 训练时间 + NRMSE（**Fig 6 输入**） |

### 2.5 CSDI 训练日志（诊断用）

| 文件 | 内容 | 价值 |
|---|---|---|
| `experiments/week2_modules/results/csdi_longrun_{full,no_noise,no_mask,vanilla}_v5_long.log` | v5_long 训练（无 centering）日志 | 对照组，证明 centering 是必需的 |
| `experiments/week2_modules/results/csdi_longrun_*_v6_center.log` | v6_center 训练（三重修复后）日志 | **paper ckpt 训练过程可复现性** |
| `experiments/week2_modules/results/ablation_with_csdi_v6_ep20*.log` | 完整 ablation 运行日志 | 诊断参考 |

---

## 3. Model Checkpoints

### 3.1 CSDI（paper M1）

| 文件 | 大小 | 用途 |
|---|:-:|---|
| **[experiments/week2_modules/ckpts/dyn_csdi_full_v6_center_ep20.pt](experiments/week2_modules/ckpts/)** | ~5 MB | **paper 使用的最佳 CSDI M1 checkpoint** — ablation_with_csdi 所有实验引用它 |
| dyn_csdi_full_v6_center_ep{10,30,40,50,60}.pt | ~5 MB 各 | 额外 epoch 用于 ep 选择实验（见 session notes） |
| dyn_csdi_no_noise_v6_center_ep{10,20,30,40,50,60}.pt | ~5 MB 各 | ablation B-only（delay_mask，无 noise_cond） |
| dyn_csdi_no_mask_v6_center_ep{10,20}.pt | ~5 MB 各 | ablation A-only（noise_cond，无 delay_mask，对照 RMSE 7.4） |
| dyn_csdi_vanilla_v6_center_ep{10,20}.pt | ~5 MB 各 | ablation baseline（无 A/B，对照 RMSE 7.4） |
| dyn_csdi_*_v{3,5}*.pt | ~5-13 MB 各 | 历史 checkpoint（v3 旧方案失败对照，v5_long 无 centering 对照） |

### 3.2 SVGP + GPR（pipeline 内置，每次训练时新建，不持久化）

---

## 4. 核心代码文件（功能索引）

### 4.1 四大 Module 实现

| 文件 | Module | 接口 |
|---|:-:|---|
| [methods/dynamics_csdi.py](methods/dynamics_csdi.py) | **M1** | `DynamicsCSDI`, `Lorenz63ImputationDataset`, 含 per-dim centering + Bayesian soft-anchor |
| [methods/dynamics_impute.py](methods/dynamics_impute.py) | M1 baseline + 分发 | `impute(observed, kind={linear, cubic, dynamics, ar_kalman, csdi})` |
| [methods/csdi_impute_adapter.py](methods/csdi_impute_adapter.py) | M1 桥接 | `set_csdi_checkpoint(path)`, `csdi_impute(observed)`；让 `kind="csdi"` 在 pipeline 里生效 |
| [methods/mi_lyap.py](methods/mi_lyap.py) | **M2** | `mi_lyap_bayes_tau`（Stage A BO）, `mi_lyap_cmaes_tau`（Stage B CMA-ES）, `robust_lyapunov`, `fraser_swinney_tau`, `random_tau` |
| [models/svgp.py](models/svgp.py) | **M3** | `SVGP`, `MultiOutputSVGP`, `SVGPConfig`（GPyTorch Matern-5/2） |
| [methods/lyap_conformal.py](methods/lyap_conformal.py) | **M4** | `SplitConformal`, `LyapConformal`（4 growth modes）, `AdaptiveLyapConformal` |

### 4.2 Pipeline + Baselines

| 文件 | 作用 |
|---|---|
| [experiments/week1/full_pipeline_rollout.py](experiments/week1/full_pipeline_rollout.py) | 端到端 pipeline：`full_pipeline_forecast(obs, pred_len, imp_kind=...)` + `full_pipeline_ensemble_forecast(...)` |
| [baselines/panda_adapter.py](baselines/panda_adapter.py) | Panda-72M zero-shot（Panda 代码在 `/home/rhl/Github/panda-src`，SafeTensors 在 `baselines/panda-72M/`） |
| [baselines/chronos_adapter.py](baselines/chronos_adapter.py) | Chronos-T5 {small, base, large} |
| [baselines/context_parroting.py](baselines/context_parroting.py) | Context-Parroting，1-NN 类方法 |

### 4.3 度量

| 文件 | 内容 |
|---|---|
| [metrics/uq_metrics.py](metrics/uq_metrics.py) | CRPS, PICP, MPIW, Winkler, reliability_curve, ECE |
| [metrics/chaos_metrics.py](metrics/chaos_metrics.py) | VPT（多阈值）, NRMSE |

### 4.4 主实验脚本

| 脚本 | 产物 |
|---|---|
| [experiments/week1/phase_transition_pilot_v2.py](experiments/week1/phase_transition_pilot_v2.py) | Phase Transition 主扫（Fig 1 输入） |
| [experiments/week1/plot_trajectory_overlay.py](experiments/week1/plot_trajectory_overlay.py) | Fig 2 |
| [experiments/week1/plot_separatrix_ensemble.py](experiments/week1/plot_separatrix_ensemble.py) | Fig 3 |
| [experiments/week2_modules/run_ablation.py](experiments/week2_modules/run_ablation.py) | 原 9-config ablation（AR-Kalman M1） |
| [experiments/week2_modules/run_ablation_with_csdi.py](experiments/week2_modules/run_ablation_with_csdi.py) | CSDI M1 9-config ablation |
| [experiments/week2_modules/merge_ablation_csdi_paperfig.py](experiments/week2_modules/merge_ablation_csdi_paperfig.py) | **合并 AR-K + CSDI 生成 Fig 4b** |
| [experiments/week2_modules/module4_horizon_calibration.py](experiments/week2_modules/module4_horizon_calibration.py) | Module 4 专项（Fig 5 + D3/D4 数据） |
| [experiments/week2_modules/plot_horizon_calibration_paperfig.py](experiments/week2_modules/plot_horizon_calibration_paperfig.py) | **D3/D4 画图** |
| [experiments/week2_modules/reliability_diagram.py](experiments/week2_modules/reliability_diagram.py) | **D5 实验 + 画图** |
| [experiments/week2_modules/lorenz96_scaling.py](experiments/week2_modules/lorenz96_scaling.py) | Fig 6 |
| [experiments/week2_modules/train_dynamics_csdi.py](experiments/week2_modules/train_dynamics_csdi.py) | CSDI 训练脚本 |
| [experiments/week2_modules/run_csdi_longrun.sh](experiments/week2_modules/run_csdi_longrun.sh) | 4-GPU 并行 CSDI 训练 launcher |
| [experiments/week2_modules/make_lorenz_dataset.py](experiments/week2_modules/make_lorenz_dataset.py) | 生成 Lorenz63 cache（v6 用 512K × seq=128，751MB） |

---

## 5. 会话记录（决策过程 + 完整推理）

| 文件 | 涵盖 |
|---|---|
| [../session_notes/2026-04-22_csdi_m1_diagnosis_fix_breakthrough.md](../session_notes/2026-04-22_csdi_m1_diagnosis_fix_breakthrough.md) | **今日核心**：CSDI 诊断、3 重修复、完整翻盘流程 |
| [../session_notes/2026-04-21_csdi_pro_v2_week1.md](../session_notes/2026-04-21_csdi_pro_v2_week1.md) | W1 Day 1-2 环境 + Day 6-7 Phase Transition pilot |
| [../session_notes/2026-04-21_csdi_pro_v2_week2_modules_ablation.md](../session_notes/2026-04-21_csdi_pro_v2_week2_modules_ablation.md) | W2 四 module 实现 + 消融实验完整归档 |
| [../session_notes/2026-04-21_separatrix_ensemble_figure3.md](../session_notes/2026-04-21_separatrix_ensemble_figure3.md) | Fig 3 separatrix ensemble 的完整 design + 结果 |

---

## 6. "可作为强力实验或消融"的 claim-by-claim 索引

| paper claim | 支撑文件 |
|---|---|
| C1: **S3 ours vs Panda 2.2× / vs Parrot 7.1×** | Fig 1 + pt_v2_with_panda_n5_small.json |
| C2: **Panda/Parrot phase drop −85/−92%** | Fig 1 同文件 |
| C3: **Ensemble std 在 separatrix 放大 45×/100×** | Fig 3 + separatrix_ensemble_seed4 .json/.npz |
| C4: **Full vs v1-like +104% NRMSE 提升** | Fig 4a + ablation_S3_n3_v2.json |
| C5: **CSDI M1 升级带来 h=4 −24%** 🆕 | **Fig 4b + ablation_final_s3_merged.{json,md}** |
| C6: **每 Module ≥ 24% 独立贡献** | Fig 4a/b + ablation_* JSONs |
| C7: **Lyap-empirical 5.5× PICP 校准改善** | Fig 5 + module4_horizon_cal_S3_n3.json |
| C8: **Horizon × Coverage 稳定贴 0.90** 🆕 | D3 horizon_coverage_paperfig.png |
| C9: **Raw GP 严重过覆盖，CP 校准必要** 🆕 | D5 reliability_diagram_paperfig.png + JSON |
| C10: **SVGP 训练时间线性 in N** | Fig 6 + lorenz96_scaling.json |
| C11: **Proposition 1/2/Theorem 1 formal** | ❌ 理论证明未写 |
| C12: **Stage B CMA-ES 1.8× 快** | Fig 7（数据在 ablation 里） |

---

## 7. 关键数字备查（paper 里直接引用的 15 条，见 [DELIVERY.md §4](DELIVERY.md)）

所有数字都应该可以从本索引的**某个 JSON 文件**加载后复算。
