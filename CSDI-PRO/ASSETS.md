# CSDI-PRO — 产物索引（Figures + 数据文件 + Checkpoints + 脚本）

> **按文件类型/用途速查 paper 和 repo 里的每个产物**。
> 合并自原 `PAPER_FIGURES.md` + `ARTIFACTS_INDEX.md`。
>
> **最后更新**：2026-04-23
>
> 其它文档：
> - [README.md](README.md) — 项目导航入口
> - [STATUS.md](STATUS.md) — 项目状态 + 完成度 + TODO
> - [EXPERIMENTS_REPORT_zh.md](EXPERIMENTS_REPORT_zh.md) — 详细数字 + 符号表

---

## 一、论文 Figures 清单（12 张 paper-ready + 超额）

按 paper 出现顺序：

| # | 文件 | 呈现内容 | 支撑 claim / 说明 |
|:-:|---|---|---|
| **Fig 1** | [figures/pt_v2_with_panda_n5_small_paperfig.png](experiments/week1/figures/pt_v2_with_panda_n5_small_paperfig.png) | Phase Transition 主图，L63 × 7 harshness × 5 methods × 5 seeds，VPT@1.0 / VPT@0.3 / NRMSE 三面板 | **核心卖点**：S3 ours 2.2× Panda / 7.1× Parrot；Panda/Parrot phase-drop −85/−92% |
| **Fig 1b** 🆕 | [figures/pt_v2_csdi_upgrade_n5.png](experiments/week1/figures/pt_v2_csdi_upgrade_n5.png) | Phase Transition CSDI M1 升级（ours vs ours_csdi），n=5 | **S2 全面碾压所有基线**（1.26-8.7×）；S4 ours_csdi vs Panda **9.4×** |
| Fig 2 | [figures/trajectory_overlay_seed3_S0_S2_S3_S5.png](experiments/week1/figures/trajectory_overlay_seed3_S0_S2_S3_S5.png) | 观测空间轨迹叠加，4 scenarios × 3 channels，5 方法的点预测 + VPT 崩断点 | Fig 1 的直观可视化版 |
| **Fig 2 CSDI** 🆕 | [figures/trajectory_overlay_seed3_S0_S2_S3_S5_with_csdi.png](experiments/week1/figures/trajectory_overlay_seed3_S0_S2_S3_S5_with_csdi.png) | 加 ours_csdi 的轨迹叠加 | Fig 1b 的定性版本，直观展示 M1 升级 |
| **Fig 3** ⭐ | [figures/separatrix_ensemble_seed4_S0_K30_ic05.png](experiments/week1/figures/separatrix_ensemble_seed4_S0_K30_ic05.png) | SVGP ensemble rollout，6 面板 = 3 channels + butterfly 相位图 + wing 直方图 + VPT 直方图 + std 曲线 | **paper 核心 novelty 图**："模型知道自己何时不确定" —— std 在 separatrix h=60/104 放大 45×/100×；30/30 正确终态 wing |
| **Fig 3 CSDI** 🆕 | [figures/separatrix_ensemble_seed4_S0_K30_csdi.png](experiments/week1/figures/separatrix_ensemble_seed4_S0_K30_csdi.png) | CSDI M1 + SVGP ensemble on S0 | VPT median 1.99Λ（与 AR-K 同），terminal wing 30/30 — CSDI 不破坏 ensemble |
| Fig 4a | [figures/ablation_S3.png](experiments/week2_modules/figures/ablation_S3.png) | 原 9-config × 4 horizons × 4 metrics，AR-Kalman M1 | Paper Table 2 原版 |
| **Fig 4b dual-M1** 🆕 | [figures/ablation_final_dualM1_paperfig.png](experiments/week2_modules/figures/ablation_final_dualM1_paperfig.png) | **dual-M1 消融**：2 sc × 3 horizons × (AR-Kalman vs CSDI paired bars) | **CSDI M1 在 7/8 configs 上 h=4 一致带来 −18% 到 −24%** |
| Fig 5 | [figures/module4_horizon_cal_S3.png](experiments/week2_modules/figures/module4_horizon_cal_S3.png) | Module 4 mixed-horizon calibration，4 growth modes + Split CP | **Lyap-empirical vs Split mean \|PICP−0.9\| 5.5× 改善** |
| **Fig 5 S2/S3 CSDI** 🆕 | [figures/module4_horizon_cal_S{2,3}_csdi.png](experiments/week2_modules/figures/) | Module 4 horizon cal @ CSDI M1 | Lyap-emp overall PICP 0.898 |
| Fig 6 | [figures/lorenz96_svgp_scaling.png](experiments/week2_modules/figures/lorenz96_svgp_scaling.png) | SVGP Lorenz96 N={10,20,40} 训练时间 vs N，NRMSE | **Proposition 2 实证**：线性 in N |
| Fig 7 | [figures/tau_lowrank_spectrum_paperfig.png](experiments/week2_modules/figures/tau_lowrank_spectrum_paperfig.png) | τ 矩阵 UU^T 奇异值谱 L={3,5,7} | **有效 rank ≈ 2-3**，验证 §2.3 low-rank ansatz |
| **D2** 🆕 | [figures/coverage_across_harshness_paperfig.png](experiments/week2_modules/figures/coverage_across_harshness_paperfig.png) | Coverage Across Harshness, S0-S6 × 3 horizons × 3 seeds | **Lyap-emp 3.2× 更准校准**（\|PICP−0.9\| 0.071→0.022） |
| **D2 CSDI** 🆕 | [figures/coverage_across_harshness_paperfig_csdi.png](experiments/week2_modules/figures/coverage_across_harshness_paperfig_csdi.png) | 同设置但 CSDI M1 | 2.3× 改善，claim 仍成立 |
| **D3** 🆕 | [figures/horizon_coverage_paperfig{,_csdi}.png](experiments/week2_modules/figures/) | Horizon × Coverage 独立图（paper 版），2 panels × 5 CP methods | **Lyap-empirical 全 horizon 稳在 [0.88, 0.92]** |
| **D4** 🆕 | [figures/horizon_piwidth_paperfig{,_csdi}.png](experiments/week2_modules/figures/) | Horizon × PI Width | **Lyap-growth 让 PI 合理扩张** |
| **D5** 🆕 | [figures/reliability_diagram_paperfig{,_csdi}.png](experiments/week2_modules/figures/) | Reliability diagram, α∈{0.01..0.5}, Raw Gaussian vs Split CP | **Raw GP 严重过覆盖**（α=0.3 PICP 0.98 vs 0.70）；Split CP 完美贴 y=x |
| **D6** 🆕 | [figures/tau_stability_paperfig.png](experiments/week2_modules/figures/tau_stability_paperfig.png) | MI-Lyap τ-stability vs noise, 6 σ × 15 seeds × 3 methods | **MI-Lyap σ=0 时 15/15 选同一 τ（std=0）**；σ=0.5 下比 Fraser std 小 47% |
| **D7 v2** 🆕 | [figures/tau_lowrank_spectrum_paperfig.png](experiments/week2_modules/figures/tau_lowrank_spectrum_paperfig.png) | τ 矩阵奇异值谱 L={3,5,7} × 5 seeds | **有效 rank ≈ 2-3**，验证 §2.3 low-rank ansatz |

**待补**：D9 EEG case study（需要 EEG 数据集，见 STATUS.md §五 T7）。

---

## 二、实验数据 JSON（所有 paper claim 的原始数字）

### 2.1 Phase Transition 主图数据

| 文件 | 内容 | 用于 |
|---|---|---|
| [experiments/week1/results/pt_v2_with_panda_n5_small.json](experiments/week1/results/pt_v2_with_panda_n5_small.json) | 175 runs (L63 × 7 × 5 × 5 seeds) 的 VPT03/05/10 + NRMSE | **Fig 1 主图 + paper Table 1 数字** |
| [pt_v2_with_panda_n5_small.md](experiments/week1/results/pt_v2_with_panda_n5_small.md) | 同数据 markdown 表 | 复制到 paper 附录 |
| [pt_v2_csdi_upgrade_n5.json](experiments/week1/results/pt_v2_csdi_upgrade_n5.json) | Phase Transition ours vs ours_csdi, 7 scenarios × 5 seeds（主）| **Fig 1b 输入（主）** |
| [pt_v2_csdi_upgrade_n3.json](experiments/week1/results/pt_v2_csdi_upgrade_n3.json) | 先导 n=3 版本（废弃） | 历史 |
| [separatrix_ensemble_seed4_S0_K30.json](experiments/week1/results/separatrix_ensemble_seed4_S0_K30.json) + [.npz](experiments/week1/results/separatrix_ensemble_seed4_S0_K30.npz) | Ensemble 30 paths 的 VPT / PICP / separatrix std / wing counts；30 条 path + truth + quantile bands（64KB） | **Fig 3 数字来源 + 可复现性** |

### 2.2 消融数据 (Fig 4 / Table 2)

| 文件 | 内容 | 用于 |
|---|---|---|
| [ablation_S3_n3_v2.json](experiments/week2_modules/results/ablation_S3_n3_v2.json) | 原 S3 × 9 configs × 3 seeds × 4 horizons（AR-Kalman M1） | **Paper Table 2 原版** |
| [ablation_S2_n3_v2.json](experiments/week2_modules/results/ablation_S2_n3_v2.json) | 同 S2 | 补充 S2 数据 |
| [ablation_with_csdi_v6_ep20.json](experiments/week2_modules/results/ablation_with_csdi_v6_ep20.json) | CSDI M1 batch 1（S2+S3 × 5 configs） | Fig 4b 输入 |
| [ablation_with_csdi_v6_ep20_9cfg_S3.json](experiments/week2_modules/results/ablation_with_csdi_v6_ep20_9cfg_S3.json) | CSDI M1 batch 2（S3 补 5 configs） | Fig 4b 输入 |
| [ablation_with_csdi_v6_ep20_9cfg_S2.json](experiments/week2_modules/results/ablation_with_csdi_v6_ep20_9cfg_S2.json) | CSDI M1 batch 3（S2 补 5 configs） | Fig 4b 输入 |
| **[ablation_final_dualM1_merged.json](experiments/week2_modules/results/ablation_final_dualM1_merged.json)** + [.md](experiments/week2_modules/results/ablation_final_dualM1_merged.md) | **合并后的 dual-M1 table（AR-Kalman vs CSDI 并排，9 configs × 3 horizons × 2 scenarios）** | **paper Table 2 最终版** |
| [ablation_final_s3_merged.json](experiments/week2_modules/results/ablation_final_s3_merged.json) + [.md](experiments/week2_modules/results/ablation_final_s3_merged.md) | legacy S3-only | 历史兼容 |

### 2.3 CP 校准数据 (Fig 5 / D2 / D3 / D4 / D5)

| 文件 | 内容 | 用于 |
|---|---|---|
| [module4_horizon_cal_S3_n3.json](experiments/week2_modules/results/module4_horizon_cal_S3_n3.json) | S3 × 4 growth modes × 8 horizons × 3 seeds PICP/MPIW | **Fig 5 / D3 / D4 输入** |
| [module4_horizon_cal_S2_n3.json](experiments/week2_modules/results/module4_horizon_cal_S2_n3.json) | 同 S2 | 同上 |
| [module4_horizon_cal_S3_n3_csdi.json](experiments/week2_modules/results/module4_horizon_cal_S3_n3_csdi.json) | S3 × 同上 × CSDI M1 | CSDI 补齐 |
| [module4_horizon_cal_S2_n3_csdi.json](experiments/week2_modules/results/module4_horizon_cal_S2_n3_csdi.json) | S2 × 同上 × CSDI M1 | CSDI 补齐 |
| [reliability_diagram_n3_v1.json](experiments/week2_modules/results/reliability_diagram_n3_v1.json) + [\_csdi.json](experiments/week2_modules/results/reliability_diagram_n3_v1_csdi.json) | α ∈ {0.01..0.5} × S2+S3 × 3 seeds, Raw/Split PICP | **D5 输入** |
| [coverage_across_harshness_n3_v1.json](experiments/week2_modules/results/coverage_across_harshness_n3_v1.json) + [\_csdi.json](experiments/week2_modules/results/coverage_across_harshness_n3_v1_csdi.json) | 7 scenarios × 3 horizons × 3 seeds, Split + Lyap-emp PICP/MPIW | **D2 输入** |

### 2.4 SVGP scaling

| 文件 | 内容 |
|---|---|
| [lorenz96_scaling_N10_20_40.json](experiments/week2_modules/results/lorenz96_scaling_N10_20_40.json) | N∈{10,20,40} × 2 seeds 训练时间 + NRMSE（**Fig 6 输入**）|

### 2.5 Module 2 专项 (Fig D6 / D7)

| 文件 | 内容 |
|---|---|
| [tau_stability_n15_v1.json](experiments/week2_modules/results/tau_stability_n15_v1.json) | 6 σ × 15 seeds × 3 方法的 τ 向量（**D6 输入**） |
| [tau_spectrum_v2.json](experiments/week2_modules/results/tau_spectrum_v2.json) | L ∈ {3,5,7} × 5 seeds 的 CMA-ES UV^T 奇异值（**D7 输入**） |

### 2.6 CSDI 训练日志（诊断用）

| 文件 | 内容 | 价值 |
|---|---|---|
| `csdi_longrun_{full,no_noise,no_mask,vanilla}_v5_long.log` | v5_long 训练（无 centering）日志 | 对照组，证明 centering 必需 |
| `csdi_longrun_*_v6_center.log` | v6_center 训练（三重修复后）日志 | **paper ckpt 训练过程可复现** |
| `ablation_with_csdi_v6_ep20*.log` | 完整 ablation 运行日志 | 诊断参考 |

---

## 三、Model Checkpoints

### 3.1 CSDI paper M1 checkpoint

| 文件 | 大小 | 用途 |
|---|:-:|---|
| **`experiments/week2_modules/ckpts/dyn_csdi_full_v6_center_ep20.pt`** | ~5 MB | **paper 使用的最佳 CSDI M1 checkpoint** — ablation_with_csdi 所有实验引用它 |
| `dyn_csdi_full_v6_center_ep{10,30,40,50,60}.pt` | ~5 MB 各 | 额外 epochs 用于 ep 选择实验 |
| `dyn_csdi_no_noise_v6_center_ep{10..60}.pt` | ~5 MB 各 | ablation B-only（delay_mask，无 noise_cond）|
| `dyn_csdi_no_mask_v6_center_ep{10,20}.pt` | ~5 MB 各 | ablation A-only（对照 RMSE 7.4）|
| `dyn_csdi_vanilla_v6_center_ep{10,20}.pt` | ~5 MB 各 | ablation baseline（RMSE 7.4）|
| `dyn_csdi_*_v{3,5}*.pt` | — | 历史 checkpoint（v3 旧方案失败对照，v5_long 无 centering 对照）|

**注意**：ckpts 在 `.gitignore` 里，本地有但 git 不存。要复制请手动拷。

### 3.2 训练数据缓存

| 文件 | 大小 | 说明 |
|---|:-:|---|
| `experiments/week2_modules/data/lorenz63_clean_512k_L128.npz` | 751 MB | CSDI v6_center 训练缓存，本地生成，git 不存 |

---

## 四、核心代码文件（功能索引）

### 4.1 四大 Module 实现

| 文件 | Module | 核心接口 |
|---|:-:|---|
| [methods/dynamics_csdi.py](methods/dynamics_csdi.py) | **M1** | `DynamicsCSDI`, `Lorenz63ImputationDataset`，含 per-dim centering + Bayesian soft-anchor |
| [methods/dynamics_impute.py](methods/dynamics_impute.py) | M1 baseline + 分发 | `impute(observed, kind={linear, cubic, dynamics, ar_kalman, csdi})` |
| [methods/csdi_impute_adapter.py](methods/csdi_impute_adapter.py) | M1 桥接 | `set_csdi_checkpoint(path)`, `csdi_impute(observed)` |
| [methods/mi_lyap.py](methods/mi_lyap.py) | **M2** | `mi_lyap_bayes_tau`（Stage A BO）, `mi_lyap_cmaes_tau`（Stage B CMA-ES）, `robust_lyapunov`, `fraser_swinney_tau`, `random_tau` |
| [models/svgp.py](models/svgp.py) | **M3** | `SVGP`, `MultiOutputSVGP`, `SVGPConfig`（GPyTorch Matern-5/2）|
| [methods/lyap_conformal.py](methods/lyap_conformal.py) | **M4** | `SplitConformal`, `LyapConformal`（4 growth modes）, `AdaptiveLyapConformal` |

### 4.2 Pipeline + Baselines

| 文件 | 作用 |
|---|---|
| [experiments/week1/full_pipeline_rollout.py](experiments/week1/full_pipeline_rollout.py) | 端到端 pipeline：`full_pipeline_forecast(obs, pred_len, imp_kind=...)` + `full_pipeline_ensemble_forecast(...)` |
| [baselines/panda_adapter.py](baselines/panda_adapter.py) | Panda-72M 零样本封装 |
| [baselines/chronos_adapter.py](baselines/chronos_adapter.py) | Chronos-T5 {small, base, large} |
| [baselines/context_parroting.py](baselines/context_parroting.py) | Context-Parroting，1-NN in context |

### 4.3 度量

| 文件 | 内容 |
|---|---|
| [metrics/uq_metrics.py](metrics/uq_metrics.py) | CRPS, PICP, MPIW, Winkler, reliability_curve, ECE |
| [metrics/chaos_metrics.py](metrics/chaos_metrics.py) | VPT（多阈值）, NRMSE |

### 4.4 主实验脚本

| 脚本 | 产物 Figure |
|---|:-:|
| [experiments/week1/phase_transition_pilot_v2.py](experiments/week1/phase_transition_pilot_v2.py) | **Fig 1 + Fig 1b** |
| [experiments/week1/plot_trajectory_overlay.py](experiments/week1/plot_trajectory_overlay.py) | Fig 2 |
| [experiments/week1/plot_separatrix_ensemble.py](experiments/week1/plot_separatrix_ensemble.py) | **Fig 3** |
| [experiments/week2_modules/run_ablation.py](experiments/week2_modules/run_ablation.py) | 原 9-config ablation（AR-Kalman M1） |
| [experiments/week2_modules/run_ablation_with_csdi.py](experiments/week2_modules/run_ablation_with_csdi.py) | CSDI M1 9-config ablation |
| [experiments/week2_modules/merge_ablation_csdi_paperfig.py](experiments/week2_modules/merge_ablation_csdi_paperfig.py) | **合并 AR-K + CSDI 生成 Fig 4b** |
| [experiments/week2_modules/module4_horizon_calibration.py](experiments/week2_modules/module4_horizon_calibration.py) | Module 4 专项（Fig 5 + D3/D4 数据） |
| [experiments/week2_modules/plot_horizon_calibration_paperfig.py](experiments/week2_modules/plot_horizon_calibration_paperfig.py) | **D3/D4 画图** |
| [experiments/week2_modules/reliability_diagram.py](experiments/week2_modules/reliability_diagram.py) | **D5 实验 + 画图** |
| [experiments/week2_modules/coverage_across_harshness.py](experiments/week2_modules/coverage_across_harshness.py) | **D2** |
| [experiments/week2_modules/tau_stability_vs_noise.py](experiments/week2_modules/tau_stability_vs_noise.py) | **D6** |
| [experiments/week2_modules/tau_lowrank_spectrum_v2.py](experiments/week2_modules/tau_lowrank_spectrum_v2.py) | **D7** |
| [experiments/week2_modules/lorenz96_scaling.py](experiments/week2_modules/lorenz96_scaling.py) | Fig 6 |
| [experiments/week2_modules/train_dynamics_csdi.py](experiments/week2_modules/train_dynamics_csdi.py) | CSDI 训练脚本 |
| [experiments/week2_modules/run_csdi_longrun.sh](experiments/week2_modules/run_csdi_longrun.sh) | 4-GPU 并行 CSDI 训练 launcher |
| [experiments/week2_modules/make_lorenz_dataset.py](experiments/week2_modules/make_lorenz_dataset.py) | 生成 Lorenz63 cache（v6 用 512K × seq=128） |

---

## 五、会话记录（决策过程 + 完整推理）

| 文件 | 涵盖 |
|---|---|
| [../session_notes/2026-04-22_csdi_m1_diagnosis_fix_breakthrough.md](../session_notes/2026-04-22_csdi_m1_diagnosis_fix_breakthrough.md) | **核心**：CSDI 诊断、3 重修复、完整翻盘流程（5 阶段） |
| [../session_notes/2026-04-21_csdi_pro_v2_week1.md](../session_notes/2026-04-21_csdi_pro_v2_week1.md) | W1 Day 1-2 环境 + Day 6-7 Phase Transition pilot |
| [../session_notes/2026-04-21_csdi_pro_v2_week2_modules_ablation.md](../session_notes/2026-04-21_csdi_pro_v2_week2_modules_ablation.md) | W2 四 module 实现 + 消融实验完整归档 |
| [../session_notes/2026-04-21_separatrix_ensemble_figure3.md](../session_notes/2026-04-21_separatrix_ensemble_figure3.md) | Fig 3 separatrix ensemble 的完整设计 + 结果 |

---

## 六、Paper Claim × 支撑文件映射

| Paper Claim | 支撑文件 |
|---|---|
| C1: S3 ours vs Panda 2.2× / vs Parrot 7.1× | Fig 1 + `pt_v2_with_panda_n5_small.json` |
| C2: Panda/Parrot phase drop −85/−92% | Fig 1 同文件 |
| C3: Ensemble std 在 separatrix 放大 45×/100× | Fig 3 + `separatrix_ensemble_seed4.{json,npz}` |
| C4: Full vs v1-like +104% NRMSE 提升 | Fig 4a + `ablation_S3_n3_v2.json` |
| **C5: CSDI M1 升级带来 h=4 −24%** 🆕 | **Fig 4b + `ablation_final_dualM1_merged.{json,md}`** |
| C6: 每 Module ≥ 24% 独立贡献 | Fig 4a/b + ablation_* JSONs |
| C7: Lyap-empirical 5.5× PICP 校准改善 | Fig 5 + `module4_horizon_cal_S3_n3.json` |
| **C8: Horizon × Coverage 稳定贴 0.90** 🆕 | D3 `horizon_coverage_paperfig.png` |
| **C9: Raw GP 严重过覆盖，CP 校准必要** 🆕 | D5 `reliability_diagram_*.json` |
| C10: SVGP 训练时间线性 in N | Fig 6 + `lorenz96_scaling.json` |
| C11: Proposition 1/2/Theorem 1 formal | ❌ **理论证明未写**（见 STATUS.md §五 T2） |
| C12: Stage B CMA-ES 1.8× 快 | Fig 7（数据在 ablation 里） |
| **C13: ours_csdi @ S4 vs Panda 9.4×** 🆕 | Fig 1b + `pt_v2_csdi_upgrade_n5.json` |
| **C14: ours_csdi @ S2 全面碾压所有基线** 🆕 | 同上 |

---

## 七、常用复现命令

```bash
# Phase Transition 主图（Fig 1）
CUDA_VISIBLE_DEVICES=0 python -m experiments.week1.phase_transition_pilot_v2 \
    --n_seeds 5 --tag with_panda_n5_small

# Phase Transition CSDI 升级（Fig 1b）
CUDA_VISIBLE_DEVICES=0 python -m experiments.week1.phase_transition_pilot_v2 \
    --n_seeds 5 --methods ours ours_csdi \
    --csdi_ckpt experiments/week2_modules/ckpts/dyn_csdi_full_v6_center_ep20.pt \
    --tag csdi_upgrade_n5

# Fig 2 轨迹叠加（带 CSDI）
CUDA_VISIBLE_DEVICES=0 python -m experiments.week1.plot_trajectory_overlay \
    --seed 3 --scenarios S0 S2 S3 S5 --tag seed3_S0_S2_S3_S5_with_csdi \
    --include_csdi \
    --csdi_ckpt experiments/week2_modules/ckpts/dyn_csdi_full_v6_center_ep20.pt

# Fig 3 Separatrix ensemble
CUDA_VISIBLE_DEVICES=0 python -m experiments.week1.plot_separatrix_ensemble \
    --seed 4 --sparsity 0 --noise 0 --K 30 --tag seed4_S0_K30_ic05

# Fig 4b dual-M1 合版
python -m experiments.week2_modules.merge_ablation_csdi_paperfig

# Fig 5 Module 4 horizon calibration
CUDA_VISIBLE_DEVICES=0 python -m experiments.week2_modules.module4_horizon_calibration \
    --n_seeds 3 --scenario S3

# D2 Coverage Across Harshness（含 CSDI 版）
CUDA_VISIBLE_DEVICES=0 python -m experiments.week2_modules.coverage_across_harshness \
    --n_seeds 3 --impute_kind csdi \
    --csdi_ckpt experiments/week2_modules/ckpts/dyn_csdi_full_v6_center_ep20.pt

# D3/D4 直接画图（无需重跑实验）
python -m experiments.week2_modules.plot_horizon_calibration_paperfig

# D5 Reliability diagram
CUDA_VISIBLE_DEVICES=0 python -m experiments.week2_modules.reliability_diagram \
    --n_seeds 3 --scenarios S2 S3

# D6 τ-stability
python -m experiments.week2_modules.tau_stability_vs_noise --n_seeds 15

# D7 τ-low-rank spectrum
python -m experiments.week2_modules.tau_lowrank_spectrum_v2 --L_list 3 5 7 --n_seeds 5

# Fig 6 SVGP scaling
CUDA_VISIBLE_DEVICES=0 python -m experiments.week2_modules.lorenz96_scaling \
    --N 10 20 40 --n_seeds 2

# CSDI M1 训练
bash experiments/week2_modules/run_csdi_longrun.sh
```

---

**End of ASSETS. 当你想找某个 figure、数据、脚本时打开本文档。**
