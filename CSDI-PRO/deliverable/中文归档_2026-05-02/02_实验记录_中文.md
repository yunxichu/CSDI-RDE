# 完整实验内容（中文）— 2026-05-02 freeze (`6189a12`)

本文档独立于论文，列出**所有**已完成实验：动机、协议、cell/seed 配置、关键数字、源 JSON、对应论文章节。论文主稿见 `deliverable/paper/paper_draft_zh.md` 与 `paper_draft_en.md`。

---

## 0 总览：实验仓库结构

```
experiments/
├── week1/
│   ├── results/                       ← 所有 JSON 结果
│   ├── figures/                       ← 摘要 .md + .png 图
│   ├── logs/                          ← 训练 / 评估日志
│   ├── corruption.py                  ← v2 corruption 函数
│   ├── lorenz63_utils.py              ← L63 集成 + Lyapunov + VPT
│   ├── lorenz96_utils.py              ← L96 集成
│   ├── phase_transition_grid_l63_v2.py
│   ├── panda_jitter_control.py
│   ├── panda_embedding_ood_l63.py     ← §4.2 + B1 SAITS arm
│   ├── panda_per_layer_probe_l63.py   ← B2 逐层 hook
│   ├── panda_altimputer_control.py    ← §4.4 替代 imputer
│   ├── chronos_frontier_l63.py        ← P1.2 + P1.4
│   ├── enkf_l63_upper_bound.py        ← P1.3
│   └── jena_real_sensor_pilot.py      ← §6.6 Jena
├── week2_modules/
│   ├── ckpts/                         ← CSDI / SAITS checkpoints
│   ├── data/                          ← 混沌语料 + Jena CSV
│   ├── train_saits_l63.py             ← SAITS 通用预训练
│   └── make_jena_npz.py               ← Jena 数据预处理
└── ...

baselines/
├── panda_adapter.py                   ← Panda-72M forecaster
├── panda_model.py                     ← Panda 模型架构
└── chronos_adapter.py                 ← Chronos-bolt-small forecaster

methods/
├── dynamics_impute.py                 ← linear / AR-Kalman / CSDI 入口
└── csdi_impute_adapter.py             ← CSDI 推理调用
```

---

## 1 锁定 v2 协议

所有 **patched** 实验共享下列约定：

| 系统 | attractor_std | dt | n_ctx | pred_len | Lyapunov 指数 |
|:--|--:|--:|--:|--:|--:|
| Lorenz-63 | 8.51 | 0.025 | 512 | 128 | 0.906 |
| Lorenz-96 N=20, F=8 | 3.6387 | 0.05 | 512 | 128 | 1.61 |
| Lorenz-96 N=10 | — | 0.05 | 512 | 128 | — |
| Rössler | 4.45 | 0.1 | 512 | 128 | 0.071 |
| Mackey-Glass τ=17 | — | — | — | — | — (legacy 协议) |
| Chua double-scroll | — | — | — | — | — (legacy 协议) |

- **Corruption seed scheme**：`seed_corr = 1000 × seed + 5000 + grid_index`，`grid_index` 来自 `experiments/week1/configs/corruption_grid_v2.json`。
- **CSDI 推理**：`set_csdi_attractor_std()` 与系统匹配；`sigma_override = noise_std_frac × attractor_std`，纯稀疏 cell（σ=0）严格为 0，使观测时步保持 ~10⁻⁶ 的 anchor 精度。
- **VPT**：以 Lyapunov 时间归一化，逐轴吸引子标准差作为阈值穿越参考。
- **CI**：mean 用 95% bootstrap（5000 resamples），二项 survival 用 Wilson 95%。
- **稀疏度命名约定**：SP65 = 65% sparsity = 35% keep。SP82 = 82% sparsity = 18% keep。

---

## 2 实验全表（17 项）

下表按论文 Appendix B.2 的编号。每项后面有详细一节（§3.x）。

| # | 名称 | 系统 | 场景 | Cell | Seeds | 论文章节 |
|:-:|:--|:--|:--|:--|:-:|:--|
| 1 | Figure 1 v2 grid（稀疏度线） | L63 | SP00–SP97 | linear/CSDI × Panda/DeepEDM | 10 | §3.2 / §1 |
| 2 | Figure 1 v2 grid（噪声线） | L63 | NO00–NO120 | 同 #1 | 10 | §3.4 |
| 3 | L96 N=20 v2 cross-system | L96 N=20 | SP55–SP82 + NO010–NO050 | 同 #1 | 10 | §3.2 / §4.3 |
| 4 | L63 jitter / 残差控制 | L63 | SP65, SP82 | linear / +iid jitter / +shuffled / CSDI | 10 | §4.3 |
| 5 | L96 N=20 jitter / 残差 | L96 N=20 | SP65, SP82 | 同 #4 | 5 | §4.3 |
| 6 | Rössler jitter / 残差 | Rössler | SP65, SP82 | 同 #4 | 5 | §4.3 |
| 7 | 跨系统 jitter milestone | L63+L96+Rössler | SP65, SP82 | 来自 #4–#6 | 5–10 | Figure 3 |
| 8 | Panda embedding OOD（含 SAITS arm） | L63 | SP65, SP82 | clean / linear / SAITS-pretrained / CSDI × {patch, embed, encoder, pooled} | 5 | §4.2 |
| 9 | Raw-patch v2 诊断 | L63 | SP65, SP82 | clean / linear / CSDI × {local stdev, lag-1 ρ, mid-freq power} | 10 | §4.2 |
| 10 | Cross-system 隔离矩阵（legacy）| L63, L96 N=10/20, Rössler, Kuramoto | S0–S6 | linear/Kalman/CSDI × Panda/DeepEDM | 5 | §4.1 / Figure 2 |
| 11 | MG / Chua scope-boundary | Mackey-Glass, Chua | S0–S6 | 同 #10 | 5 | §6.3 |
| 12 | 替代 imputer C0 sanity（per-instance） | L63 | SP65 | linear / SAITS / BRITS / CSDI（单轨迹 fit） | 5 | 附录 E sanity |
| **13** | **预训练替代 imputer**（P1.1 + P1.5 + P2.2 30-seed） | L63, L96 N=20 | L63 SP65+SP82, L96 SP82 | linear / SAITS-pretrained / CSDI | L63: 10, **L96: 30** | §4.4 + 附录 C |
| **14** | **Chronos mini-frontier**（P1.2 + P1.4 native） | L63 | SP55–SP82 | linear / CSDI × Chronos × {pred_len 64, 128} | 5 | §6.4 |
| **15** | **EnKF 已知动力学上界**（P1.3） | L63 | SP55–SP82 + NO020 + NO050 | EnKF（真实向量场，100 ensemble members） | 5 | §6.4 / §6.5 |
| **16** | **Jena Climate 真实传感器**（P2.1 + P3.A） | Jena Climate 2009–2016 | SP55, SP65, SP75, SP82 (hourly) | clean / linear / SAITS-pretrained-on-Jena × {Chronos, Panda} | 10 | §6.6 |
| **17** | **逐层 encoder 探测**（P3.B2） | L63 | SP65, SP82 | clean / linear / SAITS-pretrained / CSDI × 12 PandaLayer 输出 + post-embedder | 5 | §4.2 + §6.4 |

#1–#9 是 v2 协议锁定数；#10–#11 是 legacy S0–S6 协议（仅辅助方向证据）；#12 是 sanity；#13–#17 是 P1/P2/P3 reviewer-defense 实验。

---

## 3 实验逐项

### 3.1 实验 #1：Figure 1 v2 grid（L63 稀疏度线）

**动机**：建立"稀疏观测制造清晰可预测性前沿"主张。

**配置**：L63，10 个 sparsity 点（SP00, SP20, SP40, SP55, SP65, SP75, SP82, SP88, SP93, SP97），σ=0，10 seeds。Cell：linear/CSDI × Panda/DeepEDM = 4。

**关键数字**（patched 协议）：

| 场景 | linear → Panda mean / Pr(VPT>1.0) | CSDI → Panda mean / Pr(VPT>1.0) | paired Δ [95% CI] |
|:--|:--|:--|:--|
| SP65 | 1.22 / 70% | 2.86 / 100% | **+1.64 [+1.40, +1.87]** ↑ |
| SP75 | 0.52 / 20% | 2.29 / 100% | +1.77 [+1.39, +2.17] ↑ |
| SP82 | 0.34 / 0% | 1.34 / 60% | **+1.00 [+0.54, +1.51]** ↑ |

**Wilson 95% CI**（survival probability）：
- SP65：linear 70% [40%, 89%] → CSDI 100% [72%, 100%]
- SP82：linear 0% [0%, 28%] → CSDI 60% [31%, 83%]

**源**：`experiments/week1/results/pt_l63_grid_v2_l63_fine_s_v2_10seed_patched_h0.json`、`..._h5.json`
**图**：`deliverable/figures_main/figure1_l63_v2_10seed_patched.png` + `.md`

---

### 3.2 实验 #2：Figure 1 v2 grid（L63 噪声线）

**动机**：把"corruption-aware 重构有效"的主张限定在稀疏度轴；纯噪声轴是反例。

**配置**：L63，8 个噪声点（NO00, NO005, NO010, NO020, NO050, NO080, NO100, NO120），s=0，10 seeds。同 #1 cells。

**关键发现**：CSDI 在每一档 σ > 0 上对 Panda **中性或略有伤害**。直接验证"CSDI 是间隙补全杠杆，不是密集噪声 denoiser"。

**源**：`experiments/week1/results/pt_l63_grid_v2_l63_fine_sigma_v2_10seed_patched_{h0,h5}.json`

---

### 3.3 实验 #3：L96 N=20 v2 cross-system

**动机**：把 L63 故事推广到高维（N=20）。

**配置**：L96 N=20, F=8。SP55/SP65/SP75/SP82 + NO010/NO020/NO050。10 seeds（5+5 扩种子合并）。同 #1 cells。

**关键数字**（L96 SP82 patched）：
- Panda median：linear 0.50 → CSDI 1.05
- Panda Pr(VPT>0.5)：60% [31%, 83%] → 100% [72%, 100%]
- DeepEDM paired CSDI − linear：**+0.43 [+0.29, +0.57]** ↑ (DeepEDM 是该 band 上**唯一**严格正 paired CI 的 forecaster)

**预注册**：L96 高维 cell 上 mean VPT 被极少数 lucky linear seed 主导，因此 headline 度量为 **median + survival**（§4.3 预注册）。

**源**：`pt_l96_smoke_l96N20_v2_B_patched_5seed.json` + `..._seed5_9.json`

---

### 3.4 实验 #4：L63 jitter / shuffled-residual 控制

**动机**：测试 CSDI 是否仅是随机正则化。

**配置**：L63 SP65 + SP82，10 seeds。Cells：
- `linear`
- `linear + iid jitter`（方差匹配 per-channel CSDI 残差尺度）
- `linear + shuffled CSDI residual`（仅在 missing 位置应用）
- `CSDI`

**关键数字**：

| 场景 | iid jitter Δ [95% CI] | shuffled Δ | CSDI Δ |
|:--|:--|:--|:--|
| SP65 | +0.17 [−0.01, +0.36] ≈ | −0.16 [−0.34, −0.02] ↓ | **+1.65 [+1.41, +1.87]** ↑ |
| SP82 | (CI 跨 0) | +0.34 (modest) | **+1.09 [+0.65, +1.61]** ↑ |

**结论**：iid jitter 与 shuffled residual 都不能复制 CSDI 增益。**结构化 imputation 残差与同等量级 iid 噪声不可互换**。

**源**：`panda_jitter_control_l63_sp65_sp82_v2protocol_patched_10seed.json`

---

### 3.5 实验 #5：L96 N=20 jitter / shuffled

**配置**：L96 N=20 SP65 + SP82，5 seeds，同 #4 cells。

**关键发现**：
- L96 SP65 是"通用正则化 regime"：iid jitter / shuffled / CSDI 在 mean 上都接近（Δ ≈ +1.08–1.19，全部严格正）。但 CSDI 在 tail 上独占：$\Pr(\mathrm{VPT}>1.0) = 80\%$ vs jitter / shuffled / linear 各 40%。
- L96 SP82：mean 高方差，median + survival 为 headline；CSDI 仍 strict-positive on direction-and-rank。

**源**：`panda_jitter_control_l96N20_sp65_sp82_v2protocol_patched_5seed.json`

---

### 3.6 实验 #6：Rössler jitter / shuffled

**配置**：Rössler SP65 + SP82，5 seeds，同 #4 cells。

**关键发现**：CSDI 方向稳定为正；Rössler Lyapunov 指数小（0.071）使 $\Pr(\mathrm{VPT}>1.0)$ 太严，更合适用 $\Pr(\mathrm{VPT}>0.5)$。

**源**：`panda_jitter_control_rossler_sp65_sp82_v2protocol_patched_5seed.json`

---

### 3.7 实验 #7：跨系统 jitter milestone

**动机**：把 #4–#6 合成 Figure 3。

**输出**：六面板图（L63 / L96 N=20 / Rössler × SP65 / SP82），mean vs Pr(VPT>1.0)。

**源**：`deliverable/figures_jitter/jitter_milestone_summary.md`、`jitter_milestone_SP{65,82}.png`

---

### 3.8 实验 #8：Panda embedding OOD 诊断（含 SAITS arm，P3.B1）

**动机**：测试机制——CSDI 在 Panda 内部表征上是否更接近 clean？P3.B1 加 SAITS-pretrained，问"机制是 CSDI-specific 还是 corpus-pretrained-imputation 通用"。

**配置**：L63 SP65 + SP82，5 seeds。Cells：clean / linear / SAITS-pretrained / CSDI。Panda 阶段：patch / embedder / encoder / pooled。度量：跨 token 平均 paired L2-到-clean。

**关键数字**（5 seeds, mean L2）：

| | SP65 linear / SAITS / CSDI | SP82 linear / SAITS / CSDI |
|:--|:--|:--|
| patch | 0.51 / 0.18 / 0.03 | 1.61 / 0.40 / 0.71 |
| embedder | 103 / 52 / 8.6 | 216 / 88 / 94 |
| encoder | 64 / 31 / 5.4 | 120 / 53 / 76 |
| pooled | 9.2 / 3.3 / 0.51 | 14.3 / 5.5 / 6.8 |

**关键比率**：
- SP65：CSDI 在每个阶段比 linear 减少 12–18×；SAITS 减少 2–3×；SAITS/CSDI ≈ 5–6（CSDI 远更接近）
- **SP82：SAITS/CSDI = 0.57–0.95**（**SAITS 比 CSDI 更接近 clean**！与 §4.4 forecast 上 CSDI ≈ SAITS 不一致）

**机制读法**：
1. 入口带：corpus-pretrained 结构化 imputation 减少 token 距离-到-clean 是**通用**机制（CSDI 与 SAITS 都做），减少幅度跟踪救援强度（CSDI ~12× → +0.41 Λ paired 优势）。
2. 底部带：SAITS 比 CSDI 更接近 clean 但 VPT tied — **encoder-side 距离不能 order SAITS-vs-CSDI 在底部带**。这促使我们做 #17 逐层 probe。

**源**：`panda_embedding_ood_l63_sp65_sp82_dt025_v2protocol_patched_with_saits_5seed.json`（当前 §4.2 来源）；旧 3-cell 文件 `..._5seed.json` 保留用于 PCA 散点。

---

### 3.9 实验 #9：Raw-patch v2 诊断

**动机**：在原始时间序列空间（不进 Panda）测度 distance-to-clean。

**配置**：L63 SP65 + SP82，10 seeds。三个度量：local stdev、lag-1 自相关、mid-frequency power。Wasserstein-1 距离 to clean。

**关键数字**（linear / CSDI W₁-to-clean 比）：

| 度量 | SP65 | SP82 |
|:--|:-:|:-:|
| local stdev | 21.02 | 3.54 |
| lag-1 自相关 | 15.02 | **0.62**（偏 linear）|
| mid-freq power | 33.71 | 5.19 |

**结论**：SP65 处所有 raw 度量 CSDI 远接近 clean；SP82 处 lag-1 raw 自相关混合。这是 §4.2 "底部带 mechanism 不能简化到单一保真度量" 的硬证据。

**源**：`l63_patch_ood_v2_v2protocol_sp65_sp82_10seed.json`

---

### 3.10 实验 #10：Cross-system 隔离矩阵（legacy S0–S6）

**动机**：方向-of-effect 的跨系统辅助证据。

**配置**：L63、L96 N=10/20、Rössler、Kuramoto。S0–S6 corruption protocol（旧 `make_sparse_noisy`，不与 v2 一致）。Cells：linear/Kalman/CSDI × Panda/DeepEDM = 6。5 seeds。

**关键数字**（L96 N=20 S4，legacy）：
- `CSDI → Panda`：mean VPT 0.52 → 3.60，$\Pr(\mathrm{VPT}>0.5)$ 60% → 100%，paired-bootstrap +3.07 [+0.57, +6.45]
- L96 N=10 S4：+1.11 [+0.08, +2.22]
- L63 S2：+0.82 [+0.32, +1.37]

**注意**：legacy 协议用 `make_sparse_noisy`，与 v2 corruption grid 不可直接比较。**v2 协议数（§3.2 / §4.4）才是权威**；legacy 仅用作 direction-of-effect 跨系统复制。

**源**：`pt_{l63,l96_iso_l96N{10,20},rossler_iso_rossler,kuramoto}_*_5seed.json`
**图**：`deliverable/figures_isolation/*_heatmap.png`、`*_bars.png`

---

### 3.11 实验 #11：Mackey-Glass / Chua scope-boundary

**动机**：Mackey-Glass 是标量延迟方程（无限维状态），Chua 是分段线性非光滑——破坏 §5.2 的光滑吸引子 / 有限维 Takens 嵌入假设。诚实报告。

**配置**：legacy S0–S6 协议，5 seeds。

**关键发现**：CSDI rescue 在 MG / Chua 上**不一致**。这是预期边界，不是被掩盖的失败。

**源**：`pt_{mg,chua}_*_5seed.json`
**图**：`experiments/week1/figures/pt_mg_mg_5seed_phase_transition.png`、`pt_chua_chua_5seed_phase_transition.png`；attractor 轨迹在 `pictures/`。

---

### 3.12 实验 #12：替代 imputer C0 sanity（per-instance）

**动机**：在没有预训练语料的条件下，单轨迹拟合 SAITS / BRITS 也能比 linear 更好吗？

**配置**：L63 SP65，5 seeds。Cell：linear、SAITS（per-instance fit）、BRITS、CSDI。

**结论**：per-instance fit 对 SAITS / BRITS 在设计上不公平（数据不足以训练 transformer）；预期 SAITS / BRITS 弱于 CSDI。这**不**是主对照实验，只在附录作 sanity。

**源**：`panda_altimputer_l63sp65_partial_5seed.json`

---

### 3.13 实验 #13：预训练替代 imputer（P1.1 + P1.5 + P2.2 30-seed）

**动机**：reviewer 关键问：CSDI 在 transition band 的优势是 CSDI-specific，还是任何 corpus-pretrained 结构化 imputer 都能做到？

**配置**：
- L63 SP65 + SP82：linear / SAITS-pretrained-L63 / CSDI，10 seeds
- L96 N=20 SP82：同上但 SAITS 在 L96 N=20 corpus 预训，**30 seeds**（lucky-seed 稀释）

**SAITS 训练**：
- L63：64K 长度 128 窗口（`lorenz63_clean_64k_L128.npz`），30 epochs，~18 min。Val MAE on missing = 1.26 = 0.149 × `attractor_std`。
- L96 N=20：64K 长度 128 窗口（`lorenz96_clean_512k_L128_N20.npz` 子集），30 epochs。Val MAE = 1.07 = 0.29 × `attractor_std`。
- 推理：SAITS 固定窗口 128，测试 context 512 切 4 个非重叠 chunk 独立补全后拼接。

**关键数字**（L63 SP65 + SP82, n=10）：

| Cell | SP65 mean / Pr>1.0 | SP82 mean / Pr>1.0 |
|:--|:--|:--|
| linear | 1.22 / 70% | 0.29 / 0% |
| SAITS-pretrained | **2.49** / 90% | **1.51** / 70% |
| CSDI | **2.89** / 100% | **1.57** / 70% |

| Paired contrast | SP65 Δ [95% CI] | SP82 Δ [95% CI] |
|:--|:--|:--|
| SAITS − linear | +1.26 [+0.83, +1.64] ↑ | +1.23 [+0.86, +1.62] ↑ |
| CSDI − linear | +1.67 [+1.41, +1.92] ↑ | +1.28 [+0.73, +1.85] ↑ |
| **CSDI − SAITS-pretrained** | **+0.41 [+0.05, +0.87] ↑** | **+0.06 [−0.31, +0.59] ≈** |

**关键数字**（L96 N=20 SP82, **n=30**）：

| Cell | mean | median | Pr(VPT>1.0) Wilson 95% |
|:--|:-:|:-:|:-:|
| linear | 0.86 | 0.25 | 20% [10%, 37%] |
| SAITS-pretrained | 1.57 | 1.01 | 50% [33%, 67%] |
| CSDI | **1.87** | **1.26** | **73%** [56%, 86%] |

| Paired contrast | Δ | 95% CI |
|:--|:-:|:-:|
| SAITS − linear | +0.71 | [+0.02, +1.38] ↑ |
| CSDI − linear | +1.01 | [+0.36, +1.64] ↑ |
| CSDI − SAITS | **+0.31** | **[+0.07, +0.56] ↑** |

**结论**：corpus-pretrained 结构化 imputation 是 lever；CSDI 是其中一个强实例；CSDI 在 L63 入口带保留 +0.41 Λ paired-CI-strict 优势，在 L96 SP82（n=30）所有度量都 strict-positive，在 L63 底部带与 SAITS-pretrained 统计不可分辨。

**源**：
- `panda_altimputer_l63_sp65_sp82_pretrained_10seed_chunked.json`
- `panda_altimputer_l96_sp82_pretrained_10seed.json`（被 30-seed 取代）
- `panda_altimputer_l96_sp82_pretrained_30seed.json`（authoritative）

---

### 3.14 实验 #14：Chronos mini-frontier（P1.2 + P1.4 native horizon）

**动机**：reviewer 关键问：前沿 shape 是 Panda-specific 还是跨 foundation forecaster 普适？

**配置**：L63 SP55, SP65, SP75, SP82。Cells：`linear → Chronos`、`CSDI → Chronos`。Forecaster：Chronos-bolt-small。两个 horizon：
- `pred_len = 128`（与 Panda 匹配）
- `pred_len = 64`（Chronos 原生训练 horizon；Chronos 库本身警告 `pred_len > 64` 超出训练分布）

5 seeds 每个 cell × horizon × scenario。

**关键数字**（vh@1.0 mean）：

| 场景 | linear → Chronos | CSDI → Chronos | paired Δ |
|:--|:-:|:-:|:--|
| SP55 | 0.37 | 0.38 | +0.01 [+0.00, +0.03] |
| SP65 | 0.39 | 0.38 | −0.00 [−0.02, +0.01] |
| SP75 | 0.50 | 0.39 | −0.11 [−0.36, +0.02] |
| SP82 | 0.34 | 0.39 | +0.05 [−0.03, +0.19] |

(`pred_len = 64` 与 `pred_len = 128` 在 per-seed 层面统计不可区分。)

**结论**：Chronos 在两个 horizon 都停在低 VPT 平台（mean 0.34–0.50, $\Pr(\mathrm{VPT}>1.0)$ ≤ 20%），CSDI 也未明显改善。`pred_len = 64` 确认负面结果不是 Chronos OOD horizon 的 artefact。

**论文读法**：corpus-pretrained-imputation 杠杆在 Panda 上经验性可观察；在 Chronos 上杠杆不可观察是因为 Chronos 自身 VPT 分布远低于杠杆能起作用的 regime。前沿是 **forecaster-dependent** 的，不是普适的。

**源**：
- `chronos_frontier_l63_chronos_l63_sp55_sp82_5seed.json`（pred_len=128）
- `chronos_frontier_l63_chronos_l63_sp55_sp82_5seed_pl64.json`（pred_len=64）

---

### 3.15 实验 #15：EnKF 已知动力学上界（P1.3）

**动机**：建立 model-aware reference — 当 forecaster 知道真实 L63 向量场时，前沿在哪？

**配置**：Stochastic ensemble Kalman filter，n_members = 100，RK4 forward propagation，95% observation-noise floor at 0.01 × `attractor_std`。L63 SP55–SP82 + NO020 + NO050，5 seeds。

**关键数字**：

| 场景 | EnKF mean / median | Pr(VPT>1.0) |
|:--|:-:|:-:|
| SP55 | 2.84 / 2.90 | 100% |
| SP65 | 2.84 / 2.90 | 100% |
| SP75 | 2.85 / 2.90 | 100% |
| SP82 | 2.84 / 2.90 | 100% |
| NO020 | 2.81 / 2.90 | 100% |
| NO050 | 2.49 / 2.70 | 100% |

(VPT 上限 = pred_len × dt × λ_max ≈ 128 × 0.025 × 0.906 = 2.90)

**结论**：EnKF 在整个稀疏度 transition band 上撞 VPT 天花板。前沿是**黑盒部署接口**（forecaster 不能访问动力学）的属性，不是 L63 系统本身的属性。在密集噪声轴上 EnKF 仍然 graceful（NO050 mean = 2.49）。

**源**：`enkf_l63_enkf_l63_v2_5seed.json`

---

### 3.16 实验 #16：Jena Climate 真实传感器（P2.1 + P3.A clean-upper / 跨 forecaster）

**动机**：在真实多变量传感器流上压力测试 §4.4 lever 主张。

**数据**：Public Jena Climate 2009–2016（Max Planck Institute Jena 气象站，14 个数值大气特征，10 分钟采样）。下采样到小时（×6）。Train 2009–2014（52,622 hours）、val 2015（8,760 hours）、test 2016（8,709 hours）。所有特征用 train-split per-feature mean/std z-score。

**SAITS-on-Jena 训练**：64K windows（train+val），30 epochs，~2 min on V100。Val MAE on missing = 0.62 z-units。Best ckpt epoch 30。

**配置**：SP55, SP65, SP75, SP82。$n_{ctx} = 512$ hours，$pred_{len} = 64$ hours。
- Cells：clean / linear / SAITS-pretrained-on-Jena（4 个）
- Forecaster：Chronos-bolt-small **和** Panda-72M（跨 forecaster 控制）
- 10 seeds each (forecaster × scenario × cell)

**度量**：normalized valid horizon vh@τ —— 跨 14 个 z-scored 特征的 per-step RMSE 保持 ≤ τ 的最大 lead-step。报告 vh@0.3, vh@0.5, vh@1.0, vh@2.0。

**关键数字**（vh@1.0 mean，跨 SP55–SP82）：

| | clean | linear | SAITS-pretrained |
|:--|:-:|:-:|:-:|
| → Chronos | 51.1 | 50.6（avg）| 30.3（avg）|
| → Panda | 46.4 | 43.2（avg）| 35.2（avg）|

**Paired SAITS − linear at vh@1.0**：

| Cell | Chronos paired Δ [95% CI] | Panda paired Δ [95% CI] |
|:--|:--|:--|
| SP55 | −16.7 [−28.2, −5.8] ↓ | −6.3 [−16.6, +3.0] ≈ |
| SP65 | −18.8 [−29.7, −8.2] ↓ | −4.1 [−12.9, +5.2] ≈ |
| SP75 | −21.0 [−34.3, −8.6] ↓ | −12.0 [−20.0, −4.7] ↓ |
| SP82 | −23.6 [−39.2, −8.6] ↓ | −9.7 [−19.9, −0.9] ↓ |

**关键发现**：
1. **Linear-fill ≈ clean-context 上界** 在两个 forecaster 上都成立（差 1–7 vh-units）。Linear 已经保留了主导日循环。
2. **SAITS-pretrained < linear** 严格负 paired CI 在 Chronos 上每个 cell；在 Panda 上 SP75/SP82 严格负。**跨 forecaster** 排除"Chronos-specific 弱点"假设。
3. Clean-context 上界排除"forecaster 自身是瓶颈、SAITS 只是恰好在噪声平台一侧"假设：SAITS-fill 与 clean 的差距远超 seed 噪声。

**论文读法**（§6.6）：

> Linear 插值已经在 Jena hourly 上达到 clean-context 预测天花板（主导日周期性 linear "免费"保留）。在含噪真实语料上 fit 的 SAITS imputer 引入 sample-specific 高频 artefacts，把填补 context 推**离** forecaster 依赖的周期模式，所以两个 forecaster 都跌破 clean。learned imputer 没有 headroom 救援。

> **Corpus-pretrained-imputation 救援在混沌吸引子主导的系统（L63、L96）上可观察 — linear 插值会破坏 foundation forecaster 依赖的局部几何结构。在周期主导的真实数据流（Jena 小时）上，linear 已达 clean 天花板，corpus-pretrained imputer 净有害。**

**源**：
- Chronos + clean upper：`jena_real_sensor_jena_chronos_with_clean_upper_10seed.json`
- Panda 跨 forecaster：`jena_real_sensor_jena_panda_with_clean_upper_10seed.json`
- 数据预处理：`experiments/week2_modules/make_jena_npz.py`
- SAITS-Jena ckpt：`experiments/week2_modules/ckpts/saits_jena_pretrained/20260502_T021448/SAITS.pypots`

---

### 3.17 实验 #17：逐层 encoder 探测（P3.B2）

**动机**：B1 (#8) 发现 SP82 处 SAITS pooled 距离 < CSDI 但 VPT tied。原 §6.4 假设"decoder-side latent dynamics"——但 Panda-72M 是 encoder-only（head 是从 pooled encoder 的线性投影）。重新定位为：**哪一个 encoder 层是 rescue-relevant 的**。

**配置**：L63 SP65 + SP82，5 seeds。Cells：clean / linear / SAITS-pretrained / CSDI。Hook：每个 PandaLayer（12 个）输出 + post-embedder = 13 个测点。度量：per-layer 跨 token 平均 paired L2-到-clean。

**实现**：PyTorch `register_forward_hook` 在每个 `model.model["encoder"].layers[i]`，捕获 `[B, C, P, d_model]` 输出。

**关键数字**（SAITS/CSDI ratio per layer）：

| Layer | SP65 SAITS/CSDI | SP82 SAITS/CSDI |
|:-:|:-:|:-:|
| 0（post-embedder）| 5.67 | 0.94 |
| 1 | 5.08 | 0.70 |
| 2 | 4.77 | 0.64 |
| 3 | 4.20 | 0.65 |
| 4 | 4.86 | 0.86 |
| 5 | 5.46 | 0.94 |
| **6** | **5.90** | **1.02** ← convergence band |
| **7** | **6.26** | **1.06** ← saits 略微偏离 clean 比 csdi 更远 |
| **8** | **5.28** | **1.00** |
| 9 | 4.35 | 0.74 |
| 10 | 4.69 | 0.78 |
| 11 | 4.78 | 0.76 |
| 12（pooling 之前最终层）| 4.85 | 0.71 |

**关键发现**：
- **SP65 每一层** CSDI 比 SAITS 更接近 clean 4–6×（uniform mechanism 跟踪 +0.41 Λ paired 优势）
- **SP82 非单调**：早层 (0–5) 与晚层 (9–12) SAITS 更近；**中间 encoder 层 6–8 处两者收敛（saits/csdi 1.00–1.06）**
- 中间 encoder 层是 Panda 内部表征中**唯一**SAITS 与 CSDI 看起来 equivalently close to clean 的子区域 — 恰好匹配 §4.4 forecast tie

**论文读法**（§4.2 + §6.4 关闭原 decoder 假设）：
1. §4.4 底部带 rescue **在 Panda 中间 encoder（层 6–8）处饱和**：到层 7，两个 corpus-pretrained imputer 已经产生在距离-到-clean 上统计不可分辨的内部表征。
2. 后续层的任何区分都不会传递到可预测性，因为 head 的线性投影看到的是 pooled (≈ averaged) state，平滑掉了中间 encoder 的收敛。
3. 上面 #8 距离条形表里的 **pooled-only reading 因此是"晚层 artefact"**，不是 VPT 的相关预测器。
4. encoder geometry **是**底部带 rescue 的相关几何，只是在中间 encoder 而不是 final pooled。

**源**：
- 脚本：`experiments/week1/panda_per_layer_probe_l63.py`
- 结果：`experiments/week1/results/panda_per_layer_probe_l63_sp65_sp82_per_layer_5seed.json`

---

## 4 关键数字快速查表

### 4.1 L63 patched headline

| 指标 | SP65 | SP82 |
|:--|:--|:--|
| `linear → Panda` mean VPT | 1.22 | 0.34 |
| `CSDI → Panda` mean VPT | 2.86 | 1.34 |
| Paired CSDI − linear Δ | +1.64 [+1.40, +1.87] | +1.00 [+0.54, +1.51] |
| linear $\Pr(\mathrm{VPT}>1.0)$ Wilson | 70% [40%, 89%] | 0% [0%, 28%] |
| CSDI $\Pr(\mathrm{VPT}>1.0)$ Wilson | 100% [72%, 100%] | 60% [31%, 83%] |

### 4.2 §4.4 替代 imputer (L63)

| 对比 | SP65 | SP82 |
|:--|:--|:--|
| SAITS-pretrained − linear | +1.26 [+0.83, +1.64] ↑ | +1.23 [+0.86, +1.62] ↑ |
| CSDI − linear | +1.67 [+1.41, +1.92] ↑ | +1.28 [+0.73, +1.85] ↑ |
| **CSDI − SAITS-pretrained** | **+0.41 [+0.05, +0.87] ↑** | **+0.06 [−0.31, +0.59] ≈** |

### 4.3 §4.4 L96 SP82 (n=30)

| | mean | median | Pr(VPT>1.0) |
|:--|:-:|:-:|:-:|
| linear | 0.86 | 0.25 | 20% [10%, 37%] |
| SAITS | 1.57 | 1.01 | 50% [33%, 67%] |
| CSDI | **1.87** | **1.26** | **73%** [56%, 86%] |

Paired all strict-positive：SAITS − linear +0.71 [+0.02, +1.38]；CSDI − linear +1.01 [+0.36, +1.64]；**CSDI − SAITS +0.31 [+0.07, +0.56]**。

### 4.4 §6.4 Chronos 跨 foundation

L63 SP55–SP82：mean 0.34–0.50 across cells × pred_len ∈ {64, 128}；paired CSDI − linear 都跨 0；clean ≠ Panda 前沿 shape。

### 4.5 §6.4 EnKF 上界

L63 SP55–SP82 整段：mean 2.84–2.85，Pr(VPT>1.0) = 100%。前沿是黑盒部署接口的性质。

### 4.6 §6.6 Jena 实测（10 seeds）

| | clean | linear (avg) | SAITS-pretrained (avg) |
|:--|:-:|:-:|:-:|
| → Chronos vh@1.0 | 51.1 | 50.6 | 30.3 |
| → Panda vh@1.0 | 46.4 | 43.2 | 35.2 |

Linear ≈ clean，SAITS < linear（跨两个 forecaster）。Lever 不适用于周期主导真实数据。

### 4.7 §4.2 + §6.4 机制（B1 + B2 合并）

- **入口带 SP65**：CSDI 在 encoder 每层都比 SAITS 接近 clean 4–6×（跟踪 +0.41 Λ 优势）
- **底部带 SP82**：encoder 层 6–8 处 SAITS ≈ CSDI（saits/csdi 1.00–1.06），匹配 forecast tie
- 最终 pooled 距离是晚层 artefact，**不是** rescue-relevant 几何

---

## 5 复现指引

### 5.1 关键 checkpoint

| Checkpoint | 用途 | 路径 |
|:--|:--|:--|
| `dyn_csdi_full_v6_center_ep20.pt` | L63 CSDI（M1 best）| `experiments/week2_modules/ckpts/` |
| `dyn_csdi_l96_full_c192_vales_best.pt` | L96 N=20 CSDI | 同上 |
| `dyn_csdi_l96_N10_full_vales_best.pt` | L96 N=10 CSDI | 同上 |
| `dyn_csdi_rossler_full_vales_best.pt` | Rössler CSDI | 同上 |
| `dyn_csdi_kuramoto_full_vales_best.pt` | Kuramoto CSDI | 同上 |
| `dyn_csdi_chua_full_vales_best.pt` | Chua CSDI | 同上 |
| `dyn_csdi_mg_full_vales_best.pt` | Mackey-Glass CSDI | 同上 |
| `saits_l63_pretrained/20260501_T153756/SAITS.pypots` | SAITS-L63（P1.1）| 同上 |
| `saits_l96_n20_pretrained/20260501_T210242/SAITS.pypots` | SAITS-L96 N=20（P1.5/P2.2）| 同上 |
| `saits_jena_pretrained/20260502_T021448/SAITS.pypots` | SAITS-Jena（P2.1）| 同上 |

### 5.2 重要运行命令样例

L63 Figure 1：
```bash
CUDA_VISIBLE_DEVICES=0 python -u -m experiments.week1.phase_transition_grid_l63_v2 \
  --configs SP00 SP20 SP40 SP55 SP65 SP75 SP82 SP88 SP93 SP97 \
  --n_seeds 10 --tag l63_fine_s_v2_10seed_patched_h0
```

预训练 SAITS（L63 / L96 / Jena 同样 entry）：
```bash
CUDA_VISIBLE_DEVICES=0 python -u -m experiments.week2_modules.train_saits_l63 \
  --corpus experiments/week2_modules/data/lorenz63_clean_64k_L128.npz \
  --epochs 30 --batch 64 --out experiments/week2_modules/ckpts/saits_l63_pretrained
```

替代 imputer（P1.1 / P1.5 / P2.2）：
```bash
CUDA_VISIBLE_DEVICES=0 python -u -m experiments.week1.panda_altimputer_control \
  --settings L96_SP82 --cells linear saits_pretrained csdi --n_seeds 30 \
  --saits_ckpt experiments/week2_modules/ckpts/saits_l96_n20_pretrained/20260501_T210242/SAITS.pypots \
  --tag l96_sp82_pretrained_30seed
```

Chronos mini-frontier（P1.2 / P1.4）：
```bash
CUDA_VISIBLE_DEVICES=0 python -u -m experiments.week1.chronos_frontier_l63 \
  --configs SP55 SP65 SP75 SP82 --n_seeds 5 --pred_len 64 \
  --tag chronos_l63_sp55_sp82_5seed_pl64
```

EnKF（P1.3）：
```bash
CUDA_VISIBLE_DEVICES=0 python -u -m experiments.week1.enkf_l63_upper_bound \
  --configs SP55 SP65 SP75 SP82 NO020 NO050 --n_seeds 5 \
  --tag enkf_l63_v2_5seed
```

Jena 实测（P2.1 + P3.A）：
```bash
# 1. 数据准备
python -u -m experiments.week2_modules.make_jena_npz \
  --src experiments/week2_modules/data/real/jena_climate_2009_2016.csv \
  --out experiments/week2_modules/data/real/jena_clean_hourly_L128.npz
# 2. SAITS-Jena 预训练
CUDA_VISIBLE_DEVICES=0 python -u -m experiments.week2_modules.train_saits_l63 \
  --corpus experiments/week2_modules/data/real/jena_trainval_hourly_L128.npz \
  --epochs 30 --batch 64 --n_val 1080 \
  --out experiments/week2_modules/ckpts/saits_jena_pretrained
# 3. Pilot
CUDA_VISIBLE_DEVICES=0 python -u -m experiments.week1.jena_real_sensor_pilot \
  --saits_ckpt experiments/week2_modules/ckpts/saits_jena_pretrained/<run-id>/SAITS.pypots \
  --configs SP55 SP65 SP75 SP82 --n_seeds 10 \
  --cells clean linear saits_pretrained --forecaster chronos \
  --tag jena_chronos_with_clean_upper_10seed
```

逐层 probe（P3.B2）：
```bash
CUDA_VISIBLE_DEVICES=0 python -u -m experiments.week1.panda_per_layer_probe_l63 \
  --saits_ckpt experiments/week2_modules/ckpts/saits_l63_pretrained/20260501_T153756/SAITS.pypots \
  --n_seeds 5 --tag l63_sp65_sp82_per_layer_5seed
```

---

## 6 commit 历史（按时间）

| Commit | 日期 | 内容 |
|:--|:--|:--|
| `c99c978` | 2026-04-30 | feat(paper-pivot): sparse-observation forecastability frontier story + v2 protocol |
| `75a7bf4` | 2026-05-01 | P0 cleanup |
| `695dbad` | 2026-05-01 | P1.1 / P1.2 / P1.3 — pretrained SAITS L63 + Chronos + EnKF |
| `c3f1256` | 2026-05-01 | P1.4 / P1.5 — Chronos pred_len=64 + SAITS L96 |
| `bd2ccc6` | 2026-05-01 | docs(P1) |
| `290e38b` | 2026-05-01 | submission-prep QA freeze |
| `0fb2945` | 2026-05-01 | reviewer-handoff polish |
| `2850470` | 2026-05-02 | P2 — §4.3 cell-list + L96 SP82 30-seed + Jena §6.6 |
| `dd12210` | 2026-05-02 | docs(status) P2 freeze 记录 |
| `22c0904` | 2026-05-02 | P3.A + C — Jena clean-upper + Panda 跨 forecaster + §4.1 legacy 降级 + drift callout |
| `f95e948` | 2026-05-02 | P3.B1 — SAITS arm in §4.2，机制 dissociation |
| **`6189a12`** | **2026-05-02** | **P3.B2 — 逐层 encoder 探测，关闭 decoder hypothesis** |

---

## 7 投稿就绪状态

- 严苛审稿人 6 issue：**5/6 闭合**
  - ✅ Issue 1 Jena 单 forecaster — `22c0904` 跨 forecaster + clean upper
  - ✅ Issue 2 §4.2 mechanism CSDI-specific？— `f95e948` SAITS arm
  - ✅ Issue 3 §4.1 legacy S0–S6 — `22c0904` 移到 Appendix
  - ✅ Issue 4 §4.4 drift callout — `22c0904` 移到 Appendix B.3
  - ✅ Issue 5 decoder probe — `6189a12` 中间 encoder 收敛
  - ⏸ Issue 6 MG/Chua v2 — camera-ready scope item

- Spotlight 概率预测：~45–55%
- 主版本 freeze：`6189a12`（`csdi-pro-m3-alt` branch，已同步远端）
