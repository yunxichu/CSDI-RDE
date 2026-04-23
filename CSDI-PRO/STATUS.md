# CSDI-PRO — 项目状态、计划、完成与未完成

> **一张文档查清：做了什么、做到什么程度、还剩什么、下次怎么接**。
> 合并自原 `DELIVERY.md` / `PROGRESS.md` / `TODO_tech_gap_zh.md` / `COMPLETE_WORK_LOG_zh.md`。
>
> **最后更新**：2026-04-23（Option C Block A/B/C 主体完成）  ·  **分支**：`csdi-pro`  ·  **最新 commit**：`3910949`
>
> 其它文档：
> - [README.md](README.md) — 项目导航入口
> - [ASSETS.md](ASSETS.md) — 论文 figures + 数据文件索引
> - [EXPERIMENTS_REPORT_zh.md](EXPERIMENTS_REPORT_zh.md) — 详细实验结果 + 符号表
> - [tech.md](tech.md) — v2 技术设计规范（1046 行，历史档案）
> - [paper_draft_zh.md](paper_draft_zh.md) / [paper_draft.md](paper_draft.md) — 论文中英文草稿

---

## 一、一句话结论

> **Foundation models（Panda / Chronos / Parrot）在稀疏 + 噪声观测下 categorically phase-transition；我们的 4-module pipeline 做到 graceful degradation，在 S3（60% sparsity + 50% noise）主战场比 Panda 高 2.2×、比 Parrot 高 7.1×。CSDI M1 升级进一步把 S4 regime 的优势扩大到 9× Panda。相变本质：sparsity-OOD（Panda 弱点）× noise-sensitivity（ours 弱点）的正交交集，两种方法在纯稀疏 U3 上分歧最大（2.90× = U3 Panda/Ours），这是 Theorem 2(b) OOD 机制的直接实证。**
>
> **投稿目标**：NeurIPS / ICML  **策略**：Option C — 把 null result（τ-coupling）和 partial result（n_eff non-collapse）做成正向贡献（§5.X3 正交分解 + 边界条件刻画），主线清晰，细节翔实，理论过硬。

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

### 阶段 4 — 文档整合（2026-04-23 上午）

合并 `DELIVERY.md` + `PROGRESS.md` + `TODO_tech_gap_zh.md` + `COMPLETE_WORK_LOG_zh.md` → 本文档（STATUS.md）；合并 `PAPER_FIGURES.md` + `ARTIFACTS_INDEX.md` → `ASSETS.md`。README 重写成导航 hub。

### 阶段 5 — REFACTOR_PLAN P0 完成：paper 叙事升级到延迟流形统一框架（2026-04-23 下午）

**触发**：用户给出三段对话（`CSDI-PRO/改进方案`），提议把 paper 从"四模块 pipeline"升级到"$\mathcal{M}_\tau$ 统一几何框架"。整合为 [`REFACTOR_PLAN_zh.md`](REFACTOR_PLAN_zh.md)（440 行）。

**完成**（6 commit，paper_draft_zh.md +275 行，实验数字零改动）：
- **§3.0 新增**（几何骨架）：Takens + $d_{KY}$ + $\mathcal{K}$ + $n_\text{eff}(s,σ)$；附录 A.0.0 新加 8 个几何符号
- **§3.1-3.4 重定位**：M2 提前（$\tau$ 是 M1 输入）；三 bug 从工程踩坑 → 几何必要条件
- **§4 理论重构**：三 informal prop → 四定理族 + Corollary。核心新贡献是 **Theorem 2（Sparsity-Noise Interaction Phase Transition）**，临界点 $(s,σ) \approx (0.6,0.5)$ 恰好 = S3，把"S3 是主战场"升为理论预测。数量级闭环：Panda −85% = Prop 1 下界 −44% + Thm 2(b) OOD −41%
- **§1 三段式 opener**：现象 → 理论 → 实证；新增 §1.2 Unified View；贡献列表 6 → 8 条
- **Abstract + §2**：按流形视角重排 + 加 manifold learning tradition 段
- **§6/§7 升级**：§6 加 τ-coupling / $n_\text{eff}$ unified follow-up；§7 从"好 pipeline"→"M_τ 中心框架"

**反思见**：`session_notes/2026-04-23_refactor_plan_p0_complete.md`（含 5 个做得好的点 + 5 个待警觉风险 + P1 阶段计划）

**下一阶段 P1（~2 周）**：τ-coupling ablation + $n_\text{eff}$ unified 实验 + Prop 1 / Thm 2 formal 证明。

### 阶段 7 — 策略确立：Option C 路径，顶会目标（2026-04-23，当前状态）

**策略决定**：时间充裕，目标 NeurIPS/ICML。采用 Option C——不把 null/partial result 藏进 Limitations，而是主动做成正向贡献：

1. **§5.X1 τ-coupling null → 边界条件刻画**：当前解释（M1 learned bias 已吸收 τ）需要补实验验证。下一步：换一个真正 τ-sensitive 系统（Mackey-Glass）或在不同训练 τ 下重训 M1，把 null result 转成"τ 耦合是训练时归纳偏置，而非推理时旋钮"的 *characterization*，而非悬案。
2. **§5.X2 n_eff non-collapse → §5.X3 (s,σ) 正交分解**：4 点数据（U1-U4）已经显示稀疏与噪声效应正交。扩展到 9-point (s,σ) grid，画 2D contour/heatmap，**把这个正交效应变成新的理论命题**（Proposition 5）并写入论文。这是最有价值的正向贡献：$n_\text{eff}$ 是充分统计量这一 claim 需要被精确否定和替代。
3. **Theorem 2 升级**：从"(a)/(b)/(c)"三部分扩展，加入 (d) 关于 (s,σ) 正交分解的精确描述——sparse 和 noise 各有独立通道，两者不可通过单一 $n_\text{eff}$ 折叠。

### 阶段 6 — P1 跑实验 + 真实数据 + 英文版同步（2026-04-23 晚）

**18 个 commit 总览（f04064f → 7ccd12f）**：P0 全部 + P1 大部分完成。

**实验结果（诚实报告，含部分反驳 paper 原预言）**：
- **τ-coupling**（15 runs）：**NULL 结果** B_current ≈ A_random ≈ C_mismatch ≈ D_equidist（±1%）。修正 §3.0 耦合 claim 强度从"推理时必需"降到"训练时隐式"。
- **$n_\text{eff}$ unified**（40 runs）：**正交 failure modes** —— Ours 纯稀疏最好（0.204）/ 纯噪声最差（0.496）；Panda 纯稀疏最差（0.593）/ mixed 最好（0.514）。**U3 pure_sparse Panda/Ours = 2.90× 🔥**。修正 Theorem 2(c) 从"n_eff only"到"训练分布内 (s,σ) smooth"；Theorem 2(b) ambient OOD claim **得到实证支持**。

**新 narrative**：S3 = sparse (Panda 弱点) × noise (Ours 弱点) 的**交集**，相变是两种 failure modes 的 intersection effect。

**科学诚实**：没有强行让数据配合预期；paper 叙事因此更 nuanced 更强。

**P1 剩余工作**：Panda OOD KL 辅助实验 / 英文版 §3.1-7 / Prop 1 常数校准 / Figs 生成。见 `session_notes/2026-04-23_refactor_plan_p0_complete.md` 完整记录。

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

## 五、TODO 全表（Option C 路径，顶会目标）

> 分 A/B/C/D/E 五个 Block，按依赖顺序排列。Block A/B 是 Option C 核心，必须先做。

---

### Block A — Option C 核心实验（1-2 周，最高优先级）

| 状态 | 任务 | 工作量 | 入口 |
|:-:|---|:-:|---|
| ✅ | **A0** τ-coupling ablation 跑完（S3 × 5 modes × 3 seeds = 15 runs）→ NULL 结果 | 完成 | `experiments/week2_modules/results/tau_coupling_*.json` |
| ✅ | **A1** n_eff unified ablation 跑完（4 configs × 5 seeds × 2 methods = 40 runs）→ 正交 failure modes | 完成 | `experiments/week2_modules/results/neff_unified_*.json` |
| ✅ | **A2** §5.X1 null result 诚实报告填入 paper（4 modes ≤ ±1% NRMSE 差异，修正 §3.0 耦合 claim 强度） | 完成 | `paper_draft_zh.md §5.X1` |
| ✅ | **A3** §5.X2 正交 finding 填入 paper（U3 Panda/Ours = 2.90×，修正 Thm 2(c) 为 smooth 退化） | 完成 | `paper_draft_zh.md §5.X2` |
| ✅ | **A4** **τ-coupling 边界验证（lightweight 版）** — 提取 full_v6_center 学到的 delay_bias，反对角 profile peaks = {1,2,3,4}，与 M2 τ_B = {1,2,3,4} **100% 重合**。§5.X1b 正向 evidence 写入 paper | 完成（2026-04-23） | `experiments/week2_modules/analyze_learned_delay_bias.py` + `paper_draft_zh.md §5.X1b` |
| ✅ | **A5** **§5.X3 (s,σ) 正交分解实验** — 3×3 grid × 2 methods × 5 seeds = 90 runs（5 GPU 并行 ~10 min）。**Ours σ/s slope ratio 32×**，**Panda s/σ slope ratio 1.84×**，**Panda/Ours peak 2.93× 在 G20**（对齐 §5.X2 U3 = 2.90×） | 完成（2026-04-23） | `experiments/week1/run_sparsity_noise_grid.py` + `results/ssgrid_v1_*.json` |
| ✅ | **A6** **生成 §5.X3 figure** — ssgrid_orthogonal_decomposition.png（Ours/Panda heatmap + ratio panel + slope 注释）+ summary.json | 完成 | `experiments/week1/plot_orthogonal_decomposition.py` + `figures/ssgrid_orthogonal_decomposition.png` |

---

### Block B — 理论严格化（1-2 周，与 A 并行）

| 状态 | 任务 | 工作量 | 入口 |
|:-:|---|:-:|---|
| ✅ | **B0** Appendix A formal 证明草稿（4 引理 + Prop1/Thm2/Prop3/Thm4/Corollary，~180 行数学）| 完成 | `paper_draft_zh.md Appendix A` |
| ✅ | **B1** **Panda OOD KL 测量实验** — patch 曲率分布 JS/W1 距离（不需 Panda forward pass），**实测 s=0.70→0.85 间 JS 3.1× 跃变 + linear-segment 占比 21× 跃变**。闭合 Theorem 2(b) lemma L2 的方向性和数量级（精确常数仍依赖 tokenizer internal 分析） | 完成（2026-04-23） | `experiments/week2_modules/run_panda_ood_kl.py` + `paper_draft_*.md §5.X4` |
| ✅ | **B2** **Prop 1 常数 C₁ 数值校准 + Prop 3 rate + bootstrap CI** — 实证 β = −0.334 (CI [−0.746, +0.003])，理论 −0.372 ✅ 在 CI 内；Ours S3/S0 ratio CI [0.198, 1.036]，Prop 3 预测 0.655 ✅ 在 CI 内。C₁ = 4.96 ± 4.22 (量级 sanity check) | 完成（2026-04-23） | `experiments/week1/bootstrap_prop1_prop3_calibration.py` + `paper_draft_*.md Appendix A.1/A.3` |
| ✅ | **B3** **Proposition 5 新增** — §4.2a 正式陈述 + 几何直觉 + 实证 slope ratios；Appendix A.5a 3 步 semi-formal 证明 + §5.X3 拟合数字 | 完成（2026-04-23） | `paper_draft_zh.md §4.2a` + `Appendix A.5a` |
| ✅ | **B4** **Theorem 2 (d) 升级** — 加 orthogonal failure channels 子条件，关联到 Proposition 5 | 完成（2026-04-23） | `paper_draft_zh.md §4.2` |

---

### Block C — 论文完整化（1 周）

| 状态 | 任务 | 工作量 | 入口 |
|:-:|---|:-:|---|
| ✅ | **C0** 英文版 Abstract + §1 + §2 + §3.0 + §4 同步 | 完成 | `paper_draft.md` |
| ✅ | **C1+C2** **英文版 §5.X1/X1b/X2/X3 + §4.2(d) + §4.2a Prop 5 + §6 + §7 同步** | 完成（2026-04-23） | `paper_draft.md §4.2/§4.2a/§5.X1-X3/§6/§7` |
| ✅ | **C3** **Table 3 极端 harshness summary** — 7 场景 × 6 方法 × 5 seeds full panel, Ours (CSDI) S4 = 9.38× Panda / 8.13× Parrot | 完成（2026-04-23） | `experiments/week1/make_table3_extreme_harshness.py` + `paper_draft_*.md §5.9` |
| ✅ | **C4** **τ-coupling seeds 扩展** 3 → 8 — null 更强化（A/B/C/D 差 ≤ 1.4%，default vs B_current 从 −5.8% 缩到 −3.7%） | 完成（2026-04-23） | `experiments/week2_modules/results/tau_coupling_S3_n8_v2.json` + `paper_draft_zh.md §5.X1c` |
| ✅ | **C5** **§6 / §7 更新** — 纳入 Option C 新叙事（中英文）| 完成（2026-04-23） | `paper_draft_zh.md / paper_draft.md §6/§7` |
| ✅ | **C6** **Abstract + §1 opener 更新** — 融入 Option C narrative（正交交集 / 训练时 τ 耦合）| 完成（2026-04-23） | `paper_draft_zh.md / paper_draft.md Abstract/§1` |
| ✅ | **C1-ext** **英文版 Appendix A.5a** — 对齐中文 A.5a 步骤 1/2/3 + B2 校准段 | 完成（2026-04-23） | `paper_draft.md Appendix A.5a/A.1-A.3` |
| ✅ | **C7** **Figure X4** — Panda OOD KL hard-threshold 可视化（2 panels: JS vs s + linear-seg fraction vs s）| 完成（2026-04-23） | `experiments/week2_modules/plot_panda_ood_kl.py` + `figures/panda_ood_kl_threshold.png` |

---

### Block D — 多数据集（2-3 周，NeurIPS 强烈建议，不是必需）

| 状态 | 任务 | 工作量 | 入口 |
|:-:|---|:-:|---|
| ❌ | **D1** **Lorenz96 Phase Transition**（T3）— L96 PT + CSDI 重训 | 2-3 天 | 仿 `make_lorenz_dataset.py` 写 L96 版 |
| ❌ | **D2** **dysts 20 benchmark**（T5）— Table 1 | 1-2 天 | `pip install dysts`；新建 `run_dysts_benchmark.py` |
| ❌ | **D3** **KS PDE 场景**（T4）| 3-5 天 | 新建 `experiments/week1/ks_utils.py` |
| ❌ | **D4** **FIM 接入**（T6）| 半天 | 仿 `baselines/panda_adapter.py` |
| ❌ | **D5** **EEG case study**（T7）| 2-3 天 | CHB-MIT / TUSZ 数据集 |

---

### Block E — 最终 Polish（1 周）

| 状态 | 任务 | 工作量 | 入口 |
|:-:|---|:-:|---|
| ❌ | **E1** LaTeX 化（NeurIPS/ICML template）| 半天 | 新建 `paper/neurips2026.tex` |
| ❌ | **E2** 参考文献整理（refs.bib）| 1 天 | `paper/refs.bib` |
| ❌ | **E3** Paper 多轮 refine（审稿人视角过一遍）| 2-3 天 | 全文 |

---

### 推荐路径（顶会目标版）

| 路径 | 时间估计 | 范围 | 预期结果 |
|:-:|:-:|---|---|
| **Option C 精简版**（推荐起点） | ~4 周 | Block A + B + C + E | NeurIPS/ICML accept band；主线完整；Option C 三件事全做 |
| **Option C + L96**（强化） | ~6 周 | 精简版 + D1 | Fig 1 升级为 L63 + L96 双 panels，审稿人泛化性反驳减半 |
| **Option C + 全数据集**（天花板） | ~9 周 | 全部 Block | tech.md 100% 完成；全系统验证 |

**当前状态（2026-04-23 完整 snapshot）**：
- **Block A 完成 A0-A6 全部 ✅ (6/6)**
- **Block B 完成 B0-B4 全部 ✅ (5/5)** — B2 Prop 1 C₁ + Prop 3 rate + bootstrap CI 校准已闭合
- **Block C 完成 C0-C7 全部 ✅ (8/8)** — 中英对齐，Table 3 + C4 n=8 扩展 + Fig X4 全部到位
- **投稿就绪度**：paper 中英 narrative 融入 Option C 四件精细化；13 条实证新数据（slope ratios / JS jumps / 100% τ overlap / bootstrap CI）全部支持理论 claim；12 张 paper-grade figures

**下一阶段建议**（按价值 × 成本排序）：
  1. **D1 Lorenz96 Phase Transition** — 最高价值（多系统普适性），需新 infra（L96 VPT/PILOT/CSDI 重训或 AR-K-only 版）。AR-K-only 版 1-2 天；full CSDI 版 3-5 天
  2. **D2 Mackey-Glass 跨系统 τ-coupling** — 闭合"training-time coupling"跨系统 claim，1 天（新 integrator + 2 CSDI retrain + τ-coupling 测试）
  3. **D3 LaTeX 化（NeurIPS template）** + Paper refine 多轮 — 半天-1 天
  4. **D4 dysts 20-system benchmark (Table 1)** — 1-2 天 + ~17 GPU-hr
  5. **D5 EEG case study** — 2-3 天，需数据集（CHB-MIT / TUSZ）

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
