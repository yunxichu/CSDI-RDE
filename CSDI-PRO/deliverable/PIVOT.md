# Strategic Pivot — 2026-04-26

> 阅读顺序：先读 `STORY_LOCK_2026-04-28.md`，再读这份，再回去看 `README.md`。`README.md` 描述当前已有的成果；这份说明这些成果如何**被重新组织**到一个更锋利的论文骨架里。
>
> **2026-04-28 correction**：后续 raw-patch 和 Panda representation diagnostics 已证伪"linear-fill 比 CSDI 更远离 clean token distribution"这个简单机制。最新锁定主线见 `STORY_LOCK_2026-04-28.md`：核心仍是 sharp failure frontier + corruption-aware intervention law，但机制不再写成 naive tokenizer-OOD closeness，而写成 forecastability/reconstruction mismatch。
>
> 实验设计补充：`SCENARIO_DESIGN_AND_LIT_REVIEW.md` 说明 S0-S6 为什么只能作为 summary path，以及下一版 sparse/noisy corruption grid 应该怎么设计。
>
> 执行计划：`RUN_PLAN_V2.md` 记录 4 GPU / 20% CPU 预算下的实际运行顺序、命令模板、以及 dry-run metadata 读数。
>
> L63 v2 试跑结论：`L63_V2_PILOT_READOUT.md` 记录 3-seed GPU pilot 的阈值读数和下一步扩种子优先级。
>
> 双模型协作规则：`DUAL_MODEL_COLLAB_PROTOCOL.md` 规定 GPT/Codex 与 Claude 在重大实验/写作决策前必须先形成书面共识。

---

## 1. 旧 narrative（已废弃）

> "我们设计了 CSDI 修复 + delaymask + random delay embedding + DeepEDM + conformal 的 pipeline，比 baseline 强。"

为什么不卖：每一块都是组合现有工作，审稿人会归类为 engineering integration。L96/Rössler 上 +125% 的数字不足以单独支撑顶会贡献。

## 2. 新 narrative（主线，2026-04-26 修订）

> **Sparse-noisy observations induce phase transitions in pretrained time-series foundation forecasters; a corruption-aware delay-manifold reconstruction pipeline avoids the OOD failure channel.**

**L96 N=20 isolation 后的更精确版本**：

> **The failure channel is not "Panda is intrinsically bad at chaos." It is "sparse-observation preprocessing can create non-physical patches that push tokenized forecasters OOD." Corruption-aware imputation is the first OOD-mitigation lever; delay-manifold forecasting is a second, dynamics-structured lever that improves robustness when paired with good imputation.**

**贡献分级**：

| Tier | 内容 | 当前证据状态 |
|------|------|-------------|
| **核心 1（现象）** | Foundation forecasters (Panda/Parrot/Chronos) 在 (s, σ) 相空间出现相变式崩塌（VPT → 0），不是平滑退化 | ✅ 4/6 系统已验证（README §2.1, §2.2 中 S5/S6 列） |
| **核心 2（机制）** | 崩塌主因是插值后非物理片段 → tokenizer / patch distribution OOD；corruption-aware imputation 可移除该通道，delay-coordinate 估计器提供另一条动力学约束通道 | 🟡 **L96 N=20 / N=10、L63、Rössler isolation 已验证 preprocessing lever**；仍需 tokenizer geometry 可视化 |
| **方法（次要）** | corruption-aware imputation (CSDI) + delay-manifold forecasting (DeepEDM) | ✅ 已实现并 5-seed 验证 |
| **边界（必含）** | 1D delay equation (MG) / PWL non-smooth (Chua) 不满足平滑流形假设 | ✅ 两个 negative result 现成 |

## 3. 砍 / 降级

- **`random delay embedding`** → ablation/appendix。除非证明随机延迟集合 ensemble 在 noise 下显著降 distortion，不能当主卖点。
- **`delaymask`** → 不再叫这个名字。改为 **Delay-Manifold Denoising** / **Takens-Masked Dynamics Modeling**。卖几何对象，不卖 mask。
- **CSDI 三 bug 修复** → 不写成"工程突破"，改写成"chaotic trajectories require corruption-aware diffusion anchoring 的必要条件"。
- **M4 Conformal coverage** → 主→副。

## 4. 必须新增的实验（按优先级）

### 4.1 Isolation Ablation（**reviewer killer，最高优先级**）

矩阵：Imputer ∈ {linear, kalman, **CSDI(ours M1)**} × Forecaster ∈ {**Panda** (ambient/tokenized), **Ours M3** (delay-manifold DeepEDM)}

| | Panda | Ours M3 (DeepEDM in delay coords) |
|---|---|---|
| Linear-fill | A1 已有（current `panda` baseline） | A4 ≈ 现 `ours_csdi_deepedm` 但 imputer 换 linear |
| Kalman-fill | A2 **新** | A5 **新** |
| CSDI-fill | A3 **新** | A6 = 现 `ours_csdi_deepedm` |

**核心解读**：
- 若 A3（CSDI→Panda）≈ A1（linear→Panda）：preprocessing 不是关键，**ambient-tokenizer 是 OOD 失败源**（论文机制论点成立）。
- 若 A3 显著 > A1，且 ≈ A6：preprocessing 是关键，**论文转向 "corruption-aware preprocessing 是核心"**（也比现在锋利）。
- A4 / A5 vs A6：测试 delay-manifold 对 imputer 质量的鲁棒性。

第一次跑：**L96 N=20，5 seeds，scenarios 聚焦 S2/S3/S4/S5**（相变带）。

**L96 N=20 5-seed readout（2026-04-26）**：

| Cell | S2 | S3 | S4 | S5 |
|---|---:|---:|---:|---:|
| Linear → Panda | 2.32 ± 2.42 [80%] | 1.46 ± 1.85 [80%] | 0.52 ± 0.48 [60%] | 0.00 ± 0.00 [0%] |
| Kalman → Panda | 2.69 ± 3.12 [80%] | 1.31 ± 1.94 [60%] | 0.35 ± 0.31 [40%] | 0.00 ± 0.00 [0%] |
| **CSDI → Panda** | **2.44 ± 1.91 [100%]** | **2.47 ± 1.83 [100%]** | **3.60 ± 3.73 [100%]** | 0.40 ± 0.81 [20%] |
| Linear → DeepEDM | 0.40 ± 0.25 [60%] | 0.24 ± 0.23 [20%] | 0.08 ± 0.13 [0%] | 0.00 ± 0.00 [0%] |
| Kalman → DeepEDM | 0.24 ± 0.13 [0%] | 0.25 ± 0.25 [40%] | 0.17 ± 0.15 [0%] | 0.00 ± 0.00 [0%] |
| **CSDI → DeepEDM** | 0.44 ± 0.22 [40%] | **0.71 ± 0.24 [80%]** | **0.72 ± 0.41 [60%]** | 0.18 ± 0.23 [20%] |

Cell format: VPT@1.0 mean ± std [Pr(VPT>0.5)].

Paired contrasts: CSDI improves Panda over linear at S3 by +1.01 Λ (95% paired CI [+0.12, +2.50]) and S4 by +3.07 Λ ([+0.57, +6.45]); CSDI improves DeepEDM at S3 by +0.47 Λ ([+0.05, +0.84]) and S4 by +0.64 Λ ([+0.20, +1.09]).

**Narrative consequence**:
- Do **not** claim "ambient/tokenized forecasters are doomed." The data says Panda is highly recoverable when the filled context is corruption-aware.
- The strongest mechanism claim is now: **linear/Kalman fill creates an OOD preprocessing artifact; CSDI removes much of that artifact.**
- Delay-manifold forecasting remains useful, but on L96 N=20 it is not the dominant lever in this matrix; it should be framed as the dynamics-structured companion to CSDI, not the sole survivor.
- This is reviewer-safe: the obvious alternative `CSDI → Panda` is now measured, and in fact becomes part of the paper's mechanism story.

**Replication readout（2026-04-26，5 seeds）**：

| System | Strongest CSDI→Panda gain | Strongest CSDI→DeepEDM gain | Interpretation |
|---|---:|---:|---|
| L63 | S2: +0.82 Λ, 95% CI [+0.32, +1.37]; S3: +0.40 Λ, [+0.13, +0.67] | S2: +0.75 Λ, [+0.30, +1.20]; S5: +0.15 Λ, [+0.02, +0.28] | CSDI clearly rescues moderate sparse-noisy contexts; S4/S5 mostly hit the low-data/noise floor. |
| L96 N=10 | S4: +1.11 Λ, [+0.08, +2.22] | S3: +0.44 Λ, [+0.08, +0.99]; S4: +0.67 Λ, [+0.30, +1.04] | Same lever appears in the smaller high-D system; delay-manifold gains are cleaner than Panda gains in S3/S4. |
| L96 N=20 | S3: +1.01 Λ, [+0.12, +2.50]; S4: +3.07 Λ, [+0.57, +6.45] | S3: +0.47 Λ, [+0.05, +0.84]; S4: +0.64 Λ, [+0.20, +1.09] | The headline reviewer-killer result: `CSDI→Panda` is strong, so preprocessing is the first lever. |
| Rössler | S3: +0.09 Λ, [+0.00, +0.19]; S4: +0.19 Λ, [-0.01, +0.39] | S3: +0.22 Λ, [+0.05, +0.43]; S4: +0.25 Λ, [+0.10, +0.39]; S5: +0.33 Λ, [+0.14, +0.53] | Rössler has lower absolute VPT in this isolation setup, but CSDI still consistently improves DeepEDM in the transition band. |

Data/figures:
- `experiments/week1/results/pt_l63_iso_l63_5seed.json`
- `experiments/week1/results/pt_l96_iso_l96N10_5seed.json`
- `experiments/week1/results/pt_l96_iso_l96N20_5seed.json`
- `experiments/week1/results/pt_rossler_iso_rossler_5seed.json`
- `experiments/week1/figures/iso_*_5seed.{md,png}`

### 4.2 方差治理

- L63、L96 N=20、Rössler 的 headline cells（S2-S5）当前已有 **5 seeds**；camera-ready 前建议把核心 cells 扩到 **10 seeds**。
- aggregator 增加 **paired bootstrap 95% CI** 和 **survival probability Pr(VPT > 0.5)** 列。
- 报 mean±std **同时** 报 survival prob，避免审稿人指控 std 太大。

### 4.3 Tokenizer OOD 可视化（机制图）

- 取 Panda 的 patch encoder（或它使用的 token 距离），把 (clean trajectory, linear-filled corrupted, CSDI-filled, delay-coord embedded) 四组喂进去画 PCA / UMAP。
- 显示 linear-filled 在 OOD 区域，CSDI-filled 部分恢复，delay-coord 完全跳出 ambient-tokenizer 视角。

## 5. 论文结构（新）

**主文 (~8 页)**：
- §1 Intro：抛出 phase-transition 现象 + delay-manifold survival 假设。
- §2 Setup：(s, σ) harshness scaling + VPT 度量 + Lyapunov normalization。
- §3 Phase-Transition Phenomenon：3 个强系统（**L63, L96 N=20, Rössler**）的 PT 图 + Pr(VPT>0) 表。
- §4 Mechanism：tokenizer OOD 几何图 + isolation ablation 表。
- §5 Method：corruption-aware imputation + delay-manifold forecasting（短，不再当主卖点）。
- §6 Theory：narrow 版 Theorem 2(a)（见下）。
- §7 Discussion：scope conditions（指向附录的 MG/Chua）。

**附录**：
- A. L96 N=10 + Kuramoto 复制实验。
- B. **Failure boundary**：MG (1D delay equation) + Chua (PWL non-smooth)。诚实陈述 scope。
- C. random delay embedding ablation。
- D. M4 conformal coverage。

**三张主图**：
1. Phase transition curve：4 系统的 VPT vs harshness。
2. Mechanism：(s, σ) 正交 failure channel grid + Panda patch OOD geometry。
3. Isolation ablation：2×3 矩阵热图。

## 6. 理论收窄

旧版（太大）：
> "Any ambient predictor under sparse-noisy observation has excess risk; delay-coordinate avoids it."

新版（更难被拳打穿）：
> **For tokenized pretrained sequence forecasters under interpolation-filled sparse observations, OOD patch geometry introduces an excess-risk term that scales with sparsity; delay-coordinate estimators avoid this specific OOD channel under smooth attractor assumptions.**

把 "ambient predictor" 换成 "tokenized pretrained forecaster"，把 "any" 换成 "interpolation-filled"。范围小但不可被简单反例打穿。

## 7. 立即开干顺序

1. ✅ 战略文档落地（本文件）。
2. ✅ Isolation ablation 代码（L96 N=20）→ run → aggregate。
3. ✅ aggregator 加 bootstrap + Pr(VPT>0 / >0.5)。
4. ✅ 用现有 5-seed JSON 生成 survival summary：`experiments/week1/results/phase_transition_survival_summary.md`。
5. ✅ Rössler / L96 N=10 / L63 isolation 复制并聚合。
6. 🟡 重写 paper §1 + §3 + §4，按 "corruption-aware imputation is the first OOD lever" 更新（中英文草稿已同步主叙事；细节章节仍需全文精修）。
7. ⏳ Tokenizer OOD 可视化（依赖能 introspect Panda patch encoder）。

---

**记住一句话**：现在每个实验/写作决策的判断尺度是——"这服务于 phase-transition 现象 + delay-manifold OOD-survival 主线吗？" 不服务的事情，要么砍，要么挪附录。
