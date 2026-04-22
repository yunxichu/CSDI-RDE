# CSDI-PRO

**Probabilistic chaotic forecasting under sparse + noisy observations.**
Four-module pipeline: **imputation → delay embedding → SVGP regression → Lyapunov-aware conformal prediction**.

> **核心结果**：在 Lorenz63 × 7 harshness levels 上，Foundation models（Panda-72M / Chronos / Context-Parroting）在 S3（60% sparsity + 50% noise）相变式崩溃（−85%/−92% VPT），而我们的 pipeline 做到 **graceful degradation**，在 S3 上比 Panda 高 **2.2×**、比 Parrot 高 **7.1×**；CSDI M1 升级进一步将 S4 regime 的优势扩大到 **9× Panda**。

---

## 📖 文档导航（按阅读优先级）

从这里点击，每个链接的说明都写清楚了是什么：

| 文档 | 作用 | 何时看 |
|---|---|---|
| **[REFACTOR_PLAN_zh.md](REFACTOR_PLAN_zh.md)** | **paper 叙事重构完整方案（延迟流形统一框架；P0/P1/P2 三阶段路径）** | **下一阶段投 ICML/NeurIPS 前的必读** |
| **[STATUS.md](STATUS.md)** | **项目状态 + 时间线 + 完成度 + 9 项 TODO + 投稿可能性** | **下次接着做前先看这个** |
| **[ASSETS.md](ASSETS.md)** | **论文 figures + 数据文件 + checkpoints + 脚本 的索引** | 当你想找某张图或某个数据文件 |
| **[EXPERIMENTS_REPORT_zh.md](EXPERIMENTS_REPORT_zh.md)** | **详细实验结果表 + 所有符号定义** | 当你想查"某数字怎么来的" |
| **[paper_draft_zh.md](paper_draft_zh.md)** / **[paper_draft.md](paper_draft.md)** | 论文中英文首版草稿（待按 REFACTOR_PLAN 重构） | 当你想审 paper |
| **[tech.md](tech.md)** | v2 完整技术设计规范（1046 行，历史档案） | 当你想看原始设计意图 |
| `改进方案` | 三段对话原文（REFACTOR_PLAN 的来源） | 当你想看重构方案的原始推理 |
| `../session_notes/2026-04-22_csdi_m1_diagnosis_fix_breakthrough.md` | CSDI 三 bug 诊断的完整过程 | 当你想深挖 CSDI 翻盘细节 |

**短路径**：
- 新来 → 先看 **STATUS.md**（15 分钟读完对项目全貌清晰）→ 有具体需求再开其它
- **准备投 ICML/NeurIPS** → 看 **REFACTOR_PLAN_zh.md**（把 paper 从"pipeline 叙事"升级到"延迟流形统一框架"，P0 纯写作 1 周 + P1 证明+τ-coupling 2 周）

---

## 四模块 Pipeline

```
稀疏含噪观测 (T, D) with NaNs
        │
   ┌────▼────────────────────────────────────────────┐
   │ M1  Dynamics-Aware Imputation                    │
   │     methods/dynamics_csdi.py  (CSDI, paper 主用)  │
   │     methods/dynamics_impute.py (AR-Kalman, fallback)│
   └────┬────────────────────────────────────────────┘
        │ 补全后的密集轨迹 (T, D)
   ┌────▼────────────────────────────────────────────┐
   │ M2  MI-Lyap 延迟嵌入 τ 选择                       │
   │     methods/mi_lyap.py                           │
   │     Stage A: KSG-MI + BayesOpt (L ≤ 10)          │
   │     Stage B: 低秩 CMA-ES (UV^⊤, L > 10)          │
   └────┬────────────────────────────────────────────┘
        │ 延迟坐标特征 (n_samples, L)
   ┌────▼────────────────────────────────────────────┐
   │ M3  Sparse Variational GP 回归                   │
   │     models/svgp.py   (Matern-5/2, 128 inducing) │
   └────┬────────────────────────────────────────────┘
        │ 预测均值 μ(t) 和不确定度 σ(t)
   ┌────▼────────────────────────────────────────────┐
   │ M4  Lyapunov-Conformal 预测区间                   │
   │     methods/lyap_conformal.py                    │
   │     4 growth modes：exp / saturating / clipped /   │
   │     **empirical**（推荐，λ-free）                 │
   └────┬────────────────────────────────────────────┘
        │ (mean forecast, lower, upper)
```

---

## 核心数字一眼望

**21 条 paper 可直接引用的硬数字**（完整表见 [STATUS.md §三](STATUS.md)）：

| 主题 | 核心数字 |
|---|---|
| Phase Transition (Fig 1) | **Ours @ S3 = 2.2× Panda, 7.1× Parrot**，Panda 相变 −85% |
| CSDI M1 升级 (Fig 1b) | S2 **+53%**，S4 **+110%** VPT，S4 vs Panda **9.4×** |
| Module 消融 (Fig 4b) | CSDI 在 S3 h=4 一致带来 **−24%** NRMSE；每 module ≥ 24% 必要性 |
| CP 校准 (Fig 5, D2) | Lyap-empirical **5.5× / 3.2×** 更准 calibration |
| MI-Lyap (D6) | σ=0 时 **15/15 选同一 τ**（std=0） |
| τ 低秩 (D7) | effective rank ≈ **2-3**（支持 rank-2 ansatz） |
| SVGP scaling (Fig 6) | 训练时间 **线性 in N**（Lorenz96 N=10/20/40） |

---

## 场景定义（S0–S6）

| 场景 | 稀疏率 $s$ | 噪声 $\sigma / \sigma_\text{attr}$ | 语义 |
|:-:|:-:|:-:|---|
| S0 | 0% | 0.00 | 完全干净（基线） |
| S1 | 20% | 0.10 | 轻度缺失 |
| S2 | 40% | 0.30 | 中度（相变预兆） |
| **S3** | **60%** | **0.50** | **主战场（相变分界线）** |
| S4 | 75% | 0.80 | 很糟 |
| S5 | 90% | 1.20 | 极糟 |
| S6 | 95% | 1.50 | 近纯噪声（物理底线） |

**S3 是论文核心对比点**：Parrot / Chronos / Panda 在此处 VPT 大幅崩溃，ours 保持 0.92 Λ。

---

## 仓库结构

```
CSDI-PRO/
├── README.md              ← 本文件（导航）
├── STATUS.md              ← 项目状态 + TODO + 时间线
├── ASSETS.md              ← Figures + 数据文件索引
├── EXPERIMENTS_REPORT_zh.md  ← 详细数字 + 符号表
├── tech.md                ← v2 技术规范（历史档案）
├── paper_draft_zh.md / paper_draft.md  ← 论文中英文草稿
│
├── methods/               # 四个核心算法 module
│   ├── dynamics_csdi.py           # M1 CSDI
│   ├── dynamics_impute.py         # M1 baseline + csdi 分发
│   ├── csdi_impute_adapter.py     # CSDI ckpt ↔ pipeline 桥接
│   ├── mi_lyap.py                 # M2（Stage A BO + Stage B CMA-ES）
│   └── lyap_conformal.py          # M4（4 growth modes）
├── models/svgp.py                 # M3
├── metrics/                       # VPT / NRMSE / CRPS / PICP / MPIW
├── baselines/                     # Panda / Chronos / Parrot
│
├── experiments/week1/     # Phase Transition 主实验 + Fig 1/2/3
├── experiments/week2_modules/     # 消融 + Module 专项 + Fig 4-7 + D2-D7
│
├── csdi/ gpr/ rde_delay/ rde_spatial/   ← v1 历史代码（已被 v2 取代，保留）
│
└── ../session_notes/      # 关键会话记录（重要的有 2026-04-22 CSDI 翻盘那份）
```

**详细的每个文件作用**见 [ASSETS.md §四](ASSETS.md)。

---

## 快速复现命令

```bash
# Phase Transition 主图（Fig 1）
CUDA_VISIBLE_DEVICES=0 python -m experiments.week1.phase_transition_pilot_v2 \
    --n_seeds 5 --tag with_panda_n5_small

# 消融 Fig 4b（dual-M1 paired bars）
python -m experiments.week2_modules.merge_ablation_csdi_paperfig

# CSDI M1 训练
bash experiments/week2_modules/run_csdi_longrun.sh
```

**完整复现命令清单**见 [ASSETS.md §七](ASSETS.md)。

---

## Git 状态

- 分支：`csdi-pro`
- 远端：`github.com:yunxichu/CSDI-RDE.git`
- 最新 commit：见 `git log --oneline -1`
- 12+ commits 全推送；commit 历史见 [STATUS.md §八](STATUS.md)

---

## 使用注意

- v1 的 `csdi/` 模块使用同目录相对导入，请从 `CSDI-PRO/csdi/` 目录运行或加 `sys.path.insert(0, 'CSDI-PRO/csdi')`。
- `rde_spatial/rde_spatial.py` 依赖 `gpr.gpr_module`，需在项目根目录（`CSDI-PRO/`）下运行。
- Panda-72M 权重需单独下载（见 `baselines/README.md`），`panda-src` 代码库在 `/home/rhl/Github/panda-src`。
- 所有 v2 实验脚本均使用 `python -m` 从 `CSDI-PRO/` 根目录运行（保证相对导入正确）。
- CSDI checkpoints（`experiments/week2_modules/ckpts/*.pt`）和训练数据（`data/*.npz`）在 `.gitignore` 里，本地生成不推远端。

---

## 下次从哪里开始？

**1 行命令**：打开 [STATUS.md](STATUS.md) → §五 **9 项未完成 TODO**，按 Level 1 (T1+T2+T8+T9, ~1 周) / Level 2 (+T3+T6) / Level 3 (+T4+T5+T7) 挑一项。每项都有"入口文件 + 运行命令"。
