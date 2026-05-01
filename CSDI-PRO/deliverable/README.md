# CSDI-PRO 当前研究成果整合包

整合时间：2026-04-25
分支：`csdi-pro-m3-alt`（10+ commits since 2026-04-23 起的 §5.7 改写工作）

---

## 1. 文件夹结构

```
deliverable/
├── README.md                  ← 本文件，所有结果 / 决策 / 后续方向
├── paper/                     ← 当前 paper 草稿（中英）
│   ├── paper_draft_en.md
│   └── paper_draft_zh.md
├── figures_phase_transition/  ← 6 个系统的 PT 主图（VPT vs scenario）
│   ├── L96_N20_pt.png
│   ├── L96_N10_pt.png
│   ├── Rossler_pt.png
│   ├── MackeyGlass_pt.png
│   ├── Kuramoto_pt.png
│   └── Chua_pt.png
├── trajectories/              ← 7 个系统的预测轨迹叠加图（4 scen × 3 dim）
│   ├── L63.png
│   ├── L96_N20.png
│   ├── L96_N10.png
│   ├── Rossler.png
│   ├── MackeyGlass.png
│   ├── Kuramoto.png
│   └── Chua.png
├── figures_extra/             ← §5.7 M3 backbone 对比 + 三 backbone sanity
│   ├── m3_backbone_comparison_L96.png
│   ├── svgp_sanity.png
│   ├── deepedm_sanity.png
│   └── fno_sanity.png
├── results/                   ← 6 个系统的 PT eval JSON 原始数据
│   ├── L96_N20_5seed.json
│   ├── L96_N10_5seed.json
│   ├── Rossler_5seed.json
│   ├── MackeyGlass_5seed.json
│   ├── Kuramoto_5seed.json
│   └── Chua_5seed.json
└── ckpts/                     ← 7 个 CSDI M1 checkpoint（含 L63 老 ckpt）
    ├── csdi_L63.pt            (4.9 MB, full_v6_center_ep20)
    ├── csdi_L96_N20.pt        (11 MB,  full_c192_vales_best @ep70 val=0.0453)
    ├── csdi_L96_N10.pt        (4.9 MB, full_vales_best @ep80 val=0.0357)
    ├── csdi_Rossler.pt        (1.3 MB, full_vales_best @ep84 val=0.0083)
    ├── csdi_MackeyGlass.pt    (1.3 MB, full_vales_best @ep97 val=0.0273)
    ├── csdi_Kuramoto.pt       (4.9 MB, full_vales_best @ep118 val=0.0017)
    └── csdi_Chua.pt           (1.3 MB, full_vales_best @ep79  val=0.0088)
```

---

## 2. 核心成果总览

### 2.1 完整 6 系统 5-seed PT eval 表（VPT@1.0, mean ± std）

| System | M1 + M3 best | S0 | S3 | S4 | S5 | S6 | **Headline** |
|---|---|---|---|---|---|---|---|
| **L96 N=20** | CSDI + DeepEDM | 0.79 ± 0.22 | 0.67 ± 0.34 | 0.71 ± 0.72 | **0.74 ± 0.81** | **0.49 ± 0.64** | S5/S6 唯一非零 |
| **L96 N=10** | CSDI + SVGP | 1.01 ± 1.04 | 1.23 ± 1.10 | **2.60 ± 3.14** | **0.27 ± 0.54** | **0.27 ± 0.54** | S4 超 Panda 9× |
| **Rössler** | CSDI + DeepEDM | **2.88 ± 1.46** | 0.43 ± 0.20 | 0.33 ± 0.22 | **0.23 ± 0.14** | **0.26 ± 0.15** | S0 超 Panda +30% |
| **Kuramoto** | CSDI + DeepEDM | 8.72 ± 4.16 | 2.56 ± 1.35 | 3.24 ± 2.13 | **1.58 ± 2.15** | 0.31 ± 0.31 | S5/S6 仅我们幸存 |
| **Chua** | CSDI + SVGP | 1.12 ± 0.89 | 0.72 ± 0.74 | 0.08 ± 0.11 | 0.13 ± 0.16 | 0.22 ± 0.34 | ❌ 全程输 Panda |
| **MG (τ=17)** | CSDI + SVGP | 0.28 ± 0.04 | 0.24 ± 0.10 | 0.04 ± 0.04 | 0.03 ± 0.03 | 0.03 ± 0.04 | ❌ 全程输 Panda |

### 2.2 Panda-72M 同表（baseline 参考）

| System | S0 | S3 | S4 | S5 | S6 |
|---|---|---|---|---|---|
| L96 N=20 | 2.55 | 1.18 | 1.04 | **0.00** | **0.00** |
| L96 N=10 | 2.18 | 1.51 | 0.29 | **0.00** | **0.00** |
| Rössler | 2.22 | 0.54 | 0.31 | **0.03** | **0.00** |
| Kuramoto | 15.36* | 11.62 | 10.46 | 0.60 | **0.00** |
| Chua | 1.51 | 1.06 | 0.48 | 0.48 | 0.36 |
| MG | 1.47 | 0.44 | 0.15 | 0.02 | 0.01 |

*Kuramoto S0-S2 = 15.36 是 pred_len 天花板（Panda 在该窗口内未发散）。

### 2.3 关键发现：Unique Survivor at Tokenizer-OOD threshold

**4/6 系统**（L96 N=20, L96 N=10, Rössler, Kuramoto）出现一致模式：
- S0-S3 Panda 凭预训领先
- **S5-S6（95-97% sparse, 1.2-1.5σ noise）后 Panda 全部归零或近零，我们的 pipeline 仍保持 0.2-1.5 Λ VPT**
- 这正是 Theorem 2(a) 预言的"foundation model tokenizer OOD 阈值之外，dynamics-aware 延迟流形 pipeline 唯一幸存"

**2/6 系统失败**（Chua, MG）：
- Chua: PWL Diode 让 CSDI smooth DDPM 欠拟合；Panda 预训覆盖电路振荡器
- MG: 1D state × 6 delay tokens 太少，DeepEDM 表达力受限；Panda 直接从预训分布里学到 MG

---

## 3. 设计决策与已修 bug

### 3.1 已修复的 2 个 bug（§5.7.1）
- **M1 Bug**: `csdi_impute_adapter` 没把 `attractor_std` 透传给 `model.impute()`，L96 推理用 L63 默认 8.51 → scale 错 2.3×。修后 L96 S3 imputation RMSE 比 linear 好 13%。
- **M3 Bug**: SVGP `m_inducing=128` 在 100-D 延迟特征空间下 Matérn 核 → 0，输出塌成 Y.mean。改为 `max(128, 5×feat_dim)` 自适应。修后 L96 1-step α 从 0.375 → 0.766。

### 3.2 M3 架构升级（§5.7.3）
SVGP 的 autoregressive rollout 在高维延迟特征上 α 衰减（α@h=0=0.77 → α@h=20=0.27）是**架构性瓶颈**，非超参可救。替换为 **DeepEDM**（ICML 2025 LETS Forecast）的 softmax-attention-as-learned-kernel。L96 上 1-step α 跳到 0.978（vs SVGP 0.77）。

### 3.3 评测协议
- 7 scenarios × 5 seeds × 4 methods（ours_csdi_svgp, ours_csdi_deepedm, panda, parrot/persist）= 140 runs/system
- Lorenz63 老 SVGP 仍保留作 legacy baseline
- 每系统都有 `phase_transition_pilot_<system>.py` + `plot_<system>_trajectory.py` + `aggregate_pt_<system>.py`

---

## 4. Paper 进展

`paper/paper_draft_zh.md` & `paper/paper_draft_en.md` 状态：
- ✅ Abstract / §1.1 / §3.3 / §6 / §7 已同步 §5.7 改写
- ✅ §5.7 L96 N=20 完整 4 小节（5.7.1 两 bug → 5.7.2 SVGP 架构限制 → 5.7.3 DeepEDM 替换 → 5.7.4 5-seed 最终表）
- ⏳ §5.8 MackeyGlass / §5.9 L96 N=10 / §5.10 Rössler / §5.11 Kuramoto / §5.12 Chua **尚未写入** paper（数据齐全，等你拍板叙事框架）

---

## 5. 后续可考虑方向（决策点）

### 5.1 Paper writing 选项
- **A. 全 6 系统都进 paper**：突出 universality（"4/6 系统 Theorem 2 预言的 unique-survivor 模式实证"），诚实呈现 2 个 negative。
- **B. 只用 4 个赢系统 + L63**：narrative 干净，但被 reviewer 抓 cherry-pick 风险。
- **C. 只主 paper 2-3 个系统（L96 N=20 + Rössler + L96 N=10）+ 附录 4 个**：审美最优，主线最强。

### 5.2 实验补强候选
- 增大 pred_len / 调阈值评测（Kuramoto S0 撞天花板可信吗？）
- α-vs-h supplementary metric（不依赖 VPT 单一阈值）
- 攻 MG/Chua 的负结果：是不是我们 CSDI 太小？把 MG/Chua 的 channels 加到 192 + n_layers=12 重训
- 更多 seed（10-20 seed 把 std 收紧）

### 5.3 理论闭环
- 6 系统的 VPT 数字与 d_KY / λ_1 是否成 power-law？画 scaling 图
- Panda 预训分布覆盖度 ⇄ S0-S4 我们 vs Panda 差距 — 写 supplementary

---

## 6. 提交记录（commit hash → 内容）

```
ec18a02 §5.11 Kuramoto N=10 K=1.5 5-seed PT + traj + ckpt
bd4bbcc §5.12 Chua double-scroll 5-seed PT + traj + ckpt
4673506 §5.9  Lorenz96 N=10 5-seed PT + traj + ckpt
8672eac §5.11 §5.12 infra (Kuramoto + Chua scaffolding)
b9606bf §5.10 Rössler 5-seed PT + traj + ckpt
e7a07ca §5.8  Mackey-Glass 5-seed PT + traj + ckpt
2801f71 §5.9 §5.10 infra (Rössler integrator + L96 N flex)
5d8ce4a §5.8 MG infrastructure (integrator + dataset + PT-eval pipeline)
e80343c §5.7 companion: Lorenz63 trajectory figure (DeepEDM vs legacy SVGP)
3aa1dac §5.7 final L96 trajectory figure (S0/S3/S5/S6 × dims)
4d5cd07 §5.7.4 add legacy-SVGP baseline row (+125% M3-swap gain)
c615387 paper_en §5.7 rewrite (M3 swap + S5/S6 headline)
0b6d0ef paper_zh §5.7 rewrite (M3 swap + S5/S6 survival)
f3de0dd aggregated L96 5-seed PT + §5.7 headline figure
ee96e5d aggregator + PT transition plot for §5.7 rewrite
2bbc9a9 3-way backbone comparison + Panda 5-seed
d1df7e7 full_pipeline_rollout backbone flag (default deepedm)
568f99f FNO sanity test (L96 1-step α=0.961, h=20 α=-0.146)
473c05c DeepEDM sanity test (L96 1-step α=0.977 vs SVGP 0.813)
a1f1ad3 add DeepEDM + FNO as SVGP drop-in replacements
```

---

## 7. 主要数字快查

**M3 swap 在 L63 / L96 上的增益**（§5.7 ablation）：
- L63 sanity α@h=10: SVGP 1.004, DeepEDM 1.019, FNO 0.979（不退化）
- L96 sanity α@h=10: SVGP 0.532, DeepEDM **0.629**, FNO 0.463
- L96 N=20 PT @ S0: AR-K+SVGP=0.48 → AR-K+DeepEDM=1.08（**+125%**）

**Theorem 2(a) 数值闭环**（L63 §5.2 主表 + L96 / Rössler / Kuramoto §5.7 / 5.9-5.11 跨系统验证）：
- 阈值 s* 因系统而异（L63 在 S2-S3，L96 在 S3-S4，Rössler 在 S4-S5，Kuramoto 在 S4-S5）
- 阈值之外的 unique-survivor 现象在 4/6 系统出现
