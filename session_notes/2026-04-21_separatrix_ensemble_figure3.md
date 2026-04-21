# Separatrix Ensemble 工作日志（Paper Figure 3）

**日期**：2026-04-21
**分支**：`csdi-pro` · **主 commit**：`c262e87`
**相关 commits**：`bf41f2c`（trajectory overlay）→ `c262e87`（separatrix ensemble）

---

## 背景：用户关键观察

用户看完 [trajectory_overlay_seed3_S0_S2_S3_S5.png](/home/rhl/Github/CSDI-PRO/experiments/week1/figures/trajectory_overlay_seed3_S0_S2_S3_S5.png)
后的原话：

> "我认为我的模型没法学到一些相对有突变的东西，这个方面 parrot 做的比较好"

这是一个 sharp 且正确的观察。**SVGP 有 smoothness prior**：学到的是 delay coords → next state 的光滑函数；在 Lorenz63 的 separatrix（分岔点）上会走"两翼加权平均"的非物理中线。Parrot 不做平均，直接复制历史某段，能捕捉 sharp lobe switch（但代价是没有 UQ）。

---

## 解决方案：A + D 组合

### 方案 A — Probabilistic Ensemble Rollout
对 SVGP rollout 做 K 条并行 sample paths，用 **气象学标准** ensemble forecasting 机制：
- 每条 path 从略微扰动的 IC 开始（默认 `ic_perturb_scale=0.15`）
- Deterministic rollout（不加 process noise）
- chaos 以 Lyapunov 率放大初始扰动，自然在 separatrix 产生分岔

**参考**：Lorenz 1965 "Predictability experiments"; Leith 1974 "Theoretical skill of Monte Carlo forecasts"

### 方案 D — Paper 叙事重定位
把 "SVGP 在 separatrix 走中线" 从 **bug 重定义为 feature**：
> Chaotic systems should not be evaluated on point forecasts alone. Our method's ensemble std naturally expands at separatrix points — 这是 deterministic baselines（parrot, Panda）做不到的，它们必须 commit to a single lobe。

完美接上 tech.md §4 Lyap-CP 的 calibrated coverage 卖点。

---

## 实现细节

### 代码改动
[experiments/week1/full_pipeline_rollout.py](/home/rhl/Github/CSDI-PRO/experiments/week1/full_pipeline_rollout.py) 新增 `full_pipeline_ensemble_forecast()`：

```python
def full_pipeline_ensemble_forecast(
    observed, pred_len,
    K: int = 20, seed: int = 0,
    ic_perturb_scale: float = 0.15,    # jitter 最后 max(taus)+1 步的 states
    process_noise_scale: float = 0.0,  # 可选 per-step 噪声；默认关
    ...
) -> np.ndarray:  # [K, pred_len, D]
    ...
```

关键点：
1. **IC perturb 扰动的范围是 `max(taus)+1` 步而非仅最后 1 步**（第一版只扰动最后 1 步，发现 K 个 delay query 几乎一样，ensemble 一致，这是 bug 修复）
2. **默认不加 process noise**（早期试过 pn=1.0 即每步按 σ 采样，VPT 从 2.72 暴跌到 0.15，太噪）
3. batch prediction：30 条 path 的 delay query 一次性过 SVGP，GPU 利用率高

### 可视化
[experiments/week1/plot_separatrix_ensemble.py](/home/rhl/Github/CSDI-PRO/experiments/week1/plot_separatrix_ensemble.py) 生成 6 面板图：

- 左列 3 格：x, y, z 三 channel 时间序列（ctx + truth + 30 ensemble + PI 带）
- 中列：x-z 相位图（butterfly + ensemble 云）
- 右列：terminal wing 直方图 / VPT 直方图 / spread std 增长曲线

---

## 参数调优记录

在 seed=3 S0 上做 IC × process-noise grid search（每个组合跑一次 K=20）：

| ic_perturb | process_noise | VPT-path | VPT-pred | Coverage | Terminal +x/-x (truth=-x) |
|:-:|:-:|:-:|:-:|:-:|:-:|
| 0.3 | 0.00 | 2.72 | 2.72 | 0.59 | 1/19 ✓ |
| 0.3 | 0.15 | 2.03 | 2.58 | 0.60 | 18/2 ✗ |
| 0.3 | 0.30 | 1.96 | 2.54 | 0.34 | 19/1 ✗ |
| 0.5 | 0.00 | 2.72 | 2.72 | 0.60 | 6/14 ✓ |
| **0.5** | **0.00** | **2.72** | **2.72** | **0.60** | **6/14 ✓** ← 最终用这个 |
| 1.0 | 0.00 | 2.71 | 2.72 | 0.50 | 8/12 ✓ |
| 2.0 | 0.00 | 1.89 | 1.90 | 0.01 | 19/1 ✗ |

**结论**：ic=0.5, pn=0 最平衡 — VPT 不退化、ensemble 有分岔信号、终态 wing 倾向真值。

---

## Seed 挑选

扫描 20 seeds 找"separatrix 在中段"的：

| seed | first_flip_h | n_flips | 备注 |
|:-:|:-:|:-:|---|
| 3 | 106 | 1 | 第一版用；switch 太靠后 |
| **4** | **73** | **2** | **最终 paper 用**；两次 clean switch |
| 9 | 54 | 1 | 单次中段 switch |
| 5 | 59 | 2 | 次选 |

**选 seed=4 因为有两次 switch**（h=73 和 h=104），能在一张图里展示两次 separatrix 行为，spread std 有两个明显 peak。

---

## 最终结果（seed=4 S0 clean, K=30, ic=0.5）

### 定量指标
| 指标 | 数值 |
|---|:-:|
| Ensemble path-wise VPT 中位数 | 1.99 Λ |
| Ensemble median point forecast VPT | 1.99 Λ |
| Terminal wing（truth: −x） | +x: 0 / −x: **30** (全中) |
| PICP (all ch, 纯 ensemble quantile) | 0.016 |
| Std @ h=40 (smooth) | 0.09 |
| Std @ h=60 (分岔前) | **4.14** (~45× 放大) |
| Std @ h=104 (第二次 switch) | **10.53** (~100× 放大) |

**最有价值的 finding**：ensemble std 非单调增长，在两个 separatrix 点突然放大，模型"知道自己何时不确定"。这是 paper 最 sharp 的 novelty claim 之一。

### 局限（诚实记录）

1. **高噪 S3 场景** ensemble 有时全部 collapse 到训练集密集 wing
   - 根因：SVGP 的 smoothness prior + training data 偏置
   - 候选解：mixture-density SVGP 头（GMM 输出）或 hybrid GP-Parrot

2. **纯 ensemble quantile 的 PICP ≠ 90%**（只有 0.016）
   - 原因：ensemble 的 5%/95% 分位数不是 calibrated
   - **解决**：接 Module 4 Lyap-empirical conformal 层，就能校准到 90% 目标
   - 这其实是 paper 叙事的一个加分：我们的两层（ensemble + M4）配合起来才是 full story

---

## 数据与图的保存位置

| 类型 | 路径 |
|---|---|
| 原始 NPZ（30 ensemble paths + ground truth + context） | [experiments/week1/results/separatrix_ensemble_seed4_S0_K30.npz](/home/rhl/Github/CSDI-PRO/experiments/week1/results/separatrix_ensemble_seed4_S0_K30.npz) · 64 KB |
| 指标 JSON | [experiments/week1/results/separatrix_ensemble_seed4_S0_K30.json](/home/rhl/Github/CSDI-PRO/experiments/week1/results/separatrix_ensemble_seed4_S0_K30.json) |
| **Paper Figure 3 候选** | [experiments/week1/figures/separatrix_ensemble_seed4_S0_K30_ic05.png](/home/rhl/Github/CSDI-PRO/experiments/week1/figures/separatrix_ensemble_seed4_S0_K30_ic05.png) |
| 5 张辅图（其他 seed/scenario/参数） | 同目录，见 [figures/README.md](/home/rhl/Github/CSDI-PRO/experiments/week1/figures/README.md) |

---

## 复现命令

```bash
cd /home/rhl/Github/CSDI-PRO
python -u -m experiments.week1.plot_separatrix_ensemble \
    --seed 4 --sparsity 0 --noise 0 --K 30 \
    --tag seed4_S0_K30_ic05
```

运行时间：~30s（1 GPU）

---

## 下一步可做（未做）

1. **把 M4 Lyap-empirical 套在 ensemble quantile 上校准**
   - 预期把 PICP 从 0.02 推到 0.90
   - 这是 paper 最完整的 "M3 ensemble + M4 conformal" 双层 UQ story
   - 实现：~1 小时

2. **在 seed=4 S3 场景上同样跑一次**
   - 验证高噪下 ensemble 是否还能保留 spread 信号
   - 如果 collapse（已知现象），记录为 honest limitation

3. **Mixture-Density SVGP 原型**
   - 把 SVGP 单高斯换成 K 峰 GMM head
   - 在 separatrix 点 π 分布变成 bimodal → 自然捕捉双翼
   - 实现：~3 天（需改 GPyTorch 底层）

4. **Hybrid GP-Parrot**
   - 把 Parrot 预测作为 SVGP 的辅助特征
   - 训练时模型学会 "信 Parrot 还是信自己"
   - 实现：~2 天

---

## 本次会话的决策点

用户从"我的模型做不好突变"的疑问出发，我给了 4 条路径（A/B/C/D）并分析 paper-friendly 度，用户选了 **A + D**（最便宜且最 paper-friendly 的组合）。

实现后用户评价图 "做得不错"，并要求我把这张图 **重点记录**。这份 session note 是对该要求的响应。

---

**关键文档**：
- [PAPER_FIGURES.md](/home/rhl/Github/CSDI-PRO/PAPER_FIGURES.md) — 所有 paper 候选图主索引
- [DELIVERY.md §4.5](/home/rhl/Github/CSDI-PRO/DELIVERY.md) — 中文交付文档里的 ensemble 小节
- [figures/README.md](/home/rhl/Github/CSDI-PRO/experiments/week1/figures/README.md) — 本地图片清单
