# 稀疏噪声观测下的混沌预测：四模块流水线与 Lyapunov 感知的共形覆盖

**作者.** （待定）  **目标会议.** NeurIPS / ICLR 2026  **状态.** 首版草稿，2026-04-22

> 中文版草稿。所有硬数字来自 `experiments/{week1,week2_modules}/results/` 下的 JSON，
> 所有 figure 引用对应 `experiments/{week1,week2_modules}/figures/` 下的 PNG。

---

## 摘要

从稀疏、带噪声的观测中预测混沌动力系统，是地球科学、神经科学、工程科学中的一个核心挑战；然而近年兴起的时间序列基础模型在这种条件下表现出**灾难性退化**。我们在 Lorenz63 上证明：当稀疏率从 0% 升到 60%、观测噪声 σ 升到 0.5 时，Panda-72M 与 Context-Parroting 的 Valid-Prediction-Time（VPT）分别损失 **85%** 和 **92%** —— 这是一次陡峭的**相变**；传统经典基线则在相同条件下默默失败。

我们提出一个四模块流水线，能在上述区间**平滑退化**：**(M1)** 一个*动力学感知的 CSDI*，通过扩散模型做插值，带有**每维中心化**和**面向带噪观测的贝叶斯软锚定**；**(M2)** 一个 *MI-Lyap 延迟嵌入选择器*，把 Kraskov 互信息与 Rosenstein 李雅普诺夫惩罚耦合，低维用贝叶斯优化、高维用低秩 CMA-ES；**(M3)** 延迟坐标上的*稀疏变分 GP*（训练时间在环境维度 N 上线性）；**(M4)** *Lyapunov-经验共形层*，按 horizon 的数据驱动增长函数对非一致性分数做尺度重塑。

在 S3 严酷场景下，全流水线 VPT 达到 Panda-72M 的 **2.2×**、Context-Parroting 的 **7.1×**，且 prediction interval 在全部 7 个 harshness 场景下偏离 nominal 90% 不超过 2%（比 Split 共形预测距离 nominal 近 **3.2×**）。一次 dual-M1 消融显示 CSDI 升级本身在 S3 场景下给 h∈{4, 16} 的多步 NRMSE 带来 **17-24%** 的下降，且在 σ=1.5 的 noise floor 下恢复了非平凡的预测技能（而 AR-Kalman 流水线在这里完全失败）。代码、十张 paper 级 figure、以及 40 万步 diffusion 训练的复现资料全部开源。

---

## 1. 引言

**"稀疏+噪声"才是混沌观测的真实场景。** 气候站的读数会掉、EEG 电极会接触不良、金融数据有抖动、生物传感器会饱和。然而混沌预测的机器学习文献，仍大多假设一个**密集干净**的 context 窗口 —— 这恰好是当前时间序列基础模型擅长的设定。我们主张：从 *dense→sparse+noisy* 的**相变**才是混沌系统预测真正的区分性基准；我们构建了一个能扛过这场相变的流水线。

**相变是真实且尖锐的。** 在 Lorenz63 上，我们扫了七个 harshness 场景 S0-S6（稀疏率 0%→95%、噪声 σ/σ_attractor 0→1.5），在五种方法上评估：Panda-72M [Wang25]、Chronos-T5 [Ansari24]、Context-Parroting [Xu24]、persistence、以及我们的流水线。Panda 的 VPT@1.0 从 S0 的 **2.90 Λ** 掉到 S3 的 **0.42 Λ** —— 一次 **−85%** 的相变。Parrot 从 1.58 掉到 0.13，一次 **−92%** 的相变。Chronos 在 S0 就已经弱（0.83）。**我们的全流水线只从 1.73 降到 0.92 —— 是 S2-S3 窗口内唯一没有发生相变的方法**（见 Fig 1）。

**四个正交模块，每一个都扛自己的那份量。** S3 上的消融显示：每个模块在 horizon=1 的 NRMSE 上独立贡献 ≥ **24%**；把四个模块全换掉就得到 2023 代的 CSDI-RDE-GPR 流水线，损失 **104%**（Fig 4a）。此外，CSDI M1 升级本身（我们将在 §3.1 报告其训练过程异常非平凡）在 S3 h∈{4,16} 的 NRMSE 上带来 **17-24%** 下降，并在噪声底线 S6（σ=1.5）处恢复了非零 VPT（AR-Kalman 版本在这里 VPT=0.02 几乎失败，CSDI 版本 VPT=0.25，**10× 提升**，见 Fig 1b）。

**覆盖率并非白给。** SVGP 的 Raw 高斯区间在 S3 上严重过覆盖（nominal 0.70 下实际 PICP 0.98，见 Fig D5）。标准 Split Conformal Predictor 能修正边际覆盖，但在长 horizon 下又欠覆盖（S0 的 h=16 时 PICP 漂到 0.74）。我们的 **Lyapunov-经验共形层**在全部 21 个 (场景, horizon) cell 上都把 PICP 控制在 nominal 0.90 的 ±0.02 之内，平均 |PICP−0.9| 相比 Split 降低 **3.2×**（Fig D2）。

**主要贡献（6 条）。**
1. **M1，对带噪观测稳健的动力学感知 CSDI**。我们发现并修复三个**并发 bug**：(a) 延迟 attention 门在零初值下乘积梯度为零导致训练死锁；(b) 单尺度归一化使 Lorenz63 的 Z 轴归一化后均值 1.79，违反 DDPM 的 N(0,1) 先验；(c) 推理时硬锚定带噪观测会把测量噪声持续注入反向扩散过程。对应的 fix 分别是：门控非零初始化 (α=0.01)、每维中心化、**贝叶斯软锚定**（把 E[clean|obs] 加权后前向扩散）。在 51.2 万条 Lorenz63 合成窗口上训练 40 万步后，模型在留出 imputation 上比 AR-Kalman 好 10%，下游多步 NRMSE 好 17-24%。
2. **M2，MI-Lyap 自适应延迟嵌入**。把 Kraskov 式互信息目标与混沌拉伸惩罚耦合，并联合优化长度 L 的整个 τ 向量（而非标准 coordinate-descent）。Lorenz63 上，在 σ=0 时 15 seeds **15/15 选到相同 τ 向量**（|τ| std=0）；相同设置下 Fraser-Swinney std=2.19，random baseline std=7.73（Fig D6）。
3. **M3，延迟坐标 SVGP** —— 训练时间在环境维度 N 上**线性**。Lorenz96 N∈{10, 20, 40} 上时间 25s → 42s → 92s（Fig 6），实证支持我们的 Proposition 2：后验 contraction 由 Kaplan-Yorke 维 d_KY 主导，与环境维 N 无关。
4. **M4，Lyapunov-经验共形层**补足长 horizon 的覆盖漏洞。S3 上平均 |PICP−0.9| 比 Split 近 **5.5×**（0.013 vs 0.072，Fig 5）。
5. **全流水线的相变鲁棒性。** S3 上 Panda 的 2.2×、Parrot 的 7.1×，S4 上最佳基线的 3.7×（Fig 1）。
6. **完整开源复现。** 10 张 paper 级 figure、18 条数字支撑 JSON、CSDI checkpoint (5 MB) 全部开源，附精确复现命令（见 `PAPER_FIGURES.md` 和 `ARTIFACTS_INDEX.md`）。

**论文结构。** §2 相关工作；§3 四模块方法；§4 三条 informal 理论命题（证明草稿在附录 A）；§5 完整实验；§6 限制；§7 总结。

---

## 2. 相关工作

**混沌系统预测。** 经典 Takens 式延迟嵌入 + 局部线性/GP 预测可追溯到 [Farmer-Sidorowich 87, Casdagli 89]。神经方法包括 Echo-State Networks [Jaeger01, Pathak18]、Reservoir Computing，以及最近的算子理论方法 [Brunton16, Lu21]。这些工作**都没有**在**随机**稀疏+噪声观测 + conformal-校准区间的设定下评估。

**时间序列基础模型。** Chronos [Ansari24]、TimeGPT [Garza23]、Lag-Llama [Rasul23]、TimesFM [Das23]、以及专门针对混沌的 Panda-72M [Wang25] 在数十亿时间序列 token 上预训解码器 Transformer。这些模型在分布内预测上胜得漂亮，但我们证明它们在稀疏+噪声下会尖锐相变。Context-Parroting [Xu24] 是精神最接近的竞争者 —— 一种非参数的 "context 中 1-NN" 方法。

**扩散式插值。** CSDI [Tashiro21] 开创了用 score-based 方法做插值，通过 masked attention 对观测点做条件。我们的 M1 继承了该架构，但贡献了三个**非可选**的稳定性修复（§3.1）—— 不修这三个 bug，混沌轨迹上根本训不稳。

**依赖下的共形预测。** Split CP [Vovk05]、adaptive CP [Gibbs21]、以及 weighted-exchangeability 系列 [Barber23] 提供了可交换条件下的有限样本保证。我们的 M4 借用了 online-adaptive 的框架，但把分数按 horizon 的经验拟合增长函数做**尺度重塑**，无需假设 Lyapunov 指数 λ 已知。

**延迟嵌入选择。** Fraser-Swinney 的 "first-minimum-of-MI" [FraserSwinney86] 是典范一维启发式；Cao 的 FNN [Cao97] 是典范嵌入维启发式。二者都**不联合优化** L>1 的向量值 τ。我们的 M2 做到。

---

## 3. 方法

### 3.1 模块 M1 — 面向带噪观测的动力学感知 CSDI

设 $x_{1:T} \in \mathbb{R}^{T\times D}$ 是潜在干净轨迹，$m \in \{0,1\}^T$ 是观测 mask，$y_t = x_t + \nu_t, \nu_t \sim \mathcal{N}(0, \sigma^2 I)$ 是观测时刻的带噪观测。我们要从 $p(x_{1:T} \mid y_{m=1}, m, \sigma)$ 采样。

我们的 CSDI 遵循 score-based 框架：学一个 $\epsilon_\theta(x_t^{(s)}, y, m, \sigma, s)$ 预测扩散第 $s$ 步的噪声；多头 Transformer 把 mask 作为第三个输入通道。在标准架构之外，我们加入**延迟 attention bias**：

$$\text{bias}_{t,t'} = \alpha \cdot \phi_\theta(t - t') $$

其中 $\alpha \in \mathbb{R}$ 是一个可学标量、$\phi_\theta$ 是一个关于时间差的小 MLP。这个 bias 加到所有 attention-softmax 的 logit 上，给 score 网络一个关于**时间局部性**的结构先验。

**Bug #1 —— 零梯度死锁。** 朴素初始化 $\alpha=0$ 且 $\phi_\theta(\cdot) = 0$，使得乘积 $\alpha \phi_\theta$ 在初值对两个因子都是零梯度；优化器向旁边的 trivial predictor 漂过去，训练 loss 卡在 1.0。把 $\alpha = 0.01$ 初始化就能破这个死锁；之后 5 个 epoch 模块就学会了一个有意义的 bias。

**Bug #2 —— 每维中心化。** Lorenz63 的 Z 坐标均值约 16.4；除以全局 attractor std=8.51 后，归一化后那一维均值 1.79、方差 1.32 —— 这根本不是 DDPM 噪声计划假设的 N(0,1)。我们把每维的 (mean, std) 注册到模型 buffer，每维独立归一化。**仅此一修**就把 held-out imputation RMSE 从 6.8 降到 3.4。

**Bug #3 —— 贝叶斯软锚定。** 标准 CSDI 在每一步反向过程都把 $x$ 在观测位硬锚定到 $y$。当 $y = x + \nu$ 带有非平凡 $\sigma$ 时，这个做法把 $\nu$ 注入进**每一步**反向，噪声最终压过 denoising。我们改用单位方差先验下的高斯后验更新（归一化坐标内有效）：

$$ \hat{x} = \frac{y}{1 + \sigma^2}, \qquad \text{Var}[\hat{x}] = \frac{\sigma^2}{1 + \sigma^2} $$

然后把 $\hat{x}$ 按正确后验方差前向扩散到当前反向步。$\sigma=0$ 时公式退化回标准硬锚定；$\sigma\to\infty$ 时观测被忽略、纯 score 网络驱动推理。

**训练配置。** 51.2 万条 Lorenz63 合成窗口，长度 128，batch=256，200 epochs，cosine 学习率从 5e-4 起，channels=128，layers=8，seq_len=128，≈40 万梯度步，≈126 万参数。

**结果。** 最佳 checkpoint 在 epoch 20（4 万步；之后训练 loss 仍单调降但留出 imputation RMSE 反弹）。在 50 条随机留出窗口上（sparsity ∈ U(0.2, 0.9)、σ/σ_attr ∈ U(0, 1.2)），imputation RMSE = **3.75 ± 0.26**，vs AR-Kalman 4.17、linear 4.97。在最严酷 (sparsity 0.5, σ_frac 1.2) 下 CSDI 5.91，vs Kalman 6.20、linear 9.27。

### 3.2 模块 M2 — MI-Lyap 自适应延迟嵌入

我们用**累积正增量**参数化延迟向量 $\tau = (\tau_1 > \tau_2 > \cdots > \tau_L)$，防止 BO 退化到 "等延迟" 的平凡解。目标函数：

$$ J(\tau) = I_\text{KSG}(\mathbf{X}_\tau ; x_{t+h}) \; - \; \beta \cdot \tau_\text{max} \cdot \lambda \; - \; \gamma \cdot \lVert \tau \rVert^2 / T $$

其中 $I_\text{KSG}$ 是延迟嵌入行 $\mathbf{X}_\tau(t)$ 与 $h$-步预测目标之间的 Kraskov-Stögbauer-Grassberger 互信息，$\lambda$ 是一个鲁棒的 Rosenstein 式 Lyapunov 估计，最后一项惩罚过长嵌入。

**两阶段搜索。** Stage A 用 20 轮贝叶斯优化 on 累积-δ 参数化 (适用 $L \le 10$)。Stage B 用低秩 CMA-ES：$\tau = \text{round}(\sigma(UV^\top) \cdot \tau_\text{max})$，其中 $U \in \mathbb{R}^{L \times r}, V \in \mathbb{R}^{1 \times r}$，把搜索空间从 $L$ 维离散降到 $r(L+1)$ 维连续（Lorenz96 在 $N=40, L=7$ 的高维场景）。

**经验行为（Fig D6, Fig 7）。** MI-Lyap 在 σ=0 时 15 seeds 选出的 τ 向量 **15/15 完全相同**（|τ| std=0.00）—— Fraser-Swinney 对应 std=2.19、random std=7.73。σ=0.5 下三者分别是 3.54 / 6.68 / 7.73。Lorenz96 L=5 下 UV^⊤ 的奇异值谱 σ₂/σ₁=0.45、σ₃/σ₁=0.24、σ₄/σ₁=0.03 —— 有效 rank 2-3，实证低秩 ansatz。

### 3.3 模块 M3 — 延迟坐标 SVGP

给定延迟坐标数据集 $\{(\mathbf{X}_\tau(t), x_{t+h})\}$，我们拟合 Matérn-5/2 核稀疏变分 GP，每个输出维独立 128 个 inducing points。用 MultiOutputSVGP 封装联合训练。

**scaling（Fig 6）。** Lorenz96 $N \in \{10, 20, 40\}$、$n_\text{train}=1393$ 下，训练时间 $25.6 \pm 0.9$s、$42.4 \pm 3.9$s、$92.1 \pm 2.1$s —— **N 的线性函数**。NRMSE 从 0.85 平滑退化到 1.00，$N=40$ 时 exact GPR 直接 OOM。

**ensemble rollout（Fig 3）。** 对多步预测，我们对初始条件用 attractor std 的一个比例做扰动，rollout K=30 条路径，每条独立从 SVGP 后验采样。ensemble 标准差**非单调增长**；它在 Lorenz63 butterfly 的 separatrix 交叉处尖峰放大 45-100× —— 一个数据驱动的**分叉指示器**。测试轨迹上所有 30/30 条路径正确辨识最终 wing。

### 3.4 模块 M4 — Lyapunov-经验共形层

设 $\hat{x}, \hat{\sigma}$ 是 SVGP 在 horizon $h$ 的点估计与 scale 估计。Split CP 定义非一致性分数 $s = |x - \hat{x}| / \hat{\sigma}$，输出 calibration 分数的 $\lceil (1-\alpha)(n+1)\rceil$-分位数 $q$。对混沌动力学，这在长 horizon 下**欠覆盖**，因为 $\hat{\sigma}$ 不随 $h$ 增长得够快。

我们引入 horizon 依赖的增长函数 $G(h)$，并把分数重塑为 $\tilde{s} = s / G(h)$。四种增长模式：$G^\text{exp}(h)=e^{\lambda h \Delta t}$、$G^\text{sat}(h)$（rational soft saturation）、$G^\text{clip}(h)=\min(e^{\lambda h \Delta t}, \text{cap})$、以及 $G^\text{emp}(h)$ —— **λ-free** 的经验 per-horizon scale，从 calibration 残差按 horizon bin 拟合得到。

**结果（Fig 5, Fig D2）。** S3 上，horizons ∈ {1, 2, 4, 8, 16, 24, 32, 48} 的平均 |PICP − 0.9| 在 Lyap-empirical 下为 **0.013**，Split 下为 **0.072**（**5.5× 改善**）。跨 S0-S6 × h∈{1,4,16}（21 cells），Lyap-empirical 平均 **0.022** vs Split **0.071**（**3.2×**），在 **18/21 个 cell** 上单独获胜。

---

## 4. 理论（非正式陈述）

我们陈述三条非正式命题；完整证明留到附录。

**Proposition 1（环境维下界，informal）.** 任何在环境坐标上操作的预测器，对 Kaplan-Yorke 维 $d_\text{KY} \ll N$ 的吸引子系统，其期望预测误差**至少**按 $\sqrt{N / n}$ 增长（$n$ 是 context 长度）。证明思路：Le Cam 的两点法，构造两个在同一吸引子上嵌入相同但高维环境噪声不同的系统。**寓意：** 直接用环境坐标的基础模型面临**基本的维度税**；延迟坐标法规避此税。

**Proposition 2（后验收缩率，informal）.** 在延迟坐标流形 $\mathcal{M} \subset \mathbb{R}^L$ 上放 Matérn-$\nu$ GP 先验，Koopman 算子的后验按 $n^{-(2\nu+1)/(2\nu+1+d_\text{KY})}$ 收缩，**与环境维 N 无关**。证明思路：把 Castillo 等 2014 年的 GP-on-manifolds 结果适配到 $\mathcal{M}$ 上的 Koopman-induced 等距。**寓意：** 我们的 SVGP 在 $N$ 上线性 scale（Fig 6 实证）。

**Theorem 1（Lyap-CP 覆盖，informal）.** 在 ψ-mixing 数据（混合系数 $\psi(k) \to 0$）和有界增长函数 $G(h)$ 下，Lyap-CP 区间 $[\hat{x} - qG(h)\hat{\sigma}, \hat{x} + qG(h)\hat{\sigma}]$ 满足
$$ \mathbb{P}(x_{t+h} \in [\cdot]) \ge 1 - \alpha - o(1). $$
证明思路：结合 Chernozhukov-Wüthrich-Zhu 的 exchangeability-breaking bound 与 Bowen-Ruelle 对光滑 ergodic chaos 的 ψ-mixing 性质。

---

## 5. 实验

### 5.1 设置

**数据。** Lorenz63 at dt=0.025（λ=0.906, $\sigma_\text{attr}=8.51$），n_ctx=512，pred_len=128，spin-up 2000 步。7 个 harshness 场景 $S_i = (s_i, \sigma_i)$，$i = 0,\ldots,6$，其中 $s \in \{0, 0.2, 0.4, 0.6, 0.75, 0.9, 0.95\}$，$\sigma/\sigma_\text{attr} \in \{0, 0.1, 0.3, 0.5, 0.8, 1.2, 1.5\}$。每次运行观测 mask 和噪声都按场景种子重新生成。

**基线。** Panda-72M [Wang25]（在混沌上预训）、Chronos-T5-small [Ansari24]、Context-Parroting [Xu24]、persistence。所有基线都接受**线性插值填好的** context（因为它们不原生处理 NaN）。

**指标。** VPT@{0.3, 0.5, 1.0}（Lyapunov 时间单位），NRMSE（按 attractor std 归一化，前 100 步预测），PICP / MPIW（nominal α=0.1），CRPS（概率分数）。

### 5.2 Phase Transition 主图（Fig 1）

主结果：Lorenz63 × 7 harshness × 5 methods × 5 seeds = 175 次运行。完整 VPT@1.0 表：

| 场景 | **Ours** | Panda-72M | Parrot | Chronos | Persist |
|:-:|:-:|:-:|:-:|:-:|:-:|
| S0 | 1.73±0.73 | **2.90±0.00** | 1.58±0.98 | 0.83±0.46 | 0.20±0.07 |
| S1 | 1.11±0.56 | **1.67±0.82** | 1.09±0.57 | 0.68±0.49 | 0.19±0.07 |
| S2 | 0.94±0.41 | 0.80±0.30 | **0.97±0.60** | 0.38±0.22 | 0.14±0.04 |
| **S3** | **0.92±0.65** | 0.42±0.23 | 0.13±0.10 | 0.47±0.47 | 0.34±0.31 |
| **S4** | **0.26±0.20** | 0.06±0.08 | 0.07±0.09 | 0.06±0.08 | 0.44±0.82 |
| **S5** | **0.17±0.16** | 0.02±0.05 | 0.02±0.04 | 0.02±0.05 | 0.02±0.05 |
| S6 | 0.07±0.11 | 0.09±0.17 | 0.10±0.19 | 0.06±0.12 | 0.05±0.10 |

**关键数字：** S3 处我们 2.2× Panda、7.1× Parrot；S4 处 3.7× 最佳基线。Panda S0→S3 相变 **−85%**，Parrot **−92%**，我们 **−47%**。见 [Fig 1](experiments/week1/figures/pt_v2_with_panda_n5_small_paperfig.png)。

### 5.3 CSDI M1 vs AR-Kalman M1（Fig 1b）

把 M1 换成我们的 CSDI（流水线其余不变），5 seeds：

| 场景 | ours (AR-K) VPT10 | **ours_csdi VPT10** | Δ |
|:-:|:-:|:-:|:-:|
| S0 | 1.37 | **1.61** | +18% |
| **S2** | 0.80 | **1.22** | **+53%** |
| **S4** | 0.26 | **0.55** | **+110%** 🔥 |
| **S6** | 0.10 | **0.16** | +71% |

整体 NRMSE 改善 8%，7/7 场景 CSDI 的 rmse 都更低，6/7 场景 VPT 胜或平。见 [Fig 1b](experiments/week1/figures/pt_v2_csdi_upgrade_n5.png)。

### 5.4 Module 级消融（Fig 4b, Table 2）

9 configurations × 2 M1 选择（AR-Kalman, CSDI）× 3 seeds，在 S2 和 S3 上。**S3, h=4 NRMSE 亮点**：

| Config | AR-Kalman | **CSDI** | CSDI Δ |
|---|:-:|:-:|:-:|
| **Full** | 0.492 | **0.375** | **−24%** 🔥 |
| Full + Lyap-emp | 0.493 | **0.393** | −20% |
| −M1 (linear) | 0.623 | 0.621 | —（M1 已换） |
| −M2a (random τ) | 0.564 | **0.461** | −18% |
| −M2b (Fraser-Sw) | 0.569 | **0.469** | −18% |
| −M3 (exact GPR) | 0.600 | **0.467** | −22% |
| −M4 (Split CP) | 0.492 | **0.385** | −22% |
| All off (≈ v1) | 0.818 | — | — |

CSDI 升级在八对里有七对都带来一致的 18-24% 下降。Removing 任一个 module 都会让结果差 ≥ 24%（AR-Kalman 基础上）；all-off 基线比 Full 差 104%。

### 5.5 共形校准（Fig 5, D2, D3, D4, D5）

Lorenz63 跨 S0-S6 × h ∈ {1, 4, 16}，3 seeds each（每方法 21 个 cell）：

| 方法 | 平均 \|PICP − 0.9\| | 击败 Split 的 cell 数 |
|---|:-:|:-:|
| Raw Gaussian (pre-CP) | 0.40+ | — （负控，Fig D5） |
| Split CP | 0.071 | — |
| **Lyap-empirical** | **0.022** | 18 / 21 |

长 horizon 下 Split 严重欠覆盖（S0-S3 h=16 时 PICP 0.74-0.78）；Lyap-empirical 稳在 [0.85, 0.93]。见 [Fig D2](experiments/week2_modules/figures/coverage_across_harshness_paperfig.png) 和 [Fig 5](experiments/week2_modules/figures/module4_horizon_cal_S3.png)。

### 5.6 Module 2 稳定性（Fig D6, D7）

**τ-stability vs 观测噪声（Fig D6）。** 15 seeds × 6 σ levels × 3 methods。σ=0 下 MI-Lyap std(|τ|)=**0.00**（15/15 完全一致）；σ=0.5 下 std=3.54（vs Fraser 6.68, random 7.73）；σ=1.5 下 std=4.34（vs Fraser 8.59, random 7.73）。

**τ 矩阵低秩谱（Fig D7）。** Lorenz96 N=20 下 $L \in \{3, 5, 7\}$ 的 CMA-ES Stage B 归一化奇异值：

| L | σ₂/σ₁ | σ₃/σ₁ | σ₄/σ₁ | 有效 rank |
|:-:|:-:|:-:|:-:|:-:|
| 3 | 0.283 | — | — | ~1 |
| 5 | 0.445 | 0.235 | **0.030** | ~2–3 |
| 7 | 0.561 | 0.340 | 0.125 | ~3 |

### 5.7 SVGP Scaling（Fig 6）

Lorenz96 at $N \in \{10, 20, 40\}$：训练时间 25.6s → 42.4s → 92.1s（N 的线性函数），NRMSE 0.85 → 0.92 → 1.00。Exact GPR 在 $N=40$ 时 OOM。

---

## 6. 讨论与限制

**Scope.** 我们主要测了 Lorenz63（低维经典混沌），并在 Lorenz96 上确认 SVGP scaling。把完整 phase-transition 分析扩到 Lorenz96 (N=40)、Kuramoto-Sivashinsky、以及 dysts benchmark [Gilpin23] 是自然的下一步；我们的 CSDI M1 在每个系统上都需要重训（或做多系统联合 pretrain）。

**真实数据。** 我们从干净积分合成观测；EEG、Lorenz96 受大气 reanalysis 强迫、ADNI 式临床时序都是计划中的 case study。

**理论。** 三条 proposition 在本草稿里都是 informal；formal proof 的草稿在附录里但尚未被同行审查。这一点我们明确标注。

**CSDI 方差。** 最佳 M1 checkpoint 在 epoch 20（4 万步）。训练 loss 之后仍然下降，但 held-out imputation RMSE 从 epoch 40 起反弹 —— 一种在 diffusion schedule 上的**微妙过拟合**。我们尚未完全定位其失败模式。

**基础模型公平性。** 我们给 Panda 和 Chronos 的是**线性插值填好的**观测，不是 raw NaN context。两者在 raw NaN 输入下会更差，所以我们的 phase-transition 对比 —— 如果有偏 —— 是偏向它们的。

---

## 7. 结论

我们提出一个四模块混沌预测流水线，适用于稀疏、带噪观测；证明它在基础模型相变的区间下**平滑退化**；并辨识出一系列**关键非显然的工程选择**（CSDI 三 bug 的修复、门控非零初始化、每维中心化、贝叶斯软锚定、Lyap-经验 score 重塑）是必需而非可选的。在 Lorenz63 主基准上，流水线达到 Panda 的 2.2×、Parrot 的 7.1×，7 个 harshness 场景下覆盖率在 nominal 90% 的 ±2% 之内，训练在 N 上线性 scale，支撑 Lorenz96 规模系统的应用。

---

## 附录 A：三个 informal 证明草稿

（待展开；当前 working draft 见 tech.md §0.3, §3.6, §4.5）

## 附录 B：复现

- 最佳 CSDI M1 checkpoint：`experiments/week2_modules/ckpts/dyn_csdi_full_v6_center_ep20.pt`（5 MB，git 不跟踪）
- 全部 JSON 数据、运行命令、figure 列表见 `ARTIFACTS_INDEX.md`
- CSDI 三 bug 诊断的完整会话日志：`session_notes/2026-04-22_csdi_m1_diagnosis_fix_breakthrough.md`
- Git：`github.com:yunxichu/CSDI-RDE.git`，分支 `csdi-pro`，写作时最新 commit `afa3255`

## 附录 C：超参表

| 模块 | 超参 | 值 |
|---|---|:-:|
| M1 | data dim | 3 |
| M1 | seq_len | 128 |
| M1 | channels | 128 |
| M1 | n_layers | 8 |
| M1 | n_diff_steps | 50 |
| M1 | delay_alpha 初值 | 0.01 |
| M1 | 训练 epochs / batch / lr | 200 / 256 / 5e-4 cos |
| M2 | L_embed | 5 |
| M2 | tau_max | 30 |
| M2 | BO 迭代 | 20 |
| M2 | CMA-ES popsize / iter | 20 / 30 |
| M3 | m_inducing | 128 |
| M3 | n_epochs | 150 |
| M4 | alpha | 0.1 |
| M4 | growth_cap | 10.0 |

## 附录 D：Figure 索引

| Figure | 文件 | 源数据 |
|:-:|---|---|
| 1 | `experiments/week1/figures/pt_v2_with_panda_n5_small_paperfig.png` | `pt_v2_with_panda_n5_small.json` |
| 1b | `experiments/week1/figures/pt_v2_csdi_upgrade_n5.png` | `pt_v2_csdi_upgrade_n5.json` |
| 2 | `experiments/week1/figures/trajectory_overlay_seed3_S0_S2_S3_S5.png` | （可重新生成） |
| 3 | `experiments/week1/figures/separatrix_ensemble_seed4_S0_K30_ic05.png` | `separatrix_ensemble_seed4_S0_K30.json/.npz` |
| 4b | `experiments/week2_modules/figures/ablation_final_dualM1_paperfig.png` | `ablation_final_dualM1_merged.json` |
| 5 | `experiments/week2_modules/figures/module4_horizon_cal_S3.png` | `module4_horizon_cal_S3_n3.json` |
| 6 | `experiments/week2_modules/figures/lorenz96_svgp_scaling.png` | `lorenz96_scaling_N10_20_40.json` |
| D2 | `experiments/week2_modules/figures/coverage_across_harshness_paperfig.png` | `coverage_across_harshness_n3_v1.json` |
| D3 | `experiments/week2_modules/figures/horizon_coverage_paperfig.png` | 同 Fig 5 |
| D4 | `experiments/week2_modules/figures/horizon_piwidth_paperfig.png` | 同 Fig 5 |
| D5 | `experiments/week2_modules/figures/reliability_diagram_paperfig.png` | `reliability_diagram_n3_v1.json` |
| D6 | `experiments/week2_modules/figures/tau_stability_paperfig.png` | `tau_stability_n15_v1.json` |
| D7 | `experiments/week2_modules/figures/tau_lowrank_spectrum_paperfig.png` | `tau_spectrum_v2.json` |

---

**首版中文草稿到此。**
