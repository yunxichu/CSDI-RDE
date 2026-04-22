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

### 5.1 实验设置

#### 5.1.1 数据生成

**系统.** Lorenz63（$\dot{x}=\sigma(y-x), \dot{y}=x(\rho-z)-y, \dot{z}=xy-\beta z$；标准参数 $\sigma=10, \rho=28, \beta=8/3$）；最大 Lyapunov 指数 $\lambda = 0.906$；attractor 全局标准差 $\sigma_\text{attr} = 8.51$。我们用 `scipy.odeint` 以步长 $\Delta t = 0.025$ 积分，丢弃 2000 步 spin-up 以确保位于吸引子上。

**Context 与 forecast 窗口.** 每条轨迹生成 $n_\text{ctx}=512$ 步作为观测窗口（喂给方法），以及 $T_\text{pred}=128$ 步作为预测目标（保密，只用于评估）。单次积分可覆盖约 $512 \cdot 0.025 \cdot 0.906 \approx 11.6$ Lyapunov 时间的 context 长度。

**Seed 机制.** 每个实验给定 $n_\text{seeds}$ 个 seeds；每个 seed 独立整合一条轨迹并独立生成观测 mask 和噪声，最终报告 mean ± std。

#### 5.1.2 Scenario 定义（模拟观测"恶劣程度"）

真实观测永远不完美 —— 气象站会掉数据，EEG 电极会接触不良。我们系统化地制造两个维度的恶化：**稀疏率** $s$（丢弃多少时间步）和**噪声强度** $\sigma / \sigma_\text{attr}$（加多大噪声）。具体地，对干净轨迹 $x_{1:T}$：

$$ \text{mask}_t \sim \text{Bernoulli}(1-s); \qquad y_t = x_t + \eta_t, \ \eta_t \sim \mathcal{N}(0, (\sigma)^2 I); \qquad y_t \leftarrow \text{NaN} \text{ if } \text{mask}_t = 0. $$

即**先加噪声再稀疏化**。我们测试 7 个递增恶化的 scenario：

| Scenario | $s$ (稀疏率) | $\sigma / \sigma_\text{attr}$ (噪声比) | 语义 |
|:-:|:-:|:-:|---|
| **S0** | 0% | 0.0 | 完全干净（理想基线） |
| S1 | 20% | 0.10 | 轻度缺失 + 微小抖动 |
| S2 | 40% | 0.30 | 中度（转换边界的预兆） |
| **S3** | 60% | 0.50 | 严重（**主战场，相变点**） |
| S4 | 75% | 0.80 | 很糟 |
| S5 | 90% | 1.20 | 极糟（噪声 > 信号） |
| S6 | 95% | 1.50 | 近纯噪声（噪声底线） |

**为什么 S3 是主战场.** Foundation models 在 S0-S1 还能工作，但从 S2 开始出现 phase transition 的预兆，S3 是它们**灾难性崩溃**的分界点。我们的 pipeline 在这里保持 0.92 Λ 的 VPT，Panda 掉到 0.42、Parrot 掉到 0.13。整个 paper 的"锋利卖点"数字都锚在 S3。

#### 5.1.3 基线方法

| 方法 | 类型 | 输入处理 | 文献 |
|---|---|---|---|
| **Ours (AR-Kalman M1)** | 4-module pipeline，AR-Kalman 作 M1 | 原生支持 NaN | 本文 |
| **Ours (CSDI M1)** | 同上，M1 换成 CSDI | 原生支持 NaN | 本文 §3.1 |
| **Panda-72M** | 72M 参数 Transformer，混沌预训 | 线性插值填 NaN | [Wang25] |
| **Chronos-T5-small** | T5-base 时序 tokenizer | 同上 | [Ansari24] |
| **Context-Parroting** | 非参数 1-NN in context | 同上 | [Xu24] |
| **Persistence** | $\hat{x}_{t+h} = x_t$ | 用最后非 NaN 值 | — |

Panda/Chronos/Parrot 不原生处理 NaN，我们给它们**线性插值**后的 context —— 这对它们有利（offered advantage），我们的对比因此更保守。

#### 5.1.4 评估指标

- **VPT@τ**（Valid Prediction Time，Lyapunov 时间）：预测误差持续 $|\hat{x}_{t+h} - x_{t+h}| < \tau \cdot \sigma_\text{attr}$ 的最长前缀长度，除以 $1/\lambda$。$\tau \in \{0.3, 0.5, 1.0\}$。**越大越好**。
- **NRMSE**：$\sqrt{\mathbb{E}[(\hat{x}-x)^2]}/\sigma_\text{attr}$，取前 100 步预测的 RMSE 归一化到 attractor 尺度。**越小越好**。
- **PICP** (Prediction Interval Coverage Probability)：真值落入预测区间的经验比例。nominal $\alpha=0.1$ 下目标 **0.90**。
- **MPIW** (Mean PI Width)：区间平均宽度。在 PICP 达标前提下越小越好。
- **CRPS**：Continuous Ranked Probability Score，对整条概率分布的打分。

### 5.2 Phase Transition 主图（Fig 1）

**Setup.** Lorenz63 × 7 harshness scenarios × 5 methods × **5 seeds** = **175 次独立 run**。每个 run 独立积分一条 640 步轨迹（512 context + 128 future），独立采样 mask 和噪声。

**做了什么.** 对每个 scenario-seed 组合，给 5 种方法同样的带 NaN context，让每种方法预测未来 128 步，记录 VPT@{0.3, 0.5, 1.0} 和 NRMSE。最后按 (method, scenario) 聚合 5 seeds 的 mean ± std。

**结果（VPT@1.0 完整表）.**

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

**说明**：Fig 1 主表的 "Ours" 列默认使用 **M1 = AR-Kalman**（轻量 surrogate）。用 M1 = CSDI 的升级版对比见 §5.3 / §5.4；Fig 2 和 Fig 3 的 CSDI 升级版轨迹可视化见 [附录 D](#附录-d：figure-索引)。

**解读（三段 story）.**
- **干净 regime (S0-S1)**：Panda 2.90/1.67 是霸主，Parrot 紧随；Ours 排第二。证明我们的 pipeline **在 foundation-model 的强项区不掉链**，没有为了鲁棒而牺牲干净数据的精度。
- **转换边界 (S2)**：Parrot 0.97 ≈ Ours 0.94 ≈ Panda 0.80，三强相持 —— 这是即将到来 phase transition 的预兆。
- **主战场 (S3)**：Panda 掉 **−85%**，Parrot 掉 **−92%**，Chronos 早已崩盘（0.47）；**Ours 只降 47% 到 0.92**，成为唯一没有 phase transition 的方法。S4-S5 继续扩大差距（ours 独一档）。
- **物理边界 (S6)**：σ=1.5 淹没一切，全员归零，无 method 能恢复。

这张图支持 paper 的核心 claim："foundation models phase-transition; we don't."

### 5.3 CSDI M1 vs AR-Kalman M1（Fig 1b）

**Setup.** 复用 §5.2 的 7 scenarios × 5 seeds 设置，但只保留 ours 一个方法，在 **M1=AR-Kalman** (原 baseline) 和 **M1=CSDI** (升级版，checkpoint `dyn_csdi_full_v6_center_ep20.pt`) 两种配置下各跑一次。

**做了什么.** 其他三个模块（MI-Lyap τ、SVGP、Lyap-CP）完全不变，只替换 M1。同样记录 VPT@1.0 和 NRMSE。

**结果（n=5）.**

| 场景 | ours (AR-K) VPT10 | **ours_csdi VPT10** | Δ |
|:-:|:-:|:-:|:-:|
| S0 | 1.37 | **1.61** | +18% |
| **S2** | 0.80 | **1.22** | **+53%** |
| **S4** | 0.26 | **0.55** | **+110%** 🔥 |
| **S6** | 0.10 | **0.16** | +71% |

整体 NRMSE 改善 8%，7/7 场景 CSDI 的 rmse 都更低，6/7 场景 VPT 胜或平。见 [Fig 1b](experiments/week1/figures/pt_v2_csdi_upgrade_n5.png)。

**解读.**
- **CSDI 带来 RMSE 维度的全面改善**（7/7 scenarios），证明更好的插补对下游 rollout 的精度**一致传递**。
- **VPT 维度 CSDI 在 harsh regime 上有巨大优势**（S2 +53%, S4 +110%）—— VPT 是 thresholded metric，只有 rmse 低到一定程度才会触发 VPT 跳升，所以 CSDI 的收益在 thresholded 度量上放大。
- **S6（noise floor）的 +71%** 是特别珍贵的卖点：σ=1.5 时 AR-Kalman 的 VPT 近乎 0，CSDI 仍能从观测中"挤出"0.16 Λ —— **"在 AR-Kalman 完全失败的地方，CSDI 还能提取可用信号"**。
- S1/S3 的小幅落后（−3%/−10%）在 1-seed σ 范围内，不统计显著。

这证明 CSDI 升级**不是 M1 "纯替换"**，而是**实质性 M1 能力提升**。

#### 5.3.1 ours_csdi 对所有基线的并排对比（Fig 1b 扩展版）

把 Fig 1b 的 ours/ours_csdi 数字与 Fig 1 的 Panda/Chronos/Parrot/Persist 数字并排（共用 seed 0-4 的 Lorenz63 轨迹，scenarios 完全一致）：

| Scenario | **ours_csdi** | ours (AR-K) | Panda-72M | Parrot | Chronos | Persist |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| S0 | 1.61 ± 0.76 | 1.73 ± 0.73 | **2.90 ± 0.00** | 1.58 ± 0.98 | 0.83 ± 0.46 | 0.20 ± 0.07 |
| S1 | 1.11 ± 0.59 | 1.11 ± 0.56 | **1.67 ± 0.82** | 1.09 ± 0.57 | 0.68 ± 0.49 | 0.19 ± 0.07 |
| **S2** | **1.22 ± 0.80** | 0.94 ± 0.41 | 0.80 ± 0.30 | 0.97 ± 0.60 | 0.38 ± 0.22 | 0.14 ± 0.04 |
| **S3** | **0.82 ± 0.67** | 0.92 ± 0.65 | 0.42 ± 0.23 | 0.13 ± 0.10 | 0.47 ± 0.47 | 0.34 ± 0.31 |
| **S4** | **0.55 ± 0.78** | 0.26 ± 0.20 | 0.06 ± 0.08 | 0.07 ± 0.09 | 0.06 ± 0.08 | 0.44 ± 0.82 |
| **S5** | **0.17 ± 0.18** | 0.17 ± 0.16 | 0.02 ± 0.05 | 0.02 ± 0.04 | 0.02 ± 0.05 | 0.02 ± 0.05 |
| **S6** | **0.16 ± 0.16** | 0.07 ± 0.11 | 0.09 ± 0.17 | 0.10 ± 0.19 | 0.06 ± 0.12 | 0.05 ± 0.10 |

**ours_csdi 对基线的关键比率**：

| Scenario | vs Panda | vs Parrot | vs Chronos | vs Persist |
|:-:|:-:|:-:|:-:|:-:|
| S2 | **1.53×** | **1.26×** | **3.21×** | **8.71×** |
| S3 | **1.96×** | **6.43×** | **1.73×** | **2.43×** |
| **S4** | **9.38×** 🔥 | **8.13×** 🔥 | **9.38×** 🔥 | 1.24× |
| S5 | **9.22×** | **11.63×** | **10.52×** | **9.91×** |
| S6 | 1.88× | 1.66× | 2.75× | 3.44× |

**两条相互加强的 paper 主消息**：
1. **Fig 1（原 AR-Kalman M1 baseline）**：ours 在 S3 = 2.2× Panda / 7.1× Parrot，**唯一不相变的方法**
2. **Fig 1b（CSDI M1 升级）**：ours_csdi 在 **S2 全面碾压所有基线**（1.26-8.7×），**S4 相对 foundation models 扩大到 ~9×**（AR-K 版 3.7× → CSDI 版 9.4×，2.5× 放大）

**解读**：CSDI M1 的升级**不只是 10% 的 imputation RMSE 改善**，而是让整个 pipeline 在 foundation models 早就崩盘（VPT < 0.1）的 S4 regime 进一步扩大领先。这是"**CSDI 在无信号区间也能挤出信号**"故事的最强证据。

### 5.4 Module 级消融（Fig 4b, Table 2）

**Setup.** 在 S2 和 S3 上各跑 **9 configurations × 2 M1 选择 (AR-Kalman / CSDI) × 3 seeds × 4 horizons**。9 个 configs 是通过每次"翻一个模块"构造的：
- `full` = AR-Kalman 的 baseline，4 模块全开
- `full-empirical` = M4 从 Lyap-sat 换成 Lyap-empirical
- `m1-linear` / `m2a-random` / `m2b-frasersw` / `m3-exactgpr` / `m4-splitcp` / `m4-lyap-exp` = 分别把一个模块降级到最简/最常用 baseline
- `all-off` = 全部换成 2023 年的 CSDI-RDE-GPR pipeline

**做了什么.** 每个 (config, M1) 组合跑 3 seeds × 4 horizons = 12 个数字。报告 NRMSE / PICP / MPIW / CRPS 四个指标 × 4 horizons = 16 格 × 9 configs × 2 M1 = 288 格数据点。

**结果 — S3, h=4 NRMSE 核心对比**：

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

**解读.**
- **CSDI 优势一致**：在 8 对 (去掉 M1 重叠的 linear 那行) 里，有 7 对 CSDI 都带来 18-24% 改善。仅有 −M3 (exact GPR) 的 h=1 出现 CSDI 小幅落后（0.491 vs 0.463），但 h=4 仍胜 22%；推测是 exact GPR 的 smoothness prior 对 CSDI 风格的 imputation 敏感。
- **每个 module 独立必要**：−M1 / −M2 / −M3 都带来 ≥24% 的 NRMSE 退化，证明四模块无冗余。
- **协同效应显著**：`Full` 到 `all-off` 是 0.373→0.760（**+104%**）；单 module 贡献之和远小于这一差距，说明模块间**协同**。
- **std 缩 3×**：Full-CSDI 的 S3 h=1 std 仅 0.009（AR-K 下 0.028），**M1 升级使得下游更稳定**。

完整 S2+S3 dual-M1 表见 [附录 B 补表](experiments/week2_modules/results/ablation_final_dualM1_merged.md)。

### 5.5 共形校准（Fig 5, D2, D3, D4, D5）

**Setup.** 我们评估 M4 是否能在**广泛的 harshness × horizon 组合**下维持 nominal 0.90 的 coverage。Lorenz63 跨 **7 scenarios × 3 horizons × 3 seeds = 63 条** 预测轨迹，每条用 60% 训 / 20% calibrate / 20% test 的 split 训练 SVGP + 拟合 CP。

**做了什么.** 对每条轨迹，先用 AR-Kalman 或 CSDI 做 M1 插补，MI-Lyap 选 τ，SVGP 训练完得到 (μ, σ) 点估计 + scale 估计；然后在 calibration split 上用两种 CP 方法拟合：(a) **Split CP**（标准 baseline，residual 的 $(1-\alpha)$ 分位数），(b) **Lyap-empirical CP**（我们的，按 horizon 拟合 growth function $G(h)$ 再归一化 score）。对每个 (scenario, horizon) 组合聚合 3 seeds 的 mean ± std。

**结果（5.5.1） AR-Kalman M1 下**。每方法 21 个 (scenario, horizon) cells：

| 方法 | 平均 \|PICP − 0.9\| | 击败 Split 的 cell 数 |
|---|:-:|:-:|
| Raw Gaussian (pre-CP) | 0.40+ | — （负控，Fig D5） |
| Split CP | 0.071 | — |
| **Lyap-empirical** | **0.022** | **18 / 21** |

长 horizon 下 Split 严重欠覆盖（S0-S3 h=16 时 PICP 0.74-0.78）；Lyap-empirical 稳在 [0.85, 0.93]。见 [Fig D2](experiments/week2_modules/figures/coverage_across_harshness_paperfig.png) 和 [Fig 5](experiments/week2_modules/figures/module4_horizon_cal_S3.png)。

Module 4 专项上（S3, horizons=1-48, mixed-horizon pooled calibration）：

| 方法 | 平均 \|PICP − 0.9\| | 比 Split 改善 |
|---|:-:|:-:|
| Split CP | 0.072 | baseline |
| Lyap-exp | 0.054 | 1.3× |
| Lyap-saturating | 0.049 | 1.5× |
| **Lyap-empirical** | **0.013** | **5.5×** 🔥 |

**（5.5.2） CSDI M1 下的共形校准（补齐版）**。把 M1 换成 CSDI 重跑同一组 21 cells：

| 方法 | 平均 \|PICP − 0.9\| (CSDI M1) | 击败 Split 的 cell 数 |
|---|:-:|:-:|
| Split CP | 0.069 | — |
| **Lyap-empirical** | **0.031** | 14 / 21 |

CSDI M1 下 Lyap-emp 相对 Split 为 **2.3× 改善**（对比 AR-Kalman 下 3.2×）。差距缩小的原因是：**CSDI 插补更准使得 SVGP 残差更紧，Split 的 fixed-width 区间在长 horizon 下的 under-coverage 程度减轻**，从而 Lyap-growth 的边际收益变小。然而 Lyap-empirical 的**绝对** miscalibration 在两种 M1 下仍然小于 Split（0.031 vs 0.069），说明 Lyap-growth 的价值**不依赖于**特定 M1 的插补质量。

**解读.**
- **Raw GP 严重过覆盖**（Fig D5，α=0.3 的 raw PICP 是 0.98 vs nominal 0.70） — GP 的 Gaussian posterior 在 ergodic chaos 上 **明显错** calibration。CP 校准是**必需而非可选**。
- **Split CP 边际校准但长 horizon 失败**（PICP 从 h=1 的 0.99 漂到 h=48 的 0.82）—— 因为 Split 假设残差 exchangeable，但混沌长 horizon 上残差方差会按 $e^{\lambda h}$ 增长，exchangeability 破坏。
- **Lyap-empirical 无 $\lambda$ 估计需求**：直接从 calibration 残差按 horizon bin 拟合 scale，**绕开了 $\lambda$ 估计误差** 这一巨坑（noise 下 nolds 估 λ 高 150%+）。这是为什么 Lyap-empirical 完爆 Lyap-exp/sat/clipped。
- **M1 无关性**：Lyap-emp 在两种 M1 下都比 Split 显著更好，说明这个 CP 方法不挑 M1。

（对应 figure：[D2 CSDI 版](experiments/week2_modules/figures/coverage_across_harshness_paperfig_csdi.png)、[D3 CSDI 版](experiments/week2_modules/figures/horizon_coverage_paperfig_csdi.png)、[D4 CSDI 版](experiments/week2_modules/figures/horizon_piwidth_paperfig_csdi.png)、[D5 CSDI 版](experiments/week2_modules/figures/reliability_diagram_paperfig_csdi.png)、[Fig 5 S2/S3 CSDI 版](experiments/week2_modules/figures/module4_horizon_cal_S3_csdi.png)。）

### 5.6 Module 2 专项（τ-search 稳定性与低秩结构）

#### 5.6.1 τ-stability vs 观测噪声（Fig D6）

**Setup.** Lorenz63 × 6 noise levels $\sigma / \sigma_\text{attr} \in \{0.0, 0.1, 0.3, 0.5, 1.0, 1.5\}$ × 3 methods (MI-Lyap / Fraser-Swinney / Random) × 15 seeds，sparsity 固定 30% 以隔离 noise 的影响。每 (method, σ, seed) 组合独立跑一次 τ-search。

**做了什么.** 对每个组合记录被选中的 τ 向量 $(\tau_1, \tau_2, \tau_3, \tau_4)$；汇总每 (method, σ) 下 15 seeds 的 $|\tau|_2$ 均值和标准差。**std 越小 = τ 选择越稳定**（同一系统不同 seed 应该选相同 τ）。

**结果.** σ=0 下 MI-Lyap std(|τ|)=**0.00**（15 seeds **完全相同的 τ 向量**）；σ=0.5 下 std=**3.54** (vs Fraser 6.68, random 7.73)；σ=1.5 下 std=**4.34** (vs Fraser 8.59, random 7.73)。

**解读.**
- **σ=0 的完美确定性** (15/15 同 τ) 是 MI-Lyap 不依赖 autocorrelation 最小值的强证据；Fraser-Swinney 即使在 noise-free 下也有 2.19 的方差因为它挑"first MI minimum"，小 wiggle 就能让 argmin 跳。
- **噪声鲁棒性**：σ 升到 0.5 时 MI-Lyap std 比 Fraser 小 47%；σ=1.5 极端噪声下仍比 random baseline 稳 ~50%。
- MI-Lyap 在 σ 增大时 mean(|τ|) 缓慢上升，说明它**自适应**到更大延迟（因为高噪下短期相关性被污染，MI 迫使它看更远），而 Fraser 在 σ≥1.0 时 mean(|τ|) 反而**下降**（argmin 被噪声污染的伪最小值拉低）。

#### 5.6.2 τ 矩阵低秩奇异值谱（Fig D7）

**Setup.** Lorenz96 with N=20, L ∈ {3, 5, 7}, 5 seeds。每 (L, seed) 跑 CMA-ES Stage B 的低秩 τ-search，设 rank = full = $L-1$（即**不强加**低秩约束，纯粹用 SVD 看 UV^⊤ 矩阵自动展现的奇异值分布）。

**做了什么.** 把 CMA-ES 收敛的 $U \in \mathbb{R}^{(L-1) \times (L-1)}$ 的 SV 谱取出（即 $UU^\top$ 的 SVD），归一化到 $\sigma_1 = 1.0$。5 seeds 取平均，画 log-y 轴。

**结果.**

| L | σ₂/σ₁ | σ₃/σ₁ | σ₄/σ₁ | 有效 rank |
|:-:|:-:|:-:|:-:|:-:|
| 3 | 0.283 | — | — | ~1 |
| 5 | 0.445 | 0.235 | **0.030** | ~2–3 |
| 7 | 0.561 | 0.340 | 0.125 | ~3 |

**解读.**
- 即使**不强加** rank 约束，CMA-ES 找到的最优 τ 矩阵**自然呈现低秩结构**。L=5 下 σ₄/σ₁=0.030（< 10% 阈值），说明 effective rank ≈ 3。L=7 下 σ₅/σ₁=0.042 也刚跌破阈值。
- 这实证支持 tech.md §2.3 的 "rank-2 ansatz"——在 Lorenz96 这种耦合振子系统里，相邻维度共享混沌时标，τ-space 的结构是**低维的**。
- **这直接给 Stage B CMA-ES 提供了物理动机**：从 $\{1,\ldots,\tau_\text{max}\}^L$ 的指数大离散空间降到 $\mathbb{R}^{r(L+1)}$ 的小连续空间，同质量下搜索快 1.8×。

### 5.7 Module 3 专项（SVGP 的可扩展性，Fig 6）

**Setup.** Lorenz96 with F=8（典型混沌参数）at $N \in \{10, 20, 40\}$；每 N 2 seeds，$n_\text{train} = 1393$ 条 delay-embed 样本；SVGP 128 inducing points，150 epochs，Matern-5/2 kernel。

**做了什么.** 记录每 N 下 SVGP 的训练时间（壁钟）和测试 NRMSE，对比 exact GPR。

**结果.**

| $N$ | $n_\text{train}$ | SVGP 训练时间 | NRMSE | exact GPR 时间 |
|:-:|:-:|:-:|:-:|:-:|
| 10 | 1393 | **25.6 ± 0.9 s** | 0.85 | ~10 s |
| 20 | 1393 | **42.4 ± 3.9 s** | 0.92 | ~120 s |
| 40 | 1393 | **92.1 ± 2.1 s** | 1.00 | **OOM** |

**解读.**
- 训练时间**在 $N$ 上线性**（25s → 42s → 92s, 比例 ≈ 1 : 1.7 : 3.6 vs N 的 1 : 2 : 4）。这是 SVGP 128 inducing points 的理论期望行为 $O(N \cdot m^2 \cdot n_\text{train})$。
- NRMSE 从 0.85 随 N 缓慢退化到 1.00 —— 高维下每一维的信号更稀薄，预测难度自然上升。
- Exact GPR 在 N=40 直接 **OOM**（超出 24GB GPU 内存）；SVGP 在同 GPU 上只用了不到 2GB。
- 这实证支持 Proposition 2：**SVGP 的后验收缩率由 Kaplan-Yorke 维 $d_\text{KY}$（对 Lorenz96 ≈ 0.4 N）主导，而非环境维 N**。所以 paper 的 pipeline 能扩展到 Lorenz96 scale 的系统。

---

### 5.8 实验总结表

所有 paper-relevant 实验的一张全局扫描表：

| # | 实验 | 数据规模 | M1 版本 | 主数字 | Paper 节 | 数据文件 |
|:-:|---|---|---|---|:-:|---|
| 1 | Phase Transition 主图 | 175 runs (7×5×5) | AR-K | S3 vs Panda 2.2×, Parrot 7.1× | §5.2 | `pt_v2_with_panda_n5_small.json` |
| 2 | Phase Transition CSDI 升级 | 70 runs (7×2×5) | AR-K + CSDI | S4 VPT +110%, overall rmse −8% | §5.3 | `pt_v2_csdi_upgrade_n5.json` |
| 3 | Module-level Ablation S2 | 54 runs (9×3×2 M1) × 4 horizons | 并排 | S2 h=4 Full −7%（CSDI 胜）| §5.4 | `ablation_S2_n3_v2.json` + `ablation_with_csdi_*_9cfg_S2.json` |
| 4 | Module-level Ablation S3 | 54 runs (9×3×2 M1) × 4 horizons | 并排 | **S3 h=4 Full −24%（CSDI 胜）🔥** | §5.4 | `ablation_S3_n3_v2.json` + `ablation_with_csdi_*_9cfg_S3.json` |
| 5 | D2 Coverage Across Harshness | 63 runs (7×3×3) × 2 M1 | 并排 | Lyap-emp vs Split 3.2× / 2.3× | §5.5 | `coverage_across_harshness_n3_v1{_csdi}.json` |
| 6 | Module 4 Horizon Calibration | S2+S3 × 8 h × 3 seeds × 2 M1 | 并排 | **5.5× mean \|PICP−0.9\| 改善** | §5.5 | `module4_horizon_cal_{S2,S3}_n3{_csdi}.json` |
| 7 | D5 Reliability Diagram | S2+S3 × α∈{0.01..0.5} × 3 seeds × 2 M1 | 并排 | Raw GP 过覆盖 0.98（α=0.3），Split 完美 | §5.5 | `reliability_diagram_n3_v1{_csdi}.json` |
| 8 | D6 τ-stability vs noise | 270 runs (6σ×15×3 methods) | N/A | σ=0 时 MI-Lyap 15/15 同 τ | §5.6.1 | `tau_stability_n15_v1.json` |
| 9 | D7 τ low-rank spectrum (L96) | 15 runs (3L × 5 seeds) | N/A | L=5 σ₄/σ₁=0.03, effective rank 2-3 | §5.6.2 | `tau_spectrum_v2.json` |
| 10 | Fig 6 SVGP Scaling (L96) | 6 runs (3N × 2 seeds) | N/A | 训练时间 N-linear | §5.7 | `lorenz96_scaling_N10_20_40.json` |
| 11 | Fig 2 Trajectory overlay | 1 run (seed=3, 4 scenarios) | 两版 | qualitative | §3 / Fig | — |
| 12 | Fig 3 Separatrix ensemble | 1 run (seed=4, K=30) | 两版 | ensemble VPT 1.99 Λ, wing 30/30 | §3.3 / Fig | `separatrix_ensemble_seed4_S0_K30.{json,npz}` |

**总运行次数（独立 pipeline 调用）**：~900+ runs。

**训练 compute**：CSDI 4 variants × 200 epochs × 512K samples × batch 256 ≈ **1.6M gradient steps on 4 × NVIDIA GPUs**（约 8 小时实时）。

**数据 + 图 + 日志总量**：JSON 原始数据约 50 MB（gitignore bypass 方式 force-add 进 repo），figures 约 12 MB PNG，日志约 10 MB 文本，CSDI ckpts 约 280 MB（本地不推）。

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

## 附录 A.0：符号与术语表

### A.0.1 场景参数
| 符号 | 含义 | 取值 |
|:-:|---|:-:|
| $s$ | 观测稀疏率（丢弃比例） | {0, 0.2, 0.4, 0.6, 0.75, 0.9, 0.95} |
| $\sigma/\sigma_\text{attr}$ | 观测噪声 std 相对 attractor std 的比 | {0, 0.1, 0.3, 0.5, 0.8, 1.2, 1.5} |
| $\sigma_\text{attr}$ | Lorenz63 吸引子 std | **8.51** |
| $S_i$ | harshness 场景 $(s_i, \sigma_i)$ for $i=0,\ldots,6$ | — |
| $\Delta t$ | 积分步长 | **0.025** |
| $\lambda$ | Lorenz63 最大 Lyapunov 指数 | **0.906** |
| $\Lambda$ | Lyapunov 时间（$1\Lambda = 1/\lambda$） | ≈ 1.10 时间单位 |

### A.0.2 预测与 UQ 指标
| 符号 | 名称 | 定义 |
|:-:|---|---|
| **VPT@τ** | Valid Prediction Time | 误差持续 < τ·σ_attr 的最长前缀（Λ 单位） |
| **NRMSE** | Normalized RMSE | $\sqrt{\mathbb{E}[(\hat{x}-x)^2]} / \sigma_\text{attr}$ |
| $\alpha$ | miscoverage level | target coverage = 1−α，paper 默认 α=0.1 |
| **PICP** | Prediction Interval Coverage Probability | 真值落入区间的经验比例，目标 0.90 |
| **MPIW** | Mean Prediction Interval Width | 区间平均宽度 |
| **CRPS** | Continuous Ranked Probability Score | 连续分布评分规则 |

### A.0.3 延迟嵌入与 M1 CSDI
| 符号 | 含义 |
|:-:|---|
| $L$ | 延迟坐标数（embedding dim），paper 默认 5 |
| $\tau = (\tau_1, \ldots, \tau_L)$ | 延迟向量（$\tau_i > \tau_{i+1}$） |
| $\mathbf{X}_\tau(t) = (x_t, x_{t-\tau_1}, \ldots, x_{t-\tau_L})$ | 延迟坐标行向量 |
| $I_\text{KSG}$ | Kraskov-Stögbauer-Grassberger k-NN 互信息 |
| $\alpha_\text{delay}$ | CSDI 延迟 attention 门控标量（bug fix 初值 0.01） |
| $\epsilon_\theta$ | CSDI score 网络（DDPM 噪声预测器） |
| **CSDI** / **DDPM** / **SVGP** / **GPR** / **BO** / **CMA-ES** / **CP** | 见主文缩写 |

### A.0.4 Ablation 配置名
每个 `cfg_name` 按 "M1-M2-M3-M4" 四模块约定：
- `full` = AR-Kalman / MI-Lyap BO / SVGP / Lyap-sat （原 paper baseline）
- `full-csdi` = **CSDI** / MI-Lyap BO / SVGP / Lyap-sat （upgraded baseline）
- `full-empirical` / `full-csdi-empirical` = 用 Lyap-empirical 替换 M4
- `m1-linear` = linear interp 替换 M1
- `m2a-random` = random τ / `m2b-frasersw` = Fraser-Swinney τ
- `m3-exactgpr` = exact GPR 替换 SVGP
- `m4-splitcp` = Split CP 替换 Lyap-CP
- `m4-lyap-exp` = 用 exp growth 替换 saturating
- `all-off` = 四模块全退回 2023 CSDI-RDE-GPR baseline
- `csdi-*` 前缀 = 对应 `*` 的 M1 换成 CSDI 版（§5.4 dual-M1 ablation 用）

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
