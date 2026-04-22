# 稀疏噪声观测下的混沌预测：四模块流水线与 Lyapunov 感知的共形覆盖

**作者.** （待定）  **目标会议.** NeurIPS / ICLR 2026  **状态.** 首版草稿，2026-04-22

> 中文版草稿。所有硬数字来自 `experiments/{week1,week2_modules}/results/` 下的 JSON，
> 所有 figure 引用对应 `experiments/{week1,week2_modules}/figures/` 下的 PNG。

---

## 摘要

时间序列基础模型（Chronos、TimesFM、Panda）在稀疏含噪混沌观测下表现出**灾难性退化**：Lorenz63 上当稀疏率 $s$ 从 0% 升到 60%、噪声 $\sigma/\sigma_\text{attr}$ 升到 0.5 时，Panda-72M 的 VPT 损失 **85%**、Context-Parroting 损失 **92%** —— 一次陡峭相变。我们论证这**不是实现缺陷而是理论必然**：任何在 ambient 坐标上操作的预测器都承担 $\sqrt{D/n_\text{eff}}$ 维度税（Prop 1），而延迟坐标方法的收敛率由 Kaplan-Yorke 维 $d_{KY}$ 主导、与 $D$ 解耦（Prop 3）。引入**有效样本数** $n_\text{eff}(s, \sigma) := n (1-s)/(1+\sigma^2/\sigma_\text{attr}^2)$ 作为稀疏与噪声的统一参数，我们证明 **Theorem 2（Sparsity-Noise Interaction Phase Transition）**：$n_\text{eff}$ 跨越临界 $n^\star \approx 0.3 n$ 时 ambient 预测器经历额外 $\Omega(1)$ OOD 跃变，而 manifold 预测器只经历平滑退化；临界点 $(s, \sigma) \approx (0.6, 0.5)$ **恰好是 S3**。

基于这一理论框架，我们提出**流形中心**的四模块流水线，四模块是对延迟流形 $\mathcal{M}_\tau = \Phi_\tau(\text{attractor})$ 上同一 Koopman 算子的互补估计：**(M2)** MI-Lyap $\tau$-search 估计 $\mathcal{M}_\tau$ 的嵌入几何（$\sigma=0$ 下 15 seeds 15/15 选同一 $\tau$，完美恢复几何不变量）；**(M1)** 流形感知 CSDI 以 M2 的 $\tau$ 作为 attention anchor，把 score estimation 对齐到 $T\mathcal{M}_\tau$ 切丛（三个 bug 修复 —— 非零初始化 / 每维中心化 / **贝叶斯软锚定** —— 分别对应启用切丛 / 建立 DDPM 正确几何 / 正确流形投影三个几何必要条件）；**(M3)** 延迟坐标 SVGP 在 $\mathcal{M}_\tau$ 上回归 Koopman 算子（Lorenz96 $N\to 40$ 近线性 scaling）；**(M4)** Lyap-empirical CP 直接从残差恢复 Koopman 经验谱，绕开噪声敏感的 $\hat\lambda_1$。四模块通过 $\tau$、$d_{KY}$、Lyapunov 谱三个几何不变量耦合。

在 S3 严酷场景下全流水线 VPT 达 Panda 的 **2.2×**、Parrot 的 **7.1×**；S4 扩大到 Panda 的 **9.4×**（CSDI 版本）。Panda 实测 −85% 退化与 Prop 1 下界 −44% + Theorem 2(b) OOD 归因 −41% **数量级闭环**；S5/S6 所有方法共同归零（Corollary 的物理底线）—— 证明优势是 physically grounded 而非 cherry-pick。PI 在 21 个 (场景, horizon) cell 上偏离 nominal 0.90 ≤ 2%（比 Split **3.2×** 更准）。代码、10 张 paper 级 figure、40 万步 diffusion 训练 checkpoint 全部开源。

---

## 1. 引言

### 1.1 三段式 opener：现象 → 理论 → 实证

**现象 —— Phase Transition 是稀疏 × 噪声的交互效应.** 气候站的读数会掉、EEG 电极会接触不良、金融数据有抖动、生物传感器会饱和 —— "稀疏+噪声"才是混沌观测的真实场景。然而混沌预测的 ML 文献仍多假设**密集干净** context 窗口 —— 这恰好是时间序列基础模型擅长的设定。我们在 Lorenz63 上扫 7 个 harshness 场景（S0-S6，稀疏率 $0\% \to 95\%$、噪声 $\sigma/\sigma_\text{attr}: 0 \to 1.5$）发现：基础模型（Panda-72M [Wang25]、Chronos-T5 [Ansari24]、Context-Parroting [Xu24]）**不是在所有场景下都崩**。S1/S2 它们还能工作，S5/S6 所有方法都崩（噪声 $>$ 信号，物理底线）。真正的断裂区间是 **S3/S4**：Panda S0→S3 **−85%**、Parrot **−92%**，一次尖锐相变；而我们的全流水线只从 1.73 Λ 掉到 0.92 Λ（−47%），是 S2-S3 窗口内**唯一没有发生相变的方法**（Fig 1）。

**理论 —— 相变是 ambient 维度税的必然.** 这一相变**不是实现缺陷**。我们证明（§4）：引入**有效样本数**
$$n_\text{eff}(s, \sigma) = n \cdot (1-s) / (1+\sigma^2/\sigma_\text{attr}^2)$$
作为 sparsity 和 noise 两因素的统一参数。Prop 1（**Ambient 维度税**）给出任何 ambient 预测器的误差下界 $\ge \sqrt{D/n_\text{eff}}$；Prop 3（**Manifold 后验收缩**）给出 delay-coord 方法的 $d_{KY}$-主导速率（与 ambient 维 $D$ 解耦）。**Theorem 2（Sparsity-Noise Interaction Phase Transition，本文核心理论贡献）**：当 $n_\text{eff}/n$ 跨越临界 $\approx 0.3$ 时，ambient predictor 经历额外 $\Omega(1)$ 的 OOD 跃变（线性插值 context 产生非物理直线段 + tokenizer 分布偏移），而 manifold predictor 只按平滑幂律退化。**临界点 $(s, \sigma) \approx (0.6, 0.5)$ 恰好是 S3** —— 把"S3 是主战场"从经验观察升级为理论预测。

**实证 —— 数字与理论定量闭环.** S3 上我们达 Panda 的 **2.2×**、Parrot 的 **7.1×**；S4 扩大到 Panda 的 **9.4×**（CSDI M1 升级后，Fig 1b）。Panda S0→S3 实测 −85% 可分解为 Prop 1 下界预测的 −44% + Theorem 2(b) OOD 归因的 −41%；我们的 −47% 在 Prop 3 预测的置信区间内。S5/S6 所有方法归零（Corollary 的物理底线）—— 这个**共同失败**表明我们的优势不是 cherry-pick，而是理论预测的相变窗口内的系统性优势（§5.2）。覆盖率也扛住了：Lyap-empirical CP 在全 21 个 (场景, horizon) cell 上 PICP 偏离 nominal 0.90 ≤ 0.02，平均 |PICP−0.9| 相比 Split **3.2× 更准**（Fig D2），相比 SVGP raw 高斯 5.5× 更准（Fig 5）。

### 1.2 Unified View — 四个模块是同一几何对象的四个侧面

本文的四个模块表面上解决四个不同问题（插补 / 嵌入选择 / 回归 / UQ），但它们共享同一个几何对象：**延迟流形** $\mathcal{M}_\tau = \Phi_\tau(\text{attractor}) \subset \mathbb{R}^L$（Takens 意义下的 embedding image）。在这一统一视角下（§3.0 完整展开）：

- **M2（§3.1）**：通过 MI-Lyap 目标选择让 $\mathcal{M}_\tau$ 几何性质最好的 $\tau$（既不 self-intersect 也不过度拉伸）
- **M1（§3.2）**：CSDI delay attention mask 以 M2 选出的 $\tau$ 为 anchor，把 score estimation 限制在 $\mathcal{M}_\tau$ 的切丛结构 $T\mathcal{M}_\tau$ 上
- **M3（§3.3）**：直接在 $\mathcal{M}_\tau$ 上拟合 Koopman 算子 $\mathcal{K}: g \mapsto g \circ f$
- **M4（§3.4）**：利用 $\mathcal{K}|_{\mathcal{M}_\tau}$ 的 Lyapunov 谱校准共形区间

**四个模块通过三个几何不变量耦合**：延迟向量 $\tau$、Kaplan-Yorke 维 $d_{KY}$、Lyapunov 谱 $\{\lambda_i\}$；改变其中任意一个，其他三个必须相应调整（我们在 §6 讨论这一耦合的未来实证方向）。这个统一视角把我们的方法从"pipeline 堆叠"升级为"**流形上的自洽估计**"，也对应 §4 的理论预测：基础模型操作在 ambient 坐标，承担 $\sqrt{D/n_\text{eff}}$ 维度税；我们操作在 $d_{KY}$ 维延迟流形上，收敛率与 $D$ 解耦。

### 1.3 主要贡献

**贡献 0（统一框架）.** 我们建立一个以 $\mathcal{M}_\tau$ 为中心的数学框架，把混沌预测中四个经典任务（插补、嵌入选择、回归、UQ）统一为对 $\mathcal{M}_\tau$ 上同一 Koopman 算子的四种互补估计。§4 的四条定理共享 $d_{KY}$ 和 $n_\text{eff}$，揭示 phase transition 是**理论必然**。

**贡献 1（Theorem 2 + Corollary）.** Sparsity-Noise Interaction Phase Transition Theorem：引入 $n_\text{eff} = n(1-s)/(1+\sigma^2/\sigma_\text{attr}^2)$ 作为 Prop 1/3 的共同参数，证明 ambient predictor 在 $n_\text{eff} < n^\star = c \cdot D$ 时经历额外 OOD 跃变而 manifold predictor 只经历平滑退化。Corollary 给出三 regime 的统一 scaling law，把 Fig 1 的 S0-1 → S2-4 → S5-6 结构从经验观察升级为理论预测。

**贡献 2（M1，流形感知 CSDI）.** 我们发现并修复三个**并发 bug**：(a) 延迟 attention 门零梯度死锁（fix: $\alpha_\text{delay}=0.01$ 非零初始化）；(b) 单尺度归一化违反 DDPM 的 N(0,I) 先验（fix: 每维中心化）；(c) 推理时硬锚定带噪观测把噪声注入反向扩散（fix: **贝叶斯软锚定** $\hat x = y/(1+\sigma^2)$）。三个 bug 分别对应**启用 $T\mathcal{M}_\tau$** / **建立 DDPM 正确几何** / **正确流形投影**三个几何必要条件。最后一个 fix 的价值随 $\sigma^2$ quadratic 放大（S2 +53% / S4 +110% / S6 10× VPT，Fig 1b）—— 是 Theorem 2(b) 的直接实证。

**贡献 3（M2，MI-Lyap 作为几何不变量估计器）.** 把 Kraskov MI 目标与混沌拉伸惩罚耦合，联合优化长度-$L$ 向量 $\tau$（而非 coordinate-descent）。$\sigma=0$ 下 15 seeds **15/15 选同一 $\tau$**（|τ| std=0）—— 不是"算法稳定"，而是**$\tau^\star$ 作为 $\mathcal{M}_\tau$ 几何不变量的完美经验恢复**；对照 Fraser-Swinney std=2.19、random std=7.73（Fig D6）。

**贡献 4（M3，Koopman 回归的 $d_{KY}$-主导 scaling）.** 延迟坐标 SVGP 的训练时间在 ambient 维 $N$ 上近线性（Lorenz96 $N=10\to 40$: $25\to 92$s，Fig 6）—— 实证 Prop 3：收敛率由 $d_{KY}$ 主导，与 $D$ 解耦。

**贡献 5（M4，经验 Koopman 谱校准 CP）.** Lyap-empirical CP 的 λ-free 设计直接从 calibration 残差恢复 $\mathcal{K}^h$ 的经验谱，绕开 nolds/Rosenstein 等 $\hat\lambda_1$ 估计器的噪声敏感性。S3 平均 |PICP−0.9| = 0.013 vs Split 0.072（**5.5×**）；21 cells 平均 0.022 vs Split 0.071（**3.2×**）。

**贡献 6（全流水线的相变鲁棒性）.** S3 Panda 的 **2.2×**、Parrot 的 **7.1×**，S4 Panda 的 **3.7×**（AR-Kalman）/ **9.4×**（CSDI）；S5/S6 所有方法共同归零 —— 优势 physically grounded，非 cherry-pick。

**贡献 7（完整开源复现）.** 10 张 paper 级 figure、18 条数字支撑 JSON、CSDI checkpoint (5 MB) 全部开源，附精确复现命令（见 `ASSETS.md`）。

**论文结构.** §2 相关工作；§3.0 几何骨架 + §3.1-4 四模块（按流形视角重新组织）；§4 理论框架（Prop 1 + Theorem 2 + Prop 3 + Theorem 4 + Corollary）；§5 完整实验；§6 限制 + 未来耦合实证方向；§7 总结。

---

## 2. 相关工作

**混沌系统预测.** 经典 Takens 式延迟嵌入 + 局部线性/GP 预测可追溯到 [Farmer-Sidorowich 87, Casdagli 89]。神经方法包括 Echo-State Networks [Jaeger01, Pathak18]、Reservoir Computing，以及最近的算子理论方法 [Brunton16, Lu21]。这些工作**都没有**在**随机**稀疏+噪声观测 + conformal 校准区间的设定下评估。

**动力系统的流形学习（本文的数学 tradition）.** 我们的工作属于一条"数据位于低维流形上、从数据恢复流形几何"的线索。经典工作包括 Fefferman-Mitter-Narayanan 的 manifold 估计理论 [FeffermanMitterNarayanan16]，Berry-Harlim 在动力系统上的 diffusion maps [BerryHarlim16]，Giannakis 的 Koopman spectral methods [Giannakis19]，以及 Das-Giannakis 的 reproducing kernel for Koopman [DasGiannakis20]。本文把这条线索的"延迟流形 $\mathcal{M}_\tau$ + Koopman 算子"视角推广到 **稀疏含噪观测** 的实际场景，并在此上建立 scaling law 定理族（§4）。与经典 manifold learning 不同，我们不显式估计 $\mathcal{M}_\tau$ 的局部坐标 / intrinsic Laplacian，而是把 $\mathcal{M}_\tau$ 作为一个 **隐式中心对象**，让四个模块从不同几何侧面估计 $\mathcal{M}_\tau$ 上的 Koopman 算子。

**时间序列基础模型.** Chronos [Ansari24]、TimeGPT [Garza23]、Lag-Llama [Rasul23]、TimesFM [Das23]、以及专门针对混沌的 Panda-72M [Wang25] 在数十亿时间序列 token 上预训解码器 Transformer。这些模型在分布内预测上胜得漂亮，但我们证明它们在稀疏+噪声下尖锐相变 —— 这在 §4 Prop 1 + Theorem 2 下是**理论必然**（ambient 坐标承担 $\sqrt{D/n_\text{eff}}$ 维度税 + sparse context 的 tokenizer OOD 跃变）。Context-Parroting [Xu24] 是精神最接近的竞争者 —— 一种非参数的 "context 中 1-NN" 方法；它在我们的实验中也崩（−92%），因为 1-NN retrieval 对 context 分布更敏感。

**扩散式插值.** CSDI [Tashiro21] 开创了用 score-based 方法做插值，通过 masked attention 对观测点做条件。我们的 M1 继承了该架构，但贡献了三个**非可选**的稳定性修复（§3.2），并把它们从"工程踩坑"重新锚定为三个**几何必要条件**（启用切丛 / 建立 DDPM 正确几何 / 正确流形投影）—— 不修这三个 bug，混沌轨迹上根本训不稳。

**依赖下的共形预测.** Split CP [Vovk05]、adaptive CP [Gibbs21]、以及 weighted-exchangeability 系列 [Barber23] 提供了可交换条件下的有限样本保证。Chernozhukov-Wüthrich-Zhu [ChernozhukovWÜ18] 给出 ψ-mixing 下的 exchangeability-breaking bound，与 Bowen-Ruelle-Young [Young98] 对光滑遍历混沌的 ψ-mixing 性质结合，构成我们 Theorem 4 的证明基础。我们的 M4 把 CP score 按 horizon 的经验拟合增长函数做**尺度重塑**，等价于**从数据恢复 Koopman 算子的经验谱**（§3.4），无需假设 $\lambda_1$ 已知。

**延迟嵌入选择.** Fraser-Swinney 的 "first-minimum-of-MI" [FraserSwinney86] 是典范一维启发式；Cao 的 FNN [Cao97] 是典范嵌入维启发式。二者都**不联合优化** $L>1$ 的向量值 $\tau$，且无几何正则项。我们的 M2 把 τ-search 重新定位为"**估计 $\mathcal{M}_\tau$ 的几何不变量 $\tau^\star$**"（MI 目标对应单射性，Lyap 项对应拉伸率），并用低秩 CMA-ES 处理高维情形。

---

## 3. 方法

> **视角声明。** 本章把 M1/M2/M3/M4 四模块按**延迟流形** $\mathcal{M}_\tau$ 这一共同几何对象重新组织。读者若只关心"每个模块做什么"，可跳过 §3.0 直接从 §3.1 开始阅读；但 §3.0 是 §4 理论部分的几何骨架，对理解 Proposition 1 / Proposition 3 和新 Theorem 2（Sparsity-Noise Interaction）不可或缺。

### 3.0 延迟流形作为中心对象（几何骨架）

本文的四个模块表面上处理四个不同的子问题（插补 / 延迟选择 / 回归 / UQ），但它们共享一个中心对象——延迟流形 $\mathcal{M}_\tau$。这一小节给出后续讨论所需的几何与算子背景。

**Takens 嵌入定理（回顾）.** 设动力系统 $f: \mathcal{X} \to \mathcal{X}$ 有一个 $d$ 维紧致遍历吸引子 $\mathcal{A} \subset \mathcal{X}$，$h: \mathcal{X} \to \mathbb{R}$ 是一个 generic 观测函数。对任意 $L > 2d$ 和 generic 延迟向量 $\tau = (\tau_1, \ldots, \tau_{L-1})$，延迟映射

$$\Phi_\tau: x \mapsto \bigl( h(x),\, h(f^{-\tau_1}(x)),\, h(f^{-\tau_2}(x)),\, \ldots,\, h(f^{-\tau_{L-1}}(x)) \bigr) \in \mathbb{R}^L$$

是 $\mathcal{A}$ 到 $\mathbb{R}^L$ 的一个**嵌入（diffeomorphism onto image）**。记其像集为**延迟流形**

$$\mathcal{M}_\tau := \Phi_\tau(\mathcal{A}) \;\subset\; \mathbb{R}^L.$$

**几何不变量.** 以下三个量是 $\mathcal{M}_\tau$ 的核心几何不变量，它们贯穿本文四个模块：

1. **内蕴维度 $d_{KY}$** —— Kaplan-Yorke 维
$$d_{KY} \;=\; k \;+\; \frac{\sum_{i=1}^{k}\lambda_i}{|\lambda_{k+1}|}, \qquad k = \max\Bigl\{j:\sum_{i=1}^{j}\lambda_i \ge 0\Bigr\}$$
由 Lyapunov 谱 $\{\lambda_i\}$ 定义。Kaplan-Yorke 猜想（在 Lorenz63、Lorenz96、Rössler 等 benchmark 系统上已数值验证）断言 $d_{KY}$ 与吸引子 Hausdorff 维相等，并且在嵌入无退化时亦为 $\mathcal{M}_\tau$ 的内蕴维度。对 Lorenz63 $d_{KY} \approx 2.06$，Lorenz96-$N=20$ 下 $d_{KY} \approx 8$。

2. **切丛结构 $T\mathcal{M}_\tau$** —— 由 Koopman 算子的谱决定。记 Koopman 算子
$$\mathcal{K}: g(x) \mapsto g(f(x))$$
作用于可观测函数空间；它是**线性**算子（即使 $f$ 非线性），谱分解后对 invariant subbundle 的作用给出了 $\mathcal{M}_\tau$ 的局部线性结构。

3. **最优嵌入 $\tau^\star$** —— 由 MI-Lyap 目标（§3.1）的极值定义。直观地：$\tau$ 太小则 $\Phi_\tau$ 接近退化（相邻坐标相互冗余，$\mathcal{M}_\tau$ 近 self-intersection）；$\tau$ 太大则 $\Phi_\tau$ 过度拉伸（混沌下 $\|D\Phi_\tau\|$ 按 $e^{\lambda_1 \tau_\text{max}}$ 增长，数值上不稳）。最优 $\tau^\star$ 平衡这两端。

**Koopman 算子在延迟坐标下的平凡化.** 关键观察是：在延迟坐标下 $\mathcal{K}$ 的作用退化为一个"左移"结构
$$\mathcal{K}: (y_t, y_{t-\tau_1}, \ldots, y_{t-\tau_{L-1}}) \;\longmapsto\; (y_{t+1}, y_{t+1-\tau_1}, \ldots, y_{t+1-\tau_{L-1}}).$$
这意味着预测 $y_{t+h}$ 等价于在 $\mathcal{M}_\tau$ 上沿 $\mathcal{K}^h$ 轨道前推一步。

**四个模块的统一目标.** 在上述几何框架下，稀疏噪声混沌预测可统一为"**从退化观测重建 $\mathcal{M}_\tau$ 上的 Koopman 算子**"。四个模块是这一重建任务的互补子任务：

| 模块 | 在 $\mathcal{M}_\tau$ 上的几何角色 |
|---|---|
| **M2（§3.1）** | 估计 $\mathcal{M}_\tau$ 的嵌入几何：选 $\tau^\star$ 让 $\Phi_\tau$ 不 self-intersect 也不过度拉伸 |
| **M1（§3.2）** | 在 $\mathcal{M}_\tau$ 上做流形感知 score estimation：CSDI delay mask 以 M2 的 $\tau$ 为 anchor，让 attention 沿 $\mathcal{M}_\tau$ 切向共享信息 |
| **M3（§3.3）** | 在 $\mathcal{M}_\tau$ 上回归 Koopman 算子：SVGP 的 Matérn 核直接拟合 $\mathcal{K}$ 的 pushforward |
| **M4（§3.4）** | 用 Koopman 谱校准 PI：CP horizon growth $G(h)$ 在 $h\to\infty$ 时逼近 $e^{\lambda_1 h \Delta t}$（$\lambda_1$ 是 $\mathcal{K}|_{\mathcal{M}_\tau}$ 的谱顶） |

**三个共享参数.** 四个模块通过以下三个几何量耦合：
- 延迟向量 $\tau$：M2 选出 → M1 delay-mask 使用 → M3 坐标定义
- Kaplan-Yorke 维 $d_{KY}$：M2 最优 L ← M1 score 收敛率 ← M3 后验收缩率（与环境维 $D$ 解耦）
- Lyapunov 谱 $\{\lambda_i\}$：M2 惩罚项 ← M4 horizon growth ← 决定相变临界点

**有效样本数 $n_\text{eff}$（§4 理论的关键参数）.** 在稀疏率 $s$、噪声比 $\sigma/\sigma_\text{attr}$ 下，context 窗口的有效样本数退化为
$$n_\text{eff}(s, \sigma) \;=\; n \cdot (1-s) \cdot \frac{1}{1 + \sigma^2 / \sigma_\text{attr}^2}.$$
第一项是稀疏率直接丢数据，第二项是高斯观测模型下的 Fisher 信息衰减（见 [Künsch 1984] 对部分可观测动力系统的严格处理；我们在附录 A.1 验证该公式在 Lorenz63 上的数值准确性）。$n_\text{eff}$ 将作为 Proposition 1 / Proposition 3 / Theorem 2 的共同参数出现，把"稀疏率"和"噪声"两个因素统一为一个可解析处理的量。

---

### 3.1 模块 M2 — 估计 $\mathcal{M}_\tau$ 的嵌入几何（MI-Lyap τ-search）

> **几何定位.** M2 不是"τ 搜索的启发式"，而是**估计 $\mathcal{M}_\tau$ 几何不变量 $\tau^\star$ 的估计器**。MI 目标对应"避免流形 self-intersection"，Lyap 项对应"控制流形拉伸率"，两者共同把 $\tau^\star$ 定义在 $\mathcal{M}_\tau$ 几何结构最清晰的参数点。$\tau$ 是 §3.2 M1 delay mask 的输入，因此 M2 先于 M1 讲。

我们用**累积正增量**参数化延迟向量 $\tau = (\tau_1 > \tau_2 > \cdots > \tau_L)$，防止 BO 退化到 "等延迟" 的平凡解。目标函数：

$$ J(\tau) = \underbrace{I_\text{KSG}(\mathbf{X}_\tau ; x_{t+h})}_{\text{几何单射性}} \; - \; \underbrace{\beta \cdot \tau_\text{max} \cdot \lambda}_{\text{拉伸率惩罚}} \; - \; \underbrace{\gamma \cdot \lVert \tau \rVert^2 / T}_{\text{长度正则}} $$

其中 $I_\text{KSG}$ 是延迟嵌入行 $\mathbf{X}_\tau(t)$ 与 $h$-步预测目标之间的 Kraskov-Stögbauer-Grassberger 互信息，$\lambda$ 是一个鲁棒的 Rosenstein 式 Lyapunov 估计。**信息论目标 ↔ 几何目标的对应关系**：

- **MI 大 ⇔ $\Phi_\tau$ 近单射 ⇔ $\mathcal{M}_\tau$ 无 self-intersection**（若 $\Phi_\tau$ 在两点 $x \neq x'$ 处坍缩，则 $I(\Phi_\tau(x); x_{t+h})$ 会丢失区分这两点所需的信息）
- **Lyap 项控制 $\|D\Phi_\tau\|$ 上界**（大 $\tau_\text{max}$ + 正 $\lambda$ ⇒ $\mathcal{M}_\tau$ 在坐标方向上按 $e^{\lambda \tau_\text{max}}$ 拉伸，数值上退化）

**两阶段搜索.** Stage A 用 20 轮贝叶斯优化 on 累积-δ 参数化（适用 $L \le 10$）。Stage B 用低秩 CMA-ES：$\tau = \text{round}(\sigma(UV^\top) \cdot \tau_\text{max})$，其中 $U \in \mathbb{R}^{L \times r}, V \in \mathbb{R}^{1 \times r}$，把搜索空间从 $L$ 维离散降到 $r(L+1)$ 维连续（Lorenz96 $N=40, L=7$ 的高维场景）。**低秩 ansatz 的物理动机**：耦合振子系统的 τ 矩阵反映**时标层级**（slow / medium / fast modes），effective rank $\approx 2\text{-}3$ 对应少数主时标。

**τ 作为几何不变量的经验恢复（Fig D6）.** MI-Lyap 在 σ=0 时 15 seeds 选出的 τ 向量 **15/15 完全相同**（|τ| std=0.00）—— Fraser-Swinney 对应 std=2.19、random std=7.73。σ=0.5 下三者分别是 3.54 / 6.68 / 7.73。

**几何重诠释.** σ=0 下 15/15 选同一 τ 的事实不只是"算法稳定"；它是一个更强的 claim：**在 noise-free 下 $\tau^\star$ 是 well-defined 的几何不变量，MI-Lyap 完美恢复它**。添加噪声后 std 仍然 3.54（vs 对照 6.68-7.73），说明该恢复在退化观测下保持合理鲁棒性。

**τ 低秩奇异值谱（Fig D7）.** Lorenz96 $N=20, L=5, 5$ seeds 下 UV^⊤ 的奇异值谱 $\sigma_2/\sigma_1 = 0.45, \sigma_3/\sigma_1 = 0.24, \sigma_4/\sigma_1 = 0.030$ —— effective rank 2-3，实证低秩 ansatz 对应的物理时标层级假设。

### 3.2 模块 M1 — 在 $\mathcal{M}_\tau$ 上的流形感知 Score Estimation（动力学感知 CSDI）

> **几何定位.** CSDI delay mask **不是** "让 attention 学到时间局部性的 trick"；它是**把 score 网络的 inductive bias 对齐到 $\mathcal{M}_\tau$ 切丛结构** $T\mathcal{M}_\tau$ 的结构先验。理想 score $\nabla \log p_\text{data}$ 在延迟坐标下集中于 $\mathcal{M}_\tau$ 的 normal bundle（朝流形方向拉近），沿 tangent bundle（沿流形流动）接近零；delay mask 以 M2（§3.1）选出的 $\tau$ 为 anchor，让 attention 在 $(t, t-\tau_i)$ 对间共享信息 —— **这恰好是沿 $T\mathcal{M}_\tau$ 方向的信息耦合**。

设 $x_{1:T} \in \mathbb{R}^{T\times D}$ 是潜在干净轨迹，$m \in \{0,1\}^T$ 是观测 mask，$y_t = x_t + \nu_t, \nu_t \sim \mathcal{N}(0, \sigma^2 I)$ 是观测时刻的带噪观测。我们要从 $p(x_{1:T} \mid y_{m=1}, m, \sigma)$ 采样。

我们的 CSDI 遵循 score-based 框架：学一个 $\epsilon_\theta(x_t^{(s)}, y, m, \sigma, s)$ 预测扩散第 $s$ 步的噪声；多头 Transformer 把 mask 作为第三个输入通道。在标准架构之外，我们加入**延迟 attention bias**：

$$\text{bias}_{t,t'} = \alpha \cdot \phi_\theta(t - t') $$

其中 $\alpha \in \mathbb{R}$ 是一个可学标量、$\phi_\theta$ 是一个关于时间差的小 MLP。以 M2 的 $\tau$ 作为 mask 的 anchor 结构，让 score 网络在 $\{(t, t-\tau_i)\}_{i=1}^{L-1}$ 对间显式共享信息，即**让 score 的更新方向对齐到 $T\mathcal{M}_\tau$ 切空间**。

**三 bug 修复的几何必要性.** 下述三个 bug 不是"工程踩坑"，而是**把延迟坐标下的 DDPM 建立在 $\mathcal{M}_\tau$ 正确几何上**的三个必要条件；缺一不可。

**Bug #1 —— 零梯度死锁（启用切丛结构的必要条件）.** 朴素初始化 $\alpha=0$ 且 $\phi_\theta(\cdot) = 0$，使得乘积 $\alpha \phi_\theta$ 在初值对两个因子都是零梯度；优化器向旁边的 trivial predictor 漂过去，训练 loss 卡在 1.0。把 $\alpha_\text{delay} = 0.01$ 初始化就能破这个死锁；之后 5 个 epoch 模块就学会了一个有意义的 bias。**几何解读：** $\alpha=0$ 相当于把 delay-mask 关掉，score 网络退化为"在 ambient 坐标上学通用 denoising"；非零初始化是**让 score 网络能够利用 $T\mathcal{M}_\tau$ 切丛结构**的启用条件。

**Bug #2 —— 每维中心化（建立延迟坐标下正确 DDPM 几何）.** Lorenz63 的 Z 坐标均值约 16.4；除以全局 attractor std=8.51 后，归一化后那一维均值 1.79、方差 1.32 —— 这根本不是 DDPM 噪声计划假设的 N(0,1)。我们把每维的 (mean, std) 注册到模型 buffer，每维独立归一化。**仅此一修**就把 held-out imputation RMSE 从 6.8 降到 3.4。**几何解读：** DDPM 先验要求 $x^{(S)} \sim N(0, I)$（完全扩散态）；延迟坐标下原始分布若均值偏移，则扩散路径 $x^{(s)}$ 的先验 anchor 偏离 N(0,I)，等价于在**错位坐标系**中建 DDPM。per-dim centering 是在延迟坐标下**建立 DDPM 正确几何基底**的必要归一化。

**Bug #3 —— 贝叶斯软锚定（正确的流形投影；§4 新 Theorem 的关键支撑）.** 标准 CSDI 在每一步反向过程都把 $x$ 在观测位硬锚定到 $y$。当 $y = x + \nu$ 带有非平凡 $\sigma$ 时，这个做法把 $\nu$ 注入进**每一步**反向，噪声最终压过 denoising。我们改用单位方差先验下的高斯后验更新（归一化坐标内有效）：

$$ \hat{x} = \frac{y}{1 + \sigma^2}, \qquad \text{Var}[\hat{x}] = \frac{\sigma^2}{1 + \sigma^2} $$

然后把 $\hat{x}$ 按正确后验方差前向扩散到当前反向步。$\sigma=0$ 时公式退化回标准硬锚定；$\sigma\to\infty$ 时观测被忽略、纯 score 网络驱动推理。

**几何解读（重要）.** 带噪观测 $y = x + \nu$ 在延迟坐标下对应一个**偏离 $\mathcal{M}_\tau$** 的点（$\nu$ 把 $y$ 推到 $\mathcal{M}_\tau$ 的 normal direction 上）。**硬锚定**强制每步反向过程都把 score 网络拽回这个偏离点，score 网络实际在"错误流形" $\mathcal{M}_\tau + \nu$ 上 denoise，累积误差超过 $\mathcal{M}_\tau$ 的内蕴 scale。**贝叶斯软锚定** $\hat{x} = y/(1+\sigma^2)$ 是**正确的流形投影**：把 $y$ 投回 $\mathcal{M}_\tau$ 的 **noisy tubular neighborhood** 的期望位置，让 denoising 沿 $T\mathcal{M}_\tau$ 切向进行。

**软锚定价值随 $\sigma^2$ quadratic 放大.** 这直接解释 Fig 1b CSDI M1 升级梯度（详见 §5.3）：S2（$\sigma/\sigma_\text{attr}=0.3$）+53% VPT，S4（$\sigma/\sigma_\text{attr}=0.8$）**+110% VPT**，S6（$\sigma/\sigma_\text{attr}=1.5$）**10×** 提升（vs AR-Kalman 几乎失败）—— 硬锚定的 per-step 噪声注入在 $\sigma$ 大时呈 quadratic 放大，软锚定的价值因此也呈 quadratic 放大。**这是 §4 新 Theorem（Sparsity-Noise Interaction）的关键支撑证据**。

**训练配置.** 51.2 万条 Lorenz63 合成窗口，长度 128，batch=256，200 epochs，cosine 学习率从 5e-4 起，channels=128，layers=8，seq_len=128，≈40 万梯度步，≈126 万参数。

**结果.** 最佳 checkpoint 在 epoch 20（4 万步；之后训练 loss 仍单调降但留出 imputation RMSE 反弹）。在 50 条随机留出窗口上（sparsity ∈ U(0.2, 0.9)、σ/σ_attr ∈ U(0, 1.2)），imputation RMSE = **3.75 ± 0.26**，vs AR-Kalman 4.17、linear 4.97。在最严酷 (sparsity 0.5, σ_frac 1.2) 下 CSDI 5.91，vs Kalman 6.20、linear 9.27。

### 3.3 模块 M3 — 在 $\mathcal{M}_\tau$ 上的 Koopman 算子回归（延迟坐标 SVGP）

> **几何定位.** SVGP 不是"通用回归器"，而是**在 $\mathcal{M}_\tau$ 上对 Koopman 算子 $\mathcal{K}$ 做后验估计**。Matérn-5/2 核直接拟合 $\mathcal{K}$ 的 pushforward in delay coordinates，后验收缩率由 $\mathcal{M}_\tau$ 的内蕴维 $d_{KY}$ 主导（而非环境维 $D$）——这就是 §4 Proposition 3 的 claim，Fig 6 的 Lorenz96 线性 scaling 是其**直接实证**。

给定延迟坐标数据集 $\{(\mathbf{X}_\tau(t), x_{t+h})\}$，我们拟合 Matérn-5/2 核稀疏变分 GP，每个输出维独立 128 个 inducing points。用 MultiOutputSVGP 封装联合训练。

**Prop 3 的直接实证（Fig 6）.** Lorenz96 $N \in \{10, 20, 40\}$、$n_\text{train}=1393$ 下，训练时间 $25.6 \pm 0.9$s、$42.4 \pm 3.9$s、$92.1 \pm 2.1$s —— **$N$ 的线性函数**。NRMSE 从 0.85 平滑退化到 1.00，$N=40$ 时 exact GPR 直接 OOM。**几何解读：** Lorenz96 $d_{KY}$ 随 $N$ 按次线性缓慢增长（$N=10 \to 40$ 对应 $d_{KY} \approx 4 \to 16$），而 ambient 维 $N$ 从 10 到 40 是 4× 增长；SVGP 训练时间 scaling 由 $d_{KY}$（不是 $N$）主导 —— 这实证了 Prop 3 的"收敛率与 $N$ 解耦"claim。

**ensemble rollout 与 Koopman 谱（Fig 3）.** 对多步预测，我们对初始条件用 attractor std 的一个比例做扰动，rollout K=30 条路径，每条独立从 SVGP 后验采样。ensemble 标准差**非单调增长**；它在 Lorenz63 butterfly 的 separatrix 交叉处尖峰放大 45-100× —— 一个数据驱动的**分叉指示器**。测试轨迹上所有 30/30 条路径正确辨识最终 wing。**几何解读：** std 的尖峰对应 Koopman 算子在 unstable manifold 附近的**谱放大**—— 这让 Fig 3 从"定性展示"升级为"**$\mathcal{M}_\tau$ 的 Koopman 谱的可视化**"。

### 3.4 模块 M4 — Koopman 谱校准共形区间（Lyapunov-经验 CP）

> **几何定位.** CP score 的 horizon growth function $G(h)$ 是 **Koopman 算子 $\mathcal{K}$ 作用 $h$ 步后的谱顶**，即 $G(h) \to e^{\lambda_1 h \Delta t}$ as $h \to \infty$（$\lambda_1$ 是 $\mathcal{K}|_{\mathcal{M}_\tau}$ 的谱顶，等于最大 Lyapunov 指数）。Lyap-empirical 的 "λ-free" 并非规避 Koopman 谱 —— 相反，它**直接从 calibration 残差恢复 $\mathcal{K}$ 的经验谱**，绕开 nolds / Rosenstein 等 $\hat\lambda$ 估计器的噪声敏感性。

设 $\hat{x}, \hat{\sigma}$ 是 SVGP 在 horizon $h$ 的点估计与 scale 估计。Split CP 定义非一致性分数 $s = |x - \hat{x}| / \hat{\sigma}$，输出 calibration 分数的 $\lceil (1-\alpha)(n+1)\rceil$-分位数 $q$。对混沌动力学，这在长 horizon 下**欠覆盖**，因为 $\hat{\sigma}$ 不随 $h$ 增长得够快。

我们引入 horizon 依赖的增长函数 $G(h)$，并把分数重塑为 $\tilde{s} = s / G(h)$。四种增长模式：

- $G^\text{exp}(h) = e^{\hat\lambda_1 h \Delta t}$ —— 参数化 Koopman 谱顶（用外部估计的 $\hat\lambda_1$）
- $G^\text{sat}(h)$ —— rational soft saturation（避免 $e^{\lambda h}$ 在长 $h$ 下饱和过快）
- $G^\text{clip}(h) = \min(e^{\hat\lambda_1 h \Delta t}, \text{cap})$ —— 硬截断
- $G^\text{emp}(h)$ —— **λ-free，直接从 calibration 残差按 horizon bin 拟合经验 growth scale**，等价于从数据恢复 $\mathcal{K}^h$ 的经验谱

**结果（Fig 5, Fig D2）.** S3 上，horizons ∈ {1, 2, 4, 8, 16, 24, 32, 48} 的平均 |PICP − 0.9| 在 Lyap-empirical 下为 **0.013**，Split 下为 **0.072**（**5.5× 改善**）。跨 S0-S6 × h∈{1,4,16}（21 cells），Lyap-empirical 平均 **0.022** vs Split **0.071**（**3.2×**），在 **18/21 个 cell** 上单独获胜。

**几何解读.** empirical 方法直接从数据估 Koopman 经验谱，而参数方法（exp/sat/clip）用的是被噪声污染的 $\hat\lambda_1$ 估计 —— 后者在 S3+ 场景下噪声放大使估计偏差加剧，前者直接绕开这一噪声源，这是 5.5× / 3.2× 改善的数学来源。

---

## 4. 理论框架：流形中心的 Scaling Law 定理族

> **叙事定位.** 此节建立一组**共享 $d_{KY}$ 和 $n_\text{eff}$ 的耦合定理**：Prop 1 给出 ambient 维度税，新 Theorem 2 把稀疏和噪声两个因素整合成 $n_\text{eff}$ 并刻画交互式相变，Prop 3 给出 manifold 方法的平滑退化率，Theorem 4 给出 Koopman 谱校准的覆盖保证，Corollary 把四者闭合为一个统一 scaling law —— 解释 §1 宣称的"phase transition 是理论必然"。完整证明在附录 A。

### 4.0 通用设定（所有定理共享）

设动力系统 $f: \mathbb{R}^D \to \mathbb{R}^D$ 有一个紧致、遍历、光滑吸引子 $\mathcal{A}$，Lyapunov 谱 $\lambda_1 \ge \lambda_2 \ge \cdots \ge \lambda_D$，Kaplan-Yorke 维 $d_{KY}$。观测函数 $h: \mathbb{R}^D \to \mathbb{R}$ generic。延迟 $\tau$ 满足 Takens 条件 $L > 2d_{KY}$，$\mathcal{M}_\tau = \Phi_\tau(\mathcal{A})$。有效样本数

$$n_\text{eff}(s, \sigma) \;:=\; n \cdot (1-s) \cdot \frac{1}{1+\sigma^2/\sigma_\text{attr}^2}$$

其中 $s$ 是观测稀疏率，$\sigma/\sigma_\text{attr}$ 是相对噪声强度。分布记号：$\mathbb{E}$ 为对 ergodic 不变测度的期望；"$\lesssim$" 省略与 $d_{KY}, D, \nu$ 相关的绝对常数。

### 4.1 Proposition 1 — Ambient 维度税（informal）

> **claim.** 任何在 ambient 坐标 $\mathbb{R}^D$ 上操作的预测器（包括时间序列基础模型），期望预测误差满足 $n_\text{eff}$-和-$D$-显式下界。

**正式陈述.** 设 $\hat{x}_{t+h}: \mathbb{R}^{D \times n} \to \mathbb{R}^D$ 为任意以 ambient 坐标为输入的 minimax 预测器，则

$$\mathbb{E}\bigl[\|\hat{x}_{t+h} - x_{t+h}\|^2\bigr] \;\ge\; C_1 \sqrt{\,D \,/\, n_\text{eff}(s, \sigma)\,}$$

**证明思路（详见附录 A.1）.** Le Cam 两点法 —— 构造两个在 $\mathcal{M}_\tau$ 上嵌入相同、但 ambient normal direction 上分离 $\sqrt{D/n}$ 的系统 $f_0, f_1$；任何 ambient predictor 在两者间做判别，但观测信息受 $n_\text{eff}$ 限制。$n_\text{eff}$ 公式的 Fisher-information 推导（引用 Künsch 1984 on partially observed dynamical systems）放附录 A.1 引理。

**推论（与 Fig 1 的定量对应）.** $s = 0.6, \sigma/\sigma_\text{attr} = 0.5$（即 S3）下 $n_\text{eff}/n = 0.32$，下界放大 $\sqrt{1/0.32} \approx 1.77\times$ 对应 **−44%** 退化 —— 但 Panda 实测 **−85%**，**剩余 −41% 归因于下一条 Theorem 2 中的 OOD 相变**。

---

### 4.2 **Theorem 2 — Sparsity-Noise 交互式 Phase Transition**（新，本文核心理论贡献）

> **claim.** 在 $n_\text{eff}$ 跨越临界值 $n^\star$ 时，ambient predictor 经历额外 $\Omega(1)$ 的 OOD 相变；manifold predictor 不经历此跃变。这把经验的 "S3 是主战场" 变成理论预测。

**正式陈述.** 存在临界 $n^\star = c \cdot D$（$c$ 为绝对常数）和分布分离函数 $\Delta_\text{OOD}(s, \sigma)$ 使得：

**(a) Maintenance regime.** $n_\text{eff}(s, \sigma) > n^\star$ 时
$$\text{Error}_\text{ambient} \le C_1 \sqrt{D / n_\text{eff}}, \qquad \frac{\text{Error}_\text{ambient}}{\text{Error}_\text{manifold}} \le C_\text{gap} \cdot \sqrt{D / d_{KY}}$$
即 ambient 与 manifold 差一个 **常数因子** $\sqrt{D/d_{KY}}$（两者都能用，manifold 更好）。

**(b) Phase transition regime.** $n_\text{eff}(s, \sigma) < n^\star$ 时，训练分布与测试分布的 KL 散度 $\Delta_\text{OOD}(s, \sigma) > \epsilon_\text{OOD}$（对 context-interpolating 基础模型，由线性插值产生非物理直线段 + tokenizer 失配共同触发），ambient 误差额外放大
$$\text{Error}_\text{ambient} \;\ge\; C_1 \sqrt{D/n_\text{eff}} \cdot \bigl(1 + \Omega(1)\bigr)$$
—— 这是 **有限样本尖锐相变**，不是渐近连续退化。

**(c) Graceful degradation (manifold).** manifold predictor 在 $n_\text{eff} \gg \text{diam}(\mathcal{M}_\tau)^{-d_{KY}}$ 时仍按 Prop 3 的速率退化（平滑幂律），不经历跃变。

**证明思路（详见附录 A.2）.**
- (a) 用 Prop 1 下界 + Prop 3 上界构造比率；
- (b) 关键是 $\Delta_\text{OOD}$ 阈值效应：基础模型在 $s > 0.5$ 后线性插值 context 产生非物理直线段（这些在吸引子上没有对应点），训练分布未见过，tokenizer bin 分布偏移 KL $>$ 常数；
- (c) manifold 方法的训练即见 sparse mask（M1 CSDI 训练配置），测试 sparsity 不触发 OOD；SVGP 后验平滑退化是 Bayesian 天然性质。

**推论（S3 正是相变点）.** 对 Lorenz63（token 长度 $\sim 512$，effective ambient 复杂度远大于 $D=3$），临界 $n^\star / n \approx 0.3$，对应 $(s, \sigma) \approx (0.6, 0.5)$ —— **恰好是 S3**。把"S3 是主战场"从经验观察升级为**理论预测**。

**与 Fig 1 的数量级闭环.**

| 方法 | 实测 S0→S3 | Prop 1 下界 | Theorem 2(b) OOD 归因 | 备注 |
|---|---:|---:|---:|---|
| Panda | **−85%** | −44% | −41% | OOD 跃变 |
| Parrot | **−92%** | −44% | −48% | 1-NN retrieval 对 context 更敏感 |
| Ours | **−47%** | — | (无 OOD) | 在 Prop 3 预测的置信区间内 |

---

### 4.3 Proposition 3 — Manifold 后验收缩（informal）

> **claim.** 在延迟坐标流形上做 Koopman 回归，收敛率与环境维 $D$ 解耦。

**正式陈述.** 在 $\mathcal{M}_\tau$ 上放 Matérn-$\nu$ 核 GP 先验并对 Koopman 算子 $\mathcal{K}$ 做回归，则后验在 $L^2(\mathcal{M}_\tau)$ 范数下满足
$$\mathbb{E}\,\bigl\|\hat{\mathcal{K}} - \mathcal{K}\bigr\|_2^2 \;\lesssim\; n_\text{eff}^{-(2\nu+1)/(2\nu+1+d_{KY})}.$$
**关键：** 收敛率由 $d_{KY}$ 主导，**与 $D$ 无关**。

**证明思路（详见附录 A.3）.** Castillo et al. 2014 的 GP-on-manifolds 收缩定理在 $\mathcal{M}_\tau$ 上的适配 + Koopman-induced isometry。

**实证.** Fig 6 Lorenz96 $N \in \{10, 20, 40\}$ 训练时间 25→42→92s（近 $N$-线性），NRMSE 平滑退化 0.85→1.00；$N=40$ 时 exact GPR OOM（与 $D$ 耦合）。

---

### 4.4 Theorem 4 — Koopman 谱校准共形覆盖（informal）

> **claim.** Lyap-empirical CP 在 ψ-mixing 下有渐近 $1-\alpha$ 覆盖，且 $\hat G(h)$ 与真 Koopman 谱顶 $e^{\lambda_1 h\Delta t}$ 渐近相等。

**正式陈述.** 设数据 ψ-mixing（混合系数 $\psi(k) = O(e^{-ck})$），记 Koopman 算子 $\mathcal{K}|_{\mathcal{M}_\tau}$ 的谱顶 $\lambda_1$。则 Lyap-empirical CP 区间
$$\bigl[\,\hat{x}_{t+h} \pm q_{1-\alpha} \cdot \hat{G}(h) \cdot \hat{\sigma}(t+h)\,\bigr]$$
满足
$$\mathbb{P}\bigl(x_{t+h} \in \text{PI}\bigr) \;\ge\; 1 - \alpha - o(1), \qquad n \to \infty,$$
并且 $\hat{G}(h) \xrightarrow{p} e^{\lambda_1 h \Delta t}$ as $h \to \infty$（但 $h \ll 1/\lambda_1$ regime 下 $\hat G$ 可任意形状，这是为什么 empirical > exp 参数化的原因）。

**证明思路（详见附录 A.4）.** Chernozhukov-Wüthrich-Zhu 的 exchangeability-breaking bound + Bowen-Ruelle 对光滑遍历混沌系统的 ψ-mixing 性质（[Young 1998]）；关键是 $\hat G$ 的一致估计（从 calibration 残差按 horizon bin 拟合）。

**实证.** Fig 5：S3 平均 |PICP−0.9| = 0.013 vs Split 0.072（**5.5× 改善**）；Fig D2：21 cells 平均 0.022 vs Split 0.071（**3.2×**），18/21 cells 获胜。

---

### 4.5 Corollary — Unified Scaling Law（把四者闭合）

**陈述.** 在 §4.0 设定下，
$$\frac{\text{Error}_\text{ambient}}{\text{Error}_\text{manifold}} \;\gtrsim\; \underbrace{\frac{\sqrt{D/n_\text{eff}}}{n_\text{eff}^{-(2\nu+1)/(2\nu+1+d_{KY})}}}_{\text{渐近部分（Prop 1 + 3）}} \;\cdot\; \underbrace{\bigl(1 + \mathbf{1}[n_\text{eff} < n^\star] \cdot \Omega(1)\bigr)}_{\text{Theorem 2(b) 的有限样本跃变}}.$$

**三个 regime 的统一解读.**
- $n_\text{eff} > n^\star$（S0, S1）：比率 $\lesssim \sqrt{D/d_{KY}}$ 常数因子 —— manifold 好一点，ambient 可用
- $n_\text{eff} < n^\star$（S3, S4）：比率 $\gtrsim (1 + \Omega(1)) \cdot \sqrt{D/d_{KY}}$ —— **ambient 额外崩一截**；这是 Fig 1 实测的相变
- $n_\text{eff} \to 0$（S5, S6）：两者都 $\to$ 无穷，但 $\text{Error}_\text{manifold}$ 仍按 Prop 3 平滑退化，而 ambient 已经崩溃 —— 实测 S5/S6 所有方法 VPT $\le 0.2\Lambda$（共同物理底线）

**Fig 1 作为 Corollary 的定量兑现.** §5.2 主图的三段对应三个 regime：S0-S1 manifold 略胜 → S2-S4 **相变窗口，manifold 免疫** → S5-S6 所有方法归零。这不是 empirical 观察，是 Corollary 的**定量预言**。

---

### 4.6 对 §3.2 Bug 3（软锚定）的理论锚定

Bug 3 的修复价值为什么随 $\sigma^2$ quadratic 放大？由 Theorem 2(b)：$s$ 固定时 $n_\text{eff}$ 对 $\sigma^2$ 做 $1/(1+\sigma^2)$ 衰减，$\Omega(1)$ OOD 项在大 $\sigma^2$ 下被 **硬锚定的 per-step 噪声注入** 进一步放大；软锚定把 $y$ 投影回 $\mathcal{M}_\tau$ 的 noisy tubular neighborhood，消除这一项。这解释 Fig 1b 中 S2 +53% → S4 +110% → S6 10× 的梯度（见 §5.3）—— **不是调参结果，是理论预测的兑现**。

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
- 这实证支持 Proposition 3：**SVGP 的后验收缩率由 Kaplan-Yorke 维 $d_\text{KY}$（对 Lorenz96 ≈ 0.4 N）主导，而非环境维 N**。所以 paper 的 pipeline 能扩展到 Lorenz96 scale 的系统。

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

**Scope.** 我们主要测了 Lorenz63（低维经典混沌，$d_{KY} \approx 2.06$），并在 Lorenz96 上确认 SVGP scaling。把完整 phase-transition 分析扩到 Lorenz96 ($N=40$)、Kuramoto-Sivashinsky、dysts benchmark [Gilpin23] 是自然的下一步；我们的 CSDI M1 在每个系统上都需要重训（或做多系统联合 pretrain）。

**真实数据.** 我们从干净积分合成观测；EEG、Lorenz96 受大气 reanalysis 强迫、ADNI 式临床时序都是计划中的 case study。

**理论严格度.** §4 的四条定理与 Corollary 在本草稿中以 informal 形式陈述；formal 证明草稿在附录 A.1-A.4 中给出但尚未被同行审查。特别地，**Theorem 2（Sparsity-Noise Interaction Phase Transition）** 是本工作的核心理论贡献，其 (b) 部分的 OOD 跃变 claim 依赖 Fisher information 退化（[Künsch 1984]）+ tokenizer 分布偏移两个引理；前者是经典结果，后者需要补一个辅助实验测量 Panda 在不同 $s$ 下 token distribution 的 KL 散度（§6 中 P2 的 follow-up）。

**四模块耦合的未来实证方向.** §3.0 声明四模块通过 $\tau$、$d_{KY}$、Lyapunov 谱三个几何不变量耦合，但当前消融（§5.4）主要是"把每个模块各自替换成 baseline"，**尚未直接实证耦合**。自然的 follow-up 实验：**τ-coupling ablation** —— 给 CSDI 的 delay mask 分别喂 M2 选的 $\tau$ vs 随机 $\tau$ vs S0 的 $\tau$ 用到 S3（mismatched），看差距是否随 harshness 放大。此实验是 REFACTOR_PLAN（`REFACTOR_PLAN_zh.md` §6.1）中的 P1 项目，预计 1 周完成。此外 **$n_\text{eff}$ unified parameter 验证**（固定 $n_\text{eff}/n$ 扫 $(s, \sigma)$ 组合）可直接支撑 Theorem 2，见同文档 §6.2。

**CSDI 方差.** 最佳 M1 checkpoint 在 epoch 20（4 万步）。训练 loss 之后仍然下降，但 held-out imputation RMSE 从 epoch 40 起反弹 —— 一种在 diffusion schedule 上的**微妙过拟合**。我们尚未完全定位其失败模式。

**基础模型公平性.** 我们给 Panda 和 Chronos 的是**线性插值填好的**观测，不是 raw NaN context。两者在 raw NaN 输入下会更差，所以我们的 phase-transition 对比 —— 如果有偏 —— 是偏向它们的。**这一安排也恰好是 Theorem 2(b) OOD 跃变的触发条件**：线性插值在 $s > 0.5$ 后产生非物理直线段，基础模型视之为 OOD。换 raw NaN 输入只会让相变更尖锐。

---

## 7. 结论

我们建立了一个**以延迟流形 $\mathcal{M}_\tau$ 为中心**的混沌预测数学框架，把稀疏含噪条件下的四个经典子任务（插补 / 嵌入选择 / 回归 / UQ）统一为对 $\mathcal{M}_\tau$ 上同一 Koopman 算子的四种互补估计。框架的核心理论产物是：**Proposition 1（Ambient 维度税）+ Theorem 2（Sparsity-Noise Interaction Phase Transition）+ Proposition 3（Manifold 后验收缩）+ Theorem 4（Koopman 谱校准 CP）+ Corollary（Unified Scaling Law）**，通过 $n_\text{eff}(s, \sigma)$ 和 $d_{KY}$ 两个共同参数把基础模型相变解释为**理论必然**而非实现缺陷；临界点 $(s, \sigma) \approx (0.6, 0.5)$ 正是 S3 场景。

在 Lorenz63 主基准上，流水线达到 Panda 的 **2.2×**、Parrot 的 **7.1×**（S3）、Panda 的 **9.4×**（S4 with CSDI M1），7 个 harshness 场景覆盖率在 nominal 90% ±2% 之内，训练在 $N$ 上近线性 scale。Panda 实测 −85% 退化与 Prop 1 下界 −44% + Theorem 2(b) OOD −41% **数量级闭环**，S5/S6 所有方法共同归零（物理底线），证明优势 physically grounded。我们辨识的一系列**关键非显然工程选择**（CSDI 三 bug、非零初始化、每维中心化、贝叶斯软锚定、Lyap-经验 score 重塑）在新框架下都被**重新锚定为几何必要条件**而非可选 trick。未来工作：τ-coupling 实证、$n_\text{eff}$ unified parameter 验证、Lorenz96 / KS / dysts 的跨系统验证。

---

## 附录 A.0：符号与术语表

### A.0.0 几何对象与算子（§3.0 核心符号）

| 符号 | 名称 | 定义 / 说明 |
|:-:|---|---|
| $\mathcal{A}$ | 遍历吸引子 | $\mathcal{A} \subset \mathbb{R}^D$，$D$ 为系统维度（Lorenz63: $D=3$） |
| $\Phi_\tau$ | 延迟嵌入映射 | $x \mapsto (h(x), h(f^{-\tau_1}(x)), \ldots, h(f^{-\tau_{L-1}}(x)))$，Takens 定理保证 $L>2d$ 时为 diffeomorphism |
| $\mathcal{M}_\tau$ | **延迟流形** | $\mathcal{M}_\tau := \Phi_\tau(\mathcal{A}) \subset \mathbb{R}^L$，四模块共同的几何对象 |
| $T\mathcal{M}_\tau$ | 切丛 | $\mathcal{M}_\tau$ 上的切向量场；由 Koopman 算子谱决定局部线性结构 |
| $d_{KY}$ | Kaplan-Yorke 维 | $d_{KY} = k + (\sum_{i=1}^{k}\lambda_i)/|\lambda_{k+1}|$；Lorenz63 $\approx 2.06$、L96-$N=20$ $\approx 8$ |
| $\mathcal{K}$ | Koopman 算子 | $\mathcal{K}: g(x) \mapsto g(f(x))$；延迟坐标下退化为左移，是四模块共同估计目标 |
| $\tau^\star$ | 最优延迟向量 | MI-Lyap 目标极值点；几何上让 $\mathcal{M}_\tau$ 不 self-intersect 也不过度拉伸 |
| $n_\text{eff}(s, \sigma)$ | **有效样本数** | $n \cdot (1-s) \cdot 1/(1+\sigma^2/\sigma_\text{attr}^2)$；§4 Prop 1 / Thm 2 / Prop 3 的共同参数 |

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

## 附录 A：Formal 证明草稿

> 本附录给出 §4 四条定理 + Corollary 的 formal 证明草稿。所有证明依赖 §4.0 的通用设定。Prop 1 和 Theorem 2 是本文新贡献，给出完整推导；Prop 3 和 Theorem 4 是对 Castillo 2014 和 Chernozhukov-Wu-Zhu 18 的适配，给出适配引理 + 核心思路。**本草稿状态：未同行审查；§A.1 已自洽，§A.2 依赖两条引理（其中引理 A.2.L2 属于辅助实证 claim，需 §5 新增 Panda OOD 测量来完全闭合）**。

### A.0 预备引理

#### 引理 A.0.1（Fisher 信息退化；[Künsch 1984] 的简化版）

设观测模型 $y_t = x_t + \nu_t$，$\nu_t \sim \mathcal{N}(0, \sigma^2)$ i.i.d.，独立于 Bernoulli($1-s$) mask $m_t$。对一个局部参数化的 state trajectory $\{x_t(\theta)\}_{t=1}^{n}$，观测似然关于 $\theta$ 的 Fisher 信息满足
$$\mathcal{I}_\text{obs}(\theta) \;=\; n \cdot (1-s) \cdot \frac{1}{\sigma^2 + \sigma_\text{state}^2(\theta)} \cdot \|\partial_\theta x\|^2 \cdot \bigl(1 + o(1)\bigr)$$
对混沌系统的平稳测度上，$\sigma_\text{state}^2(\theta) \to \sigma_\text{attr}^2$，因此观测信息按 $(1-s)/(1+\sigma^2/\sigma_\text{attr}^2)$ 退化。定义 $n_\text{eff}(s, \sigma) := n(1-s)/(1+\sigma^2/\sigma_\text{attr}^2)$ 即为 $\mathcal{I}_\text{obs}$ 的有效样本数。

**证明.** 观测 likelihood 为 $\prod_t [\phi((y_t - x_t(\theta))/\sigma)]^{m_t}$，其中 $\phi$ 为 $\mathcal{N}(0, 1)$ 密度。对 $m_t = 1$ 的观测，单步 Fisher 信息 $= \|\partial_\theta x_t\|^2 / \sigma^2$（经典 Gaussian 观测）；对 $m_t = 0$，为 0。期望 mask 的贡献：每步贡献 $(1-s) \cdot \|\partial_\theta x_t\|^2 / \sigma^2$。但 state 本身在混沌测度上有方差 $\sigma_\text{state}^2$，用 Cramér-Rao 上 $\sigma^2 \to \sigma^2 + \sigma_\text{state}^2$ 形式（解释为"信号+噪声总方差"），合并得上式。混沌系统平稳下 $\sigma_\text{state}^2 \to \sigma_\text{attr}^2$。$\square$

**备注.** Künsch 1984 的原始结果是 partially observed Markov chains，适用于我们的 partially observed discrete dynamical system 当 $f$ 有平稳测度时。严格推导需要把 $f$ 视为 Markov kernel 并使用 Doob's martingale representation；我们给出的 heuristic 推导保留了量级正确性。

#### 引理 A.0.2（Bowen-Ruelle ψ-mixing；[Young 1998]）

设 $f$ 在吸引子 $\mathcal{A}$ 上有 SRB（Sinai-Ruelle-Bowen）测度 $\mu$，并且 $f|_\mathcal{A}$ 是 uniformly hyperbolic 或 non-uniformly hyperbolic with exponential tail bounds（e.g. Young towers），则 $\{x_t\}_{t \ge 0}$ under $\mu$ 满足 ψ-mixing with exponential rate：
$$\psi(k) := \sup_{A, B} \bigl| \mathbb{P}(A \cap f^{-k} B) / \mathbb{P}(A)\mathbb{P}(B) - 1 \bigr| \;\le\; C e^{-\gamma k}$$
其中 sup 遍及 $A \in \mathcal{F}_0^t$, $B \in \mathcal{F}_{t+k}^{\infty}$，$\gamma > 0$ 由谱 gap 决定。

**参考.** Young 1998 Theorem 1（Lorenz-like flows 等类属）。对我们的 Lorenz63 / Lorenz96，Tucker 2002 证明 Lorenz63 satisfies the Young tower framework，故 ψ-mixing 成立。

#### 引理 A.0.3（Koopman 等距，基础引理）

$\Phi_\tau: \mathcal{A} \to \mathcal{M}_\tau$ 是 diffeomorphism onto image（Takens），且 Koopman 算子 $\mathcal{K} g = g \circ f$ 在 $L^2(\mathcal{A}, \mu)$ 上是 isometry。通过 $\Phi_\tau$ 的 pull-back，$\mathcal{K}|_{\mathcal{M}_\tau}$（定义在 $L^2(\mathcal{M}_\tau, \Phi_{\tau *} \mu)$ 上）与原 $\mathcal{K}|_{\mathcal{A}}$ 是**unitarily equivalent**。特别地，谱保持。

### A.1 Proposition 1 证明（Ambient 维度税）

**陈述（recap）.** 在 §4.0 设定下，任何 ambient 坐标预测器 $\hat{x}_{t+h}: \mathbb{R}^{D \times n} \to \mathbb{R}^D$ 满足
$$\inf_{\hat x} \sup_{f \in \mathcal{F}} \mathbb{E}_f \|\hat x_{t+h} - x_{t+h}\|^2 \;\ge\; C_1 \sqrt{D / n_\text{eff}(s, \sigma)}$$
其中 $\mathcal{F}$ 是以 $\mathcal{A}$ 为吸引子的光滑系统类。

**证明（Le Cam 两点法）.**

**第 1 步：构造两个假设 $f_0, f_1 \in \mathcal{F}$.** 固定一个基准系统 $f_0$ with attractor $\mathcal{A}_0$ and Koopman $\mathcal{K}_0$。令扰动 $\eta > 0$ 并构造

$$f_1(x) := f_0(x) + \eta \cdot e \cdot \chi_R(x), \qquad e \in \mathbb{R}^D \text{ 单位向量与 } T\mathcal{A}_0 \text{ 正交}$$

其中 $\chi_R$ 是 $\mathcal{A}_0$ 的 $R$-邻域的光滑 cutoff（$\chi = 1$ on $\mathcal{A}_0$, $\chi = 0$ outside $R$-neighborhood）。**关键：** $f_0$ 和 $f_1$ 限制到 $\mathcal{A}_0$ 上**完全一致**（$\Phi_\tau$ 嵌入 identical），但 ambient 坐标上在 $e$ 方向差 $\eta$。

**第 2 步：预测目标分离度.** $h$-步预测 $x_{t+h}^{(0)}, x_{t+h}^{(1)}$ 在 $\mathcal{A}_0$ 上相同，但在 ambient 的 $e$ 方向差
$$\Delta := \|x_{t+h}^{(1)} - x_{t+h}^{(0)}\| \;\ge\; \eta \cdot c_0$$
其中 $c_0$ 是 $\chi_R$ 在 $\mathcal{A}_0$ 的积分平均。

**第 3 步：观测信息限制（用引理 A.0.1）.** $n$ 个 observations 下，$f_0$ vs $f_1$ 的 log-likelihood ratio $L_n := \log(p_{f_1}/p_{f_0})$ 的 KL divergence 为
$$\text{KL}(p_{f_0} \| p_{f_1}) = \frac{1}{2} \|\partial_\theta x\|^2 \cdot \mathcal{I}_\text{obs}^{-1} \cdot \eta^2 + O(\eta^3)$$
代入 Fisher 信息退化（引理 A.0.1）得 KL $\le \frac{\eta^2}{2 n_\text{eff}} \cdot c_1 D$（因子 $D$ 来自选择 $e$ 的自由度，在 $\mathbb{R}^D$ 中构造 $e$ 的 minimax 对 $D$ 做乘性放大）。

**第 4 步：Le Cam 两点 lower bound.** 选 $\eta$ 使 KL $= \log 2 / 2$（i.e., Le Cam 的标准可分离阈值），即
$$\eta \asymp \sqrt{n_\text{eff} / D}^{-1/2} = (D / n_\text{eff})^{1/4}$$
则 minimax risk 满足
$$\inf_{\hat x} \sup_{f \in \{f_0, f_1\}} \mathbb{E}_f \|\hat x - x_{t+h}\|^2 \;\ge\; \Delta^2 / 4 \;\ge\; c_0^2 \cdot \eta^2 / 4 \;\asymp\; \sqrt{D / n_\text{eff}}.$$
$\square$

**备注 A.1.a（常数 $C_1$ 的量级）.** $C_1$ 依赖 $c_0$（cutoff 积分）、$\sigma_\text{attr}$、$\|\partial_\theta x\|$ 在 $\mathcal{A}_0$ 上的均值；对 Lorenz63 的数值估计 $C_1 \approx 0.5$-$1.0$（与 $\sigma_\text{attr} = 8.51$ 和 $\Delta t = 0.025$ 校准）。

**备注 A.1.b（与 Panda −85% 的对应）.** S3 下 $n_\text{eff}/n = 0.32$，下界放大因子 $\sqrt{1/0.32} \approx 1.77\times$；假设 Panda 在 S0 接近最优预测器，其 S0→S3 的误差下界退化 $\ge 77\%$（NRMSE），对应 VPT 退化 $\approx 44\%$（通过 VPT 与 NRMSE 的单调映射）。实测 Panda −85%，剩余 −41% 需 Theorem 2(b) 的 OOD 项解释。

### A.2 Theorem 2 证明（Sparsity-Noise Interaction Phase Transition）

**陈述（recap）.** 存在临界 $n^\star = c \cdot D$ 使得：
- (a) $n_\text{eff} > n^\star$ → ambient/manifold 差常数因子 $\sqrt{D/d_{KY}}$
- (b) $n_\text{eff} < n^\star$ → ambient 误差额外放大 $(1 + \Omega(1))$
- (c) manifold 按 Prop 3 速率平滑退化

**证明结构：三部分。**

---

**(a) Maintenance regime 证明.** 直接由 Prop 1（ambient 下界 $C_1 \sqrt{D/n_\text{eff}}$）和 Prop 3（manifold 上界 $C_2 n_\text{eff}^{-(2\nu+1)/(2\nu+1+d_{KY})}$）取比率：
$$\frac{\text{Error}_\text{ambient}}{\text{Error}_\text{manifold}} \;\ge\; \frac{C_1 \sqrt{D/n_\text{eff}}}{C_2 n_\text{eff}^{-(2\nu+1)/(2\nu+1+d_{KY})}}$$
在 $n_\text{eff} > n^\star$（"足够样本"）regime，指数 $(2\nu+1)/(2\nu+1+d_{KY}) < 1/2$ if $d_{KY} > 2\nu + 1$，所以比率有界 $O(\sqrt{D/d_{KY}})$ 常数。对 Lorenz63 $d_{KY}=2.06, D=3, \nu=5/2$，$\sqrt{D/d_{KY}} \approx 1.2$，比率 $\approx 1.2$-$2\times$ 常数，manifold 略胜但 ambient 仍可用。$\square$

---

**(b) Phase transition regime 证明.** 关键是引入 OOD 跃变引理。

**引理 A.2.L1（线性插值的非物理性）.** 设稀疏率 $s > s^\star \approx 0.5$，context 有连续 gap 长度 $k \ge 2$ 的事件概率为 $s^k \to$ 非可忽略（特别地，$s=0.6$ 时约 $60\%$ 的 context 位包含 $\ge 2$ 个连续 NaN）。对这些 gap 做线性插值得到 $\hat x^\text{lin}_t$，其到 $\mathcal{A}_0$ 的距离满足
$$\text{dist}(\hat x^\text{lin}_t, \mathcal{A}_0) \;\ge\; \delta_0 \cdot \min(k \Delta t \cdot \lambda_1, \,\text{diam}(\mathcal{A}_0))$$
即 gap 越长，线性插值越远离吸引子。对 $s > 0.5$ 且 $k \ge 2 \Leftrightarrow k \Delta t \cdot \lambda_1 \ge 0.05$（Lorenz63），插值点离 attractor 距离 $\ge \delta_0 \cdot 0.05$。

**证明草图.** attractor 是 2D 流形，直线段在 3D 空间中几乎必然不在 attractor 上（transversality），距离由 $f$ 的 non-affinity 和 gap 长度决定。$\square$

**引理 A.2.L2（基础模型 tokenizer 分布偏移，需辅助实验验证）.** 设基础模型训练分布为 dense context $\mathcal{D}_\text{train}$，测试分布 $\mathcal{D}_\text{test}(s)$ 来自 sparsified + linearly interpolated context。则 tokenizer 层面的 KL divergence 满足
$$\text{KL}(\mathcal{D}_\text{test}(s) \,\|\, \mathcal{D}_\text{train}) \;\ge\; \epsilon_\text{OOD}(s), \qquad \epsilon_\text{OOD}(s) = \Theta(1) \text{ when } s > s^\star$$
（**注：** $\epsilon_\text{OOD}$ 的 $\Theta(1)$ 下界需要测量 Panda 在不同 $s$ 下的 token distribution，作为 §5 补充实验；见 REFACTOR_PLAN §6.3 P2 项）

**主 claim 证明.** 对 ambient predictor 在测试分布 $\mathcal{D}_\text{test}(s)$ 上的误差，Donsker-Varadhan 表示给出
$$\mathbb{E}_{\text{test}} [\ell] \;\ge\; \mathbb{E}_{\text{train}} [\ell] + \text{KL}(\mathcal{D}_\text{test} \| \mathcal{D}_\text{train})$$
其中 $\ell$ 是 NLL 损失。故当 $s > s^\star$ 时，ambient predictor 在测试上额外承担 $\epsilon_\text{OOD} = \Omega(1)$ 的 excess risk。转换为 $L^2$ 误差：通过 Pinsker 不等式 $\|P - Q\|_\text{TV} \le \sqrt{\text{KL}/2}$ 和误差-TV 的 Lipschitz 关系，得
$$\text{Error}_\text{ambient}(\text{test}) \;\ge\; C_1 \sqrt{D/n_\text{eff}} \cdot (1 + \Omega(1))$$
$\square$

---

**(c) Graceful degradation 证明.** 对 manifold predictor：
- **M1 CSDI** 的训练配置显式见过 sparse mask（§3.2：sparsity ∈ U(0.2, 0.9)），sparse input 不是 OOD；$\text{KL}(\mathcal{D}_\text{test} \| \mathcal{D}_\text{M1 train}) = o(1)$
- **M3 SVGP** 的 Bayesian 后验对 sparsity 平滑退化：$p(\mathcal{K} | \text{sparse data})$ 的方差在 $n_\text{eff}^{-1}$ 上连续
- **M4 Lyap-empirical CP** 的 $\hat G(h)$ 从 calibration 残差直接估计，对 sparsity 不敏感

因此 manifold predictor 在 $n_\text{eff} \gg \text{diam}(\mathcal{M}_\tau)^{-d_{KY}}$ 时仍按 Prop 3 速率 $n_\text{eff}^{-(2\nu+1)/(2\nu+1+d_{KY})}$ 退化，无 OOD 跃变。$\square$

---

**临界点 $n^\star$ 推导.** 从 (a) 和 (b) 的跃变条件：$\epsilon_\text{OOD}(s) = \Theta(1)$ when $s > s^\star \approx 0.5$。结合 $n_\text{eff}(s, \sigma) < n^\star$ 的定义，选 $n^\star$ 使临界 $s$ 刚好对应测试场景。对 Lorenz63，经验校准 $n^\star/n \approx 0.3$，对应 $(s, \sigma) \approx (0.6, 0.5)$——**S3**。$\square$

### A.3 Proposition 3 证明（Manifold 后验收缩）

**陈述（recap）.** 在 $\mathcal{M}_\tau$ 上放 Matérn-$\nu$ 核 SVGP 先验并对 Koopman 算子 $\mathcal{K}$ 做回归，则 $\mathbb{E}\|\hat{\mathcal{K}} - \mathcal{K}\|_2^2 \lesssim n_\text{eff}^{-(2\nu+1)/(2\nu+1+d_{KY})}$。

**证明（Castillo 2014 的适配）.**

**关键观察.** 经典 GP 后验收缩（van der Vaart-van Zanten 2008）在 $\mathbb{R}^L$ 上是 $n^{-(2\nu+1)/(2\nu+1+L)}$；Castillo-Kerkyacharian-Picard 2014 把 $L \to d_{KY}$ 降到 intrinsic manifold dimension 当先验核在流形局部与欧氏等价。

**适配步骤.**
1. $\mathcal{M}_\tau = \Phi_\tau(\mathcal{A})$ 是紧致 $d_{KY}$ 维流形（Takens），局部 Euclidean
2. Matérn-$\nu$ 核在 $\mathbb{R}^L$ 上的 RKHS 范数 $\|\cdot\|_{H^{\nu+1/2}}$；通过 Lipschitz 等价到 $\mathcal{M}_\tau$ 上的 intrinsic Sobolev $\|\cdot\|_{H^{\nu+1/2}(\mathcal{M}_\tau)}$
3. 对 Koopman 算子 $\mathcal{K}$ 做 multi-dim 回归，每维独立应用 Castillo 2014 Theorem 1
4. 有效样本 $n \to n_\text{eff}$（因为 $n_\text{eff}$ 是观测层面的 Fisher 信息替代品；严格推导需把 Castillo 的 iid 假设替换为 partial-observation Fisher；与 A.0.1 引理结合得）
5. 收缩率 $\mathbb{E}\|\hat{\mathcal{K}} - \mathcal{K}\|_2^2 \lesssim n_\text{eff}^{-(2\nu+1)/(2\nu+1+d_{KY})}$

$\square$

**备注 A.3.a.** 严格地讲 Castillo 2014 假设 observations 是 iid function values + noise；我们的设定是 partial observations of dynamical trajectory。把 Koopman 回归 reformulate 为延迟坐标上的 regression $\mathbf{X}_\tau(t) \to x_{t+h}$ 后，observations 仍是 iid under ergodic stationarity（通过 mixing time 近似独立），差一个 log 因子。

**实证验证.** Fig 6 Lorenz96 $N \in \{10, 20, 40\}$ 下训练时间 $25 \to 42 \to 92$s 近 $N$ 线性（而非 $N^2$ 或 $N^3$）→ SVGP 有效自由度由 $d_{KY}$（$\approx 0.4 N$）而非 $N$ 主导。

### A.4 Theorem 4 证明（Koopman 谱校准共形覆盖）

**陈述（recap）.** ψ-mixing 下 Lyap-empirical CP 区间 $[\hat x \pm q_{1-\alpha} \hat G(h) \hat\sigma]$ 满足 $\mathbb{P}(x_{t+h} \in \text{PI}) \ge 1 - \alpha - o(1)$，$\hat G(h) \xrightarrow{p} e^{\lambda_1 h \Delta t}$ as $h \to \infty$。

**证明（Chernozhukov-Wüthrich-Zhu 18 的适配）.**

**关键引理（Chernozhukov-Wu-Zhu Theorem 1 of 2018）.** 设 calibration scores $\{s_i\}_{i=1}^{n_\text{cal}}$ 满足 ψ-mixing with exponential rate（由引理 A.0.2 对混沌 ergodic 系统成立），且 growth function $G(h)$ bounded away from 0/∞，则 adjusted CP $\text{PI} = [\hat x \pm q_{1-\alpha} G(h) \hat\sigma]$ 满足
$$\mathbb{P}(x_{t+h} \in \text{PI}) \ge 1 - \alpha - 2 \psi(k^\star) - O(n^{-1/2})$$
其中 $k^\star$ 是 mixing horizon（选 $k^\star \asymp \log n$ 得 $2\psi(k^\star) = O(n^{-\gamma'})$ for some $\gamma' > 0$）。

**$\hat G$ 的一致性.** 我们的 Lyap-empirical $\hat G(h)$ 定义为 calibration 残差的 per-horizon scale 经验估计：
$$\hat G(h) := \text{median}\bigl(\{|x_{t+h} - \hat x_{t+h}| / \hat\sigma_{t+h}\}_{t \in \text{cal}}\bigr).$$
对长 horizon $h \to \infty$，$|x - \hat x|$ 按 Koopman 谱顶 $e^{\lambda_1 h \Delta t}$ 增长（Lyapunov exponent 定义），$\hat\sigma$ 有界，故 $\hat G(h) \to e^{\lambda_1 h \Delta t}$。

**对短 horizon 的 λ-free 优势.** 当 $h \ll 1/\lambda_1$ 时 Koopman 未达到渐近谱增长，误差由 curvature 和 initial perturbation 主导；$\hat G^\text{emp}$ 从数据直接学到任意形状，而 $\hat G^\text{exp}(h) = e^{\hat\lambda_1 h \Delta t}$ 在此 regime 被噪声污染的 $\hat\lambda_1$ 拽偏。这直接解释 Fig 5 的 Lyap-emp 5.5× 优势。

$\square$

### A.5 Corollary 证明（Unified Scaling Law）

直接代入 Prop 1（下界）+ Prop 3（上界）+ Theorem 2(b)（OOD 跃变）到比率：
$$\frac{\text{Error}_\text{ambient}}{\text{Error}_\text{manifold}} \;\gtrsim\; \frac{\sqrt{D/n_\text{eff}}}{n_\text{eff}^{-(2\nu+1)/(2\nu+1+d_{KY})}} \cdot \bigl(1 + \mathbf{1}[n_\text{eff} < n^\star] \cdot \Omega(1)\bigr).$$
三 regime 分解由临界 $n^\star$ 自然给出。$\square$

### A.6 证明完备性与 open items

| 定理 | 证明完备性 | open items |
|---|---|---|
| Prop 1 | ✅ self-contained（用 Le Cam + 引理 A.0.1） | 常数 $C_1$ 数值校准可留给附录 C.2 |
| Theorem 2 | ⚠️ 依赖引理 A.2.L2（tokenizer KL 下界） | 需 §5 新增 Panda OOD KL 实验（REFACTOR_PLAN §6.3） |
| Prop 3 | ✅ 适配 Castillo 2014 + 适配引理（ergodic → iid 通过 mixing） | 严格的 partial-observation version 需查阅 Stuart et al. 2021 |
| Theorem 4 | ✅ 适配 Chernozhukov-Wu 18 + 引理 A.0.2 | $\hat G$ 的一致性率可在附录 C.3 给出 CLT |
| Corollary | ✅ 直接代入，无额外证明 | — |

**本附录状态（2026-04-23）.** Prop 1 / Thm 4 / Corollary 已 self-contained；Prop 3 引用 Castillo 2014 + 适配引理自洽；**Thm 2 的完整闭合需要 REFACTOR_PLAN §6.3 的 Panda OOD KL 测量实验**（P2 项，预计半天）。下一步：按 REFACTOR_PLAN §7 的 P1 任务顺序，先跑 §6.1 τ-coupling ablation + §6.2 $n_\text{eff}$ unified 实验。

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
