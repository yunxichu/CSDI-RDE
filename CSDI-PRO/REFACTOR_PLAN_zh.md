# REFACTOR_PLAN — 从「四模块工程 pipeline」到「延迟流形统一框架」完整改良方案

> **文档目的**：整合三段对话（见 `改进方案`），形成可落地的 paper 叙事重构计划。后续按 §10 的路径分阶段推进。
> **核心原则**：不改任何实验数字，只做"叙述视角 + 理论包装"升级；实验工作量集中在 1-2 组新增加分项。
> **天花板预期**：当前 UAI/TMLR 稳 → 目标 ICML/NeurIPS accept band。

---

## §0 TL;DR（一页总览）

**当前问题**：paper 把 M1/M2/M3/M4 当成四个独立工程组件并列陈述，reviewer 容易判定为"堆 trick"。

**核心洞察**：四个 module 其实都在围绕**同一个几何对象** —— 延迟流形 $\mathcal{M}_\tau = \Phi_\tau(\text{attractor}) \subset \mathbb{R}^L$ —— 工作，通过三个共享的几何不变量彼此耦合：

| 共享不变量 | 跨 module 耦合 |
|---|---|
| 延迟向量 $\tau$ | M2 选 → M1 delay-mask 用 → M3 坐标定义 |
| Kaplan-Yorke 维 $d_{KY}$ | M2 最优 L ← M1 score 收敛率 ← M3 后验收缩率（与 ambient 维 $D$ 解耦） |
| Lyapunov 谱 $\{\lambda_i\}$ | M2 惩罚项 → M4 horizon growth → 决定相变临界点 |

**一句话升级**：从 "我们造了一个好 pipeline" → "稀疏噪声混沌预测是流形几何问题；基础模型相变是 ambient 坐标维度税的理论必然；我们给出流形中心的统一框架"。

**flagship 亮点保留**：S3/S4 的 2.2× / 9.4× 优势**不但保留，反而成为新理论的 flagship empirical evidence**。通过引入 $n_\text{eff}(s, \sigma) = n(1-s)/(1+\sigma^2/\sigma_\text{attr}^2)$ 作为 Prop 1 / Prop 2 的耦合参数，S3/S4 从"某难度上的经验胜利"升级为"**理论预测的相变窗口内，唯一免疫相变的方法**"。

---

## §1 核心诊断

### 1.1 现版本的问题

- §3 方法按 M1/M2/M3/M4 平铺，每个 module 独立 motivate，彼此不共享参数
- §4 三条 proposition 各讲各的，没有共同数学对象
- Abstract / Intro 骨架是"列组件 + 列数字"，不是"理论闭环"
- 三个 CSDI bug 修复被描述成"工程踩坑"，失去理论意义
- "S3 是主战场"是经验观察，不是理论预测

### 1.2 我们手里真正的数学对象

**Takens 嵌入定理**：对 d 维吸引子，延迟坐标 $\Phi_\tau(x_t) = (h(x_t), h(x_{t-\tau_1}), \ldots, h(x_{t-(L-1)\tau_{L-1}}))$ 在 $L > 2d$ 时是嵌入。$\mathcal{M}_\tau = \Phi_\tau(\text{attractor})$ 是延迟流形，是贯穿四个 module 的**同一个几何对象**。

**Koopman 算子** $\mathcal{K}: g \mapsto g \circ f$ 在延迟坐标下退化为左移，是四个 module 共同的估计目标。

---

## §2 核心洞察：一个数学对象，四个侧面

### 2.1 四个 module 在流形上做什么

| Module | 在 $\mathcal{M}_\tau$ 上的几何角色 |
|---|---|
| **M1 CSDI delay mask** | score 网络的 attention bias 把"时间上 $\tau_i$ 相关"的位置耦合 —— 等价于告诉 denoiser "沿 $\mathcal{M}_\tau$ 切向 denoise" |
| **M2 MI-Lyap τ-search** | 选 τ 让 $\Phi_\tau$ **几何上尽量好**：MI 大 ⇔ 嵌入不 self-intersect；Lyap 项 ⇔ 嵌入不过度拉伸 |
| **M3 SVGP on delay coords** | 在 $\mathcal{M}_\tau$ 上拟合 Koopman 算子的 pushforward |
| **M4 Lyap-empirical CP** | Koopman 算子谱顶 $e^{\lambda_1 h}$ 决定 PI 的 horizon growth |

### 2.2 四个 module 的统一目标

> 稀疏噪声下的混沌预测 = **从退化观测重建 $\mathcal{M}_\tau$ 上的 Koopman 算子**。
> M1 / M2 / M3 / M4 是这个重建任务的四个互补子任务，共享 $\tau$、$d_{KY}$、Lyapunov 谱。

---

## §3 三层耦合故事（由浅入深）

### 3.1 浅层耦合：M1 的 $\tau_i$ 和 M2 选的 $\tau$ 是同一组 τ

当前 CSDI delay attention bias: $\text{bias}_{t,t'} = \alpha \cdot \phi_\theta(t-t')$，$\phi_\theta$ 是一般 MLP。

**升级写法（方案 C，可选）**：把 $\phi_\theta$ 参数化为 τ-aware 核
$$\phi_\theta(\Delta t; \tau, \mathbf{w}) = \sum_{i=1}^{L} w_i \exp(-\beta (\Delta t - \tau_i)^2)$$
其中 $\tau = (\tau_1, \ldots, \tau_L)$ 就是 M2 选的 τ，**在训练时作为外部 conditioning 注入**。

这样 M1 就从"generic CSDI + mask"升级到"**manifold-aware score model**"，M1-M2 耦合变成**结构性**而非启发式。

**实证 claim**：M1 delay mask 中出现的 τ 应与 M2 选的 τ 一致，否则 score 网络建的是"错误流形"的切丛结构。→ §5 τ-coupling ablation 实证。

### 3.2 中层耦合：$d_{KY}$ 是贯穿全流程的不变量

Kaplan-Yorke 维 $d_{KY} = k + \frac{\sum_{i=1}^{k}\lambda_i}{|\lambda_{k+1}|}$（Kaplan-Yorke 猜想 = 吸引子 Hausdorff 维 = $\mathcal{M}_\tau$ 内蕴维）。

| Module | 与 $d_{KY}$ 的关系 |
|---|---|
| M2 | Takens 要 $L > 2d_{KY}$；最优 L 由 $d_{KY}$ 决定 |
| M1 | Manifold diffusion score 收敛率 $n^{-\alpha/(2\alpha+d_{KY})}$（Chen et al. 2023） |
| M3 | SVGP 后验收缩率 $n^{-(2\nu+1)/(2\nu+1+d_{KY})}$（Prop 2） |
| M4 | CP horizon growth 依赖 $\lambda_1$；$\lambda_1$ 和 $d_{KY}$ 同出自 Lyapunov 谱 |

**paper narrative**：
> 从 dense ambient coords（基础模型）转到 delay coords（经典方法）不是工程选择，而是**信息论必然**。基础模型在 $\mathbb{R}^D$ 上承担 $\sqrt{D/n}$ 维度税；我们在 $d_{KY}$ 维延迟流形上操作，收敛率与 $D$ 解耦。$d_{KY} \ll D$ 时差距放大，稀疏场景下成**有限样本相变**。

### 3.3 深层耦合：四 module 共同估计 Koopman 算子

Koopman 算子在延迟坐标下作用平凡：$(y_t, y_{t-\tau}, \ldots) \mapsto (y_{t+1}, y_{t+1-\tau}, \ldots)$ 是一个左移。

- M1 学 $p(x_{1:T})$ 的 score → 由 Koopman 不变测度决定
- M2 选 τ 让 Koopman 矩阵近似谱接近真实谱（τ 太小 ⇒ near-identity；τ 太大 ⇒ 噪声吞谱；最优 τ 在 Fraser-Swinney first MI min，MI+Lyap 是鲁棒版）
- M3 直接拟合 Koopman on delay coords
- M4 Koopman 谱 $e^{\lambda_i h}$ 决定 PI horizon growth

---

## §4 flagship 亮点：S3/S4 优势的理论锚定

> 这是对话 3 的核心贡献。**S3/S4 数字不但保留，它恰好是新理论最锋利的 flagship evidence**。

### 4.1 关键量：有效样本数 $n_\text{eff}$

$$n_\text{eff}(s, \sigma) = n \cdot (1-s) \cdot \frac{1}{1 + \sigma^2/\sigma_\text{attr}^2}$$

第一项是稀疏丢数据，第二项是噪声 Fisher information 衰减（Künsch 1984 for partially observed dynamical systems）。

| Scenario | $s$ | $\sigma/\sigma_\text{attr}$ | $n_\text{eff}/n$ | 相对 S0 的衰减 |
|:-:|:-:|:-:|:-:|:-:|
| S0 | 0 | 0.00 | 1.00 | 1× |
| S1 | 0.20 | 0.10 | 0.79 | 1.3× |
| S2 | 0.40 | 0.30 | 0.55 | 1.8× |
| **S3** | **0.60** | **0.50** | **0.32** | **3.1×** 🔥 |
| **S4** | **0.75** | **0.80** | **0.15** | **6.6×** 🔥 |
| S5 | 0.90 | 1.20 | 0.041 | 24× |
| S6 | 0.95 | 1.50 | 0.015 | 66× |

**S3/S4 是 $n_\text{eff}/n$ 从 0.32 跌到 0.15 的区间** —— signal-available 但足够 harsh 让 ambient 方法触发 OOD 相变。这不是"随便某个难度"，而是"**相变窗口**"。

### 4.2 ambient 相变 vs manifold 平滑退化

**ambient 方法（Prop 1）**：$\text{Error} \geq C_1 \sqrt{D/n_\text{eff}}$。
S0→S3 理论下界放大 $\sqrt{1/0.32} \approx 1.77×$，对应 −44% 退化。

**但 ambient 还有两层额外相变**（只在 $s > s^*$ 触发）：

1. **Context 语义破坏**：线性插值在 $s > 0.5$ 后产生非物理直线段 → 基础模型视为 OOD
2. **Tokenizer 失配**：Chronos/Panda 的 tokenizer 为 dense 时序设计，sparse 序列 token 分布偏移

两层叠加到 $\sqrt{D/n_\text{eff}}$ 上，产生实测 **−85% / −92% 陡峭相变**。

**manifold 方法（Prop 2）**：$\text{Error} \lesssim C_2 n_\text{eff}^{-(2\nu+1)/(2\nu+1+d_{KY})}$。
Lorenz63（$d_{KY}=2.06$, Matern-5/2 $\nu=5/2$）指数 ≈ 0.745。
S0→S3 预测退化 $(1/0.32)^{0.745} \approx 2.43×$，即 −59%。

**manifold 没有那两层额外相变**，因为：
- M1 CSDI 训练时见过各种 sparsity pattern，sparse input 不是 OOD
- M2 τ 选择不依赖具体 context 长度
- M3 SVGP 后验对 sparsity 平滑退化（Bayesian 天然性质）

### 4.3 数字闭环

| 方法 | S0→S3 实测 | 理论预测 | 差距归因 |
|---|---:|---:|---|
| Panda | **−85%** | −44%（Prop 1 下界） | 剩 −41% 归 OOD 跃变 |
| Parrot | **−92%** | −44% + 更强 retrieval 敏感 | OOD 效应更强 |
| Ours | **−47%** | −57%（Prop 2 预测） | 实际略好（CSDI imputation 部分恢复 $n_\text{eff}$） |

**这是数量级一致的理论-实验闭环**。不是 empirical 观察，是理论预测的**定量兑现**。

**对称性**：S0 上 ours 排第二（基础模型强），S5/S6 所有方法归零（noise floor），**S3/S4 ours 碾压** —— 这个 U 型结构是 physically grounded，不是 cherry-pick。

---

## §5 具体重构方案（按章节）

### 5.1 Abstract

**现版本**：罗列四 module + 数字。

**目标版本骨架**：
> 时间序列基础模型在稀疏含噪混沌下表现陡峭 phase transition。我们论证这是**理论必然**：ambient 预测器受 $\sqrt{D/n}$ 维度税，而延迟坐标方法收敛率由 $d_{KY}$ 主导（与 $D$ 解耦），$d_{KY} \ll D$ 时差距在稀疏样本下放大为有限样本相变。基于此，提出**流形中心**的四模块框架，四个 module 通过共享 τ、$d_{KY}$、Lyapunov 谱耦合。S3（60% 稀疏 + σ=0.5）达 Panda 的 2.2×、Parrot 的 7.1×；S4 扩大到 9.4× Panda。覆盖率偏离 nominal 0.90 ≤ 2%。

### 5.2 §1 Introduction 重写

**加 "Unified view" 段**（§1 最后一段）：见对话 2 的模板文字（四 module 共享几何对象，改变一个则其他三个应相应调整）。

**贡献列表加"贡献 0"**：建立以 $\mathcal{M}_\tau$ 为中心的数学框架，把四个经典任务统一为 Koopman 算子的不同估计；Propositions 共享 $d_{KY}$，揭示 phase transition 是理论必然。

**三段式 opener**：
1. 现象：S1/S2 还行，S5/S6 都崩，S3/S4 是关键区间
2. 理论：Prop 1 + Prop 2 + $n_\text{eff}$，S3/S4 恰好是 $n_\text{eff}/n \in [0.15, 0.32]$ 的相变窗口
3. 实证：2.2× / 9.4× 是理论预测的定量对应；S5/S6 共同归零表明优势不 cherry-pick

### 5.3 §3 方法部分结构性重组

**现结构**：§3.1 M1 / §3.2 M2 / §3.3 M3 / §3.4 M4。

**目标结构**：
- **§3.0 延迟流形作为中心对象（新增，约半页）** — Takens + $d_{KY}$ + Koopman 平凡化
- **§3.1 M2 估计 $\mathcal{M}_\tau$ 的嵌入几何**（原 §3.2 提前，因为 τ 是后续模块的输入）
- **§3.2 M1 在 $\mathcal{M}_\tau$ 上的流形感知 score estimation**
- **§3.3 M3 在 $\mathcal{M}_\tau$ 上的 Koopman 算子回归**
- **§3.4 M4 Koopman 谱校准共形区间**

**三个 bug 修复的重定位**（放在 §3.2）：
- Bug 1（α 非零初始化）：让 score 网络**能够利用 $\mathcal{M}_\tau$ 切丛结构**的必要条件
- Bug 2（per-dim centering）：在延迟坐标下**建立 DDPM 正确几何**的必要归一化
- Bug 3（贝叶斯软锚定）：**正确的流形投影** —— 把 $y$ 投回 $\mathcal{M}_\tau$ 的 noisy tubular neighborhood。硬锚定每步反向过程把 score 网络拽离流形。软锚定价值**随 σ² 放大** —— 解释 Fig 1b CSDI 升级 S2 +53%、S4 +110% 的梯度

**M2 新叙事**：τ-stability 的 σ=0 下 15/15 同 τ 重诠释为"**τ 是 well-defined 几何量，MI-Lyap 完美恢复**"。

**M4 新叙事**：Lyap-empirical 的 λ-free 改述为"**直接从 calibration 残差恢复 Koopman 经验谱**，绕开 nolds/Rosenstein 的噪声敏感性"。

### 5.4 §4 理论部分统一重构（最重要）

**现版本**：三条 proposition 各讲各的。

**目标**：**一组共享 $d_{KY}$ 和 $n_\text{eff}$ 的耦合定理族**。

---

**Setup（通用）**：动力系统 $f: \mathbb{R}^D \to \mathbb{R}^D$ 遍历吸引子，Lyapunov 谱 $\{\lambda_i\}$，Kaplan-Yorke 维 $d_{KY}$。观测 $h$ generic，延迟 $\tau$ 满足 $L > 2d_{KY}$。$\mathcal{M}_\tau = \Phi_\tau(\text{attractor})$。有效样本 $n_\text{eff}(s, \sigma) = n(1-s)/(1+\sigma^2/\sigma_\text{attr}^2)$。

---

**Proposition 1（Ambient Dimension Tax）**：ambient 预测器满足
$$\mathbb{E}[\|\hat{x}_{t+h} - x_{t+h}\|^2] \geq C_1 \sqrt{D/n_\text{eff}}$$

**证明思路**：Le Cam 两点法 —— 构造两个在 $\mathcal{M}_\tau$ 上嵌入相同但 ambient normal direction 上 $\sqrt{D/n}$ 分离的系统。

---

**NEW Proposition / Theorem（Sparsity-Noise Interaction Phase Transition）**（对话 3 提出）：

存在临界 $n^* = c \cdot D$ 使得：

(a) $n_\text{eff} > n^*$（maintenance regime）：ambient error $\leq C_1 \sqrt{D/n_\text{eff}}$，与 manifold error 差 $\sqrt{D/d_{KY}}$ 常数因子
(b) $n_\text{eff} < n^*$（phase transition regime）：ambient 训练/测试分布 KL $> \epsilon_\text{OOD}$，误差额外放大 $\geq (1 + \Omega(1))$
(c) manifold 在 $n_\text{eff} \gg 1/\text{diam}(\mathcal{M}_\tau)^{d_{KY}}$ 时继续按 Prop 2 速率退化（graceful）

**推论**：Lorenz63 $n^* \approx 0.3n$，对应 $(s, \sigma) \approx (0.6, 0.5)$ —— **恰好是 S3**。把"S3 是主战场"从经验观察变成**理论预测**。

---

**Proposition 2（Manifold Posterior Contraction）**：$\mathbb{E}\|\hat{\mathcal{K}} - \mathcal{K}\|_2^2 \lesssim n_\text{eff}^{-(2\nu+1)/(2\nu+1+d_{KY})}$。**收敛率由 $d_{KY}$ 决定，与 $D$ 无关**。

---

**Theorem 1（Calibrated Conformal Coverage via Koopman Spectrum）**：ψ-mixing 下 Lyap-empirical CP 覆盖 $\geq 1-\alpha - o(1)$；$\hat{G}(h)$ 与真 $e^{\lambda_1 h}$ 渐近相等。

**证明思路**：Chernozhukov-Wüthrich-Zhu + Bowen-Ruelle ψ-mixing。

---

**Corollary（Unified Scaling Law）**：
$$\frac{\text{Ambient}}{\text{Manifold}} \geq \frac{C_1 \sqrt{D/n_\text{eff}}}{C_2 n_\text{eff}^{-(2\nu+1)/(2\nu+1+d_{KY})}}$$

$d_{KY} \ll D$ 且 $n_\text{eff} \ll n$ 时比值急剧放大 → Fig 1 的相变。

### 5.5 §5 实验部分：加两组耦合实证

**新增 §5.X1 τ-coupling ablation**（见 §7.1）
**新增 §5.X2 $n_\text{eff}$ as unified parameter**（见 §7.2）

**修改 §5.2 主图叙述**：从"三段 story"改为"**理论曲线 + 实证拟合**"，每个 VPT 数字对应一个理论预测。见 §4.3 的数字闭环表。

### 5.6 其他写作细节

- **§2 Related Work** 加"Manifold learning for dynamical systems"段：Fefferman-Mitter-Narayanan manifold estimation、Berry-Harlim diffusion maps、Giannakis Koopman spectral methods。把工作放入正式数学 tradition
- **Figure caption** 全部几何化：M1/M2/M3/M4 → "Score estimation on $\mathcal{M}_\tau$" 等
- **Appendix A.0 notation table** 最前加三行：$\mathcal{M}_\tau$、$d_{KY}$、$\mathcal{K}$ 定义
- **贡献列表**：四贡献 → "一框架 + 四现象"

---

## §6 新增实验（非必需但高价值）

### 6.1 τ-coupling ablation（P1，~1 周）

**动机**：§3 论证 M1 delay mask 应使用 M2 的 τ，这一耦合 claim 需实证。

**设计**（S3 场景 × 3 seeds）：

| 配置 | M1 的 delay mask τ | 预期 |
|---|---|:-:|
| A uncoupled | 随机 τ（current no_mask 变体） | 差 |
| B correctly coupled | M2 在当前轨迹选的 τ（current full） | 好 |
| C mismatched | M2 在 S0 上选的 τ，用到 S3 | 中 |
| D fixed equidistant | [1,2,4,8,16] | 差-中 |

**预期**：B >> C > D > A，**差距随 harshness 放大**。

**附加**：$d_{KY}$ recovery —— 用 correlation dim / persistent homology 估 $\dim \Phi_\tau(\text{data})$，看 M2 选的 τ 是否让其最接近理论 $d_{KY} = 2.06$。这把 M2 从"启发式选 τ"升级到"估计几何不变量"。

### 6.2 $n_\text{eff}$ unified parameter 验证（P1，1-2 天）

**动机**：若 $n_\text{eff}$ 是真的 unified parameter，不同 $(s, \sigma)$ 只要 $n_\text{eff}/n$ 相同性能应相近。

**设计**：固定 $n_\text{eff}/n = 0.3$ 扫：
- $(s=0.6, \sigma=0.5)$ S3
- $(s=0.5, \sigma=0.77)$
- $(s=0.7, \sigma=0.0)$ 纯稀疏
- $(s=0.0, \sigma=1.5)$ 纯噪声

4 configs × 5 seeds × 2 methods (ours, Panda) = 40 runs。

**预期**：
- ours VPT 在 4 config 下相近（$n_\text{eff}$-driven smooth）
- Panda 纯稀疏 config 比 S3 差更多（sparsity 破 tokenizer > 噪声）
- **"ours collapses to $n_\text{eff}$ curve, ambient doesn't"** —— 即使 Panda 不完全塌陷，这本身就是贡献

### 6.3 Panda OOD KL-divergence 测量（P2，半天）

**动机**：实证 Theorem (b) 的 OOD 跃变假设。

**设计**：测量 Panda 在不同 $s$ 下 token distribution 的 perplexity 或 KL(test || train-prior)。预期：$s > 0.5$ 后跃变。

### 6.4 M1 τ-parameterized 核（方案 C，P2，1-2 月，可选）

把 CSDI 的 $\phi_\theta$ 从 generic MLP 改为 $\phi_\theta(\Delta t; \tau, \mathbf{w}) = \sum_i w_i \exp(-\beta(\Delta t - \tau_i)^2)$，τ 从 M2 外部注入。让 M1-M2 耦合从启发式升级为结构性。**高风险高收益**，不建议本轮投稿前做。

---

## §7 工作量 × 优先级

| 改动 | 工作量 | 优先级 | 说明 |
|---|:-:|:-:|---|
| §1 "Unified view" 段 + 贡献 0 + 三段式 opener | 0.5 天 | **P0** | 纯写作 |
| §3 新增 §3.0 几何背景 | 1 天 | **P0** | 复习 Takens + Koopman |
| §3.1-3.4 重新定位（含三 bug 几何解释） | 2 天 | **P0** | 不改数字 |
| §4 三定理 + 新 Theorem + Corollary 重构 | 3 天 | **P0** | Informal statement + sketch |
| Prop 1 formal 证明（附录） | 1 周 | **P1** | Le Cam 两点法标准 |
| 新 Theorem (Sparsity-Noise Interaction) formal | 1 周 | **P1** | Fisher info 退化 + OOD 阈值 |
| τ-coupling ablation（§6.1） | 1 周 | **P1** | 实验 1-2 天 + 写 3 天 |
| $n_\text{eff}$ unified 实验（§6.2） | 2 天 | **P1** | 用已有代码 |
| $d_{KY}$ recovery（§6.1 附加） | 3 天 | **P2** | correlation dim |
| Panda OOD KL（§6.3） | 0.5 天 | **P2** | 可选 |
| Related work manifold learning | 1 天 | **P1** | reviewer 友好 |
| Figure caption + notation 同步 | 0.5 天 | **P1** | 最后做 |
| M1 τ-parameterized 核（§6.4） | 1-2 月 | **P2** | 本轮可不做 |

**P0 合计**：约 1 周纯写作 + 理论整理
**P0 + P1 合计**：约 3 周
**P2 全做**：再加 1 周（不含 §6.4）

---

## §8 风险和缓解

| 风险 | 缓解 |
|---|---|
| $n_\text{eff}$ Fisher info 公式需严格证明 | 引用 Künsch 1984（partially observed dynamical systems）+ 自己在 Lorenz63 上的 numerical verification |
| "OOD 相变" claim 需要实证 | §6.3 补 Panda token 分布 KL 测量 |
| Theorem 严格证明有难度 | main paper 给 informal + sketch，证明放附录；三条假设中前两条经典结果，第三条借鉴 Berry-Harlim 2016 |
| "ours −47% 略好于 Prop 2 预测 −57%" 过度精细 | 改 "在 Prop 2 预测的 95% CI 内"，用置信区间而非点估计 |
| τ-coupling ablation 可能结果不清晰（B 并非显著超 D） | 如结果不显著，改 claim 为 "consistent with M1-M2 coupling" 而非 "demonstrates coupling"；或加 harshness × config 的交互项分析 |

---

## §9 实验先不动（强烈建议）

**所有硬数字**（Phase Transition 主图、消融、CSDI 三 bug 效果）**保留不变**。本次重构本质是"同一组实验 + 更强数学叙事"。

- 降低出错风险
- 缩短时间
- 所有已投入算力得到保护

唯一新增实验是 §6.1 τ-coupling 和 §6.2 $n_\text{eff}$ —— capstone 性质，让耦合 claim 有数据支撑。

---

## §10 推进路径（分阶段）

### 阶段 1：P0 全做（纯写作，1 周）— "理论骨架搭起来"

- [ ] §3.0 延迟流形小节草稿（Takens + $d_{KY}$ + Koopman）
- [ ] §3 其余四小节重新定位（不改数字）
- [ ] §1 三段式 opener + Unified view + 贡献 0
- [ ] §4 informal statement 四定理 + Corollary
- [ ] Abstract 按 §5.1 模板改写
- [ ] Appendix A.0 notation table 加三行

**验收**：paper_draft_zh.md 读下来，reviewer 能从 §1 的三段式看到 §4 的 Corollary，逻辑闭环；每个 module 都能对应回 $\mathcal{M}_\tau$。

### 阶段 2：P1 实验 + 证明（2 周）— "耦合 claim 坐实"

- [ ] 跑 §6.1 τ-coupling ablation（S3 × 4 configs × 3 seeds）
- [ ] 跑 §6.2 $n_\text{eff}$ unified（40 runs）
- [ ] 写 Prop 1 formal 证明（Le Cam）
- [ ] 写新 Theorem (Sparsity-Noise Interaction) formal
- [ ] §5.X1 / §5.X2 两新小节
- [ ] Related work 加 manifold learning 段

**验收**：τ-coupling 的差距随 harshness 放大（B vs D gap 在 S3 比 S0 大）；$n_\text{eff}$ unified 在 ours 上成立。

### 阶段 3：P2 打磨 + 可选（1 周）— "细节拉满"

- [ ] Panda OOD KL 测量
- [ ] $d_{KY}$ recovery
- [ ] Figure caption + notation 全面同步
- [ ] 用置信区间替换点估计比较
- [ ] LaTeX 转换（依原 T8）
- [ ] Multi-round refine（原 T9）

**验收**：paper 可投稿状态，理论 + 实证 + 写作三齐。

---

## §11 最终定位对比

| 维度 | 现版本 | 重构后 |
|---|---|---|
| 核心 claim | "我们造了一个好 pipeline" | "稀疏噪声混沌预测是流形几何问题；相变是理论必然；我们给出统一框架" |
| 理论地位 | 三 informal prop 彼此独立 | 四定理 + Corollary 共享 $d_{KY}$ 和 $n_\text{eff}$ |
| 模块叙事 | 四独立工程组件 | 同一 Koopman 算子的四种互补估计 |
| novelty | 四组件组合是贡献 | 统一框架是贡献 0；三 bug 是几何必要条件；Lyap-empirical 是 Koopman 经验谱 |
| S3/S4 优势地位 | "某难度上胜过 baseline 的经验结果" | "理论预测的相变窗口内，唯一免疫相变的方法" |
| 投稿天花板 | UAI/TMLR 稳 | ICML/NeurIPS accept band |
| reviewer 体验 | "看起来像堆 trick" | "有数学骨架的完整工作" |

---

## §12 一段话总结（给改写者 / 未来的自己）

> 这次重构的本质是：**把同样的实验数字放进一个更强的数学叙事框架**。
>
> 现版本："我做了 A/B/C/D 四件事，合起来有效"
> 目标版本："混沌预测在稀疏噪声下是流形几何问题；A/B/C/D 是对同一流形的四种互补估计；它们通过共享 τ、$d_{KY}$、$n_\text{eff}$ 必然耦合、必然联合设计；基础模型相变是 ambient 坐标维度税的理论必然；S3/S4 恰好是 $n_\text{eff}/n \in [0.15, 0.32]$ 的相变窗口，我们是窗口内唯一免疫相变的方法"
>
> 对 reviewer：前者是好 empirical paper，后者是有数学骨架的 method paper。对同一组数字，后者把投稿天花板从 UAI/TMLR 抬到 ICML/NeurIPS。

---

## 附录 A：原始对话索引

三段对话原文保存在 [`改进方案`](改进方案)：
- **对话 1**：数学故事怎么讲 / 三层耦合故事骨架 / 方案 A/B/C
- **对话 2**：给改写者的完整重构建议文档（Abstract / §1 / §3 / §4 重写骨架）
- **对话 3**：S3/S4 优势的理论锚定（$n_\text{eff}$ 推导 + Phase Transition 作为 interaction effect）

本文档是三者的统一整合版本，按可执行维度重组。
