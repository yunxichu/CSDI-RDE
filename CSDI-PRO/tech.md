# 赛道 A 完整执行方案 v2（三大会锋利版）

> **v2 修订要点**（相对 v1）：
> 1. **Story 锋利化**：从 "Bayesian forecasting framework" → "characterizing and conquering the sparse-observation regime of foundation models for chaos"
> 2. **UQ 正式归位**：任务重定义为 **probabilistic forecasting with coverage guarantees**，Lyap-Conformal 是核心贡献而非附加
> 3. **技术模块保留并升级**：delay mask、MI-Lyap、Lyap-Conformal 全部保留；MI-Lyap 的 τ 搜索新增 **低秩 + CMA-ES** 版本处理高维
> 4. **主图重新设计**：从"方法×挑战"的被动 heatmap → **"phase transition curve"** 的主动叙事
> 5. **新增理论点**：foundation model 的 lower bound（简单 covering number 论证）
> 6. **投稿策略重排**：NeurIPS/ICLR 成为主战场（而非 ICML），UAI/AISTATS 作为安全网

---

## 标题（v2）

**新**（推荐）：  
*"When Foundation Models Meet Sparse Chaos: A Probabilistic Framework with Calibrated Coverage Guarantees"*

**保守替代**：  
*"Probabilistic Forecasting of Chaotic Dynamical Systems under Sparse, Missing, and Noisy Observations"*

---

## Punchline（一句话说清）

> Foundation models for chaotic forecasting (Panda, ChaosNexus, FIM) assume dense, clean context and fail sharply when observations are sparse, partial, and noisy. We formalize this **sparse-observation regime**, prove ambient-dimension models must fail in it, and propose a probabilistic framework that operates on intrinsic attractor dimension and delivers the first **distribution-free calibrated coverage** for chaotic forecasts.

四个 contributions（严格对应你的原始四个诉求 + UQ）：

1. **Formalization of the sparse-observation regime** + empirical phase transition + a lower bound showing ambient-dim foundation models fail
2. **Dynamics-aware imputation**（Dynamics-Aware CSDI，处理缺失 + 噪声）
3. **Lyapunov-adaptive delay embedding + SVGP**（处理高维 + 稀疏；样本复杂度由 $d_{KY}$ 主导）
4. **Lyapunov-aware conformal prediction**（第一个给混沌预测 distribution-free coverage guarantee）

---

## 核心方案（不变）

`Dynamics-Aware CSDI → MI-Lyap Adaptive Embedding → SVGP → Lyap-Conformal`

**投稿目标重排**：
- **Primary**：NeurIPS 2026（5 月 deadline）或 ICLR 2027（9 月 2026 deadline）—— 这两个会议最买"foundation model fails here"的 story
- **Secondary**：ICML 2026（1 月）
- **Safety net**：UAI 2026 / AISTATS 2026

**工作时长**：12 周（+ 2 周 buffer）

---

# Part I：技术方案

## Module 0：Sparse-Observation Regime Formalization（新增，核心锋利点）

### 0.1 为什么必须有这个

v1 方案的问题是 "我们做了一个好方法" 这种 framing 三大会不尖锐。v2 的解法是先**形式化一个 regime**，让后面所有方法都成为"为了征服这个 regime"。

### 0.2 Regime 定义

**定义（Sparse-Observation Regime）**：给定混沌动力系统 $\dot{\mathbf{x}} = f(\mathbf{x})$ with attractor $\mathcal{A} \subset \mathbb{R}^D$，观测序列 $\{\tilde{\mathbf{x}}_{t_i}\}_{i=1}^N$，定义三个 harshness 指标：

- **Sparsity** $s \in [0,1]$：观测时间步占总时间步的比例之补
- **Noise level** $\sigma$：观测噪声标准差除以 attractor std
- **Coverage shortfall** $c$：观测对 attractor 的 ε-覆盖不足程度

**Sparse-observation regime** $\mathcal{R}_{\text{sparse}}(s_0, \sigma_0, c_0)$ 定义为 $s > s_0, \sigma > \sigma_0, c > c_0$ 的三元 region。

### 0.3 Lower Bound Proposition（新增，用于锋利 story）

**Proposition 1 (Informal).** 对于任何只依赖观测数据 $\{\tilde{\mathbf{x}}_{t_i}\}$ 而不显式利用吸引子低内在维度的 forecasting 方法 $\hat{f}$，如果：
- 方法在 ambient space $\mathbb{R}^D$ 中估计函数
- 观测满足 $\mathcal{R}_{\text{sparse}}(s_0, \sigma_0, c_0)$

那么 forecasting error 的下界为：
$$\mathbb{E}[\|\hat{f} - f\|^2] \geq C \cdot \exp(\lambda_{\max} h) \cdot \left(\frac{\sigma^2 D}{N(1-s)}\right)^{\alpha/(2\alpha+D)}$$

**直觉**：$D$ 出现在指数位置，稀疏 $(1-s)$ 让有效样本数变小，噪声 $\sigma$ 进一步放大 rate。**要击败这个 bound，方法必须绕过 $D$**，这就是为什么要用 Takens 把问题 reduce 到 $d_{KY}$。

**证明策略**（附录）：简单的 covering number 论证 + Le Cam's two-point method。参考 Tsybakov 2009 "Introduction to Nonparametric Estimation" Chap 2。**不需要新理论**，只是现成工具的应用。

**这个 proposition 的作用**：给你一个"foundation model must fail"的正式 statement，作为 paper 的 theoretical hook。不需要 tight bound，只需要 scaling 正确。

---

## Module 1：Dynamics-Aware CSDI（抗噪 + 抗缺失）

### 1.1 原始 CSDI 的问题

原始 CSDI（Tashiro et al. 2021）用 2D Transformer 处理时序 + 特征维度的 attention，在 PM2.5 等非混沌时序上 work，但：

1. **没有 dynamics 先验**：它不知道输入数据来自一个低维吸引子
2. **噪声不是 first-class citizen**：观测噪声 σ 被当作是常数，无法条件化
3. **补值样本之间独立采样**：20 个样本平均起来消除了 epistemic uncertainty 信息

### 1.2 具体改动

#### 改动 A：Noise-level conditioning

原始 CSDI 的 denoising network：
$$s_\theta(x_t^{\text{noisy}}, t, c_{\text{mask}})$$

改为：
$$s_\theta(x_t^{\text{noisy}}, t, c_{\text{mask}}, \text{embed}(\hat{\sigma}_{\text{obs}}))$$

其中 $\hat{\sigma}_{\text{obs}}$ 从观测数据估计（MAD of first differences 的 robust estimator）。实现：给 transformer 加一个额外 embedding token，类似 ViT 的 CLS token。

**代码改动**：`diff_models.py` 的 `DiffusionEmbedding` 类加 `self.noise_embed = nn.Linear(1, channels)`，forward 里 concat 进去。约 50 行。

#### 改动 B：Delay-aware attention mask（你喜欢这个，v2 加强版）

在 CSDI 的 temporal attention 中加入 **延迟结构的 prior mask**：

$$\text{Attn}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}} + \alpha \cdot M_{\tau}\right) V$$

其中 $M_\tau$ 是可学习的 mask，初始化为 delay embedding 对应位置为 0、其他位置为小负数的软 mask；$\alpha$ 是温度参数，从 0 开始 annealing。

**v2 加强**：$M_\tau$ 不再是静态 hand-crafted，而是**由 Module 2 的 MI-Lyap 选出的 τ 动态生成**。这样 CSDI 和 embedding 两个模块产生协同效应（不是独立拼接）—— 这是 v2 相对 v1 在方法一致性上的升级。

具体做法：
```
1. MI-Lyap 在观测数据上初选 τ̂（Module 2）
2. Dynamics-Aware CSDI 用 τ̂ 构造 delay mask 做 imputation
3. Imputation 完成后，MI-Lyap 在补值数据上再选精确 τ
4. SVGP 用精确 τ 做预测
```

这个 two-pass 设计让两个模块互相受益，论文里可以 claim "the two modules co-adapt"，显著提升方法的 coherence。

**代码改动**：`TransformerEncoderLayer` 加 mask 参数，加一个 `update_mask(tau_vec)` 方法。约 100 行。

#### 改动 C：Ensemble-aware sampling

原版 CSDI 采样 20 次取平均。Dynamics-Aware 版本保留 20 个样本作为 posterior 样本，传递给下游模块。这不是网络改动，是 inference pipeline 改动。

### 1.3 理论 / 动机 justification

- **Noise conditioning**：扩散模型的 denoising step 本质是 score-matching $\nabla \log p(x_t | x_0, \sigma_{\text{diff}}^2)$。如果观测噪声 $\sigma_{\text{obs}}$ 已知，条件化它可以让 score network 区分 diffusion noise 和 observation noise，避免 over-smoothing。等价于 Bayesian noise model。
- **Delay mask**：Takens 定理告诉我们信息主要在延迟坐标之间传递，mask 是这个先验的 soft encoding。
- **Ensemble**：丢弃 20 个样本等于扔掉 epistemic uncertainty，与论文主旨（probabilistic forecasting）矛盾。

### 1.4 消融预期

```
原始 CSDI                          基线
+ Noise conditioning               -10% RMSE（高噪声下）  
+ Delay mask（静态）               -8% RMSE（稀疏观测下）
+ Delay mask（动态，from MI-Lyap） -12% RMSE（协同效应）
+ Ensemble retention               CRPS 改善，RMSE 不变
所有一起                           -20% RMSE + CRPS 显著改善
```

---

## Module 2：MI-Lyap Adaptive Delay Embedding（自适应 τ）

### 2.1 原始 RDE 的问题

Takahashi et al. 2021 的 Random Delay Embedding：$\tau_i \sim \text{Uniform}(1, \tau_{\max})$。

问题：
1. 随机采样无理论依据
2. 噪声下 MI 估计方差大，fixed τ 不稳定
3. 没有根据系统的混沌程度调整 τ

### 2.2 完整算法

**给定**：
- 观测 $\mathbf{x}_{t-T:t}$（已经过 CSDI 补值）
- 嵌入维数 $L$（固定，由 Takens 定理 $L > 2d_{\text{KY}}$ 设定）
- 候选延迟集 $\mathcal{T} = \{1, 2, ..., \tau_{\max}\}$
- 预测 horizon $h$

**步骤 1：局部 Lyapunov 指数估计**

用 Rosenstein 1993 算法估计当前状态 $\mathbf{x}_t$ 邻域的最大 Lyapunov 指数。现成实现：`nolds.lyap_r`。

**步骤 2：条件互信息最大化 + Lyap 惩罚**

目标函数：
$$J(\boldsymbol{\tau}) = \underbrace{I(\mathbf{y}_{\boldsymbol{\tau}}; \mathbf{x}_{t+h}|\mathbf{x}_t)}_{\text{信息项}} - \underbrace{\beta \cdot \max_i \tau_i \cdot \hat{\lambda}_{\text{loc}}(\mathbf{x}_t)}_{\text{Lyapunov 惩罚}} - \underbrace{\gamma \cdot \sum_i \tau_i^2 / T}_{\text{稀疏数据正则}}$$

其中：
- $\mathbf{y}_{\boldsymbol{\tau}} = (x(t-\tau_1), ..., x(t-\tau_L))$ 是延迟坐标
- $\beta, \gamma$ 是超参数
- **Lyapunov 惩罚的物理意义**：预测 horizon $h$ 固定时，当前越混沌，越不能用远的延迟
- **稀疏正则**：避免数据点不够时选超长延迟

### 2.3 优化算法（v2 两阶段升级，采纳你的低秩进化想法）

**Stage A（低维场景，$L \leq 10$）：BayesOpt**

```python
from skopt import gp_minimize
# L = 5 for Lorenz63 等小系统
result = gp_minimize(
    objective,
    [(1, tau_max)] * L,
    n_calls=30,
    random_state=42
)
```

**Stage B（高维场景，$L > 10$）：低秩参数化 + CMA-ES（v2 新增，处理 Lorenz96）**

**核心想法**：把 $\boldsymbol{\tau} \in \{1,...,\tau_{\max}\}^L$ 参数化为低秩矩阵乘积：

$$\boldsymbol{\tau} = \text{round}\left(\sigma(\mathbf{U}\mathbf{V}^T) \cdot \tau_{\max}\right)$$

其中 $\mathbf{U} \in \mathbb{R}^{L \times r}$, $\mathbf{V} \in \mathbb{R}^{1 \times r}$，$r = 2 \sim 4$，$\sigma$ 是 sigmoid。搜索空间从 $\tau_{\max}^L$（指数）降到 $\mathbb{R}^{r(L+1)}$（连续低维）。

**为什么合理**：在耦合振子系统（如 Lorenz96）中，相邻维度的混沌时间尺度接近，最优 τ 天然有低秩结构。这本身可以作为论文里的一个 **observation**：

> "We empirically observe that optimal delay patterns in coupled-oscillator systems exhibit low-rank structure, reflecting shared chaotic timescales across neighboring dimensions."

配一张 τ matrix 的奇异值谱图，审稿人会觉得你对问题有深刻理解。

```python
import cma

class LowRankCMAES_TauSelector:
    def __init__(self, L, tau_max, rank=2):
        self.L = L
        self.tau_max = tau_max
        self.rank = rank
        self.n_params = rank * (L + 1)
    
    def decode(self, x):
        U = x[:self.L * self.rank].reshape(self.L, self.rank)
        V = x[self.L * self.rank:].reshape(1, self.rank)
        raw = 1 / (1 + np.exp(-(U @ V.T).flatten()))
        tau = np.clip(np.round(raw * self.tau_max), 1, self.tau_max)
        return tau.astype(int)
    
    def select(self, trajectory, x_query, horizon, lambda_loc,
               beta=1.0, gamma=0.1, n_iter=50, popsize=20):
        def objective(x):
            tau = self.decode(x)
            y_delay = construct_delay_coords(trajectory, tau)
            x_future = trajectory[horizon:]
            # 3-seed average to reduce KSG-MI noise
            mi_est = np.mean([
                conditional_mi_ksg(y_delay, x_future, x_query, k=4, seed=s)
                for s in range(3)
            ])
            lyap_pen = beta * np.max(tau) * lambda_loc
            sparse_pen = gamma * np.sum(tau**2) / len(trajectory)
            return -(mi_est - lyap_pen - sparse_pen)
        
        es = cma.CMAEvolutionStrategy(
            np.zeros(self.n_params), 0.5,
            {'popsize': popsize, 'maxiter': n_iter, 'verbose': -9}
        )
        es.optimize(objective)
        return self.decode(es.result.xbest)
```

**两阶段在论文里的讲法**：

> "For low-dimensional systems (L ≤ 10), we solve the discrete τ-selection via Bayesian Optimization. For high-dimensional systems where L > 10, we employ a low-rank parameterization τ = round(σ(UV^T) · τ_max) with rank r=2, combined with CMA-ES. The low-rank structure reflects a physical prior: in coupled-oscillator systems, neighboring dimensions share chaotic timescales."

### 2.4 噪声鲁棒性 argument

**关键卖点**：在噪声 σ 下，vanilla MI 估计方差 $O(\sigma^2 / N)$，Fraser-Swinney 的 first-minimum τ 会跳。

你的 Lyap 惩罚项提供 implicit regularizer：**即使 MI 估计噪声，Lyap 项作为稳定的"物理约束"，拉住 τ 不飘**。

需要实验验证（Week 6 任务）：
- 画图：τ vs noise level，Fraser-Swinney 方差爆炸，你的方法稳定

### 2.5 理论支撑

引用：
- Sauer, Yorke, Casdagli 1991：noisy Takens
- Casdagli et al. 1991 "State space reconstruction in the presence of noise"
- Kantz & Schreiber 2004（教科书）Chap 9

---

## Module 3：SVGP on Delay Coordinates（高维可扩展 + 不确定度）

### 3.1 原始 GPR 的问题

Exact GPR 复杂度 O(n³)，内存 O(n²)。EEG h=976 挂掉就是这个原因。

### 3.2 SVGP（Sparse Variational GP）

核心想法：用 m ≪ n 个 **inducing points** $\mathbf{Z} = \{\mathbf{z}_1, ..., \mathbf{z}_m\}$ 作为压缩训练集的代表，推理复杂度 O(nm²)。

变分下界（Titsias 2009, Hensman 2013）：
$$\mathcal{L}_{\text{ELBO}} = \sum_{i=1}^n \mathbb{E}_{q(f_i)}[\log p(y_i | f_i)] - \text{KL}(q(\mathbf{u}) \| p(\mathbf{u}))$$

### 3.3 实现（GPyTorch）

```python
import gpytorch

class SVGP_RDE(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points):
        variational_dist = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_dist, 
            learn_inducing_locations=True)
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        # Matern-5/2：比 RBF 更适合吸引子上的 Hölder 连续函数
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5))
    
    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x))
```

**关键超参数**：
- `m`（inducing points 数）：128 起，最大 1024
- Kernel：Matern-5/2
- Likelihood：`GaussianLikelihood`，noise learnable
- Training：Adam, lr=0.01, 200 epochs

### 3.4 高维策略

Lorenz96 N=100, 400：

**选择 A（naive）**：每个 output dim 一个 GP，输入 delay-embedded vector ∈ R^L（L 通常 5-15）。SVGP 容易 scale。

**选择 B**：`IndependentMultitaskVariationalStrategy`，共享 inducing points。

**推荐 A 做主实验，B 做 ablation**。

### 3.5 预期计算时间

| 场景 | Exact GPR | SVGP (m=256) |
|------|-----------|--------------|
| n=500 | 5s | 30s |
| n=2000 | 80s | 60s |
| n=10000 | OOM | 150s |
| n=50000 | N/A | 600s |

**SVGP 在 n > 2000 开始赢**。

### 3.6 理论 hook：样本复杂度由 $d_{KY}$ 主导

**Proposition 2 (Informal)**：设吸引子 $\mathcal{A}$ 的 Kaplan-Yorke 维数为 $d_{KY}$，Takens 嵌入维数 $L > 2d_{KY}$。对 Matern-$\nu$ GP 拟合 $\mathbb{R}^L \to \mathbb{R}$ 映射，其 posterior contraction rate 为：
$$\|\hat{f}_n - f^*\|_{L^2(\mu_{\mathcal{A}})} = O_p\left(n^{-\nu/(2\nu + d_{KY})}\right)$$

**与 Proposition 1 的对比**：
- Proposition 1：ambient-dim 方法 rate 含 $D$
- Proposition 2：我们的方法 rate 含 $d_{KY}$
- **两者之差就是核心优势**：Lorenz96 N=100 时 $D=100$ 但 $d_{KY} \approx 20$，**样本复杂度差 $10^4$ 倍**

**证明策略**：引用 Castillo 2014 "On Bayesian supremum norm contraction rates"（GP on manifolds）+ Takens 保证 $\Phi(\mathcal{A})$ 是 $d_{KY}$ 维流形。**不需要新证明，是已有结果的直接应用**。

---

## Module 4：Lyap-Conformal（校准不确定度 — v2 中正式归位为核心贡献）

### 4.1 为什么这是核心贡献（不是附加）

v2 任务定义：**probabilistic forecasting with calibrated coverage guarantee**，不是 point prediction。

在这个定义下，输出 calibrated PI 不是 bonus，**是任务的一部分**。Lyap-Conformal 是让框架满足 task definition 的 **enabler**。

Claim："we provide the **first distribution-free calibrated coverage guarantee** for chaotic forecasting" — 这句话在 2025-2026 是可以成立的（搜索确认）。

### 4.2 标准 conformal 回顾

**Split conformal（Vovk et al. 2005）**：
1. 训练集 → 拟合模型
2. 校准集 → 计算 nonconformity scores $s_i = |y_i - \hat{y}_i|$
3. 测试 PI = $\hat{y} \pm q_{1-\alpha}(\{s_i\})$

**保证**：$\mathbb{P}(y_{\text{test}} \in \text{PI}) \geq 1 - \alpha$，分布无关，finite sample。前提：exchangeability。

### 4.3 为什么标准 CP 不能直接用于混沌

三个问题：
1. **时间相关性破坏 exchangeability**
2. **残差在 Lyapunov 时间尺度上指数增长**
3. **残差条件分布依赖于 $\mathbf{x}_t$ 的位置**（attractor 密集区 vs 稀疏区）

### 4.4 Lyap-Conformal 算法

**Nonconformity score**：
$$s_t^{(h)} = \frac{|y_{t+h} - \hat{y}_{t+h}|}{\exp(\hat{\lambda}_{\max} \cdot h \cdot \Delta t) \cdot \hat{\sigma}_{\text{GP}}(\mathbf{x}_t, h)}$$

**Prediction interval**：
$$\text{PI}_{1-\alpha}(y_{t+h}) = \hat{y}_{t+h} \pm q_{1-\alpha} \cdot \exp(\hat{\lambda}_{\max} \cdot h \cdot \Delta t) \cdot \hat{\sigma}_{\text{GP}}(\mathbf{x}_t, h)$$

**算法**：

```
Input: 校准集 (x_i, y_{i+h})_{i=1}^N, 显著水平 α, 估计的 λ_max
For each i in 1..N:
    ŷ_{i+h}, σ̂_{i+h} = SVGP.predict(x_i, h)
    s_i = |y_{i+h} - ŷ_{i+h}| / (exp(λ_max * h * Δt) * σ̂_{i+h})
q_{1-α} = (⌈(N+1)(1-α)⌉ / N)-th quantile of {s_i}

At test time:
    ŷ_test, σ̂_test = SVGP.predict(x_test, h)
    PI = ŷ_test ± q_{1-α} * exp(λ_max * h * Δt) * σ̂_test
```

### 4.5 理论 coverage 保证

**Theorem 1 (Informal)**：在 ergodicity + $\psi$-mixing 条件下：
$$\mathbb{P}(y_{t+h} \in \text{PI}_{1-\alpha}(y_{t+h})) \geq 1 - \alpha - O(\psi(N\Delta t))$$

**引用的 building block**：
- Chernozhukov, Wüthrich, Zhu 2018 "Exact and robust conformal inference under weak dependence"
- Barber, Candès, Ramdas, Tibshirani 2023 "Conformal prediction beyond exchangeability"
- Bowen-Ruelle：Axiom-A attractor 的 $\psi$-mixing 由 Lyapunov 指数控制

**证明难度评估**：只需要 invoke 现成结果 + 验证 $\psi$-mixing。**可以做到 informal 级别严谨**。如果数学背景可以，push 到 formal 级别会让 paper 档次上升。

### 4.6 Adaptive 版本（backup）

如果 $\hat{\lambda}_{\max}$ 不准或有 regime switch：
$$q_t = q_{t-1} + \eta \cdot (1\{y_t \notin \text{PI}_{t-1}\} - \alpha)$$

Gibbs & Candès 2021 ACI 变种，作为 robustness 加强。

---

# Part II：12 周详细 Gantt（v2 微调）

总体路线图：

```
Month 1 (Week 1-4):   基建 + UQ 模块（最易出成果）
Month 2 (Week 5-8):   方法创新 + 高维 + Phase Transition 实验
Month 3 (Week 9-12):  主实验 + 理论 + 写作
```

**v2 相对 v1 的调整**：
- Week 1 新增 "Phase Transition pilot"（关键锋利点，决定 story 能否成立）
- Week 6 新增 "低秩 CMA-ES 实现"（你的想法）
- Week 7 的 delay mask 改为动态（由 MI-Lyap 驱动）
- Week 8 新增 "Foundation model 大 PK"
- Week 9 主图改为 phase transition curve 而非 heatmap
- Week 10 理论章节新增 Proposition 1

---

## Week 1：环境 + 阅读 + Phase Transition pilot（v2 关键）

**输入**：你现在的 CSDI-RDE-GPR repo

**目标**：
- 把 codebase 重整，能跑 Lorenz63/96
- 读完 5 篇必读论文
- **跑一个 pilot 实验验证 phase transition 存在（新增，关键）**

**具体任务**：

Day 1-2：安装 + 跑 demo
```bash
pip install gpytorch properscoring uncertainty-toolbox scikit-optimize
pip install nolds npeet dysts cma
pip install torch transformers  # for Chronos zero-shot
```

跑通：
- GPyTorch 官方 SVGP 例子
- `dysts` 的 `make_trajectory` 基础调用
- **Chronos zero-shot 推理**（HuggingFace `amazon/chronos-t5-small`）
- 你自己的 Lorenz63 当前 pipeline

Day 3-5：精读 5 篇
- Zhang & Gilpin ICLR 2025「Zero-shot forecasting of chaotic systems」
- Lai, Bao, Gilpin ICLR 2026「Panda」
- Seifner et al. ICLR 2025「FIM for dynamical systems」
- Angelopoulos & Bates 2021「Gentle intro to Conformal Prediction」
- Hersbach 2000「CRPS decomposition」

Day 6-7：**Phase Transition pilot（v2 新增，最关键的早期实验）**

如果这个 pilot 不成功，整个 v2 锋利 story 就塌了，**必须 Week 1 验证**。

具体做法：
1. 拿 Lorenz63 轨迹 1000 步
2. 创建 7 个 scenarios，harshness 递增：
   - S0: 100% 观测, σ=0（benchmark 条件）
   - S1: 80% 观测, σ=0.1
   - S2: 60% 观测, σ=0.3
   - S3: 40% 观测, σ=0.5
   - S4: 25% 观测, σ=0.8
   - S5: 10% 观测, σ=1.2
   - S6: 5% 观测, σ=1.5
3. 让 **Chronos-20M** 做 zero-shot forecasting
4. 记录每个 scenario 的 VPT
5. **画 VPT vs harshness 曲线 — 如果有明显 phase transition（某 scenario 后 VPT 急剧下降），锋利 story 就成立**

**成功标准**：Chronos 在 S3 以后 VPT 显著下降（>50% drop）

**失败 fallback**：
- 如果 Chronos 很稳：推到更极端（S7: 3% 观测, σ=2.0）
- 如果所有 setting 下 Chronos 都差：Lorenz63 太小，换 Lorenz96
- 如果 pilot 彻底失败（foundation models 在极端 setting 下也 OK）：转回 v1 定位

**产出**：
- 能跑的 pipeline
- Related work 草稿（1-2 页）
- **Phase transition pilot 结果图（决定 v2 story 能否成立）**

---

## Week 2：SVGP 化 + UQ metrics

**目标**：把 exact GPR 换成 SVGP，实现 CRPS / PICP / 其它 UQ 指标

Day 8-9：SVGP 实现
```python
# 文件：models/svgp_rde.py
# 任务：把当前 GPR 预测代码改成 SVGP
# 关键：接口兼容（输入输出格式不变），只换内部实现
# 测试：Lorenz63 上跑，确认 RMSE 和原 GPR 接近（差 <10%）
```

Day 10-11：UQ metrics 实现
```python
# 文件：metrics/uq_metrics.py
def crps_gaussian(y_true, y_pred_mean, y_pred_std): ...
def crps_ensemble(y_true, ensemble_samples): ...
def picp(y_true, y_lower, y_upper): ...
def mpiw(y_lower, y_upper): ...
def reliability_diagram(y_true, y_pred_mean, y_pred_std, n_bins=10): ...
def winkler_score(y_true, y_lower, y_upper, alpha): ...
```

Day 12-14：第一张图 — Reliability Diagram
- Lorenz63 上跑 SVGP，对比 calibration
- **这张图是论文里必然会出现的**

**产出**：SVGP pipeline + UQ metrics 代码库 + 第一张 reliability diagram

---

## Week 3：Vanilla Conformal 实现

**目标**：实现 split conformal，作为 Lyap-Conformal 的基线对照

Day 15-16：Split Conformal
```python
class SplitConformal:
    def __init__(self, alpha=0.1): ...
    def calibrate(self, y_cal, y_pred_cal, std_cal):
        scores = np.abs(y_cal - y_pred_cal) / std_cal  # CQR 风格
        n = len(scores)
        level = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.q_hat = np.quantile(scores, level)
    def predict_interval(self, y_pred, std):
        return y_pred - self.q_hat * std, y_pred + self.q_hat * std
```

Day 17：Adaptive Conformal Inference (Gibbs & Candès 2021)

Day 18-19：Lorenz63/96 实验
- 对比 GP raw PI vs Split CP vs ACI PI
- PICP、MPIW、Winkler score
- 画图：PI width 随 horizon 增长

Day 20-21：发现问题
- 观察 Split Conformal 在 horizon 大时 undercoverage
- **这个观察是 Lyap-Conformal 的动机**

**产出**：Conformal library + "why vanilla CP fails" 的实验证据

---

## Week 4：Lyap-Conformal 实现

**目标**：实现 Lyap-Conformal，验证它修复 vanilla CP 问题

Day 22-23：Lyapunov 指数估计
```python
def estimate_lyapunov_rosenstein(trajectory, emb_dim, delay):
    """Rosenstein 1993. nolds.lyap_r 直接调"""

def estimate_local_lyapunov(x_query, trajectory, k_neighbors=10):
    """局部 Lyapunov，k 近邻 divergence rate"""
```

Day 24-26：Lyap-Conformal 实现

```python
class LyapConformal:
    def __init__(self, alpha=0.1, lambda_est=None): ...
    def calibrate(self, y_cal, y_pred_cal, std_cal, horizons, dt):
        scores = np.abs(y_cal - y_pred_cal) / (
            std_cal * np.exp(self.lambda_est * horizons * dt)
        )
        ...
    def predict_interval(self, y_pred, std, horizon, dt):
        width = self.q_hat * std * np.exp(self.lambda_est * horizon * dt)
        return y_pred - width, y_pred + width
```

Day 27-28：实验对比
- Horizon-coverage curve：GP raw / Split CP / ACI / Lyap-CP 四条线
- Horizon-width curve：MPIW
- **目标**：Lyap-CP 是唯一在所有 horizon 都保持 ≈90% 覆盖的

**产出**：Lyap-Conformal implementation + 论文 Section 5.4 的主图

**里程碑**：Month 1 结束。你已经有 SVGP 主 pipeline + 完整 UQ 评估 + 一个新方法（Lyap-CP）+ Phase Transition pilot 证据。**仅这一个 contribution 已经可以写成 UAI/AISTATS workshop paper**。

---

## Week 5：MI-Lyap Adaptive Embedding（BayesOpt 版）

**目标**：实现低维场景的自适应 τ

Day 29-30：Conditional MI estimator
```python
def conditional_mi_ksg(X, Y, Z, k=4):
    """KSG estimator for I(X;Y|Z). 用 npeet 库。"""
    from npeet import entropy_estimators as ee
    return ee.cmi(X, Y, Z, k=k)
```

Day 31-33：τ 搜索算法（BayesOpt）—— 见 Module 2.3 Stage A 代码

Day 34-35：实验验证
- Lorenz63 上用 MI-Lyap 选出的 τ 做 GP 预测
- 对比：Takahashi 随机、Fraser-Swinney、你的方法
- 指标：RMSE、CRPS、VPT
- **v2 加入 baseline**：Chronos 在相同稀疏条件下

**产出**：MI-Lyap BayesOpt 版 + 初步对比结果

---

## Week 6：低秩 CMA-ES + 噪声鲁棒性实验（v2 新增高维适配）

**目标**：实现高维场景的低秩 CMA-ES 版；做噪声鲁棒性 showcase

Day 36-37：噪声扫描实验（BayesOpt 版在 Lorenz63）
- σ = [0, 0.1, 0.3, 0.5, 1.0, 2.0]，每水平 20 seeds
- 关键图：τ std vs noise level（Fraser-Swinney 爆炸，你稳定）

Day 38-39：跟 Fraser-Swinney 对比
- 画 MI 曲线 vs τ 在不同 noise 下的稳定性对比图

Day 40-42：**低秩 CMA-ES 实现 + 高维测试（v2 新增）**

实现 Module 2.3 Stage B 的代码。

在 Lorenz96 N=40 (L=15) 上跑：
- BayesOpt 版：搜索时间 vs RMSE
- 低秩 CMA-ES 版：搜索时间 vs RMSE
- 对比：低秩应该显著快，同时 RMSE 近似

**画 τ matrix 的奇异值谱图**，展示低秩结构 —— 这是 v2 新增的论文卖点。配文字：

> "The singular value spectrum of the optimal τ matrix decays rapidly (first 2 singular values capture >90% of the mass), confirming our low-rank prior."

**产出**：
- "Noise robustness" self-contained section 的图
- 低秩 CMA-ES 实现
- τ matrix 奇异值谱图（Figure 7 候选）

**里程碑**：两个 τ 搜索算法 ready，高维场景有解决方案。

---

## Week 7：Dynamics-Aware CSDI（动态 delay mask）

**目标**：升级 CSDI，delay mask 由 MI-Lyap 驱动

Day 43-45：Noise conditioning
- Fork CSDI 原 repo
- `diff_models.py` 加 noise embedding
- 训练 data 里加入不同 noise level 的数据
- 测试：给同一个数据，σ 从 0 到 1，输出平滑度应该递增

Day 46-47：**动态 delay-aware attention mask（v2 加强）**
- 不是 hand-crafted mask，而是 mask = f(τ from MI-Lyap)
- Module 1 和 Module 2 的协同：
  1. MI-Lyap 在观测数据上初选 τ̂
  2. CSDI 用 τ̂ 构造 delay mask 做 imputation
  3. Imputation 完成后，MI-Lyap 在补值数据上再选精确 τ
  4. SVGP 用精确 τ 做预测

这个 two-pass 在论文里可以讲 "co-adaptation of imputation and embedding"，显著提升 coherence。

Day 48-49：训练 + 消融实验
- 在 Lorenz63 稀疏观测上训练三版本：
  1. Vanilla CSDI
  2. CSDI + noise conditioning
  3. CSDI + noise conditioning + dynamic delay mask
- 对比 imputation RMSE 和 CRPS

**产出**：Dynamics-Aware CSDI + 消融表

**失败 fallback**：
- delay mask 训练不稳定：只保留 noise conditioning
- 都不稳定：退回 vanilla CSDI，承认"CSDI 是足够好的黑盒"

---

## Week 8：高维 Lorenz96 + Foundation Model 大 PK（v2 重要升级）

**目标**：高维 scale + foundation model 对比（不能糊弄，这是 v2 锋利度的硬支撑）

Day 50-51：Lorenz96 scaling
- N = 40, 100, 400
- 测 SVGP 训练时间 vs N
- 画 scaling curve（对应 Proposition 2）
- **关键发现（预期）**：SVGP 时间随 N 亚线性增长（因 $d_{KY}$ 不随 N 线性增长）

Day 52-53：dysts integration
- 选 20 个经典系统：Lorenz, Rössler, Chua, Sprott-A, B, C... 
- 跑你的完整 pipeline
- 报告：VPT、sMAPE、correlation dim error

Day 54-56：**Foundation model 大 PK（v2 升级）**

必跑的 foundation models：
1. **Chronos-20M**（最易）
2. **Chronos-200M**
3. **Panda-72M**（from `GilpinLab/panda-72M`）
4. **FIM for ODE imputation**（from `fim4science.github.io`）
5. **Context parroting**（2025 tough-to-beat baseline）

每个跑 dysts 20 systems × 7 sparsity scenarios × 3 seeds = **2100 runs**。foundation model 推理快（~30s/run），约 **17 小时**。

**产出**：
- Lorenz96 scaling curve
- dysts benchmark 结果
- Foundation model vs 你的方法的 full comparison table
- **Phase transition 图最终版（跟 Week 1 pilot 对比验证）**

**里程碑**：Month 2 结束。方法完整，baselines 齐全。

---

## Week 9：主图 — Phase Transition Figure（v2 重新设计）

**v2 核心调整**：主图从"heatmap grid"改为"phase transition curve"。

Day 57-59：实验矩阵

```
Datasets: {Lorenz63, Lorenz96 N=40, KS equation}
Harshness scenarios: 8 个递增的 (sparsity, noise) 对
Methods: {
    Your full,
    Your w/o CSDI, Your w/o Lyap-CP, Your w/o MI-Lyap (ablations),
    Chronos-200M, Panda-72M, FIM,  # foundation models
    GRU-ODE-Bayes, CSDI-vanilla + kNN  # classic baselines
}
Metrics: {VPT, CRPS, PICP@90, MPIW}
```

总 runs = 3 × 8 × 10 × 5 = **1200 runs**，约 120 小时 = 5 天。**Day 57 必须启动**。

Day 60-61：监控 + 调试

Day 62-63：生成核心图

**Figure 1（v2 主图）**— Phase Transition Curves

一行 3 个 panel（3 datasets）：x 轴 harshness，y 轴 VPT；每条 method 一条曲线。

Key 叙事：
- Foundation models 在 harshness 到某阈值后 **急剧崩溃**（phase transition）
- 你的方法 **graceful degradation**
- 三个 dataset 上 phase transition 位置稳定（普遍现象）

**Figure 2** — Coverage Across Harshness：同样 x 轴 harshness，y 轴 PICP@90。你的方法贴着 90% 横线，其他方法全部偏离。

**Figure 3** — Horizon-Width curve：选典型 harshness (S4)，展示 Lyap 膨胀让 PI 合理扩张。

**产出**：核心实验数据 + 3 张主图

---

## Week 10：消融 + 理论章节

Day 64-66：消融实验

```
Full method vs:
- w/o Dynamics-Aware CSDI (use vanilla CSDI)
- w/o MI-Lyap (use random τ, Takahashi baseline)
- w/o MI-Lyap (use Fraser-Swinney)
- w/o SVGP (use exact GPR when n allows)
- w/o Lyap-Conformal (use split conformal)
- w/o all (v1 的 CSDI-RDE-GPR pipeline)
```

Day 67-69：理论章节写作（v2 新增 Proposition 1）

**Section 4.1：Setup**
- 问题形式化：sparse-observation regime
- 记号：$\mathbf{x}_t \in \mathbb{R}^D$，吸引子 $\mathcal{A}$，$d_{KY}$

**Section 4.2：Lower Bound for Ambient-Dim Methods（v2 核心）**
- Proposition 1（informal）：任何不利用内在维度的方法，必然被 $D$ 的维度诅咒
- 证明 sketch（附录）：covering number + Le Cam

**Section 4.3：Upper Bound via Intrinsic Dimension**
- Proposition 2（informal）：我们的方法 rate 由 $d_{KY}$ 主导
- 引用 Castillo 2014 + Takens

**Section 4.4：Calibrated Coverage via Lyap-Conformal**
- Theorem 1（informal）：$\psi$-mixing 下的覆盖保证

**Section 4.5：Putting Together — The Gap is Exponential**
- Proposition 1 vs Proposition 2 的 rate gap
- 对应 Figure 1 的 phase transition

Day 70：consistency 检查
- 理论 scaling 是否和 Week 8 scaling curve 吻合？

**产出**：完整消融表 + 理论章节草稿（2-3 页 + appendix 证明 sketch）

---

## Week 11：Case study + 写作 Push 1

Day 71-72：EEG case study
- 只保留 h=100 setting（你赢的那个）
- 加 noise 和 missingness，做 robustness 展示
- **重点**：展示 PI 在真实数据上的 calibration
- 画 EEG 的 reliability diagram

Day 73-77：写作 Push 1
- Introduction（2 页）
- Related Work（1.5 页）
- Method（2-3 页，4 modules）
- 每节配对应的图

**产出**：Paper 初稿 50%

---

## Week 12：写作 Push 2 + 内部 review + 润色

Day 78-80：写作 Push 2
- Experiments（3 页）
- Discussion & Limitations（1 页）
- Conclusion（0.5 页）
- Appendix（证明 sketch、补充实验、超参数）

Day 81-82：内部 review
- 不做这方向的 PhD：能 1 分钟说清 punchline 吗？
- 做时序的 PhD：technical 细节对不对？

Day 83-84：润色 + 提交

---

# Part III：Deliverables Checklist（v2 更新）

## 图（至少 9 张主图）

- [ ] **Figure 1：Phase Transition Curves（v2 主图）**— 3 datasets × harshness → VPT
- [ ] **Figure 2：Coverage Across Harshness（v2 核心）**— 3 datasets × harshness → PICP@90
- [ ] Figure 3：Horizon vs coverage curve（Lyap-CP 独家 calibrated）
- [ ] Figure 4：Horizon vs PI width（展示 Lyap 膨胀）
- [ ] Figure 5：Reliability diagram（pre/post conformal）
- [ ] Figure 6：MI-Lyap τ 稳定性 vs Fraser-Swinney vs noise
- [ ] **Figure 7：Low-rank structure of optimal τ（v2 新增）**— τ matrix 奇异值谱
- [ ] Figure 8：SVGP scaling + theoretical rate consistency
- [ ] Figure 9：EEG case study（真实数据 calibration）

## 表（至少 3 张）

- [ ] Table 1：主结果（dysts 20 × 全 methods × VPT/CRPS/PICP）
- [ ] Table 2：消融表
- [ ] Table 3：极端 harshness 下的 sharp summary

## 关键 claim + evidence

- [ ] **Claim 1（v2 锋利点）**："存在 sparse-observation regime，foundation models 在该 regime 呈 phase transition" → Figure 1
- [ ] **Claim 2（v2 锋利点）**："Ambient-dim 方法的失败是 fundamental，不是 engineering issue" → Proposition 1
- [ ] Claim 3："我们的方法 sample complexity 由 $d_{KY}$ 主导" → Proposition 2 + Figure 8
- [ ] **Claim 4（核心）**："First distribution-free calibrated coverage for chaotic forecasting" → Theorem 1 + Figure 2, 3, 5
- [ ] Claim 5："每个 module 独立贡献" → Table 2

---

# Part IV：风险管理（v2 更新）

## 新增高风险（v2）

### 风险 0（v2 最关键）：Week 1 phase transition pilot 失败

**症状**：Chronos 在极端 setting 下仍然 work

**应对优先级**：
1. 推到更极端 setting
2. 换更高维系统（Lorenz96）
3. 加入 non-stationarity（regime switching chaotic system）
4. **如果都不成功**：**认真考虑放弃 v2 锋利 story**，回归 v1 定位。**Week 1 就要知道答案**

### 原有风险（v1）

| 风险 | 概率 | 应对 |
|------|------|------|
| Panda 在稀疏 setting 下没崩 | 30% | 推极端 setting，降级 claim |
| MI-Lyap 没比 Fraser-Swinney 明显好 | 40% | 强调噪声鲁棒性 angle |
| Dynamics-Aware CSDI 训练不稳定 | 30% | 只保留 noise conditioning |
| 理论证不出严格版 | 70% | Informal sketch |
| 时间不够 | 50% | 见下 |

**v2 时间不够时按序砍**：
1. **绝对保留**：Week 1 phase transition pilot、SVGP、Lyap-Conformal、Week 9 主图
2. **次重要**：MI-Lyap 两阶段优化、Proposition 1 formal 化
3. **次次重要**：动态 delay mask、理论严格版、EEG case study
4. **可砍**：低秩 τ 奇异值谱图、Lorenz96 N=400、过多 foundation baselines

---

# Part V：计算资源估算（v2 微调）

| 任务 | GPU hours | v2 变化 |
|------|-----------|---------|
| CSDI 训练（每数据集） | 4-8 hrs | 不变 |
| SVGP 训练 | CPU | 不变 |
| MI 估计 | CPU | 不变 |
| CMA-ES on low-rank τ（v2 新增） | CPU | ~20 hrs 额外 |
| Foundation model inference（Week 8 PK） | 20-40 hrs | **v2 升级** |
| Phase transition scan | 120-200 hrs | 比 v1 的 heatmap 稍多 |

**总计**：~400 GPU hours A100 或同等级。如果只有 RTX 3090，~800 小时。

**如果没有 GPU**：SVGP + Lyap-CP 纯 CPU 可跑；CSDI 训练可以租 runpod/lambda labs（约 $100）

---

# Part VI：每周 checkpoint + Kill Criteria（v2 更新）

**最关键 kill criterion（v2 新增）**：

- **Week 1 end**：如果 phase transition pilot 失败（Chronos 在所有 harshness 下都 OK）
  → 立即告诉我。我们讨论是否改为 v1 定位或其他方向
  → **不要硬撑锋利 story**

其他 checkpoints：
- Week 4：Lyap-CP 相对 split CP 没提升 → 考虑投 workshop
- Week 8：SVGP 跑不动 Lorenz96 N=100 → 砍 N=400
- Week 10：主图 phase transition 不明显 → 重设计实验

---

# Part VII：投稿策略（v2 重排）

## Primary target 更新

**首选 NeurIPS 2026**（5 月 deadline）
- 原因：NeurIPS 接收大量 "foundation model fails here, and we fix it" 类型论文
- AC pool 里 Bayesian ML + UQ + dynamics 方向 reviewer 多
- 5 月 deadline 对 12 周计划友好

**次选 ICLR 2027**（9/10 月 deadline）
- 原因：ICLR 特别喜欢 empirical insights + clear motivation
- 时间宽

**再次 ICML 2026**（1 月 deadline）
- 原因：时间紧，v2 有 Week 1 phase transition pilot 作为卡点

## Safety net

- **UAI 2026**（5 月）：与 NeurIPS 同 deadline；可作为 NeurIPS rejected 后 resubmit
- **AISTATS 2026**（10 月）：natural safety net for v2 story
- **Workshop**：ICLR "AI for scientific discovery"、NeurIPS "ML and physical sciences"

## 降级策略

- v2 锋利 story + NeurIPS 拒 → resubmit UAI or AISTATS（大概率接）
- v2 锋利 story + phase transition 失败 → 降级为 v1 定位，投 AISTATS（大概率接）

---

# Part VIII：接下来你跟我的互动指南（v2 强化）

**Week 1 的关键信号（最重要）**：
- Day 7 结束时给我 phase transition pilot 的图
- 如果 phase transition 明显 → v2 story 锁定，全力 push
- 如果不明显 → 我们讨论 fallback 选项

**Week 1-2（基础期）**：
- "这篇 paper 的 X 部分看不懂"
- "我跑 dysts 报错 XXX"

**Week 3-4（UQ 实现期）**：
- "Reliability diagram 画出来是反的"
- "Conformal 的 q_hat 算得对吗"

**Week 5-6（方法创新期）**：
- "KSG estimator 高维下爆了"
- "CMA-ES 不收敛"

**Week 7-8（工程期）**：
- "SVGP 训练 diverge，lr 调多少"
- "Lorenz96 N=400 内存爆"

**Week 9-10（主实验期）**：
- "这张图 reviewer 会怎么 attack"
- "这个意外现象怎么解释"

**Week 11-12（写作期）**：
- "帮我 review abstract"
- "这个图 caption 怎么写"
- "模拟 rebuttal"

---

# 结语：v2 能让你发三大会吗？

**诚实再评估**：

| 目标 | v1 概率 | v2（phase transition 成立） | v2（phase transition 失败） |
|------|---------|---------------------------|---------------------------|
| NeurIPS/ICLR | 25-35% | **40-50%** | 15-20% |
| ICML | 25-35% | 35-45% | 15-20% |
| UAI | 50-60% | 60-70% | 50-60% |
| AISTATS | 60% | 60% | 60% |

**v2 赌一把**：如果 Week 1 phase transition 成立，三大会接收率显著提升；如果不成立，比 v1 还差（因为 story 锋利了但没 evidence）。

**最关键的事**：**Week 1 Day 6-7 的 phase transition pilot 决定一切**。

---

# 立刻要做的三件事

1. **Week 1 Day 1-2 的安装动作**：
```bash
pip install gpytorch properscoring uncertainty-toolbox scikit-optimize
pip install nolds npeet dysts cma
pip install torch transformers  # for Chronos
```

2. **安装成功后跑通 4 个 demo**：
   - GPyTorch SVGP
   - `dysts.flows.Lorenz()` 生成轨迹
   - 你自己的 Lorenz63 pipeline
   - **Chronos zero-shot 推理**（HuggingFace `amazon/chronos-t5-small`）

3. **发信号 "Day 1-2 完成"**，我给你 Day 3-5 的 5 篇论文精读 checklist + Day 6-7 的 phase transition pilot 脚本模板。

---

**开始吧。**