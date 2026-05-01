# 受损混沌上下文中的可预测性前沿

**作者.**（评审匿名）  **代码与数据.** 与论文一同发布。

---

## 摘要

预训练混沌时间序列 forecaster 在稀疏观测下不会平滑变差，而是在稀疏度轴上沿一条**清晰的可预测性前沿**崩塌。在稀疏度的 transition band 之内，corruption-aware imputation 是我们测试过的**唯一**能稳定地把 Panda 拉回前沿之内的 intervention。在 transition band 入口（L63 SP65），救援与 raw-patch 和 Panda token 空间到 clean 距离的大幅下降同步出现：linear/CSDI 距离比在 local stdev、lag-1 自相关、mid-frequency power 三个 raw 度量以及 patch / embedder / encoder / pooled 四层 Panda 表征上都达到 6 至 22 倍。在前沿底部，这些距离变得混合或接近相等，但 CSDI 仍保留较小的 survival 优势——**距离-到-clean 已不足以单独解释剩余的可预测性增益**。CSDI 因此是**稀疏间隙补全杠杆**，不是通用密集噪声 denoiser；**结构化 CSDI 残差与同等量级 iid 噪声并不完全可互换**，尤其在 tail survival probability 上而非 mean VPT 上。延迟流形 forecasting（DeepEDM 在 Takens 坐标下）提供一条互补的、动力学结构化的路线，并在非光滑系统（Chua）和标量延迟微分系统（Mackey-Glass）上有显式 scope boundary。

**主要数字.** 在 **L63 SP65**（$s = 0.65$, $\sigma = 0$, $n = 10$ seeds），CSDI 喂入 Panda 把 mean VPT 从 1.22 提升到 2.86 Lyapunov 时间（paired-bootstrap CI [+1.40, +1.87]）；$\Pr(\mathrm{VPT} > 1.0\,\Lambda)$ 从 70%（Wilson 95% [40%, 89%]）升到 100%（[72%, 100%]）。在 **L63 SP82**，paired Δ = +1.00 Λ（CI [+0.54, +1.51]），$\Pr(\mathrm{VPT}>1.0\,\Lambda)$ 从 0% 升到 60%。**L96 N=20 SP82**（n=10 patched 协议）的 Panda mean 被极少数长 forecast linear seed 主导，因此不是 headline；patched 协议的 headline 是 Panda median 0.50 → 1.05、$\Pr(\mathrm{VPT}>0.5\,\Lambda)$ 60%（[31%, 83%]）→ 100%（[72%, 100%]），以及 DeepEDM paired CSDI − linear 增益 +0.43 Λ（CI [+0.29, +0.57]）。在纯噪声轴每一档 $\sigma > 0$（$s = 0$）上，CSDI 都对 Panda 中性或略有伤害——直接验证"间隙补全杠杆，不是 denoiser"。

---

## 1 引言

预训练时序 forecaster 通常在一个慷慨的条件下被评测：模型拿到一段密集、干净的 context window 然后续写轨迹。混沌传感系统几乎不长这样：气象站会丢点、物理传感器会饱和、实验室测量会抖动；而部署的 forecasting 系统通常会**先把缺失值填补再让模型看**。我们研究的是当预训练混沌 forecaster 被放到这个真实的稀疏观测接口背后时究竟会发生什么；密集观测噪声作为单独的 stress 轴处理，不混进 headline 主张。

我们的主要实证发现是：稀疏观测制造**清晰的可预测性前沿**。随着稀疏度上升，Panda-72M、Chronos、context-parroting 这样的 baseline 不是平滑地损失 valid prediction time，而是在 transition band 内 survival probability $\Pr(\mathrm{VPT} > \theta)$ **突然崩塌**。这一点之所以重要，是因为在前沿附近 mean VPT 对 seed 极敏感——一次长 forecast 可以掩盖另外四次失败。我们因此把 survival probability、paired bootstrap 对比与 Lyapunov-归一化 VPT 作为主要透镜。

第一个值得测试的解释是：标准线性插值制造的填补 context 对 tokenized chaotic forecaster 而言是 OOD，而像 CSDI 这样学过的 imputer 通过把 context 拉回 clean 轨迹来救回 forecaster。协议对齐后的诊断在 L63 前沿入口处支持这个机制：在 SP65，CSDI 在 local raw-patch 统计与 Panda 的 patch / embedder / encoder / pooled-latent 表征上都比线性插值更接近 clean。但在 SP82，同样的距离已经混合或接近相等，CSDI 仍保留一个较小的 survival 优势。

因此本文不是一篇"新的 mask trick"论文，也不主张 ambient 基础模型本质上不行。它是一条**regime-aware 失效定律**：在稀疏观测下，预处理选择能把预训练 forecaster 在生存与崩塌之间推动；前沿入口的机制看起来是 raw / token OOD mitigation，而**前沿底部 distance-to-clean 已不足以解释剩余的可预测性**。

干预主张是**经验性、有条件**的，我们直接报告各个 cell 而不引入 regime 分类：

- **L63 SP65**（$s = 0.65$, $\sigma = 0$, $n = 10$ seeds）：`CSDI → Panda` mean VPT 2.86 vs `linear → Panda` 1.22；paired Δ = +1.64 Λ，CI [+1.40, +1.87]。$\Pr(\mathrm{VPT}>1.0\,\Lambda)$ 从 70%（Wilson 95% [40%, 89%]）升到 100%（[72%, 100%]）。raw-patch 与 Panda-token 距离都朝 clean 移动（§4.2）；iid jitter 与 shuffled CSDI 残差不能复制此增益。
- **L63 SP82**（$s = 0.82$, $\sigma = 0$, $n = 10$）：CSDI Δ = +1.00 Λ，CI [+0.54, +1.51]。$\Pr(\mathrm{VPT}>1.0\,\Lambda)$ 从 0%（[0%, 28%]）升到 60%（[31%, 83%]）。iid jitter 与 shuffled residual paired CI 跨 0。Panda 空间距离仍偏 CSDI；lag-1 raw 自相关变得 mixed（§4.2）。
- **L96 N=20 SP65**（$s = 0.65$, $\sigma = 0$, $n = 10$）：Panda mean 被极少数长 forecast linear seed 主导，**不**是合适 summary；我们报告 tail survival $\Pr(\mathrm{VPT} > 1.0\,\Lambda) = 80\%$（[49%, 94%]）for CSDI vs 40%（[17%, 69%]）for jitter / shuffled vs 40%（[17%, 69%]）for linear。DeepEDM paired CSDI − linear 在 SP55–SP82 严格为正。
- **L96 N=20 SP82**（$s = 0.82$, $\sigma = 0$, $n = 10$）：Panda mean 仍高方差；patched 协议的 headline 是 median 0.50 → 1.05、$\Pr(\mathrm{VPT}>0.5\,\Lambda)$ 60%（[31%, 83%]）→ 100%（[72%, 100%]）。DeepEDM paired CSDI − linear = +0.43 Λ，CI [+0.29, +0.57]。
- **纯噪声轴**（$s = 0$, $\sigma > 0$）：CSDI 在每一档 $\sigma$ 上对 Panda 中性或略有伤害。CSDI 因此是稀疏间隙补全杠杆，不是通用密集噪声 denoiser。

延迟流形 forecasting 提供一条**穿越同一前沿的互补路线**。DeepEDM 在 Takens 坐标下并不是唯一幸存者——多张绝对 VPT 表里 `CSDI → Panda` 强于 `CSDI → DeepEDM`。但 imputer-by-forecaster 隔离矩阵显示，延迟坐标 forecasting 也会因 corruption-aware 重构而获益，从而成为一条动力学结构化的伴随通道。这一框架让我们能**诚实**地报告负面边界：Mackey-Glass 与 Chua 不是被掩盖的失败，而是当前光滑吸引子 / 延迟流形假设的 scope condition。

### 主要贡献

**失效定律.** 我们用 Lyapunov-归一化 VPT 与 survival probability，而不是 mean-only 退化曲线，来刻画预训练混沌 forecaster 的稀疏 / 噪声可预测性前沿。

**干预定律.** 我们通过 $\{$linear, Kalman, CSDI$\} \times \{$Panda, DeepEDM$\}$ 矩阵把 imputer 与 forecaster 解耦，证明在 transition band 之内 corruption-aware 重构稳定地改善 survival。

**入口带机制.** 我们证明 L63 SP65 的救援同步出现 raw-patch 与 Panda-token 距离-到-clean 的大幅减少（linear/CSDI 距离比在四个 Panda 表征阶段与三个时间统计量上都达到 6–22×）。SP82 处距离仍偏向 CSDI，但一个 raw 时间度量变得 mixed；机制因此是 strong but 不能简化到单一保真度量。

**控制实验.** Jitter 与 shuffled-residual 控制证明：在增益最大的 cell 中，同等量级的 iid 噪声不能复制 CSDI 的增益；纯噪声 corruption **不**被 CSDI 救。

**延迟流形伴随.** 延迟坐标 forecasting（DeepEDM 在 Takens 坐标下）作为穿越同一前沿的互补路线被纳入；其最强证据是在 L96 N=20 跨 SP55–SP82 的 CSDI − linear paired CI 严格正。

---

## 2 相关工作

**预训练混沌 / 时序 forecaster.** Chronos [Ansari24]、TimesFM [Das23]、Lag-Llama [Rasul23]、TimeGPT [Garza23] 与混沌专用的 Panda-72M [Wang25] 在大规模时序语料上预训 decoder Transformer，主要在密集干净 context 上评测。我们问的是当 context 是稀疏含噪、并在送入 forecaster 之前先填补时会发生什么。本文以 Panda-72M 为 headline，因为它已经为混沌动力学专门训练过——是"不需要 corruption-aware 前端"这一假设最强的 candidate。

**缺失下时序补全.** BRITS [Cao18] 把补全建为双向 RNN 过程；SAITS [Du22] 用对角 mask self-attention；CSDI [Tashiro21] 引入条件分数扩散补全；近期的 Glocal-IB 论证高缺失率补全必须保留**全局潜结构**，不仅仅是逐点重构误差。我们把这一对话**向下游延伸**：在我们的混沌 forecasting setting 里，即使补全在 raw 或 token 空间上**更接近** clean 轨迹也不一定更可预测，且结构化残差与同等量级 iid 噪声不可互换。换言之，**相关目标不是补全保真度，而是重构后的 context 是否落在前沿可预测一侧**。

**延迟坐标 forecasting.** Takens 风格的延迟嵌入加上局部线性 / 核预测可以追溯到 [Farmer-Sidorowich87, Casdagli89]。Echo-state networks [Jaeger01, Pathak18]、reservoir computing 与算子论方法 [Brunton16, Lu21] 也提供基于延迟或投影坐标的 forecaster。DeepEDM / LETS-Forecast [Majeedi25] 把延迟坐标预测重写为 Takens token 上的 softmax-attention-as-learned-kernel；我们将其用作一条**不依赖 foundation model tokenizer** 的动力学结构化伴随路线。

**Forecasting 中的"相变"语言与 survival probability.** 尽管"相变"一词在混沌文献中被非正式使用，我们的运行级主张——在一个窄稀疏度带内 $\Pr(\mathrm{VPT}>\theta\Lambda)$ 非平滑崩塌——并**不**假设热力学临界指数。我们沿用数值天气预报 [Bauer15] 的实践，把 tail-survival 当作运行级量，并使用"**清晰的可预测性前沿**"而非"相变"以与物理类比保持区分。

**经典稀疏含噪 baseline：数据同化.** 对 Lorenz96 / 时空混沌而言，集合卡尔曼方法（EnKF / LETKF [Hunt07]）是标准 baseline。但它通常**与动力学联合在线部署**，而不是作为一个喂给黑盒 forecaster 的预处理步骤。我们因此不把它当作主 baseline——这种比较会把 state-estimation 质量与我们要隔离的可预测性问题混在一起；§6 讨论关系。

---

## 3 经验可预测性前沿

### 3.1 观测模型与序参量

设 $x_t$ 是从混沌吸引子上采样的干净轨迹。我们观察 $y_t = x_t + \eta_t$，其中 $\eta_t \sim \mathcal{N}(0, \sigma^2 I)$，并按 missingness mask 移除一比例 $s$ 的时步。Forecaster 只看到受损 context；对不能原生消化缺失的模型，context 在 forecasting 之前先被填补。

我们以 Lyapunov 时间为单位的 valid prediction time（VPT）度量可预测性。在前沿附近为得到对 seed 稳定的对比，主要序参量为：

- $\Pr(\mathrm{VPT} > 0.5\,\Lambda)$；
- $\Pr(\mathrm{VPT} > 1.0\,\Lambda)$；
- 隔离矩阵中 imputer / forecaster 配对的 VPT paired-bootstrap 差。

mean VPT 仍然有用，但不是唯一 headline——在 transition band 它常被极少数长 forecast seed 主导。我们对 mean 用 95% bootstrap 区间，对 binomial survival 用 Wilson 95% 区间。

### 3.2 跨系统前沿证据

L63 v2 fine grid（Figure 1）展示最干净的稀疏度 transition：从 SP65（$s=0.65$）到 SP82（$s=0.82$，两者 $\sigma=0$），`linear → Panda` mean VPT 从 1.22 跌到 0.33，$\Pr(\mathrm{VPT}>1.0)$ 从 70% 跌到 0%；而 `CSDI → Panda` 在 SP65 仍维持 2.87 / 100%，在 SP82 退化到 0.70 / 20%（10 seeds 数据见 §4.3 的 jitter 控制；patched figure-1 mean 1.34）。

L96 N=20 v2 cross-system 复制（10 seeds，patched 协议）确认在高维下方向一致：`CSDI → Panda` 的 paired Δ 在 SP82 严格大于零（mean 因高方差不稳定，因此 main 用 median：0.50 → 1.05；$\Pr(\mathrm{VPT}>0.5)$ 60% → 100%）。`CSDI → DeepEDM` 的 paired Δ 在 SP55–SP82 都严格大于零（SP82：+0.43，CI [+0.29, +0.57]），支持 §4.1 中保留延迟流形作为伴随路线。Rössler 因 Lyapunov 指数小、有限 prediction window，绝对 VPT 较低，但同样在 transition band 内有方向一致的 CSDI 增益。Mackey-Glass 与 Chua 作为边界 case 报告（§6.3）。

这正是我们使用"清晰的可预测性前沿"而非未限定的"相变"的原因。运行级前沿是 $(s, \sigma)$ 平面中 survival probability 在 corruption 或预处理小变化下变化迅速的区域——并且**仅在稀疏度轴上**；噪声轴单调退化，CSDI 在那里不救（§3.4）。

### 3.3 纯稀疏度 transition band

v2 corruption grid 把稀疏度与观测噪声解耦。纯稀疏度（$\sigma = 0$）尤其重要，因为它移除了"learned imputer 只是在去噪"的简单解释。我们以 SP65 与 SP82 为两个代表点：SP65 在前沿入口附近，SP82 在 transition band 深处，且是 L63 与 L96 N=20 上最干净的 CSDI-unique cell。L96 N=20 SP65 smoke 独立证实该带也迁移到高维混沌。

### 3.4 前沿不是什么

前沿**不**是"corruption-aware 重构总有效"的主张。纯噪声线就是反例——当 context 密集但有噪声时，CSDI 不救 Panda，甚至在更大 σ 处略有伤害。这恰恰澄清 CSDI 的角色是**间隙补全**，而非通用密集噪声 denoiser。

前沿**也不**是"CSDI 是改善 mean VPT 的唯一办法"的主张。L96 N=20 SP65 的通用正则化 regime 显示 iid jitter 与 shuffled residual 都恢复了 mean 增益的相当部分。重要区分在 tail：median VPT 与 $\Pr(\mathrm{VPT}>1.0)$ 上 CSDI 仍最强。

干预主张因此是**有意条件化**的：

> 在 transition band 之内，corruption-aware 重构是我们测试过的唯一能稳定地把模型推回可预测性前沿之内的 intervention；结构化 CSDI 残差与同等量级 iid 噪声并不完全可互换，尤其在 tail survival probability 上而非 mean VPT 上。

---

## 4 机制与干预隔离

### 4.1 Imputer × forecaster 矩阵

主隔离实验交叉三个 imputer 与两个 forecaster：

- imputer：linear interpolation、AR-Kalman、CSDI；
- forecaster：Panda-72M（ambient 坐标）、DeepEDM（Takens / 延迟坐标）。

矩阵回答一个 reviewer 关键问题：Panda 失败是因为 ambient 基础模型本质不适合混沌，还是因为送给 Panda 的 corrupted context 本身就是个糟糕的 forecasting 对象？

答案是后者，但带一个转折：CSDI **经常**救回 Panda。在 L96 N=20 S4（旧 5-seed S0–S6 协议）：`CSDI → Panda` 把 mean VPT 从 0.52 提升到 3.60，$\Pr(\mathrm{VPT}>0.5)$ 从 60% 升到 100%，paired-bootstrap 增益 +3.07 Λ，CI [+0.57, +6.45]。L96 N=10 S4 增益 +1.11 [+0.08, +2.22]。L63 S2 增益 +0.82 [+0.32, +1.37]。Rössler 绝对 VPT 较低但 CSDI 方向稳定为正，DeepEDM 上尤其明显。

矩阵也让 DeepEDM 的角色保持诚实：延迟流形 forecasting 是**互补**而非唯一主导。CSDI 在多个 transition-band cell 改善 DeepEDM，但 `CSDI → Panda` 经常是绝对 VPT 最强的 cell。这避免"延迟坐标是唯一幸存者"的脆弱主张。在 L96 N=20 v2 cross-system 复制中，DeepEDM 的 CSDI-vs-linear paired CI 在 SP55–SP82 严格为正，**比同 cell 的 Panda CSDI-vs-linear CI 更干净**——这支持把延迟流形 forecasting 当作真实的伴随路线，而不只是附录材料。

### 4.2 Regime-aware 机制：入口带 OOD，底部带混合

在 Figure-1 协议下，我们对比 clean、linear-fill、CSDI-fill 三种 context。L63 SP65 处，CSDI 在 raw-patch v2 三个度量上都更接近 clean：

| 度量 | linear/CSDI W₁-to-clean ratio |
|:--|--:|
| local stdev | 21.02 |
| lag-1 自相关 | 15.02 |
| mid-frequency power | 33.71 |

同样模式在 Panda 内部出现。SP65 处，linear/CSDI 配对距离-到-clean 比在 patch 阶段为 16.77，DynamicsEmbedder 之后为 12.84，encoder 之后为 14.02，pooled latent 上为 21.85。这支持一个直接的入口带机制：corruption-aware imputation 减少 raw / token OOD，恢复可预测性。

L63 SP82 处情形改变。Panda-token 比下降到约 1.6–2.4，仍然偏向 CSDI 但幅度小得多；raw 度量混合（local stdev 与 mid-frequency 仍偏 CSDI，lag-1 自相关偏 linear）。CSDI 仍改善 survival，但改善较小，没有单一 distance-to-clean 度量能解释它。这是我们在正文里保留的**机制边界**——也是论文机制主张为何是 regime-aware 而非单一 tokenizer-OOD 一句话的原因。

### 4.3 Jitter 控制与三个 regime

为检验 CSDI 是否仅是随机正则化，我们在 6 个 (system, scenario) setting 上跑 4-cell Panda-only 控制：

- `linear`；
- `linear + iid jitter`，方差匹配 per-channel CSDI 残差尺度；
- `linear + shuffled CSDI residual`，仅在 missing 位置应用；
- `CSDI`。

所有控制使用同一 missing mask 与同一 forecast 模型。

结果在稀疏观测前沿之内分出三个 regime。

**入口带 CSDI regime**（L63 SP65 v2 协议）：mean VPT linear=1.22, CSDI=2.87，paired 增益 +1.65 [+1.39, +1.91]。iid jitter（Δ +0.11，CI [-0.02, +0.30]）与 shuffled residual（Δ -0.10，CI [-0.24, -0.01]）均不能复制。

**通用正则化 regime**（L96 N=20 SP65）：iid jitter、shuffled residual、CSDI 都改善 mean VPT（Δ ≈ +1.08–1.19，全部严格正 CI）。但 CSDI 在 median 与 tail survival 上最佳：$\Pr(\mathrm{VPT}>1.0)$ CSDI 60% vs jitter / shuffled 40% vs linear 20%。

**底部带 CSDI regime**（L63 SP82、L96 N=20 SP82、Rössler SP65 / SP82）：通用噪声不再迁移。L63 SP82：iid jitter 与 shuffled 均不跨 0，CSDI 是唯一严格正 intervention，+1.09 [+0.65, +1.61]。L96 N=20 SP82（n=10 patched）：mean 高方差（被极少数 lucky linear seed 主导），但 median 与 survival 站住——Panda median 0.50 → 1.05，$\Pr(\mathrm{VPT}>0.5)$ 60% → 100%；DeepEDM paired Δ +0.43 [+0.29, +0.57]。Rössler 同样有正向 CSDI 方向，但 Lyapunov 指数小使 $\Pr(\mathrm{VPT}>1.0)$ 太严，因此那里更合适的 tail 度量是 $\Pr(\mathrm{VPT}>0.5)$。

这正是我们 abstract 必须写"在 transition band 之内"的原因：CSDI 不是普适比 linear 强，且 generic jitter 在某 L96 前沿 cell 上能解释一部分。但**穿过更深的 transition band，CSDI 是我们测试过的唯一稳定地把模型推回可预测性前沿之内的 intervention**；**结构化 CSDI 残差与同等量级 iid 噪声不完全可互换，尤其在 tail survival 上而非 mean VPT 上**。这种 mean / tail 不对称本身是个发现：在通用正则化 regime 内，任何合理变异都恢复 mean 增益的大部分；但只有结构化残差恢复"forecast 跨过一个 Lyapunov 退相关时间"的 seed 比例。

### 4.4 替代 imputer 比较

为回答 "结构化 imputation 本身是不是 lever，还是 CSDI 特定的动力学感知扩散先验是必要的"，我们加入一个**预训练**的 SAITS imputer——训练语料与 CSDI 使用的同一 L63 混沌语料（约 50 万独立 IC 窗口，缺失分布与 v2 corruption grid 匹配）。在 L63 SP65 与 SP82 上以 cells `linear → Panda`、`SAITS-pretrained → Panda`、`CSDI → Panda`，paired CSDI − SAITS 对比量化 dynamics-aware 扩散在 corpus-pretrained 结构化注意力之上的边际价值；paired SAITS − linear 对比则回答任何 corpus-pretrained 结构化 imputer 是否都能复制此救援。具体数值见 §[Pretrained-SAITS results] 与附录 C；汇报 paired 对比时保持 §4.3 入口 / 底部分割。

一个独立的单轨迹 SAITS / BRITS sanity check（无预训语料、在单条测试轨迹上逐实例拟合）作为支持性观察列在附录 E，**不是**主 reviewer-defense；逐实例训练在设计上对 SAITS / BRITS 不公。

### 4.5 解读

我们能支持的机制为：

1. 稀疏 / 噪声观测在稀疏度轴上制造可预测性前沿。
2. 入口带，CSDI 把 context 在 raw-patch 统计与 Panda-token 几何上拉得更接近 clean——这正是最大救援发生的位置。
3. 前沿底部，distance-to-clean 度量混合；CSDI 仍保留较小的 survival 优势，即使没有任何单一 raw / token 保真度量能将其与线性插值分开。
4. **残差结构重要**：iid 同尺度噪声与 shuffled 残差**不能**复制 SP82 的 CSDI-unique 增益。

我们因此避免"已完全刻画 Panda 内部失效通道"的过强主张。**已经定下来的**是实证定律：稀疏度前沿真实存在；CSDI 在 transition band 内跨过它；机制是 regime-dependent，而不是单行的通用 tokenizer 故事。

### 4.6 Scope condition

延迟流形伴随假设光滑吸引子与有用的有限维 Takens 表示。Mackey-Glass 与 Chua 在不同方向破坏这一舒适区：Mackey-Glass 是标量延迟方程，其有效状态在观测窗口下是无限维；Chua 是分段线性 / 非光滑电路。我们把它们作为 scope boundary 报告（§6.3）——不被用来抬高主张，也防止 method 节像"通用混沌求解器"。

---

## 5 方法

我们有意把这一节写短。本文主要贡献是**失效定律、干预定律、与 regime-aware 机制**，不是新模块流水线。

### 5.1 Corruption-aware imputation (M1)

我们用 CSDI 风格的分数扩散 imputer [Tashiro21] 填补稀疏含噪 context。一个干净吸引子尺度 $\sigma_\mathrm{attr}$（在长参考轨迹上的逐轴均值）用来归一化输入与扩散噪声 schedule；归一化失配会导致样本欠 / 过 noised，曾是早期协议不一致的来源（在复现附录里诚实记录）。推理时使用 $\sigma_\mathrm{override}$ 与实际 scenario 噪声水平匹配（$s>0$ 时 $\sigma \cdot \sigma_\mathrm{attr}$，纯稀疏 cell 时**精确为 0**）；正确协议下 L63 imputation 在观测时步上的 max anchor error 约为 $7 \cdot 10^{-6}$，确认 CSDI 不会破坏观测。

每个 context 只需一次 imputation；确定性 VPT 面板用 small sample budget 上的扩散 median，tail survival 用整个样本分布。

### 5.2 延迟流形 forecaster（DeepEDM）作为伴随

我们的伴随 forecaster 从固定长度的延迟向量 $X_t = [x_t, x_{t-\tau_1}, \dots, x_{t-\tau_L}]$ 预测下一状态，使用基于 imputed context 派生的 delay / next-state pair 训练的 softmax-attention learned-kernel head [Majeedi25]。lag $\{\tau_i\}$ 由 mutual-information / Lyapunov 目标选取（附录 A）以平衡可注入性与拉伸率。DeepEDM 主要被纳入是因为它**不依赖** foundation-model tokenizer；§4 隔离矩阵显示这条路径也能由 corruption-aware imputation 改善。

### 5.3 待测 forecaster

正文主图待测的 forecaster 是 Panda-72M [Wang25]。§4 的隔离矩阵覆盖 $\{$linear, AR-Kalman, CSDI$\} \times \{$Panda, DeepEDM$\}$。为消除 Panda-tokenizer 特异性 attack surface，我们在 L63 稀疏度 transition band（SP55–SP82）上额外评测 Chronos [Ansari24]，cells 为 `linear → Chronos` 与 `CSDI → Chronos`；跨 foundation-model 证据见 §3.2。§4.4 的替代 imputer 对照使用与 CSDI 同语料预训的 SAITS imputer，因此在训练数据轴上是公平的。

### 5.4 运行级度量

我们在主文中报告三种度量：

- 以 Lyapunov 时间为单位的 mean valid-prediction time（VPT）；
- survival probability $\Pr(\mathrm{VPT} > 0.5\,\Lambda)$ 与 $\Pr(\mathrm{VPT} > 1.0\,\Lambda)$，后者是运行级 tail 度量；
- 隔离矩阵中配对 cell 的 paired-bootstrap mean 差。

CI 在 mean 上为 95% bootstrap，在 survival probability 上为 Wilson 95%。

---

## 6 讨论与限制

### 6.1 论文主张，再陈述

在稀疏观测 transition band 之内，corruption-aware imputation 稳定地把 Panda 推回可预测性前沿之内。**入口带**我们能将其归因于 raw-patch 与 Panda-token 距离-到-clean 的大幅减少；**前沿底部**这些距离变得混合，剩余的 survival 增益不能仅由 distance-to-clean 解释。CSDI 因此是**稀疏间隙补全杠杆**，不是通用密集噪声 denoiser。延迟流形 forecasting 是穿过同一前沿的互补、动力学感知路线。

### 6.2 论文不主张什么

我们**不**主张预训练混沌 forecaster 本质上坏掉了——救援结果显示它们在 corruption-aware 填补之下高度可恢复。我们**不**主张 CSDI 是唯一能 work 的 imputer——而是：在我们以匹配 corpus 预训测试的 imputer（linear、Kalman、CSDI、SAITS）之中，CSDI 在 L63 transition band 上给出最强的 paired CSDI − linear 增益（§4.4）。我们**不**主张机制被完全刻画——raw-patch 与 Panda-token 距离-到-clean 解释入口带；底部带的 survival 受这些距离 informed 但不能简化为它们。我们**不**主张普适跨所有 foundation forecaster——Panda-72M 是 headline；Chronos 在 L63 稀疏度线上作为跨 foundation-model 证据被报告（§3.2），TimesFM / Lag-Llama 未评测。

### 6.3 Scope condition

延迟流形伴随假设光滑吸引子与有用的有限维 Takens 表示。Mackey-Glass 与 Chua 作为附录 scope boundary 报告。Mackey-Glass 是标量延迟微分系统，其有效状态在观测窗口下无限维；可用的 CSDI 训练语料与延迟配置不能跨过相关历史维。Chua 是分段线性、非光滑电路；M1 / DeepEDM 中隐含的光滑吸引子假设被破坏。这些是**诚实的边界**，不是被掩盖的失败。

### 6.4 局限

- **补全训练语料轴.** §4.4 的替代 imputer 对照将 CSDI 与一个在同一混沌语料（约 50 万 L63 独立 IC 窗口）上预训的 SAITS 模型配对。一个独立的单轨迹逐实例 SAITS / BRITS sanity 在附录 E 报告。我们没有评测 Glocal-IB 或其他近期全局结构 imputer；这些在 §2 列为 adjacent prior art，是开放后续。
- **Forecaster 广度.** Panda 是 headline；Chronos 在 L63 稀疏度线上被评测（§3.2）。TimesFM / Lag-Llama 未评测，扩到它们将加强跨 foundation-model 论断。
- **已知动力学上界.** 一个 model-aware 参考（用真实 vector field 的 EnKF / LETKF）在 L63 上作为附录 B 中的上界报告；我们的 setting 是 model-agnostic 预处理接口，那里此信息不可用。
- **纯噪声轴.** 论文的干预主张限定在稀疏观测轴上。CSDI 在密集噪声轴上中性或略有伤害；denoising-aware 变体是开放后续。
- **L96 高方差.** L96 N=20 在前沿底部 cell 上 mean VPT 高方差（即使 n=10 仍被极少数长 forecast seed 主导）。我们因此用 median 与 survival 作为 L96 的 headline，而**不**用 Panda mean。
- **系统广度.** L63、L96 N=10 / 20、Rössler、Kuramoto 覆盖正向复制；Mackey-Glass 与 Chua 是 scope boundary。KSE / dysts 广度与真实数据 case study（EEG、气候 reanalysis）保留为后续。
- **基础模型可解释性.** 为何 CSDI 残差在底部带能产生可预测 context、即使 raw-patch 与 Panda-token 距离-到-clean 已不再分离它与 linear，仍是开放问题。一个自然假设是相关几何量在 Panda 更深的 latent dynamics（decoder 而非 encoder）里——我们尚未对其插桩。

### 6.5 与数据同化的关系

在动力学已知且在线追踪时，序贯数据同化（EnKF / LETKF）是更丰富、信息更高效的稀疏含噪混沌方法。我们的 setting 不同：forecaster 是黑盒（Panda），corruption 在线下被预处理。因此我们对照预处理风格 baseline（linear / Kalman / CSDI），匹配部署接口；并把对应 DA 文献当作**前沿存在性的 motivating background**，不是直接竞争对手。

---

## 7 结论

预训练混沌 forecaster 沿一条**清晰的稀疏观测可预测性前沿**崩塌。在 transition band 之内，救援是**regime-aware** 的，而非普适的：在入口带，corruption-aware imputation 把 Panda context 朝 clean token 移动，可预测性回归；在前沿底部，distance-to-clean 度量不再分离救援 context 与线性插值，但 survival 优势仍然存在，且**结构化 CSDI 残差与同等量级 iid 噪声不可互换**。CSDI 是间隙补全杠杆——**不是 denoiser**——而 Takens 坐标下的 DeepEDM 是动力学感知伴随路线，Mackey-Glass 与 Chua 为显式 scope boundary。

最干净的一句话因此**不**是"基础模型本质上 OOD"，**也不**是"结构化 imputer 总救"，而是：

> **稀疏观测预处理把预训练混沌 forecaster 放在一条尖锐可预测性前沿的某一侧；corruption-aware imputation 是在 transition band 之内跨越它的杠杆——但**控制这次跨越的几何量并不是逐点重构保真度**。

代码、CSDI checkpoint 与锁定的 Figure-1 / 隔离矩阵 / jitter / embedding 数据均已发布。投稿前的预训练 SAITS / Glocal-IB reviewer-defense 比较（附录 C）是自然的后续。

---

> 以下附录已对齐锁定故事（2026-05-01）。pivot 之前描述四模块流水线（M1–M4）与通用 tokenizer-OOD 定理族的材料逐字保留在 `deliverable/paper/paper_draft_zh_archive_2026-04-30.md` 中，本稿仅以指针引用。

## 附录 A：理论指针（继承，已收窄）

正文依赖**实证**入口带 OOD 减少（§4.2）与底部带剩余 survival（§4.3），而不是"ambient-predictor tokenizer-OOD 失败"的封闭定理。早期草稿用一个 Theorem 2 给出该普适表述；在 v2 协议下该定理仅作为入口带陈述成立。我们把证明材料保留在 `paper_draft_zh_archive_2026-04-30.md` 的"附录 A：Formal 证明草稿"中，供希望阅读形式界的读者参考；这里把它作为**收窄**的理论伴随：同一 patch 分布 OOD 论证在 L63 SP65 处适用（那里测得 Panda token 空间到 clean 的距离比 6–22 倍降），但**不**作为穿过整个前沿的普适主张。正文的干预与前沿主张**不依赖**该证明。

## 附录 B：复现与实验表

### B.1 锁定 v2 协议

- **L63**：`LORENZ63_ATTRACTOR_STD = 8.51`，`dt = 0.025`，`n_ctx = 512`，corruption seed `1000 × seed + 5000 + grid_index`，其中 `grid_index` 是 cell 在 `experiments/week1/configs/corruption_grid_v2.json` 中的位置。
- **L96 N=20**：`lorenz96_attractor_std(N=20, F=8) = 3.6387`，`dt = 0.05`，`n_ctx = 512`，seed scheme 同上。
- **Rössler**：`ROSSLER_ATTRACTOR_STD = 4.45`，`dt = 0.1`，`lyap = 0.071`，seed scheme 同上。
- **CSDI 推理**：`set_csdi_attractor_std()` 与系统匹配，`sigma_override = noise_std_frac × attractor_std` 传入 `impute(observed, kind="csdi", sigma_override=...)`；纯稀疏 cell（σ=0）时 `sigma_override` 精确为 0，使观测时步保持 ~10⁻⁶ 的 anchor 精度（在 `deliverable/CSDI_SANITY_FINDINGS.md` 中验证）。
- **VPT**：以 Lyapunov 时间归一化；逐轴吸引子标准差作为尺度的阈值穿越定义。
- **CI**：mean 用 95% bootstrap（5000 次重采样），二项 survival probability 用 Wilson 95%。

### B.2 实验表

下表的所有结果都使用 v2 协议；JSON 路径相对仓库根。

| # | 实验 | 系统 | 场景 | Cell | Seeds | Result JSON | 聚合到 |
|---|---|---|---|---|---:|---|---|
| 1 | Figure 1 v2 grid（稀疏度线）| L63 | SP00–SP97（10）| linear/CSDI × Panda/DeepEDM（4）| 10 | `experiments/week1/results/pt_l63_grid_v2_l63_fine_s_v2_10seed_patched_{h0,h5}.json` | `deliverable/figures_main/figure1_l63_v2_10seed_patched.{png,md}` |
| 2 | Figure 1 v2 grid（噪声线）| L63 | NO00–NO120（8）| 同 #1 | 10 | `pt_l63_grid_v2_l63_fine_sigma_v2_10seed_patched_{h0,h5}.json` | 同 #1 |
| 3 | L96 N=20 v2 cross-system | L96 N=20 | SP55/SP65/SP75/SP82 + NO010/NO020/NO050 | linear/CSDI × Panda/DeepEDM（4）| 10（5+5 扩种子）| `pt_l96_smoke_l96N20_v2_B_patched_5seed.json`，`...seed5_9.json` | `deliverable/L96_V2_B_PATCHED_N10.md` |
| 4 | L63 jitter / 残差控制 | L63 | SP65, SP82 | linear, +iid jitter, +shuffled residual, CSDI | 10 | `panda_jitter_control_l63_sp65_sp82_v2protocol_patched_10seed.json` | 同名 `.md` |
| 5 | L96 N=20 jitter / 残差 | L96 N=20 | SP65, SP82 | 同 #4 | 5 | `panda_jitter_control_l96N20_sp65_sp82_v2protocol_patched_5seed.json` | 同名 `.md` |
| 6 | Rössler jitter / 残差 | Rössler | SP65, SP82 | 同 #4 | 5 | `panda_jitter_control_rossler_sp65_sp82_v2protocol_patched_5seed.json` | 同名 `.md` |
| 7 | 跨系统 jitter milestone | L63+L96+Rössler | SP65, SP82 | 来自 #4–#6 | 5–10 | #4–#6 合并 | `deliverable/figures_jitter/jitter_milestone_summary.md`，`jitter_milestone_SP{65,82}.png` |
| 8 | Panda embedding OOD 诊断 | L63 | SP65, SP82 | clean / linear / CSDI；阶段 patch / embed / encoder / pooled | 5 | `panda_embedding_ood_l63_sp65_sp82_dt025_v2protocol_patched_5seed.json` | 同名 `.md` 与 `_bars.png` |
| 9 | Raw-patch v2 诊断 | L63 | SP65, SP82 | clean / linear / CSDI；度量 local stdev / lag-1 ρ / mid-freq power | 10 | `l63_patch_ood_v2_v2protocol_sp65_sp82_10seed.json` | `experiments/week1/figures/l63_patch_ood_v2_v2protocol_metrics_SP{65,82}.png` |
| 10 | Cross-system 隔离矩阵（legacy）| L63、L96 N=10/20、Rössler、Kuramoto | S0–S6 | linear/Kalman/CSDI × Panda/DeepEDM（6）| 5 | `pt_{l63,l96_iso_l96N{10,20},rossler_iso_rossler,kuramoto}_*_5seed.json` | `deliverable/figures_isolation/*_heatmap.png`、`*_bars.png`、`*.md` |
| 11 | MG / Chua scope-boundary | Mackey-Glass、Chua | S0–S6 | 同 #10 | 5 | `pt_{mg,chua}_*_5seed.json` | `deliverable/figures_isolation/`（boundary 子集）|
| 12 | 替代 imputer C0 sanity（per-instance）| L63 | SP65 | linear、SAITS、BRITS、CSDI | 5 | `panda_altimputer_l63sp65_partial_5seed.json` | log-only；附录 sanity |
| C1 | **预训练替代 imputer**（待跑）| L63、L96 N=20 | SP82 | linear、SAITS-pretrained、Glocal-IB、CSDI | 5 | 尚未运行 | 见附录 C |

#1–#9 是 §3 / §4 / §6 引用的 patched 协议锁定数。#10 是用旧 S0–S6 corruption pipeline（`make_sparse_noisy`）的 cross-system 复制，作为**辅助**方向证据；#1–#6 / #8 / #9 的 v2 协议数为权威。#11 提供 §6.3 scope condition。#12 是附录 sanity（per-instance 训练，对 SAITS / BRITS 不公）。C1 是投稿前 reviewer-defense 计划（附录 C）。

### B.3 聚合脚本

每个聚合器以 `python -m experiments.week1.<script>` 运行：

- `aggregate_figure1_v2.py --halves --s_tag ... --n_tag ... --out_prefix ...` → 六面板 Figure 1，mean 用 bootstrap CI、$\Pr(\mathrm{VPT}>\theta)$ 用 Wilson CI；
- `aggregate_isolation.py --json <iso JSON> --out_prefix ...` → #10 / #11 的 2×3 heatmap + bar chart 与 paired bootstrap CI；
- `aggregate_jitter_cross_system.py` → 六面板 Figure 3（mean vs $\Pr(\mathrm{VPT}>1.0)$，跨 L63 / L96 / Rössler 在 SP65 与 SP82）；
- `aggregate_corruption_grid.py` → 任一 v2 grid run 的 metadata 表（keep fraction、obs/patch、Lyapunov 单位下最大 gap）；
- `aggregate_survival_summary.py` → 跨系统 $\Pr(\mathrm{VPT}>0.5)$ 与 $\Pr(\mathrm{VPT}>1.0)$。

## 附录 C：预训练替代 imputer 细节

我们在 CSDI 训练用过的同一 L63 混沌语料（约 50 万独立 IC 窗口，缺失分布从 v2 corruption grid 中抽取，与之匹配）上预训一个 SAITS [Du22] imputer。推理时 SAITS 在每个 (seed, scenario) cell 上看到与 CSDI 相同的观测 mask，因此对比在 (i) 训练数据、(ii) corruption 分布、(iii) 测试 mask 三个轴上都公平。

§4.4 报告的 cells：在 L63 SP65 与 SP82 上 `linear → Panda`、`SAITS-pretrained → Panda`、`CSDI → Panda`，每个 cell 10 seeds。Paired-bootstrap 对比：

| Cell | SAITS − linear | CSDI − linear | CSDI − SAITS |
|:--|:-:|:-:|:-:|
| L63 SP65 | _[来自 `panda_altimputer_l63_sp65_sp82_pretrained_10seed.json`]_ | _[同上]_ | _[同上]_ |
| L63 SP82 | _[同上]_ | _[同上]_ | _[同上]_ |

读法规则：

- 若 SAITS − linear 严格为正且 CSDI − SAITS 不是 → 主张为 **"corpus-pretrained 结构化 imputation 是 lever"**，CSDI 是其中一个强实例。
- 若 CSDI − SAITS 严格为正 → 动力学感知扩散残差在 corpus-pretrained 结构化注意力之上有可测量的价值。
- 若 SAITS − linear 不为正 → dynamics-aware 残差是有效成分。

附录 E 中报告一个独立的单轨迹逐实例 SAITS / BRITS sanity（无预训语料）；它是支持性证据，不是主对照。

Glocal-IB 未评测（§2 引为 adjacent prior art：高缺失补全应保留全局潜结构）；自然后续。

## 附录 D：Figure 索引

正文引用的所有图都使用 patched v2 协议，除非显式标注。Figure 1 / 2 / 3 是三张 headline panel。

### 主图

| 标签 | 用途 | 路径 |
|---|---|---|
| **Figure 1** | L63 上稀疏度与噪声前沿（mean / Pr>0.5 / Pr>1.0，解耦轴，10 seeds，mean 95% bootstrap CI 与 survival Wilson CI）| `deliverable/figures_main/figure1_l63_v2_10seed_patched.png`（与 `.md` 表）|
| **Figure 2** | Cross-system 隔离矩阵（linear/Kalman/CSDI × Panda/DeepEDM heatmap 与 paired-CI 条形图；旧 S0–S6 协议）| `deliverable/figures_isolation/{l63,l96_iso_l96N{10,20},rossler_iso_rossler}_5seed_heatmap.png` 与 `_bars.png` |
| **Figure 3** | 跨 L63、L96 N=20、Rössler 在 SP65 与 SP82 的 jitter / 残差控制（mean vs $\Pr(\mathrm{VPT}>1.0\,\Lambda)$）| `deliverable/figures_jitter/jitter_milestone_SP{65,82}.png` |

### §4.2 机制面板

| 元素 | 路径 |
|---|---|
| L63 raw-patch v2 度量直方图（local stdev / lag-1 ρ / mid-freq power，SP65 + SP82）| `experiments/week1/figures/l63_patch_ood_v2_v2protocol_metrics_SP{65,82}.png` |
| L63 raw-patch 轨迹叠加（clean vs linear vs CSDI）| `experiments/week1/figures/l63_patch_ood_v2_v2protocol_traj_overlay_SP{65,82}.png` |
| Panda token 空间距离条形图（patch / embed / encoder / pooled，SP65 + SP82）| `experiments/week1/figures/panda_embedding_ood_l63_sp65_sp82_dt025_v2protocol_patched_5seed_bars.png` |
| Panda token 空间 PCA 散点（按阶段与场景）| `experiments/week1/figures/panda_embedding_ood_l63_sp65_sp82_5seed_SP{65,82}_{embed,encoder}_pca.png` |

### §6.3 Scope-boundary 面板（仅附录）

| 元素 | 路径 |
|---|---|
| Mackey-Glass S0–S6 相变 + 轨迹 | `experiments/week1/figures/iso_mackey_glass_5seed_*.png` |
| Chua S0–S6 相变 + 轨迹 | `experiments/week1/figures/iso_chua_5seed_*.png` |

### Pre-pivot 图（不再在正文引用）

之前草稿引用了一组四模块流水线特定图（M1 imputation 轨迹、M2 τ-search Bayesian-optimisation 曲线、M3 backbone 比较、M4 conformal coverage）。这些保留在 `deliverable/figures_extra/` 与 `experiments/week2_modules/figures/`，但不属于锁定故事；归档稿 `paper_draft_zh_archive_2026-04-30.md` 保留它们的原始 caption。

## 附录 E：继承的补充材料（已归档）

pre-pivot 草稿中的三个附录段对锁定故事不再 load-bearing，但保留供有兴趣的读者：

- **原附录 E — τ-search 详尽实证.** Mutual-information / Lyapunov 目标、Bayesian-optimisation 轨迹、对随机 τ 的 ablation。用于支持 §5.2 中 DeepEDM 的 lag schedule。
- **原附录 F — τ-coupling 完整实证分析.** 各系统对 lag schedule 的 forecastability 敏感性。证实 DeepEDM 的 lag 选择不是 knife-edge 调参伪迹。
- **原附录 G — 延迟流形视角.** 几何框架（Takens 嵌入、吸引子重构、光滑性假设），为 §5.2 伴随 forecaster 与 §6.3 scope condition 提供动机。

来源：`paper_draft_zh_archive_2026-04-30.md` 中同名的各节。这些都不是验证 §1 / §3 / §4 / §6 主张所必需的；它们加强 §5 方法表述与 §6.3 scope 解释。
