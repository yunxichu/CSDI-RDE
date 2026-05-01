# 受损混沌上下文中的可预测性前沿

**作者.**（评审匿名）  **代码与数据.** 与论文一同发布。

---

## 摘要

一个预训练混沌 foundation forecaster（Panda-72M）在稀疏观测下沿一条**清晰的可预测性前沿**崩塌；第二个预训练 forecaster（Chronos）则在同一稀疏度线上停在低 VPT 平台，因此 transition 形状是 forecaster-dependent 的，而**不是**普适的（§6.4）。在 Panda 前沿的 transition band 之内，**corpus-pretrained 结构化 imputation** 是把 Panda 推回前沿之内的杠杆：CSDI（corruption-aware 扩散 imputer）和在同一混沌语料上预训的 SAITS 都能强烈救援 Panda；CSDI 在入口带保留一个小但 paired-CI-strict 的优势（L63 SP65：CSDI − SAITS = +0.41 Λ，95% CI [+0.05, +0.87]），在底部带两者统计上不可分辨（SP82：+0.06 Λ，[−0.31, +0.59]）。机制在入口带最强：CSDI 在 raw-patch 和 Panda-token 距离-到-clean 上产生大幅减少（linear/CSDI 距离比在 local stdev、lag-1 自相关、mid-frequency power 三个 raw 度量以及 patch / embedder / encoder / pooled 四层 Panda 表征上达到 12 至 34 倍；逐阶段细分见 §4.2）。在前沿底部，Panda-token 距离仍偏 CSDI 但一个 raw 时间度量变得 mixed，因此距离-到-clean 是 informative 但不足以完全解释 tail survival。CSDI 因此是**稀疏间隙补全杠杆**，不是通用密集噪声 denoiser；**结构化 imputation 残差与同等量级 iid 噪声并不完全可互换**，尤其在 tail survival probability 上而非 mean VPT 上。延迟流形 forecasting（DeepEDM 在 Takens 坐标下）提供一条互补的、动力学结构化的路线，并在非光滑系统（Chua）和标量延迟微分系统（Mackey-Glass）上有显式 scope boundary。

**主要数字.** 在 **L63 SP65**（$s = 0.65$, $\sigma = 0$, $n = 10$ seeds），CSDI 喂入 Panda 把 mean VPT 从 1.22 提升到 2.86 Lyapunov 时间（paired-bootstrap CI [+1.40, +1.87]）；$\Pr(\mathrm{VPT} > 1.0\,\Lambda)$ 从 70%（Wilson 95% [40%, 89%]）升到 100%（[72%, 100%]）。在 **L63 SP82**，paired Δ = +1.00 Λ（CI [+0.54, +1.51]），$\Pr(\mathrm{VPT}>1.0\,\Lambda)$ 从 0% 升到 60%。**L96 N=20 SP82**（n=10 patched 协议）的 Panda mean 被极少数长 forecast linear seed 主导，因此不是 headline；patched 协议的 headline 是 Panda median 0.50 → 1.05、$\Pr(\mathrm{VPT}>0.5\,\Lambda)$ 60%（[31%, 83%]）→ 100%（[72%, 100%]），以及 DeepEDM paired CSDI − linear 增益 +0.43 Λ（CI [+0.29, +0.57]）。在纯噪声轴每一档 $\sigma > 0$（$s = 0$）上，CSDI 都对 Panda 中性或略有伤害——直接验证"间隙补全杠杆，不是 denoiser"。

---

## 1 引言

预训练时序 forecaster 通常在一个慷慨的条件下被评测：模型拿到一段密集、干净的 context window 然后续写轨迹。混沌传感系统几乎不长这样：气象站会丢点、物理传感器会饱和、实验室测量会抖动；而部署的 forecasting 系统通常会**先把缺失值填补再让模型看**。我们研究的是当预训练混沌 forecaster 被放到这个真实的稀疏观测接口背后时究竟会发生什么；密集观测噪声作为单独的 stress 轴处理，不混进 headline 主张。

我们的主要实证发现是：稀疏观测对 Panda-72M（专门为混沌动力学预训的 foundation forecaster）制造**清晰的可预测性前沿**。随着稀疏度上升，Panda 的 survival probability $\Pr(\mathrm{VPT} > \theta)$ 不是平滑下降，而是在一条窄 transition band 内**突然崩塌**。Chronos 在我们的 setting 下**不**展示这种 transition 形状——它在同一稀疏度线上停留在低 VPT 平台（mean 0.34–0.50；§6.4）——因此经验前沿是为 Panda 建立的；跨 foundation 推广作为 forecaster-dependent 观察被报告，而**不是**普适主张。这一点之所以重要，是因为在前沿附近 mean VPT 对 seed 极敏感——一次长 forecast 可以掩盖另外四次失败。我们因此把 survival probability、paired bootstrap 对比与 Lyapunov-归一化 VPT 作为主要透镜。

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

**入口带机制.** 我们证明 L63 SP65 的救援同步出现 raw-patch 与 Panda-token 距离-到-clean 的大幅减少（linear/CSDI 距离比在四个 Panda 表征阶段与三个时间统计量上达到 12–34×；逐阶段细分见 §4.2）。SP82 处距离仍偏向 CSDI，但一个 raw 时间度量变得 mixed；机制因此是 strong but 不能简化到单一保真度量。

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

前沿**也不**是"CSDI 是改善 mean VPT 的唯一办法"的主张。L96 N=20 SP65 的通用正则化 regime 显示 iid jitter 与 shuffled residual 都恢复了 mean 增益的相当部分。重要区分在 tail：CSDI 的 $\Pr(\mathrm{VPT}>1.0\,\Lambda) = 80\%$ vs linear / iid / shuffled 各 40%（n=10），Wilson 95% CI 重叠但每个 seed 上 rank order 都被保留；median VPT 上 CSDI 也占优。L96 SP65 的 paired-CI strict 主张因此在 direction-and-rank 上而非 mean 上。

干预主张因此是**有意条件化**的：

> 在 transition band 之内，**corpus-pretrained 结构化 imputation** 是我们测试过的能稳定把模型推回可预测性前沿之内的 lever；CSDI 与同语料预训的 SAITS 都能跨过前沿，CSDI 在入口带保留小但 paired-CI-strict 的优势；结构化 imputation 残差与同等量级 iid 噪声并不完全可互换，尤其在 tail survival probability 上而非 mean VPT 上。

---

## 4 机制与干预隔离

### 4.1 Imputer × forecaster 矩阵

主隔离实验交叉三个 imputer 与两个 forecaster：

- imputer：linear interpolation、AR-Kalman、CSDI；
- forecaster：Panda-72M（ambient 坐标）、DeepEDM（Takens / 延迟坐标）。

矩阵回答一个 reviewer 关键问题：Panda 失败是因为 ambient 基础模型本质不适合混沌，还是因为送给 Panda 的 corrupted context 本身就是个糟糕的 forecasting 对象？

答案是后者。Cell 级别的权威证据来自 §3.2 与 §4.4 的 v2 协议数（10-seed Figure 1 grid；30-seed L96 SP82 alt-imputer）；一个更粗的 5-seed S0–S6 隔离扫描（旧 `make_sparse_noisy` corruption 流水线，Figure 2）跨 L63 / L96 / Rössler / Kuramoto 复制 direction-of-effect，详细数字见附录 B。

矩阵把 DeepEDM 在主文中的角色锁定到一个硬事实：在 L96 N=20 v2 cross-system 复制中，**DeepEDM 是 SP55–SP82 transition band 上每个 cell 都拿到严格正 paired CSDI − linear CI 的唯一 forecaster**，SP82 处 +0.43、[+0.29, +0.57]。在同一 band 上，Panda mean 被极少数 lucky linear seed 主导（§4.3），所以高维下最干净的 cross-system CSDI − linear 证据走的是延迟坐标通道而非 ambient 通道。这正是 DeepEDM 留在主文而不进附录的原因，但**不**过度主张其支配性：在低维 cell（如 L63 SP65）上 `CSDI → Panda` 才是绝对 VPT 最强的 cell，§3.2 / §4.4 主张并不依赖 DeepEDM。

### 4.2 Regime-aware 机制：入口带 OOD，底部带混合

在 Figure-1 协议下，我们对比 clean、linear-fill、CSDI-fill 三种 context。L63 SP65 处，CSDI 在 raw-patch v2 三个度量上都更接近 clean：

| 度量 | linear/CSDI W₁-to-clean ratio |
|:--|--:|
| local stdev | 21.02 |
| lag-1 自相关 | 15.02 |
| mid-frequency power | 33.71 |

同样模式在 Panda 内部出现。SP65 处，linear/CSDI 配对距离-到-clean 比在 patch 阶段为 16.77，DynamicsEmbedder 之后为 12.84，encoder 之后为 14.02，pooled latent 上为 21.85。这支持一个直接的入口带机制：corruption-aware imputation 减少 raw / token OOD，恢复可预测性。

L63 SP82 处情形改变。Panda-token 比下降到约 1.6–2.4，仍然偏向 CSDI 但幅度小得多；raw 度量混合（local stdev 与 mid-frequency 仍偏 CSDI，lag-1 自相关偏 linear）。CSDI 仍改善 survival，但改善较小，没有单一 distance-to-clean 度量能解释它。这是我们在正文里保留的**机制边界**——也是论文机制主张为何是 regime-aware 而非单一 tokenizer-OOD 一句话的原因。

### 4.3 Jitter 与 shuffled-residual 控制（cell-wise 观察）

为检验 CSDI 是否仅是随机正则化，我们在 6 个 (system, scenario) setting 上跑 4-cell Panda-only 控制：

- `linear`；
- `linear + iid jitter`，方差匹配 per-channel CSDI 残差尺度；
- `linear + shuffled CSDI residual`，仅在 missing 位置应用；
- `CSDI`。

所有控制使用同一 missing mask 与同一 forecast 模型。我们逐 cell 报告，**不**引入 regime 分类——§4.4 的替代 imputer 对照已经把"floor-band CSDI 最强"与"任何结构化残差都跨过"之间的边界重塑（后者是 §4.4 SP82 处对 SAITS-pretrained 的发现）。

**L63 SP65**（入口带，$n=10$）：`linear → Panda` mean VPT 1.22，`CSDI → Panda` mean VPT 2.87，paired Δ +1.65，CI [+1.41, +1.87]。iid jitter Δ = +0.17，CI [−0.01, +0.36]；shuffled residual Δ = −0.16，CI [−0.34, −0.02]。两个 magnitude-matched 控制都不能复制 CSDI 增益。§4.4 的替代 imputer 比较加上：`SAITS-pretrained → Panda` 也跨过此处前沿，paired CSDI − SAITS = +0.41 [+0.05, +0.87]（CSDI 严格正但小）。

**L96 N=20 SP65**（$n=10$）：iid jitter、shuffled residual、CSDI 都把 Panda mean 推向正方向，但因 Panda 有极少数长 forecast seed，mean 上没有 cleanly separation。每个 seed 上 rank order 都被保留，CSDI 在 tail survival 上最强：$\Pr(\mathrm{VPT}>1.0\,\Lambda)$ CSDI 80%（Wilson 95% [49%, 94%]）vs linear / iid / shuffled 各 40%（[17%, 69%]）。我们**预注册** median + survival 作为 Panda mean 高方差的 L96 cell 的 headline 度量。

**L63 SP82**（底部带，$n=10$）：iid jitter CI 不跨 0，shuffled residual 助力轻微（Δ +0.34），CSDI 给出 Δ +1.09，CI [+0.65, +1.61]，$\Pr(\mathrm{VPT}>1.0\,\Lambda) = 70\%$。§4.4 替代 imputer 比较加上：`SAITS-pretrained → Panda` 在此 cell 与 CSDI **统计不可分辨**（paired CSDI − SAITS = +0.06，[−0.31, +0.59]）——所以底部带的发现是"任何 corpus-pretrained 结构化 imputer 都跨过，并且都在 magnitude-matched 控制之上"，而**不是** "CSDI 特定地"。

**L96 N=20 SP82**（$n=10$）：Panda mean 被极少数 lucky linear seed 主导（如 seed 2 的 `keep_frac = 0.15` 恰好对齐一个可预测 Panda token 序列，所有 cell 的 VPT@1.0 都达 10.75）；我们因此**预注册 median + survival 作为高维高方差 L96 cell 的 headline 度量**，而**不**把 mean 当作 primary read。在这两个度量上次序干净：linear < SAITS-pretrained < CSDI 在 median（0.50 / 0.84 / 1.13）与 Pr(VPT>1.0)（30 / 40 / 60%）上同时成立。在 forecaster 层面，DeepEDM 在该 cell 给出唯一的**严格正 paired CSDI − linear CI**（+0.43，[+0.29, +0.57]；§3.2）。

**Rössler SP65 / SP82**（$n=5$）：相同的正向 CSDI 方向，但 Lyapunov 指数小使 $\Pr(\mathrm{VPT}>1.0\,\Lambda)$ 太严；$\Pr(\mathrm{VPT}>0.5\,\Lambda)$ 是更合适的 tail 度量。

**纯噪声轴**（$s=0$，$\sigma > 0$）：CSDI 在每一档 $\sigma$ 上对 Panda 中性或略有伤害（Figure 1 噪声线）。CSDI 因此是稀疏间隙补全杠杆，不是通用密集噪声 denoiser。

这些 cell-wise 观察是 abstract 必须写"在 transition band 之内"的原因：iid jitter 与 magnitude-matched shuffled 残差都不能复制 **corpus-pretrained 结构化 imputation** 所做的——§4.4 替代 imputer 对照显示 CSDI 与 corpus-pretrained SAITS 都跨过前沿、而这两个 magnitude-matched 控制都不；CSDI 在 L63 入口带保留小但 paired-CI-strict 的优势。**来自 corpus-pretrained imputer 的结构化残差因此与同等量级 iid 噪声不完全可互换**，尤其在 tail survival 上而非 mean VPT 上。

### 4.4 替代 imputer 比较

为回答 "结构化 imputation 本身是不是 lever，还是 CSDI 特定的动力学感知扩散先验是必要的"，我们加入一个**预训练**的 SAITS imputer——在 CSDI 使用的同一 L63 混沌语料（约 64K 独立 IC、长度 128 窗口，缺失分布与 v2 corruption grid 匹配）上预训。推理时以 SAITS 预训上下文长度（128）切非重叠 chunk，每个 chunk 独立补全。

| Cell | L63 SP65（n=10）| L63 SP82（n=10）|
|:--|:--:|:--:|
| `linear → Panda` mean VPT | 1.22 | 0.29 |
| `SAITS-pretrained → Panda` | **2.49** | **1.51** |
| `CSDI → Panda` | **2.89** | **1.57** |

| Paired 对比 | SP65 Δ（95% CI）| SP82 Δ（95% CI）|
|:--|:-:|:-:|
| SAITS-pretrained − linear | +1.26 [+0.83, +1.64] ↑ | +1.23 [+0.86, +1.62] ↑ |
| CSDI − linear | +1.67 [+1.41, +1.92] ↑ | +1.28 [+0.73, +1.85] ↑ |
| **CSDI − SAITS-pretrained** | **+0.41 [+0.05, +0.87] ↑** | **+0.06 [−0.31, +0.59] ≈** |

Tail survival probability $\Pr(\mathrm{VPT}>1.0\,\Lambda)$（Wilson 95% CI）：

| Cell | SP65 | SP82 |
|:--|:-:|:-:|
| linear | 70% [40%, 89%] | 0% [0%, 28%] |
| SAITS-pretrained | 90% [60%, 98%] | 70% [40%, 89%] |
| CSDI | 100% [72%, 100%] | 70% [40%, 89%] |

**跨系统复制：L96 N = 20 SP82（n = 30）.** 我们另在 L96 N = 20 混沌语料（`lorenz96_clean_512k_L128_N20.npz`，64K 长度 128 窗口，同 v2-grid 匹配缺失分布）上预训第二个 SAITS imputer，val MAE = 1.07 = 0.29 × `attractor_std`。为稀释 §4.3 预注册中标注的 lucky-seed 效应，我们在此 cell 上跑 **30 seeds**（全文唯一 30-seed cell）：

| Cell | mean VPT | median VPT | $\Pr(\mathrm{VPT}>1.0\,\Lambda)$ Wilson 95% |
|:--|:-:|:-:|:-:|
| `linear → Panda` | 0.86 | 0.25 | 20% [10%, 37%] |
| `SAITS-pretrained → Panda` | 1.57 | 1.01 | 50% [33%, 67%] |
| `CSDI → Panda` | **1.87** | **1.26** | **73%** [56%, 86%] |

Paired-bootstrap on means（5000 resamples，$n = 30$）：

| Paired contrast | Δ | 95% CI | sign |
|:--|:-:|:-:|:-:|
| SAITS-pretrained − linear | +0.71 | [+0.02, +1.38] | ↑ |
| CSDI − linear | +1.01 | [+0.36, +1.64] | ↑ |
| CSDI − SAITS-pretrained | **+0.31** | **[+0.07, +0.56]** | **↑** |

30 seeds 下 lucky-seed 稀释消除了 10 seeds 时 L96 mean 上的歧义：每个度量（mean / median / Pr(VPT > 0.5) / Pr(VPT > 1.0)）都单调 `linear < SAITS-pretrained < CSDI`，三个 paired 对比都严格正。CSDI − SAITS-pretrained 在均值上变成严格正 paired CI，与 L63 SP65 入口带匹配。

L63 SP65 + SP82 + L96 SP82 共同把 §1 intervention 主张从 "CSDI 是唯一测试过的 intervention" 显著收窄为 "**corpus-pretrained 结构化 imputation 是 lever**，CSDI 在 L63 入口带保留小但 paired-CI-strict 的优势、在 L96 SP82（n=30）means 与 median + survival 上都给出严格正 paired CI、在 L63 底部带与 SAITS-pretrained 不可分辨"。这一现象——某个 corpus-pretrained 结构化 imputer 在 linear 插值崩塌的稀疏观测 transition band 处仍能跨过——**不是 CSDI 独有**；linear < SAITS-pretrained < CSDI 的次序在第二个 corpus-pretrained imputer 上被复制，且在两个不同的混沌系统（3-D L63 与 20-D L96）上同时观察到。我们**不**主张更宽广的推广；§6.6 的 Jena Climate 实测案例显示该 lever 在周期主导的真实数据上**不**复现，定义了主张的边界。

一个独立的单轨迹 SAITS / BRITS sanity check（无预训语料、在单条测试轨迹上逐实例拟合）作为支持性观察列在附录 E。逐实例训练在设计上对 SAITS / BRITS 不公，所以这**不是**主对照实验。

Glocal-IB 与其他近期全局结构 imputer 未评测；它们在 §2 列为 adjacent prior art，是自然后续。

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

正文主图待测的 forecaster 是 Panda-72M [Wang25]。§4 的隔离矩阵覆盖 $\{$linear, AR-Kalman, CSDI$\} \times \{$Panda, DeepEDM$\}$。为消除 Panda-tokenizer 特异性 attack surface，我们在 L63 稀疏度 transition band（SP55–SP82）上额外在两个 horizon 下评测 Chronos [Ansari24]（`pred_len = 128` 与 Panda 匹配，`pred_len = 64` 与 Chronos 原生训练 horizon 匹配），cells 为 `linear → Chronos` 与 `CSDI → Chronos`；跨 foundation 观察见 §6.4（Chronos 停在低 VPT 平台、不展示 Panda 的前沿形状；CSDI 救援不可见因为 Chronos 自身的 VPT 分布低于杠杆能起作用的 regime）。§4.4 的替代 imputer 对照使用与 CSDI 同语料预训的 SAITS imputer，因此在训练数据轴上是公平的。

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

我们**不**主张预训练混沌 forecaster 本质上坏掉了——救援结果显示它们在 corruption-aware 填补之下高度可恢复。我们**不**主张 CSDI 是唯一能 work 的 imputer——而是：在我们以匹配 corpus 预训测试的 imputer（linear、Kalman、CSDI、SAITS）之中，CSDI 在 L63 transition band 上给出最强的 paired CSDI − linear 增益（§4.4）。我们**不**主张机制被完全刻画——raw-patch 与 Panda-token 距离-到-clean 解释入口带；底部带的 survival 受这些距离 informed 但不能简化为它们。我们**不**主张普适跨所有 foundation forecaster——Panda-72M 是 headline；Chronos 在 L63 稀疏度线上作为跨 foundation 观察被报告（§6.4 — Chronos 停在低 VPT 平台、不展示 Panda 的 transition 形状），TimesFM / Lag-Llama 未评测。

### 6.3 Scope condition

延迟流形伴随假设光滑吸引子与有用的有限维 Takens 表示。Mackey-Glass 与 Chua 作为附录 scope boundary 报告。Mackey-Glass 是标量延迟微分系统，其有效状态在观测窗口下无限维；可用的 CSDI 训练语料与延迟配置不能跨过相关历史维。Chua 是分段线性、非光滑电路；M1 / DeepEDM 中隐含的光滑吸引子假设被破坏。这些是**诚实的边界**，不是被掩盖的失败。

### 6.4 局限

- **补全训练语料轴.** §4.4 的替代 imputer 对照将 CSDI 与一个在同一混沌语料（约 64K 独立 IC、长度 128 L63 窗口，缺失分布与 v2 corruption grid 匹配；见附录 C）上预训的 SAITS 模型配对。一个独立的单轨迹逐实例 SAITS / BRITS sanity 在附录 E 报告。我们没有评测 Glocal-IB 或其他近期全局结构 imputer；这些在 §2 列为 adjacent prior art，是开放后续。
- **Forecaster 广度.** Panda 是 headline。我们在两个 horizon 下评测 Chronos-bolt-small 在 L63 稀疏度线上（SP55–SP82, 5 seeds, `linear → Chronos` 与 `CSDI → Chronos`）：`pred_len = 128`（与 Panda 匹配）与 `pred_len = 64`（Chronos 原生训练 horizon，因为 Chronos 库本身警告 `prediction_length > 64` 超出训练分布）。两个 horizon 下 Chronos 绝对 VPT 都远低于 Panda（mean 0.34–0.50, $\Pr(\mathrm{VPT}>1.0)$ ≤ 20%），CSDI 也未明显改善（paired Δ 在 0 附近，CI 跨 0）。`pred_len = 64` 的 per-seed VPT 与 `pred_len = 128` 在统计上不可区分，确认负面结果**不是** Chronos OOD horizon 的 artefact。我们读为："corpus-pretrained-imputation 杠杆在 Panda 上经验性可观察；在 Chronos 上杠杆不可观察是因为 Chronos 自身的 VPT 分布远低于杠杆能起作用的 regime。" TimesFM / Lag-Llama 等其他大型预训练时序 forecaster 在匹配 horizon 下的评测留作未来工作。
- **已知动力学上界.** 一个 model-aware 参考（用真实 L63 向量场的 stochastic EnKF, 100 ensemble members）在整个稀疏观测 transition band 上撞 VPT ceiling（SP55–SP82 mean 2.84–2.85, $\Pr(\mathrm{VPT}>1.0) = 100\%$；附录 B）。前沿因此是**黑盒部署接口**（forecaster 不能访问动力学）的属性，不是系统本身的属性。
- **纯噪声轴.** 论文的干预主张限定在稀疏观测轴上。CSDI 在密集噪声轴上中性或略有伤害；denoising-aware 变体是开放后续。
- **L96 高方差.** L96 N=20 在前沿底部 cell 上 mean VPT 高方差（即使 n=10 仍被极少数长 forecast seed 主导）。我们因此用 median 与 survival 作为 L96 的 headline，而**不**用 Panda mean。
- **系统广度.** L63、L96 N=10 / 20、Rössler、Kuramoto 覆盖正向复制；Mackey-Glass 与 Chua 是 scope boundary。Jena Climate 小时尺度作为真实传感器**负向** case study 报告（§6.6），把杠杆限定在混沌吸引子主导的 regime。KSE / dysts 广度与其他真实数据 case study（EEG、气候 reanalysis）保留为后续。
- **基础模型可解释性.** 为何 CSDI 残差在底部带能产生可预测 context、即使 raw-patch 与 Panda-token 距离-到-clean 已不再分离它与 linear，仍是开放问题。一个自然假设是相关几何量在 Panda 更深的 latent dynamics（decoder 而非 encoder）里——我们尚未对其插桩。

### 6.5 与数据同化的关系

在动力学已知且在线追踪时，序贯数据同化（EnKF / LETKF）是更丰富、信息更高效的稀疏含噪混沌方法。我们的 setting 不同：forecaster 是黑盒（Panda），corruption 在线下被预处理。因此我们对照预处理风格 baseline（linear / Kalman / CSDI），匹配部署接口；并把对应 DA 文献当作**前沿存在性的 motivating background**，不是直接竞争对手。

### 6.6 真实传感器 case study：Jena Climate（杠杆的边界）

为在真实多变量传感器流上压力测试 §4.4 的"corpus-pretrained 结构化 imputation 是 lever"主张，我们在公开的 Jena Climate 2009–2016 数据集（14 个数值大气特征，10 分钟采样，下采样到小时；train 2009–2014, val 2015, test 2016，详见附录 C.2）上跑同一稀疏-context-fill 协议。我们额外纳入：(a) **clean-context 上界**（不 corrupt，ctx_true → forecaster 直接），作为任何 imputer-then-forecaster 路径能希望达到的天花板；(b) **跨 forecaster 控制**（Chronos-bolt-small *和* Panda-72M），用以区分"imputation 轴"效应与 forecaster-specific 弱点。Imputer：linear 与 SAITS-pretrained-on-Jena（在 train split 上预训，val MAE 0.62 z-units）。10 seeds × {SP55, SP65, SP75, SP82}，$n_{ctx} = 512$ 小时，$pred_{len} = 64$ 小时。度量：normalized valid horizon vh@τ —— 跨 14 个 z-scored 特征的 per-step RMSE 保持 ≤ τ 的最大 lead-step。

**Clean-context 上界与跨 forecaster.**

| | clean | linear | SAITS-pretrained |
|:--|:-:|:-:|:-:|
| `→ Chronos`（vh@1.0 mean，SP55–SP82 平均）| 51.1 | 50.6（avg）| 30.3（avg）|
| `→ Panda`（vh@1.0 mean，SP55–SP82 平均）| 46.4 | 43.2（avg）| 35.2（avg）|

两个事实凸显：

1. **Linear-fill ≈ clean-context** 在两个 forecaster 上都成立。Chronos 上 per-cell linear-fill mean（51.1 / 50.9 / 48.5 / 50.9）跟踪 clean 上界（51.1）在 1–3 vh-units 之内；Panda 上 linear means（45.7 / 41.8 / 46.0 / 39.2）也保持在 clean 46.4 的 1–7 之内。Linear 插值已经保留了主导日循环的足够信号，让 forecaster 触及 clean-context 天花板。
2. **SAITS-pretrained 把两个 forecaster 都拖到 clean 之下.** Chronos 上 mean vh@1.0 跌到 27–34；Panda 上跌到 30–39。Paired SAITS − linear 在 vh@1.0 上（5000-resample bootstrap，$n = 10$）：

| Cell | Chronos paired Δ（95% CI）| Panda paired Δ（95% CI）|
|:--|:-:|:-:|
| SP55 | −16.7 [−28.2, −5.8] ↓ | −6.3 [−16.6, +3.0] ≈ |
| SP65 | −18.8 [−29.7, −8.2] ↓ | −4.1 [−12.9, +5.2] ≈ |
| SP75 | −21.0 [−34.3, −8.6] ↓ | −12.0 [−20.0, −4.7] ↓ |
| SP82 | −23.6 [−39.2, −8.6] ↓ | −9.7 [−19.9, −0.9] ↓ |

SAITS-hurts 模式是**跨 forecaster 的** —— 在 Chronos 上每个 cell 都严格负，在 Panda 上 SP75 与 SP82 严格负、SP55 / SP65 方向负 —— 因此这**不是** Chronos-specific artefact。Clean-context 上界还排除了"forecaster 自身是瓶颈、SAITS 只是恰好在噪声平台一侧"的假设：SAITS-fill 与 clean 的差距在 Chronos 上是 17–24 vh-units、Panda 上是 7–17，远超 seed-to-seed 噪声。

**读.** §4.4 lever **不**适用于 Jena，且机制**不是** "Chronos 太弱"：

> Linear 插值已经在 Jena hourly 上达到 clean-context 预测天花板，因为主导时间结构是确定性日周期性，linear 已经"免费"保留了它。在含噪真实语料上 fit 的 SAITS imputer 引入 sample-specific 高频 artefacts，把填补 context 推**离** forecaster 依赖的周期模式，所以两个 forecaster 都跌破 clean。learned imputer 没有 headroom 可以救援。

这干净地界定了 §4.4 主张：

> **Corpus-pretrained-imputation 救援在混沌吸引子主导的系统（L63、L96）上可观察 —— 这种系统中 linear 插值会破坏 foundation forecaster 依赖的局部几何结构。在周期主导的真实数据流（Jena 小时）上，linear 插值已经达到 clean-context 天花板，corpus-pretrained imputer 在两类 forecaster（broad time-series 的 Chronos 和 chaos-pretrained 的 Panda）上都净有害。**

前沿故事因此是**混沌系统性质**，不是普适的稀疏-context-fill 主张。我们把这一 case study 留在 §6 而不提到 §3，因为它是一个**负向**结果，定义了主张的边界，而 headline 前沿陈述（§3.2）保持不变。来源：
- Chronos + clean 上界：`experiments/week1/results/jena_real_sensor_jena_chronos_with_clean_upper_10seed.json`
- Panda 跨 forecaster 控制：`experiments/week1/results/jena_real_sensor_jena_panda_with_clean_upper_10seed.json`

---

## 7 结论

一个预训练混沌 foundation forecaster（Panda-72M）沿一条**清晰的稀疏观测可预测性前沿**崩塌；第二个预训练 forecaster（Chronos）展示一种不同的失效模式（低 VPT 平台），因此 transition 形状是 forecaster-dependent 的。在 Panda 的 transition band 之内，救援是**regime-aware** 的，而非普适的：在入口带，corruption-aware imputation 把 Panda context 朝 clean token 移动，可预测性回归；在前沿底部，distance-to-clean 度量不再分离救援 context 与线性插值，但 survival 优势仍然存在，且**结构化 CSDI 残差与同等量级 iid 噪声不可互换**。CSDI 是间隙补全杠杆——**不是 denoiser**——而 Takens 坐标下的 DeepEDM 是动力学感知伴随路线，Mackey-Glass 与 Chua 为显式 scope boundary。

最干净的一句话因此**不**是"基础模型本质上 OOD"，**也不**是"结构化 imputer 总救"，而是：

> **稀疏观测预处理把预训练混沌 forecaster 放在一条尖锐可预测性前沿的某一侧；corruption-aware imputation 是在 transition band 之内跨越它的杠杆——但**控制这次跨越的几何量并不是逐点重构保真度**。

代码、CSDI checkpoint、预训练 SAITS baseline 与锁定的 Figure-1 / 隔离矩阵 / jitter / embedding 数据均已发布。Glocal-IB、真实数据案例与 Panda decoder-side 插桩是自然的 camera-ready 后续。

---

> 以下附录已对齐锁定故事（2026-05-01）。pivot 之前描述四模块流水线（M1–M4）与通用 tokenizer-OOD 定理族的材料逐字保留在 `deliverable/paper/paper_draft_zh_archive_2026-04-30.md` 中，本稿仅以指针引用。

## 附录 A：理论指针（继承，已收窄）

正文依赖**实证**入口带 OOD 减少（§4.2）与底部带剩余 survival（§4.3），而不是"ambient-predictor tokenizer-OOD 失败"的封闭定理。早期草稿用一个 Theorem 2 给出该普适表述；在 v2 协议下该定理仅作为入口带陈述成立。我们把证明材料保留在 `paper_draft_zh_archive_2026-04-30.md` 的"附录 A：Formal 证明草稿"中，供希望阅读形式界的读者参考；这里把它作为**收窄**的理论伴随：同一 patch 分布 OOD 论证在 L63 SP65 处适用（那里测得 Panda token 空间到 clean 的距离比 12–22 倍降，跨四个表征阶段），但**不**作为穿过整个前沿的普适主张。正文的干预与前沿主张**不依赖**该证明。

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
| 13 | **预训练替代 imputer（P1.1 + P1.5 + P2.2 30-seed）** | L63、L96 N=20 | L63 SP65 + SP82、L96 SP82 | linear、SAITS-pretrained、CSDI | L63: 10, L96: **30** | `panda_altimputer_l63_sp65_sp82_pretrained_10seed_chunked.json`、`panda_altimputer_l96_sp82_pretrained_30seed.json` | §4.4 + 附录 C |
| 14 | **Chronos mini-frontier（P1.2）** | L63 | SP55, SP65, SP75, SP82 | linear、CSDI（forecaster: Chronos，`pred_len ∈ {64, 128}`）| 5 | `chronos_frontier_l63_chronos_l63_sp55_sp82_5seed.json`、`..._5seed_pl64.json` | §6.4 跨 foundation 观察；pred_len=64 确认负面结果不是 Chronos OOD horizon 的 artefact |
| 15 | **EnKF 已知动力学上界（P1.3）** | L63 | SP55–SP82, NO020, NO050 | EnKF（真实向量场，100 ensemble members）| 5 | `enkf_l63_enkf_l63_v2_5seed.json` | §6.5 / 附录 B 参考 |
| 16 | **真实传感器 case study（P2.1 + P3.A clean-upper / 跨 forecaster）** | Jena Climate 2009–2016 | SP55, SP65, SP75, SP82（hourly, $n_{ctx}=512$, $pred_{len}=64$）| clean、linear、SAITS-pretrained-on-Jena × {Chronos-bolt-small, Panda-72M} | 10 | `jena_real_sensor_jena_chronos_with_clean_upper_10seed.json`、`jena_real_sensor_jena_panda_with_clean_upper_10seed.json` | §6.6 边界 case study 加 clean-context 上界 + 跨 forecaster 控制；度量为 z-RMSE 单位下的 normalized valid horizon vh@τ |

#1–#9 是 §3 / §4 / §6 引用的 patched 协议锁定数。#10 是用旧 S0–S6 corruption pipeline（`make_sparse_noisy`）的 cross-system 复制，作为**辅助**方向证据；#1–#6 / #8 / #9 的 v2 协议数为权威。#11 提供 §6.3 scope condition。#12 是附录 sanity（per-instance 训练，对 SAITS / BRITS 不公）。#13–#15 是 P1 reviewer-defense 实验：预训练 SAITS 替代 imputer 对照（§4.4 / 附录 C）、Chronos 跨 foundation mini-frontier（§6.4）、EnKF 已知动力学上界（§6.5 / 附录 B）。

### B.3 复现性说明：CSDI 采样器随机性跨 §3.2 / §4.3 / §4.4

CSDI 是条件分数扩散 imputer；其 `impute(...)` 调用是随机的，所以**相同**的 corruption draws 上独立运行会产生略有不同的 filled context、进而略有不同的 per-seed VPT。§3.2（Figure 1 v2 grid）、§4.3（jitter 控制）、§4.4（替代 imputer）的锁定数来自三次独立 CSDI 推理 run，各自有独立的 sampler seed。L63 SP65 paired CSDI − linear 在 §3.2 / §1 是 +1.64，在 §4.3 是 +1.65，在 §4.4 是 +1.67；三者 CI 重叠，定性结论（L63 入口带 paired CSDI − linear 严格正）稳定。同一 caveat 也适用于 §3.2 L96 SP82 跨系统复制（Panda median 0.50 → 1.05）与 §4.4 L96 SP82 替代 imputer 对照（10 seeds 时 median 0.50 → 1.13；30 seeds 时 0.25 → 1.26）之间的小差：相同 v2 corruption seed scheme、相同 Panda checkpoint，但独立 CSDI 推理 run。我们没有跨实验冻结 diffusion seed；逐 seed VPT 的差（typically < 0.5 Λ）反映 CSDI 采样器随机性，**不是**协议漂移。所有实验都作为独立 JSON 发布（附录 B.2 行 1 / 4 / 13），per-run 数字可重新派生。

### B.4 聚合脚本

每个聚合器以 `python -m experiments.week1.<script>` 运行：

- `aggregate_figure1_v2.py --halves --s_tag ... --n_tag ... --out_prefix ...` → 六面板 Figure 1，mean 用 bootstrap CI、$\Pr(\mathrm{VPT}>\theta)$ 用 Wilson CI；
- `aggregate_isolation.py --json <iso JSON> --out_prefix ...` → #10 / #11 的 2×3 heatmap + bar chart 与 paired bootstrap CI；
- `aggregate_jitter_cross_system.py` → 六面板 Figure 3（mean vs $\Pr(\mathrm{VPT}>1.0)$，跨 L63 / L96 / Rössler 在 SP65 与 SP82）；
- `aggregate_corruption_grid.py` → 任一 v2 grid run 的 metadata 表（keep fraction、obs/patch、Lyapunov 单位下最大 gap）；
- `aggregate_survival_summary.py` → 跨系统 $\Pr(\mathrm{VPT}>0.5)$ 与 $\Pr(\mathrm{VPT}>1.0)$。

## 附录 C：预训练替代 imputer 细节

**训练.** 我们在 `experiments/week2_modules/data/lorenz63_clean_64k_L128.npz`（约 64K 独立 IC L63 长度 128 窗口）上预训一个 SAITS [Du22] imputer。每个训练窗口的稀疏度从 v2 `fine_s_line` 网格（`{0, 0.20, 0.40, 0.55, 0.65, 0.75, 0.82, 0.88, 0.93, 0.97}`）均匀抽取，应用 iid_time mask，使 SAITS 训练 corruption 分布与 v2 评测分布一致。架构：2 SAITS 层、$d_{model} = 64$、4 heads、$d_k = d_v = 16$、$d_{ffn} = 128$。30 epochs，batch 64，~18 min on 1 GPU。最佳 ckpt 在 epoch 30；最终训练 MAE = 0.47，验证 MSE = 8.28，验证 missing entries MAE = 1.26（= 0.149 × $\sigma_\text{attr}$）。

Checkpoint：`experiments/week2_modules/ckpts/saits_l63_pretrained/<run-id>/SAITS.pypots`。

**推理.** SAITS 需要固定输入长度匹配位置编码。测试 context（长度 512）切成 4 个非重叠 length-128 chunk，每个独立补全后拼接。CSDI 的变长推理不变。

**结果（10 seeds，L63 SP65 + SP82, σ = 0）.**

| Cell | SP65 mean | SP82 mean | SP65 Pr>1.0 | SP82 Pr>1.0 |
|:--|--:|--:|--:|--:|
| linear | 1.22 | 0.29 | 70% | 0% |
| SAITS-pretrained | 2.49 | 1.51 | 90% | 70% |
| CSDI | 2.89 | 1.57 | 100% | 70% |

| Paired 对比 | SP65 Δ（95% CI）| SP82 Δ（95% CI）|
|:--|:-:|:-:|
| SAITS − linear | +1.26 [+0.83, +1.64] ↑ | +1.23 [+0.86, +1.62] ↑ |
| CSDI − linear | +1.67 [+1.41, +1.92] ↑ | +1.28 [+0.73, +1.85] ↑ |
| CSDI − SAITS | +0.41 [+0.05, +0.87] ↑ | +0.06 [−0.31, +0.59] ≈ |

**读.** 预训练 SAITS 复现了 L63 transition band 救援的大部分。CSDI 在入口带保留小但 paired-CI-strict 的优势，在底部带与 SAITS-pretrained 统计不可分辨。我们因此把 §1 / abstract 的 intervention 主张收窄为 "corpus-pretrained 结构化 imputation 是 lever；CSDI 是其中一个强实例，在入口带有小 advantage"。

**L96 N = 20 SP82 跨系统复制（10 seeds, σ = 0）.** 在 `lorenz96_clean_512k_L128_N20.npz`（64K 长度 128 窗口，相同 v2-grid 匹配缺失分布、相同架构）上预训第二个 SAITS imputer。Best ckpt 在 epoch 27；validation MAE on missing = 1.07（= 0.29 × $\sigma_\text{attr}^{(\mathrm{L96})}$）。逐 seed VPT@1.0：

| seed | linear | SAITS-pretrained | CSDI |
|:-:|--:|--:|--:|
| 0 | 0.50 | 0.67 | 0.76 |
| 1 | 0.00 | 0.84 | 1.85 |
| 2 | **10.75** | 4.96 | 5.12 |
| 3 | 0.42 | 0.76 | 0.76 |
| 4 | 3.78 | 4.12 | 4.12 |
| 5 | 1.09 | 1.09 | 1.01 |
| 6 | 0.50 | 1.34 | 1.26 |
| 7 | 0.25 | 0.84 | 0.92 |
| 8 | 0.50 | 0.50 | 0.50 |
| 9 | 0.25 | 0.50 | 1.43 |
| **mean** | 1.81 | 1.56 | 1.77 |
| **median** | 0.50 | 0.84 | **1.13** |

seed 2 是 linear 的 clean-context fluke（linear seed-2 keep-fraction = 0.15 恰好对齐了一个可预测的 Panda token 序列，三个 cell 都得到长 forecast）。按 §6.4 标注的 L96 高方差局限，mean 不可靠；以 median + survival 为 headline：

| Cell | $\Pr(\mathrm{VPT}>1.0\,\Lambda)$ | Wilson 95% CI |
|:--|:-:|:-:|
| linear | 30% | [11%, 60%] |
| SAITS-pretrained | 40% | [17%, 69%] |
| CSDI | **60%** | [31%, 83%] |

Paired-bootstrap（5000 resamples）on means：CSDI − SAITS-pretrained = +0.21 [+0.00, +0.49]（恰好触及 0）；CSDI − linear 与 SAITS − linear 跨 0（被 linear-seed-2 离群驱动）。median + survival 上的定性次序与 L63 SP65 一致：linear < SAITS-pretrained < CSDI。我们读为 "结构化 imputation lever 在 20-D 系统上以 L96 cell 使用的 survival 度量推广；mean 噪声太大无法直接读"。

Glocal-IB 未评测（§2 引为 adjacent prior art：高缺失补全应保留全局潜结构）；自然后续。

来源：
- L63：`experiments/week1/results/panda_altimputer_l63_sp65_sp82_pretrained_10seed_chunked.json`
- L96：`experiments/week1/results/panda_altimputer_l96_sp82_pretrained_10seed.json`

## 附录 D：Figure 索引

正文引用的所有图都使用 patched v2 协议，除非显式标注。Figure 1 / 2 / 3 是三张 headline panel。

### 主图

| 标签 | 用途 | 路径 |
|---|---|---|
| **Figure 1** | L63 上稀疏度与噪声前沿（Panda mean VPT、$\Pr(\mathrm{VPT}>0.5\,\Lambda)$、$\Pr(\mathrm{VPT}>1.0\,\Lambda)$，$s$ 与 $\sigma$ 解耦轴；**10 seeds**；**patched v2 协议**；mean 95% bootstrap CI、survival Wilson 95% CI）| `deliverable/figures_main/figure1_l63_v2_10seed_patched.png`（与 `.md` 表）|
| **Figure 2** | Cross-system 隔离矩阵：linear / Kalman / CSDI × Panda / DeepEDM（heatmap 与 paired-bootstrap CI 条形图；**5 seeds**；**legacy S0–S6 协议** — 辅助方向证据；§3.2 的 v2 协议数为权威）| `deliverable/figures_isolation/l63_iso_l63_5seed_heatmap.png`、`l96_iso_l96N10_5seed_heatmap.png`、`l96_iso_l96N20_5seed_heatmap.png`、`rossler_iso_rossler_5seed_heatmap.png`（每个有匹配的 `_bars.png`）|
| **Figure 3** | 跨 L63、L96 N=20、Rössler 在 SP65 与 SP82 的 jitter / 残差控制（Panda mean VPT vs $\Pr(\mathrm{VPT}>1.0\,\Lambda)$）；L63 **10 seeds**，L96 / Rössler **5 seeds**；**patched v2 协议**；每个 Δ 的 paired-bootstrap CI | `deliverable/figures_jitter/jitter_milestone_SP65.png`、`jitter_milestone_SP82.png` |

### §4.2 机制面板（patched v2 协议）

| 元素 | 路径 |
|---|---|
| L63 raw-patch 度量直方图（local stdev / lag-1 ρ / mid-freq power，SP65 + SP82，10 seeds）| `experiments/week1/figures/l63_patch_ood_v2_v2protocol_metrics_SP65.png`、`..._SP82.png` |
| L63 raw-patch 轨迹叠加（clean vs linear vs CSDI，10 seeds）| `experiments/week1/figures/l63_patch_ood_v2_v2protocol_traj_overlay_SP65.png`、`..._SP82.png` |
| Panda token 空间距离条形图（patch / embed / encoder / pooled，SP65 + SP82，5 seeds，paired-bootstrap CI）| `experiments/week1/figures/panda_embedding_ood_l63_sp65_sp82_dt025_v2protocol_patched_5seed_bars.png` |
| Panda token 空间 PCA 散点（按阶段与场景，5 seeds）| `experiments/week1/figures/panda_embedding_ood_l63_sp65_sp82_5seed_SP65_embed_pca.png`、`..._SP65_encoder_pca.png`、`..._SP82_embed_pca.png`、`..._SP82_encoder_pca.png` |

### §4.4 / 附录 C 替代 imputer（P1.1 + P1.5）

| 元素 | 路径 |
|---|---|
| L63 SP65 + SP82 替代 imputer 摘要表（linear / SAITS-pretrained / CSDI；10 seeds；paired-bootstrap CI；survival Wilson CI）| `experiments/week1/figures/panda_altimputer_l63_sp65_sp82_pretrained_10seed_chunked.md` |
| L96 N=20 SP82 替代 imputer 摘要表（linear / SAITS-pretrained / CSDI；10 seeds；median + survival Wilson CI；mean 上 paired-bootstrap）| `experiments/week1/figures/panda_altimputer_l96_sp82_pretrained_10seed.md` |

### §6.3 Scope-boundary 面板（仅附录，**legacy S0–S6 协议**，5 seeds）

| 元素 | 路径 |
|---|---|
| Mackey-Glass S0–S6 相变曲线 | `experiments/week1/figures/pt_mg_mg_5seed_phase_transition.png` |
| Mackey-Glass 吸引子轨迹 | `pictures/mackey_glass_trajectory_final.png` |
| Chua S0–S6 相变曲线 | `experiments/week1/figures/pt_chua_chua_5seed_phase_transition.png` |
| Chua 双涡卷轨迹 | `pictures/chua_trajectory_final.png` |

### Pre-pivot 图（不再在正文引用）

之前草稿引用了一组四模块流水线特定图（M1 imputation 轨迹、M2 τ-search Bayesian-optimisation 曲线、M3 backbone 比较、M4 conformal coverage）。这些保留在 `deliverable/figures_extra/` 与 `experiments/week2_modules/figures/`，但不属于锁定故事；归档稿 `paper_draft_zh_archive_2026-04-30.md` 保留它们的原始 caption。

## 附录 E：继承的补充材料（已归档）

pre-pivot 草稿中的三个附录段对锁定故事不再 load-bearing，但保留供有兴趣的读者：

- **原附录 E — τ-search 详尽实证.** Mutual-information / Lyapunov 目标、Bayesian-optimisation 轨迹、对随机 τ 的 ablation。用于支持 §5.2 中 DeepEDM 的 lag schedule。
- **原附录 F — τ-coupling 完整实证分析.** 各系统对 lag schedule 的 forecastability 敏感性。证实 DeepEDM 的 lag 选择不是 knife-edge 调参伪迹。
- **原附录 G — 延迟流形视角.** 几何框架（Takens 嵌入、吸引子重构、光滑性假设），为 §5.2 伴随 forecaster 与 §6.3 scope condition 提供动机。

来源：`paper_draft_zh_archive_2026-04-30.md` 中同名的各节。这些都不是验证 §1 / §3 / §4 / §6 主张所必需的；它们加强 §5 方法表述与 §6.3 scope 解释。
