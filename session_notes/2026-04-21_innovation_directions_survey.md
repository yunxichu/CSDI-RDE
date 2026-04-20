# CSDI-RDE-GPR 顶会顶刊创新方向调研

**日期**：2026-04-21
**目标刊物**：ICML / NeurIPS / ICLR / KDD / AAAI / TPAMI
**调研时间窗口**：2023–2026（以 2024–2025 为主）

---

## 现状诊断（方法的 5 个可攻击点）

| # | 痛点 | 对应可改进方向 |
|---|---|---|
| 1 | GPR O(n³) → EEG h=976 不能跑 | **A. 可扩展 GP** |
| 2 | CSDI 只做补值，未做预测，两阶段割裂 | **B. 端到端扩散预测** / **E. 联合训练** |
| 3 | RDE 随机采样 (d,τ) 不 principled | **C. 可学习延迟嵌入** |
| 4 | KDE 融合是启发式，PICP@2σ 不是 coverage-valid | **D. Conformal Prediction** |
| 5 | 混沌系统无物理约束，长期预测会漂 | **F. 物理约束 / invariant measure** |

---

## 方向 A：可扩展高斯过程 / 深度核学习

**一句话**：用 SVGP / Neural Process 家族打穿 O(n³)。

- **LBANP** (ICLR 2023, [arXiv:2211.08458](https://arxiv.org/pdf/2211.08458v1))：latent bottleneck tokens 把 TNP 的 O(N²) 压到常数。**借鉴点**：整体替换"s 个 GPR + KDE 融合"。
- **New SVGP Bounds** (2025, [arXiv:2502.08730](https://arxiv.org/html/2502.08730v2))：更紧 collapsed bound，inducing points 更少。**借鉴点**：每个 (d,τ) 的 GPR 换成 SVGP，M=64–128 inducing points → 支持 h=976。
- **Transformer Neural Processes (TNP)**：meta-learning 输出 predictive distribution。**借鉴点**：principled 融合替 KDE。

**最优改进建议**：**SVGP-DKL + LBANP 融合头**。CNN/Transformer 把延迟嵌入映射到 kernel 空间，SVGP 共享 inducing points 处理长 h，LBANP 在 s 通道做 principled posterior mixing。预计 EEG 上显存从 8GB → 1GB，h 从 256 → 1000+。

---

## 方向 B：扩散模型直接做预测（而非只做补值）

**一句话**：把扩散从"插值器"推进到"预测器+UQ"。

- **TMDM** (ICLR 2024)：Transformer 抽历史先验注入扩散。**借鉴点**：CSDI 同一个 score net 同时输出"补值+未来"。
- **MG-TSD** (ICLR 2024)：多粒度真值做 guide，6 个 benchmark 提升 4.7–35.8%。**借鉴点**：延迟嵌入当"粗粒度"、CSDI 补值当"细粒度"。
- **Diffusion-TS** (ICLR 2024)：sample prediction + Fourier loss。**借鉴点**：对 Lorenz/EEG 强周期信号合适。
- **TimeDiT** (KDD 2025)：DiT 架构时序 foundation model。

**最优改进建议**：MG-TSD 式**单一扩散**替代两阶段，同时吃下补值 + 预测 + UQ。适合直接投 ICLR/NeurIPS probabilistic forecasting track。

---

## 方向 C：可学习的延迟嵌入 / Takens embedding

**一句话**：把"纯随机 (d,τ)"升级为可微可学的延迟组合。

- **Universal Delay Embedding (UDE)** ([arXiv:2509.12080](https://arxiv.org/abs/2509.12080))：Hankel patch + self-attention + Koopman 线性演化，MSE 比 SOTA foundation model 低 20%+。**借鉴点**：直接替掉 RDE 随机采样。
- **Delay Embedding Theory of Neural Sequence Models** ([arXiv:2406.11993](https://arxiv.org/html/2406.11993v1))：证明 SSM/Transformer 隐藏态就是 delay embedding。**借鉴点**：加 Mamba branch 学"隐式延迟"。
- **Attraos** (NeurIPS 2024, [arXiv:2402.11463](https://arxiv.org/abs/2402.11463))：attractor memory + MDMU SSM，参数量 1/12 于 PatchTST 而 SOTA on Lorenz96。**注意这是威胁也是借鉴对象**。

**最优改进建议**：RDE 改写为 **"可微 Hankel patch + Gumbel-softmax lag selection"**，end-to-end 训练。回应"随机采样不 principled"最直接的审稿人质疑。

---

## 方向 D：Conformal Prediction / 现代 UQ

**一句话**：把 KDE 启发 + PICP@2σ 升级为 distribution-free 有覆盖保证。

- **CT-SSF** (NeurIPS 2024)：latent semantic 空间做 weighted non-conformity。**借鉴点**：s 个延迟嵌入作为 semantic feature，替掉 ±2σ。
- **Relational CP** (ICML 2025)：graph + quantile 显式处理相关时序。**借鉴点**：PM2.5 全 36 站 graph。
- **CPTC** (NeurIPS 2025, [arXiv:2509.02844](https://arxiv.org/abs/2509.02844))：online CP + change-point detector。**借鉴点**：EEG/Lorenz 突变场景。

**最优改进建议**：s 个 GPR 的 mean/var 作为 base，外层套 **CT-SSF + CPTC online 更新** → marginally valid 90/95% coverage。

---

## 方向 E：联合训练 / 端到端

- **TSDE** (KDD 2024)：扩散作为 representation extractor 支撑下游 6 项任务。
- **TSDiff** (NeurIPS 2023)：unconditional pretrain + inference-time self-guidance，**零额外训练**贯通两阶段。

**最优改进建议**：最干净路线是 **TSDiff 式 inference-time guidance**——CSDI 保持已训好，推理步骤注入 "下游 GPR 预测损失的梯度" 作为 guidance，瞬间贯通且不破坏 pretrain。

---

## 方向 F：物理约束（差异化竞争）

- **Attraos** (NeurIPS 2024)：Lorenz96 SOTA，**对我们是威胁**。
- **Dissipativity-Informed Learning** (NeurIPS 2024 ML4PS)：约束 trajectory 收敛到 invariant set。
- **PINN + Invariant Measure Score Matching** (2024–2025)：score matching 对 invariant measure，和 CSDI 完美契合。

**最优改进建议**：CSDI 的 score-matching 目标加一项 **invariant measure matching loss**，走"物理 + 概率"差异化，避开与 Attraos 在确定性 RMSE 硬拼，换成 **"小样本 + 覆盖率 + 长期稳定"三元优势**做 selling point。

---

## 投稿策略：3 条可组合路径

### 路径 1（推荐）：SVGP-DKL + Conformal → NeurIPS / ICLR main
- **组合**：A（LBANP/SVGP）+ D（CT-SSF）+ C 轻量（Gumbel 延迟）
- **卖点**：同时解决 O(n³) + principled UQ + 可学延迟
- **周期**：3–4 个月实现 + 1 个月实验
- **目标**：**NeurIPS 2026 / ICLR 2027 Bayesian + Time-series track**

### 路径 2（高新颖度）：端到端扩散 + 物理 → ICLR / TPAMI
- **组合**：B（MG-TSD 端到端）+ E（TSDiff guidance）+ F（invariant measure）
- **卖点**：CSDI 从补值器 → 混沌系统 physics-aware 扩散预测器
- **周期**：6 个月
- **目标**：**TPAMI / ICLR long paper**

### 路径 3（差异化应用）：延迟嵌入 Koopman + CP → KDD / AAAI
- **组合**：C（UDE + Koopman）+ D（Relational CP for PM2.5 graph）
- **卖点**：解决 PM2.5 36 站 + EEG 小样本两个痛点 + Koopman 可解释
- **目标**：**KDD / AAAI / TKDE**

---

## 推荐决策

**如果时间紧（2026 NeurIPS 五月 deadline）** → 走路径 1，重点实现 SVGP + Conformal。
**如果想冲 best-paper-potential** → 路径 2，但实验量极大。
**不建议**：在未做路径 1 的情况下直接上路径 2。

**最小可行第一步**：把 `GaussianProcessRegressor` 换成 GPyTorch 的 `SVGP`，在 EEG h=976 上跑通 baseline，这一步本身就能破当前 O(n³) 瓶颈。
