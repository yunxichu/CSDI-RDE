# CSDI-RDE 论文相关材料

本目录包含 CSDI-RDE 方法的论文和相关实验材料。

## 目录结构

```
paper_CSDI_RDE/
├── README.md                          # 本说明文件
├── paper_CSDI_RDE.md                  # 论文主文档（Markdown格式）
├── experiments/                        # 实验结果目录
│   ├── pm25/                          # PM2.5数据集实验结果
│   └── lorenz/                        # Lorenz系统实验结果
└── figures/                           # 论文图表目录
```

## 论文概要

**标题**：CSDI-RDE: 融合条件分数扩散模型与随机分布嵌入的高维时空稀疏数据预测框架

**核心思想**：
- 使用条件分数扩散模型（CSDI）对高维时空稀疏数据进行补值
- 使用随机分布嵌入（RDE）结合高斯过程回归（GPR）进行预测
- 基于拓扑学理论，利用低维吸引子嵌入高维空间的性质
- 将传统单点预测转化为概率分布估计，提供不确定性量化

## 方法概述

### CSDI-RDE 框架

```
原始稀疏数据 → CSDI补值 → 完整数据 → RDE-GPR预测 → 预测结果+不确定性
```

### 关键特点

1. **数据补值**：CSDI生成高质量补值样本
2. **随机嵌入**：随机选择L个变量组合，降低维度
3. **GPR预测**：对每个嵌入训练GPR模型
4. **集成学习**：集成多个模型的预测结果
5. **不确定性量化**：提供预测的置信区间

## 实验结果

### PM2.5 数据集

| 方法 | RMSE | MAE |
|------|------|-----|
| GRU-ODE-Bayes | 13.8425 | 9.2047 |
| **RDE-GPR (Ours)** | **13.8856** | **9.2692** |
| NeuralCDE | 15.8941 | 11.8540 |

### 关键发现

- CSDI补值显著提升预测性能（RMSE提升约44%）
- RDE-GPR性能与深度学习方法接近，但计算更高效
- 方法能够提供可靠的不确定性估计

## 文件说明

### 论文文件
- `paper_CSDI_RDE.md` - 完整论文（Markdown格式，包含LaTeX公式）

### 实验结果
- 详细实验结果请参考 `experiments/` 目录

### 图表
- 论文图表请参考 `figures/` 目录

## 引用

如果您使用了本方法，请引用：

```
[待补充]
```

## 参考文献

[1] Tashiro, Y., Song, J., Song, Y., & Ermon, S. (2021). CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation. NeurIPS 2021.

[2] Wang, Y., Smola, A., & Maddox, T. (2021). Random Distribution Embeddings for Time Series Forecasting. ICML 2021.

[3] Rubanova, Y., Chen, R. T., & Duvenaud, D. (2019). Latent Ordinary Differential Equations for Irregularly-sampled Time Series. NeurIPS 2019.

[4] Kidger, P., et al. (2020). Neural Controlled Differential Equations for Irregular Time Series. NeurIPS 2020.

[5] Takens, F. (1981). Detecting strange attractors in turbulence. Dynamical Systems and Turbulence.

## 联系方式

如有问题，请联系：[待补充]
