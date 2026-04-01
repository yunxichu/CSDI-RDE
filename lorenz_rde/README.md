# Randomly Distributed Embedding (RDE) 实现

本目录实现了Randomly Distributed Embedding (RDE)方法，用于时间序列预测。

**注意**：本目录使用Randomly Distributed Embedding方法，即**没有时间延迟**，从空间维度随机选择特征组合。

## 目录结构

```
lorenz_rde/
├── models/               # 模型模块
│   ├── main_model.py     # CSDI模型
│   ├── gpr_module.py     # GPR模块
│   └── diff_models.py    # 扩散模型
├── data/                 # 数据模块
│   └── dataset_lorenz.py  # Lorenz系统数据生成
├── training/             # 训练脚本
│   ├── exe_lorenz.py     # CSDI模型训练
│   └── utils.py          # 训练工具
├── inference/            # 推理脚本
│   ├── test.py           # CSDI补值
│   ├── test2.py          # RDE-GPR预测
│   ├── test_comb.py      # 组合流程（推荐）
│   └── lstm_module.py    # LSTM基线
├── config/               # 配置文件
├── save/                 # 模型保存
└── README.md             # 本说明文件
```

## 两种方法的区别

| 方法 | 目录 | 嵌入方式 | 说明 |
|------|------|----------|------|
| **Randomly Distributed Embedding (RDE)** | `lorenz_rde/` | 从空间维度随机选择特征 | **没有时间延迟** |
| **Randomly Delay Embedding (RDE-Delay)** | `lorenz_rde_delay/` | 从时间维度随机选择延迟 | 有时间延迟 |

## RDE方法原理

### 核心思想
RDE方法通过**随机选择空间维度特征组合**来构建嵌入向量：

1. **随机特征组合**：从D维特征中随机选择L个特征的组合
2. **构建嵌入向量**：使用同一时间点t的不同特征组合：`[x_1(t), x_3(t), x_7(t), ...]`
3. **集成学习**：采样S个不同的特征组合，训练S个GPR模型
4. **聚合预测**：通过核密度估计合并多个GPR的预测

### 具体实现

```python
# 对于每个随机采样的特征组合
comb = [d_1, d_2, ..., d_L]  # L个随机选择的特征索引

# 构建嵌入向量（同一时间点）
x(t) = [x_{d_1}(t), x_{d_2}(t), ..., x_{d_L}(t)]

# 使用GPR建模
y(t+1) = f(x(t)) + ε

# 集成S个GPR模型的预测
y_pred = KDE([f_s(x(t)) for s in 1..S])
```

## 快速开始

### 运行完整流程（包含RDE-GPR和LSTM）

```bash
cd /home/rhl/Github/lorenz_rde/inference
python test_comb.py
```

### 单独运行RDE-GPR预测

```bash
cd /home/rhl/Github/lorenz_rde/inference
python test2.py
```

## 方法说明

### CSDI (Conditional Score-based Diffusion Models)
基于条件分数的扩散模型，用于时序数据填补。

### RDE-GPR (Randomly Distributed Embedding + Gaussian Process)
随机分布嵌入集成高斯过程回归：
- **随机嵌入**：从D维空间特征中随机选择L维组合
- **集成预测**：采样S个不同组合
- **GPR建模**：每个组合用高斯过程回归
- **KDE融合**：用核密度估计合并多个预测结果

### LSTM基线
标准LSTM深度学习方法，用于对比。

## 参数说明

### 预测参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `trainlength` | 30 | 训练序列长度 |
| `L` | 4 | 嵌入维度（特征组合） |
| `s` | 100 | 随机组合数量 |
| `j` | 0 | 目标变量索引 |
| `n_jobs` | 4 | 并行进程数 |
| `steps_ahead` | 1 | 预测步长 |

### LSTM参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `hidden_size` | 64 | LSTM隐藏层大小 |
| `num_layers` | 2 | LSTM层数 |
| `epochs` | 50 | 训练轮数 |
| `batch_size` | 8 | 批处理大小 |
