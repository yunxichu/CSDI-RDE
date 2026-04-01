# CSDI-RDE-GPR: Lorenz系统预测

本目录实现CSDI补值 + RDE/RDE-Delay预测的完整流程。

## 目录结构

```
lorenz_rde_delay/
├── config/               # 配置文件
│   └── lorenz.yaml      # CSDI模型配置
├── data/                 # 数据模块
│   └── dataset_lorenz.py # Lorenz系统数据生成
├── models/               # 模型模块
│   ├── main_model.py    # CSDI模型
│   ├── diff_models.py   # 扩散模型结构
│   ├── gpr_module.py    # GPR模块
│   └── rde_module.py    # RDE-Delay模块
├── training/             # 训练脚本
│   ├── exe_lorenz.py    # CSDI模型训练
│   └── utils.py         # 训练工具
├── inference/            # 推理脚本
│   ├── test.py          # CSDI补值测试
│   ├── test_comb_rde.py # 完整对比流程（推荐）
│   └── lstm_module.py   # LSTM基线
├── save/                 # 模型保存
│   ├── model.pth        # 训练好的模型权重
│   └── config.json      # 训练配置
├── results/              # 输出结果
└── README.md            # 本说明文件
```

## 核心功能

### 1. CSDI补值
将50点稀疏数据补值为100点（间隔采样）。

### 2. RDE（无时间延迟）
Randomly Distributed Embedding - 从空间维度随机选择特征组合。

### 3. RDE-Delay（有时间延迟）
Randomly Delay Embedding - 从时间维度随机选择延迟嵌入。

## 运行

### 完整对比流程
```bash
cd /home/rhl/Github/lorenz_rde_delay/inference
python test_comb_rde.py
```

### 单独测试CSDI补值
```bash
cd /home/rhl/Github/lorenz_rde_delay/inference
python test.py --device cpu --n_samples 10
```

## 数据流程

1. 原始完整数据: 400步 (t=0,1,...,399)
2. 稀疏采样: 每8步采样 → 50点 (t=0,8,16,...,392)
3. CSDI补值: 50点 → 100点（奇数位放已知，偶数位补值）
4. 预测: 稀疏50点预测20步，补值100点预测40步（取前20步对比）

## 结果

CSDI补值 RMSE ≈ 0.01
RDE（补值）效果优于RDE（稀疏）