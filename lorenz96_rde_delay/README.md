# Lorenz96-RDE-Delay

Lorenz96系统的CSDI补值 + RDE/RDE-Delay预测实验。

## 目录结构

```
lorenz96_rde_delay/
├── config/
│   └── lorenz96.yaml           # 配置文件
├── data/
│   └── dataset_lorenz96.py     # Lorenz96数据生成
├── models/
│   ├── main_model.py           # CSDI模型
│   ├── diff_models.py          # 扩散模型组件
│   ├── gpr_module.py          # GPR模块
│   └── rde_module.py          # RDE-Delay模块
├── training/
│   ├── exe_lorenz96.py        # 训练脚本
│   ├── utils.py               # 训练工具
│   └── save/                  # 保存的模型
├── inference/
│   └── test_comb_rde.py       # 完整流程测试
└── results/                    # 实验结果
```

## 快速开始

### Step 1: 训练CSDI模型

```bash
cd /home/rhl/Github/lorenz96_rde_delay/training
python exe_lorenz96.py --device cuda:0
```

模型会保存到 `training/save/lorenz96_YYYYMMDD_HHMMSS/`

### Step 2: 运行完整流程

```bash
cd /home/rhl/Github/lorenz96_rde_delay/inference
python test_comb_rde.py
```

## Lorenz96系统

Lorenz96是一个用于研究大气对流的高维混沌系统：

- **维度**: N=100个状态变量
- **方程**: dx_j/dt = (x_{j+1} - x_{j-2}) * x_{j-1} - x_j + F
- **Forcing**: F=8

## 数据流程

1. **生成数据**: 400步完整时间序列
2. **稀疏采样**: 每8步采样 → 50个数据点
3. **CSDI补值**: 50点 → 100点
4. **RDE预测**: 稀疏预测 + 补值后预测

## 输出

- 补值质量可视化
- 预测结果对比图
- RMSE、不确定性覆盖率等指标