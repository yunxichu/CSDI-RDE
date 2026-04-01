# PM2.5预测: CSDI补值 + RDE/RDE-Delay预测

本目录实现PM2.5数据的完整预测流程。

## 目录结构

```
rde_gpr/
├── csdi/                    # CSDI模型相关
│   ├── main_model.py       # CSDI_PM25模型定义
│   ├── diff_models.py      # 扩散模型组件
│   ├── utils.py           # 训练和评估工具
│   └── save/              # 训练好的CSDI模型
│
├── pm25_complete_workflow.py    # 一体化流程脚本（补值+预测+可视化）
├── pm25_test_comb.py            # 一体化流程脚本（旧版）
├── pm25_CSDIimpute_after-RDEgpr.py  # 独立预测脚本
├── results/                       # 输出结果目录
└── README.md                      # 本说明文件

baselines/
└── no_impute_baseline/           # 基线方法（不补值）
    ├── pm25_neuralcde_noimpute.py     # NeuralCDE基线
    └── pm25_gruodebayes_noimpute.py   # GRU-ODE-Bayes基线
```

## 完整流程

### Step 1: 训练CSDI模型

```bash
cd /home/rhl/Github
python experiments/exe_pm25.py --device cuda:0
```

### Step 2: 补值 + RDE/RDE-Delay预测 + 可视化

```bash
python rde_gpr/pm25_complete_workflow.py \
    --device cuda:0 \
    --run_folder pm25_validationindex0_XXXXXXXXX
```

### Step 3: 基线方法对比（不补值）

#### NeuralCDE
```bash
python baselines/no_impute_baseline/pm25_neuralcde_noimpute.py \
    --missing_path ./data/pm25/Code/STMVL/SampleData/pm25_missing.txt \
    --ground_path ./data/pm25/Code/STMVL/SampleData/pm25_ground.txt \
    --split_ratio 0.5 --horizon_days 1 \
    --device cuda
```

#### GRU-ODE-Bayes
```bash
python baselines/no_impute_baseline/pm25_gruodebayes_noimpute.py \
    --missing_path ./data/pm25/Code/STMVL/SampleData/pm25_missing.txt \
    --ground_path ./data/pm25/Code/STMVL/SampleData/pm25_ground.txt \
    --split_ratio 0.5 --horizon_days 1 \
    --device cuda
```

## 方法说明

### 1. CSDI补值 + RDE/RDE-Delay（本文方法）

| 组件 | 方法 | 说明 |
|------|------|------|
| 补值 | CSDI | 条件分数扩散模型 |
| 预测 | RDE | 随机维度集成 |
| 预测 | RDE-Delay | 随机延迟嵌入 |

**特点**：先补值后预测，利用补值后的完整数据进行预测。

### 2. NeuralCDE（基线）

神经控制微分方程，直接处理缺失数据。

**特点**：
- 无需补值，直接处理不规则时间序列
- 利用神经ODE建模连续时间动态
- 适合有缺失的真实数据

### 3. GRU-ODE-Bayes（基线）

GRU + 常微分方程 + 贝叶斯方法。

**特点**：
- 同时建模确定性和随机动态
- 自然处理缺失数据
- 概率预测输出

## 参数说明

### pm25_complete_workflow.py

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--run_folder` | 必填 | 模型文件夹名 |
| `--device` | `cuda:0` | 运行设备 |
| `--nsample` | 100 | CSDI补值采样次数 |
| `--trainlength` | 4000 | RDE训练序列长度 |
| `--L` | 4 | RDE嵌入维度 |
| `--s` | 50 | RDE随机组合数 |
| `--max_delay` | 50 | RDE-Delay最大延迟 |
| `--M` | 4 | RDE-Delay嵌入维度 |
| `--num_samples` | 100 | RDE-Delay采样数 |
| `--target_dim` | 0 | 可视化目标维度 |
| `--n_jobs` | 8 | 并行进程数 |

### NeuralCDE / GRU-ODE-Bayes

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--missing_path` | 必填 | 缺失数据路径 |
| `--ground_path` | 必填 | 真实数据路径 |
| `--split_ratio` | 0.5 | 训练/测试分割 |
| `--horizon_days` | 1 | 预测天数 |
| `--device` | `cuda` | 运行设备 |
| `--epochs` | 10 | 训练轮数 |

## 输出说明

### 主流程输出

```
rde_gpr/results/pm25_YYYYMMDD_HHMMSS/
├── pm25_imputation_quality_*.png    # 补值质量可视化
├── pm25_prediction_comparison_*.png   # 预测对比可视化
├── imputed_data.csv                  # 补值结果
├── predictions_rde.csv               # RDE预测结果
├── predictions_rde_delay.csv        # RDE-Delay预测结果
├── stds_rde.csv                     # RDE不确定性
├── stds_rde_delay.csv              # RDE-Delay不确定性
└── Summary.json                    # 结果摘要
```

### 基线方法输出

基线方法输出包含预测值文件和评估指标。

## 数据说明

PM2.5数据集：
- **时间序列长度**: 约24000小时（10个月）
- **空间维度**: 36个监测站点
- **数据来源**: `/home/rhl/Github/data/pm25/Code/STMVL/SampleData/`

## 注意事项

1. **模型训练**: 首次运行需要先训练CSDI模型
2. **硬件要求**: CSDI模型和基线方法建议使用GPU加速
3. **数据准备**: 确保数据文件在正确位置
4. **对比公平性**: 基线方法不经过补值，直接预测

---
*更新时间: 2026-03-20*