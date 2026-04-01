# Physio数据集: CSDI补值 + RDE/RDE-Delay预测

本目录实现Physio数据集的完整预测流程。

## 目录结构

```
csdi/
├── main_model.py       # CSDI_Physio模型定义
├── diff_models.py      # 扩散模型组件
├── utils.py           # 训练和评估工具
├── save/             # 训练好的CSDI模型
│   └── physio_fold*_XXXXXXXX/   # 模型文件夹
└── results/          # 输出结果目录

baselines/
└── no_impute_baseline/           # 基线方法（不补值）
    ├── physio_neuralcde_noimpute.py     # NeuralCDE基线
    └── physio_gruodebayes_noimpute.py   # GRU-ODE-Bayes基线
```

## 完整流程

### Step 1: 训练CSDI模型

```bash
cd /home/rhl/Github
python experiments/exe_physio.py --device cuda:0
```

**参数说明**：
- `--device`: 运行设备（cuda:0 或 cpu）
- `--nfold`: 折数 [0-4]，用于5折交叉验证
- `--testmissingratio`: 测试集缺失比例（默认0.1）
- `--seed`: 随机种子

**输出**：
- 模型保存到: `csdi/save/physio_fold{nfold}_YYYYMMDD_HHMMSS/`

### Step 2: 补值 + RDE/RDE-Delay预测 + 可视化

```bash
python physio_complete_workflow.py \
    --device cuda:0 \
    --run_folder physio_fold0_XXXXXXXX
```

**参数说明**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--run_folder` | 必填 | 模型文件夹名 |
| `--device` | `cuda:0` | 运行设备 |
| `--nsample` | 100 | CSDI补值采样次数 |
| `--trainlength` | 40 | RDE训练序列长度 |
| `--L` | 4 | RDE嵌入维度 |
| `--s` | 50 | RDE随机组合数 |
| `--max_delay` | 10 | RDE-Delay最大延迟 |
| `--M` | 4 | RDE-Delay嵌入维度 |
| `--num_samples` | 100 | RDE-Delay采样数 |
| `--target_dim` | 0 | 可视化目标维度 |
| `--n_jobs` | 8 | 并行进程数 |

### Step 3: 基线方法对比（不补值）

#### NeuralCDE
```bash
python baselines/no_impute_baseline/physio_neuralcde_noimpute.py \
    --data_path ./data/physio/ \
    --device cuda
```

#### GRU-ODE-Bayes
```bash
python baselines/no_impute_baseline/physio_gruodebayes_noimpute.py \
    --data_path ./data/physio/ \
    --device cuda
```

## 数据说明

Physio数据集：
- **来源**: ICU患者生理数据 (PhysioNet Challenge 2012)
- **时间序列长度**: 48小时
- **特征维度**: 35个生理指标
- **特征列表**: HR, Temp, pH, Na, K, Glucose, 等
- **数据路径**: `/home/rhl/Github/data/physio/set-a/`

## 方法说明

### 1. CSDI补值 + RDE/RDE-Delay（本文方法）

| 组件 | 方法 | 说明 |
|------|------|------|
| 补值 | CSDI | 条件分数扩散模型 |
| 预测 | RDE | 随机维度集成 |
| 预测 | RDE-Delay | 随机延迟嵌入 |

**特点**：先补值后预测，利用补值后的完整数据进行预测。

### 2. NeuralCDE / GRU-ODE-Bayes（基线）

直接处理缺失数据，无需补值。

## 输出说明

```
csdi/results/physio_YYYYMMDD_HHMMSS/
├── physio_imputation_quality_*.png    # 补值质量可视化
├── physio_prediction_comparison_*.png   # 预测对比可视化
├── imputed_data.csv                   # 补值结果
├── predictions_rde.csv                # RDE预测结果
├── predictions_rde_delay.csv         # RDE-Delay预测结果
└── summary.json                     # 结果摘要
```

## 注意事项

1. **模型训练**: 首次运行需要先训练CSDI模型
2. **硬件要求**: CSDI模型建议使用GPU加速
3. **数据准备**: 确保数据文件在正确位置

---
*更新时间: 2026-03-20*