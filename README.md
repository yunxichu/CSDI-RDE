# CSDI-RDE-GPR: Conditional Score-based Diffusion Models with Random Delay/Dimension Ensemble and Gaussian Process Regression

时序预测项目，结合CSDI扩散模型与RDE/RDE-Delay方法进行时间序列填补和预测。本项目在多个数据集上进行了测试：Lorenz63系统、Lorenz96系统、PM2.5（真实数据）和Physio。

## 项目结构

```
CSDI-RDE-GPR/
├── lorenz_rde_delay/            # Lorenz63系统实现（含RDE和RDE-Delay）
│   ├── models/                 # 模型模块
│   │   ├── rde_module.py      # RDE-Delay模块
│   │   ├── gpr_module.py      # GPR模块
│   │   └── main_model.py      # CSDI模型
│   ├── data/                  # 数据模块
│   ├── training/              # 训练脚本
│   ├── inference/             # 推理脚本
│   ├── config/               # 配置文件
│   ├── save/                # 模型保存
│   └── experiment_summary.md # 实验总结
│
├── lorenz96_rde_delay/          # Lorenz96系统实现（高维混沌系统）
│   ├── models/                 # 模型模块
│   │   ├── rde_module.py      # RDE-Delay模块
│   │   ├── gpr_module.py      # GPR模块
│   │   └── main_model.py      # CSDI模型
│   ├── data/                  # 数据模块
│   │   └── dataset_lorenz96.py # Lorenz96数据生成
│   ├── training/              # 训练脚本
│   ├── inference/             # 推理脚本
│   ├── config/               # 配置文件
│   └── results/              # 实验结果
│
├── lorenz/                    # Lorenz63系统原始实现
│   ├── models/               # 模型模块
│   ├── data/                 # 数据模块
│   ├── training/             # 训练脚本
│   ├── inference/            # 推理脚本
│   ├── config/               # 配置文件
│   └── save/                 # 模型保存
│
├── rde_gpr/                   # RDE-GPR预测模块
│   ├── pm25_CSDIimpute_after-RDEgpr.py  # PM2.5预测
│   └── weather_CSDIimpute_after-RDEgpr.py  # Weather预测
│
├── datasets/                   # 数据集处理脚本
│   ├── dataset_weather.py     # Weather数据集
│   └── dataset_physio.py     # Physio数据集
│
├── csdi/                      # CSDI模型脚本
│   ├── weather_train.py      # Weather训练
│   └── main_model.py         # CSDI模型定义
│
├── imputation/                 # 补值脚本
│   ├── weather_generate_missing.py  # Weather缺失数据生成
│   ├── weather_CSDIimpute.py       # Weather补值
│   └── physio_CSDIimpute.py        # Physio补值
│
├── baselines/                 # 基线方法
│   ├── pm25_neuralcde_forecast.py    # NeuralCDE
│   └── pm25_gruodebayes_forecast.py  # GRU-ODE-Bayes
│
├── best_record/               # 实验结果
├── paper_CSDI_RDE/           # 论文和相关实验
├── config/                   # 配置文件
├── data/                    # 数据目录
├── save/                    # 保存目录
├── requirements.txt          # 依赖
└── README.md                # 本说明文件
```

## 数据集说明

### 1. Lorenz63系统（经典混沌系统）
- **类型**：耦合Lorenz63系统，混沌动力系统
- **特点**：确定性但高度非线性，适合测试预测方法的性能
- **数据生成**：`lorenz_rde_delay/data/dataset_lorenz.py`
- **完整流程**：数据生成 → CSDI补值 → RDE/RDE-Delay预测 → 结果分析
- **方法**：CSDI补值 + RDE/RDE-Delay预测

### 2. Lorenz96系统（高维混沌系统）
- **类型**：Lorenz96系统，高维混沌动力系统
- **特点**：N=100维，具有空间耦合特性，更接近真实大气系统
- **数据生成**：`lorenz96_rde_delay/data/dataset_lorenz96.py`
- **完整流程**：
  - 原始数据：400时间步，100维
  - 稀疏采样：每8步采样 → 50个稀疏点
  - CSDI补值：50点 → 100点（每4步一个点）
  - RDE/RDE-Delay预测
- **方法**：CSDI补值 + RDE/RDE-Delay预测
- **关键参数**：N=100, T=400, sample_step=8, forcing=8.0

### 3. PM2.5（真实数据）
- **类型**：环境监测数据，时空序列
- **特点**：真实世界数据，包含缺失值和噪声
- **数据来源**：`data/pm25/`
- **处理流程**：CSDI补值 → RDE-GPR预测
- **方法**：CSDI补值 + RDE-GPR预测

## 快速开始

### Lorenz96系统（推荐，高维）

```bash
# Step 1: 训练CSDI模型
cd lorenz96_rde_delay && python training/exe_lorenz96.py --config config/lorenz96.yaml --device cpu --epochs 200

# Step 2: 运行完整流程（CSDI补值 + RDE/RDE-Delay预测）
cd lorenz96_rde_delay && python inference/test_comb_rde.py
```

**关键说明**：
- 训练序列长度必须与推理一致（均为100个时间点）
- CSDI补值：50个稀疏点 → 100个密集点
- 预测：训练60步，预测40步

### Lorenz63系统（推荐）

```bash
# Step 1: 训练CSDI模型
cd lorenz_rde_delay/training && python exe_lorenz.py --device cuda --seq_len 100 --seq_count 1000 --batch_size 32 --Nnodes 5

# Step 2: 运行完整流程（CSDI补值 + RDE/RDE-Delay预测）
cd lorenz_rde_delay/inference && python test_comb_rde.py

# 单独测试CSDI补值
python test.py --device cpu
```

### Lorenz63系统（原始版本）

```bash
# 训练CSDI模型
cd lorenz/training && python exe_lorenz.py --device cpu

# 运行完整流程
cd lorenz/inference && python test_comb.py
```

### PM2.5预测

**两步流程**：

```bash
# Step 1: 训练CSDI模型（如果还没有）
cd /home/rhl/Github && python experiments/exe_pm25.py --device cuda:0

# Step 2: CSDI补值（历史段）
python imputation/pm25_CSDIimpute_3.py \
  --run_folder ./rde_gpr/csdi/save/pm25_validationindex0_20260321_151826 \
  --ground_path ./data/pm25/Code/STMVL/SampleData/pm25_ground.txt \
  --missing_path ./data/pm25/Code/STMVL/SampleData/pm25_missing.txt \
  --meanstd_path ./data/pm25/pm25_meanstd.pk \
  --split_ratio 0.5 \
  --device cuda:0 \
  --impute_n_samples 50

# Step 3: RDE-GPR预测（使用补值后的历史）
python rde_gpr/pm25_CSDIimpute_after-RDEgpr.py \
  --imputed_history_path ./save/pm25_history_imputed_split0.5_seed42_XXXXXXXX/history_imputed.csv \
  --ground_path ./data/pm25/Code/STMVL/SampleData/pm25_ground.txt \
  --split_ratio 0.5 \
  --horizon_days 1 \
  --L 4 --s 50 --trainlength 4000 --n_jobs 8 \
  --target_indices 0,1,2
```

**输出文件**：
- Step 2 输出：`history_imputed.csv`（补值后的历史数据）
- Step 3 输出：`future_pred.csv`（预测结果）、`metrics.json`（评估指标）

### Physio预测

**三步流程**：

```bash
# Step 1: 训练CSDI模型（如果还没有）
cd /home/rhl/Github && python experiments/exe_physio.py --device cuda:0 --nfold 0

# Step 2: CSDI补值（历史段）
python imputation/physio_CSDIimpute.py \
  --run_folder ./csdi/save/physio_fold0_20260324_202330 \
  --data_path ./data/physio/ \
  --missing_ratio 0.1 \
  --split_ratio 0.5 \
  --device cuda:0 \
  --seed 1 \
  --impute_n_samples 20

# Step 3: RDE-GPR预测（样本内预测：前36步预测后12步）
python rde_gpr/physio_CSDIimpute_after-RDEgpr.py \
  --imputed_history_path ./save/physio_history_imputed_split0.5_seed1_XXXXXXXX/history_imputed.npy \
  --missing_positions_path ./save/physio_history_imputed_split0.5_seed1_XXXXXXXX/history_missing_positions.csv \
  --history_timesteps 36 \
  --horizon_timesteps 12 \
  --use_ground_truth_sliding \
  --L 8 --s 100 --trainlength 36 --n_jobs 2 \
  --target_indices 0,1,2,3 \
  --max_samples 100
```

**输出文件**：
- Step 2 输出：
  - `history_imputed.npy`（补值后的历史数据）
  - `history_full.npy`（原始真实值）
  - `history_missing_positions.csv`（缺失位置记录）
- Step 3 输出：
  - `future_pred.npy`（预测结果）
  - `future_truth.npy`（真实值）
  - `metrics.json`（评估指标）
  - `plot_dim*_full.png`（可视化，包含补值位置标注）

**参数说明**：
- `--missing_ratio`: 缺失率（默认0.1）
- `--impute_n_samples`: CSDI采样次数（推荐20-50）
- `--history_timesteps`: 用于预测的历史时间步数（默认36）
- `--horizon_timesteps`: 预测的未来时间步数（默认12）
- `--use_ground_truth_sliding`: 使用真实值更新滑窗（推荐）
- `--L`: 随机嵌入维度（推荐8-12）
- `--s`: 采样组合数（推荐50-100）
- `--trainlength`: 训练序列长度（推荐20-36）
- `--n_jobs`: 并行进程数（建议2-4）

**滑窗更新模式**：
- **真值滑窗**（`--use_ground_truth_sliding`）：每预测一步，用真实值更新滑窗
  - 优点：误差不累积，评估模型本身能力
  - 缺点：不符合真实预测场景
- **预测值滑窗**（默认）：每预测一步，用预测值更新滑窗
  - 优点：符合真实预测场景
  - 缺点：误差会累积

**可视化说明**：
- 蓝色实线：真实值（完整48步）
- 红色虚线：预测值（只显示预测的12步）
- 红色阴影：95%置信区间
- 橙色阴影：缺失位置（补值位置）
- 绿色虚线：预测起始点（第36步）

**批量测试L值**：
```bash
# 测试L从8到15的效果
bash test_L_values.sh

# 查看结果
tail -f logs/test_L_values.log
```

### Weather预测

**四步完整流程**：

```bash
cd /home/rhl/Github

# Step 1: 生成缺失数据
python imputation/weather_generate_missing.py \
  --data_path ./data/weather/weather.npy \
  --missing_ratios 0.1 \
  --modes random \
  --seed 42

# Step 2: 训练CSDI模型
python csdi/weather_train.py \
  --device cuda:5 \
  --missing_ratio 0.1 \
  --missing_mode random \
  --seed 42 \
  --epochs 200

# Step 3: CSDI补值（历史段）
# 运行后会在 ./save/ 下生成 weather_history_imputed_random_ratio0.1_split0.5_seed42_XXXXXXXX 目录
python imputation/weather_CSDIimpute.py \
  --run_folder ./save/weather_random_ratio0.1_fold0_20260328_034411 \
  --missing_ratio 0.1 \
  --missing_mode random \
  --split_ratio 0.5 \
  --device cuda:5 \
  --seed 42 \
  --impute_n_samples 50

# Step 4: RDE-GPR预测
# 使用 Step 3 生成的补值目录（如 weather_history_imputed_random_ratio0.1_split0.5_seed42_20260328_155218）
python rde_gpr/weather_CSDIimpute_after-RDEgpr.py \
  --imputed_history_path ./save/weather_history_imputed_random_ratio0.1_split0.5_seed42_20260328_155218/history_imputed.npy \
  --missing_positions_path ./save/weather_history_imputed_random_ratio0.1_split0.5_seed42_20260328_155218/history_missing_positions.csv \
  --impute_meta_path ./save/weather_history_imputed_random_ratio0.1_split0.5_seed42_20260328_155218/impute_meta.json \
  --missing_ratio 0.1 \
  --missing_mode random \
  --horizon_steps 24 \
  --L 4 --s 50 --n_jobs 2 \
  --target_indices 0,1,2,3,4,5
```

**Weather数据集说明**：
- **数据形状**: (52696, 21) - 52696个时间步，21个特征
- **训练数据**: 70% = 36887个时间步
- **测试数据**: 15% = 7904个时间步
- **验证数据**: 15% = 7904个时间步

**缺失模式**：
| 模式 | 说明 |
|------|------|
| `uniform` | 均匀缺失：按固定间隔挖去数据 |
| `random` | 随机缺失：随机选择位置挖去 |

**参数说明**：
- `--impute_meta_path`: 补值脚本生成的impute_meta.json路径（用于对齐split_point）
- `--target_indices`: 要预测的特征索引（默认0-5）
- `--horizon_steps`: 预测的未来时间步数（默认24）
- `--history_timesteps`: 用于预测的历史时间步数（默认72）

**输出文件**：
- Step 1 输出：
  - `weather_ground.npy`（原始数据备份）
  - `weather_missing_{mode}_ratio{X}_seed{Y}.npy`（缺失数据）
  - `missing_patterns_overview.png`（缺失模式可视化）
- Step 3 输出：
  - `history_imputed.npy`（补值后的历史数据）
  - `history_full.npy`（原始真实值）
  - `history_missing_positions.csv`（缺失位置记录）
  - `impute_meta.json`（元数据，包含total_len和split_point）
  - `imputation_*.png`（补值可视化）
- Step 4 输出：
  - `future_pred.csv`（预测结果）
  - `metrics.json`（评估指标）
  - `prediction_detail_dim{X}.png`（预测详情）
  - `prediction_multi_dim.png`（多维度预测对比）
  - `prediction_rmse_per_dim.png`（各维度RMSE柱状图）
  - `prediction_error_heatmap.png`（误差热力图）

**快速测试命令**（减少epoch和采样次数）：
```bash
# 训练（减少epoch）
python csdi/weather_train.py --device cuda:5 --missing_ratio 0.1 --missing_mode random --seed 42 --epochs 50

# 补值（减少采样）
python imputation/weather_CSDIimpute.py --run_folder ./save/weather_random_ratio0.1_fold0_XXXXXXXX --missing_ratio 0.1 --missing_mode random --split_ratio 0.5 --device cuda:5 --seed 42 --impute_n_samples 10

# 预测（减少目标维度）
python rde_gpr/weather_CSDIimpute_after-RDEgpr.py --imputed_history_path ./save/weather_history_imputed_XXX/history_imputed.npy --impute_meta_path ./save/weather_history_imputed_XXX/impute_meta.json --missing_ratio 0.1 --missing_mode random --horizon_steps 24 --L 4 --s 50 --n_jobs 2 --target_indices 0,1,2
```

### Baseline方法对比

Weather数据集支持与以下Baseline方法对比：

**1. NeuralCDE**
```bash
cd /home/rhl/Github

python baselines/weather_neuralcde_forecast.py \
    --imputed_history_path ./save/weather_history_imputed_random_ratio0.1_split0.5_seed42_20260328_155218/history_imputed.npy \
    --impute_meta_path ./save/weather_history_imputed_random_ratio0.1_split0.5_seed42_20260328_155218/impute_meta.json \
    --ground_path ./data/weather/weather_ground.npy \
    --horizon_steps 24 --history_timesteps 72 \
    --hidden_channels 64 --num_layers 3 \
    --epochs 100 --batch_size 128 --lr 1e-3 --seed 42
```

**2. GRU-ODE-Bayes**
```bash
python baselines/weather_gruodebayes_forecast.py \
    --imputed_history_path ./save/weather_history_imputed_random_ratio0.1_split0.5_seed42_20260328_155218/history_imputed.npy \
    --impute_meta_path ./save/weather_history_imputed_random_ratio0.1_split0.5_seed42_20260328_155218/impute_meta.json \
    --ground_path ./data/weather/weather_ground.npy \
    --horizon_steps 24 --history_timesteps 72 \
    --hidden_size 64 --p_hidden 32 \
    --epochs 100 --batch_size 128 --lr 1e-3 --seed 42
```

**3. 综合对比可视化**
```bash
python visualization/weather_full_comparison.py \
    --ground_path ./data/weather/weather_ground.npy \
    --impute_meta_path ./save/weather_history_imputed_random_ratio0.1_split0.5_seed42_20260328_155218/impute_meta.json \
    --methods rdegpr,neuralcde,gruodebayes \
    --output_dir ./save/weather_comparison \
    --plot_dims 0,1,2,3,4,5
```

**输出文件**：
- `full_comparison.png`（综合对比图，包含轨迹对比、误差对比、散点图、RMSE柱状图）
- `comparison_metrics.json`（各方法评估指标）

## 方法说明

### CSDI (Conditional Score-based Diffusion Models)
基于条件分数的扩散模型，用于时序数据填补。通过学习数据的分数函数，逐步从噪声中恢复缺失的数据。

**Lorenz96中的CSDI应用**：
- 输入：50个稀疏时间点（每8步采样）
- 输出：100个密集时间点（每4步一个点）
- 训练策略：gt_mask标记奇数位置（索引99,97,...,1）作为已知条件
- 预测目标：偶数位置（索引98,96,...,0）需要补值
- 关键要点：训练序列长度必须与推理一致（均为100），否则后半部分补值质量会下降

### RDE (Random Dimension Ensemble)
随机维度集成高斯过程回归：
- **随机嵌入**：从D维特征中随机选择L个特征组合
- **同一时刻**：使用同一时间点t的不同维度特征 `[x_d1(t), x_d2(t), ..., x_dL(t)]`
- **集成预测**：采样s个不同组合
- **GPR建模**：每个组合用高斯过程回归
- **KDE融合**：用核密度估计合并多个预测结果

### RDE-Delay (Random Delay Embedding)
随机延迟嵌入高斯过程回归：
- **时间延迟嵌入**：从时间维度选择延迟τ，利用历史信息
- **特征向量**：`[x_d1(t-τ1), x_d2(t-τ2), ..., x_dM(t-τM)]`
- **自适应延迟上限**：τ_max = trainlength // (M + 1)
- **优点**：捕获时间依赖性，预测更准确

### 两种方法对比

| 方法 | 嵌入方式 | 适用场景 |
|------|----------|----------|
| RDE | 空间维度组合 | 高维耦合系统，无需时间延迟 |
| RDE-Delay | 时间延迟嵌入 | 需要捕获时间依赖性 |

### Lorenz96系统原理

**动力学方程**：
```
dx_i/dt = (x_{i+1} - x_{i-2}) * x_{i-1} - x_i + F
```

其中：
- i = 1, 2, ..., N（N=100维）
- F = 8.0（强迫参数）
- 边界条件：周期性（x_{N+1} = x_1, x_0 = x_N, x_{-1} = x_{N-1}）

**系统特性**：
1. **高维混沌**：N=100维，比经典Lorenz63系统（3维）复杂得多
2. **空间耦合**：相邻变量通过非线性项耦合
3. **能量级联**：能量从大尺度向小尺度传递
4. **可预测性**：混沌特性导致长期预测困难

**数值积分**：
- 时间步长：dt = 0.01
- 积分方法：Euler方法
- 初始条件：平衡态 + 小随机扰动
- Burn-in：1000步消除瞬态

**数据流程**：
```
原始数据(400步) → 稀疏采样(每8步) → 50点 → CSDI补值 → 100点 → RDE预测
```

## 实验结果

### Lorenz96系统

**实验设置**：
- N=100维，T=400时间步
- 稀疏采样：每8步采样 → 50个稀疏点
- CSDI补值：50点 → 100点
- 预测参数：trainlength=30/60, L=4, s=100

**实验结果**：

| 方法 | RMSE | MaxErr | 2σ Coverage(%) |
|------|------|--------|----------------|
| RDE（稀疏原始） | 1.1949 | 2.1785 | 100.0 |
| RDE-Delay（稀疏原始） | 0.8663 | 1.3295 | 100.0 |
| **RDE（补值后）** | **0.5233** | 1.0493 | 100.0 |
| **RDE-Delay（补值后）** | **0.3402** | 0.6831 | 100.0 |

**CSDI补值RMSE**: 0.1065

**关键发现**：
1. CSDI补值后，RDE预测RMSE从1.19降至0.52（提升56%）
2. CSDI补值后，RDE-Delay预测RMSE从0.87降至0.34（提升61%）
3. RDE-Delay在补值后表现最佳，RMSE仅为0.34
4. 所有方法的2σ覆盖率均达到100%

### Lorenz63系统（36组实验统计）

| 方法 | 平均RMSE | 标准差 |
|------|----------|--------|
| RDE（稀疏原始） | 2.10 | 1.21 |
| RDE-Delay（稀疏原始） | 3.78 | 1.19 |
| **RDE（补值后）** | **1.32** | 0.87 |
| RDE-Delay（补值后） | 1.43 | 0.66 |

**关键发现**：CSDI补值后预测效果提升37%-62%

### PM2.5

结果保存在`best_record/`目录，包含各种方法的性能对比

## 基线方法

| 方法 | 适用数据集 | 说明 |
|------|------------|------|
| LSTM | Lorenz63 | 深度学习基线方法 |
| NeuralCDE | PM2.5 | 神经控制微分方程 |
| GRU-ODE-Bayes | PM2.5 | GRU+ODE+贝叶斯方法 |

## EEG 预测对比实验

### 实验设置

- **数据集**：EEG 脑电信号（64通道，1000时间点）
- **任务**：单步滚动预测（预测未来24步）
- **目标维度**：前3个通道（dim 0, 1, 2）
- **历史长度**：100个时间点
- **缺失率**：50%随机缺失

### 对比方法

| 方法 | 类型 | 训练数据 | 特点 |
|------|------|----------|------|
| **RDE-GPR** | 非参数方法 | 即时学习 | 无需预训练，适应性强 |
| GRU | 参数方法 | 100点预训练 | 需要大量训练数据 |
| LSTM | 参数方法 | 100点预训练 | 需要大量训练数据 |
| NeuralCDE | 参数方法 | 100点预训练 | 神经控制微分方程 |
| GRU-ODE-Bayes | 参数方法 | 100点预训练 | GRU+ODE+贝叶斯 |

### 公平对比设置

为确保公平对比，所有方法采用相同设置：
- **训练数据**：ground truth 的 history 部分（100点）
- **预测模式**：单步滚动预测，每步用 ground truth 更新滑窗

### 实验结果

| 方法 | RMSE | MAE | 相对提升 |
|------|------|-----|----------|
| **RDE-GPR** | **7.53** | **6.23** | - |
| GRU-ODE-Bayes | 9.62 | 8.08 | +28% |
| LSTM | 9.98 | 8.28 | +33% |
| GRU | 9.84 | 8.28 | +31% |

### 关键发现

1. **RDE-GPR 效果最优**：RMSE 比最好的基线（GRU-ODE-Bayes）好 **22%**

2. **非参数方法优势**：
   - RDE-GPR 是非参数方法，不需要大量训练数据
   - 每次预测时即时从当前数据学习，适应性强
   - 深度学习方法（GRU/LSTM）只有100个训练样本，数据量不足

3. **Delay Embedding 有效**：
   - 能有效捕获时间序列动态
   - 不使用未来数据，无数据泄露

### 预测模式分析

| 模式 | 训练数据 | 测试输入 | 评估目标 |
|------|----------|----------|----------|
| 自回归 | imputed数据 | 预测值 | 真实应用场景 |
| 单步滚动 | ground truth | ground truth | 公平对比预测能力 |

**注意**：自回归模式下 GRU/LSTM 效果可能更好，但这不是公平对比：
- GRU/LSTM 用 imputed 数据训练，分布一致
- RDE-GPR 用 ground truth 即时学习，分布不一致
- 单步滚动模式才是公平对比

### 运行命令

```bash
# 公平对比（推荐）
python baselines/eeg_forecast_comparison.py \
  --imputed_path ./save/eeg_imputed_random_ratio0.5_seed42_20260331_131907/eeg_imputed.npy \
  --ground_path ./data/eeg/eeg_ground.npy \
  --horizon_steps 24 \
  --history_timesteps 100 \
  --target_dims "0,1,2" \
  --rdegpr_L 7 \
  --rdegpr_s 50 \
  --rdegpr_trainlength 100 \
  --rdegpr_max_delay 20 \
  --gru_epochs 100 \
  --n_jobs 2 \
  --device cpu \
  --use_teacher_forcing \
  --use_ground_truth_train
```

### 输出文件

- `eeg_forecast_comparison.png`：预测轨迹对比图
- `eeg_forecast_metrics.png`：RMSE/MAE柱状图
- `comparison_summary.csv`：各方法指标汇总
- `comparison_results.json`：完整结果（含预测值）

## 注意事项

1. **Lorenz96系统**：
   - 训练序列长度必须与推理一致（均为100个时间点）
   - 如果训练时序列长度为50，推理时用100，会导致后半部分补值质量严重下降
   - gt_mask标记策略：从末尾向前标记奇数位置（索引99,97,...,1）
   - 模型路径：`lorenz96_rde_delay/training/save/lorenz96_20260323_163834/`

2. **Lorenz63系统**：推荐使用`lorenz_rde_delay/`目录，包含最新的RDE-Delay方法

3. **PM2.5**：主要处理脚本为`rde_gpr/pm25_CSDIimpute_after-RDEgpr.py`

4. **模型训练**：首次运行需要训练CSDI模型

5. **硬件要求**：CSDI模型建议使用GPU加速

6. **配置文件**：位于`config/`目录，包含Lorenz和PM2.5的配置

## 依赖安装

```bash
pip install -r requirements.txt
```