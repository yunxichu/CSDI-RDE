# CSDI-RDE-GPR 对比表 (完整数据)

生成时间: 2026-04-19 03:01

## 对比设计说明
- **Track-A 预处理对齐**: 所有方法（基线 + 我的）都用 **CSDI 补值后的数据** → 比较纯预测能力
- **Track-B 完整 pipeline**: 基线直接吃**原始稀疏/缺失数据** → 展示 CSDI+RDE-GPR 整套 pipeline 的价值
- RDE-GPR (ours) / RDE-Delay-GPR (ours) = CSDI-RDE-GPR 完整方法, 见 `/home/rhl/Github/README.md`

## Lorenz63

| Track | Method | RMSE | MAE | 说明 | 来源 |
|-------|--------|------|-----|------|------|
| Track-A | **NeuralCDE** | 6.048 | 4.224 | CSDI 补值输入 | `experiments_v2/lorenz63/neuralcde` |
| Track-A | **GRU-ODE-Bayes** | 5.969 | 4.029 | CSDI 补值输入 | `experiments_v2/lorenz63/gruodebayes` |
| Track-A | **SSSD_v1** | 18.801 | 15.205 | CSDI 补值输入 | `experiments_v2/lorenz63/sssd` |
| Track-A | **SSSD_v2** | 15.209 | 12.018 | CSDI 补值输入 | `experiments_v2/lorenz63/sssd_v2` |
| Track-A | **RDE-GPR (ours)** 🏆 | 0.573 ± 0.144 | — | CSDI 补值 → RDE-GPR (空间集成), 5 seeds 均值 | `experiments_v2/lorenz63/rde_delay/summary_mean.json (5 seeds)` |
| Track-A | **RDE-Delay-GPR (ours)** 🏆 | 1.403 ± 0.413 | — | CSDI 补值 → RDE-Delay-GPR (延迟嵌入), 5 seeds 均值 | `experiments_v2/lorenz63/rde_delay/summary_mean.json (5 seeds)` |
| Track-B | **NeuralCDE** | 6.887 | 5.230 | 基线直接吃 sparse_50, h=20 | `experiments_v2/lorenz63/neuralcde_sparse` |
| Track-B | **GRU-ODE-Bayes** | 2.692 | 2.086 | 基线直接吃 sparse_50, h=20 | `experiments_v2/lorenz63/gruodebayes_sparse` |

## Lorenz96

| Track | Method | RMSE | MAE | 说明 | 来源 |
|-------|--------|------|-----|------|------|
| Track-A | **NeuralCDE** | 9.939 | 7.159 | CSDI 补值输入 | `experiments_v2/lorenz96/neuralcde` |
| Track-A | **GRU-ODE-Bayes** | 4.105 | 3.257 | CSDI 补值输入 | `experiments_v2/lorenz96/gruodebayes` |
| Track-A | **SSSD_v1** | 5.592 | 4.432 | CSDI 补值输入 | `experiments_v2/lorenz96/sssd` |
| Track-A | **SSSD_v2** | 6.659 | 5.180 | CSDI 补值输入 | `experiments_v2/lorenz96/sssd_v2` |
| Track-A | **RDE-GPR (ours)** 🏆 | 0.284 ± 0.099 | — | CSDI 补值 → RDE-GPR (空间集成), 5 seeds 均值 | `experiments_v2/lorenz96/rde_delay/summary_mean.json (5 seeds)` |
| Track-A | **RDE-Delay-GPR (ours)** 🏆 | 0.265 ± 0.107 | — | CSDI 补值 → RDE-Delay-GPR (延迟嵌入), 5 seeds 均值 | `experiments_v2/lorenz96/rde_delay/summary_mean.json (5 seeds)` |
| Track-B | **NeuralCDE** | 3.533 | 2.631 | 基线直接吃 sparse_50, h=20 | `experiments_v2/lorenz96/neuralcde_sparse` |
| Track-B | **GRU-ODE-Bayes** | 1.140 | 0.609 | 基线直接吃 sparse_50, h=20 | `experiments_v2/lorenz96/gruodebayes_sparse` |

## PM2.5

| Track | Method | RMSE | MAE | 说明 | 来源 |
|-------|--------|------|-----|------|------|
| Track-A | **NeuralCDE** | 15.064 | 10.440 | CSDI 补值输入 | `experiments_v2/pm25/neuralcde` |
| Track-A | **GRU-ODE-Bayes** | 20.986 | 15.561 | CSDI 补值输入 | `experiments_v2/pm25/gruodebayes` |
| Track-A | **SSSD_v1** | 105.211 | 95.319 | CSDI 补值输入 | `experiments_v2/pm25/sssd` |
| Track-A | **RDE-GPR (ours)** 🏆 | 17.205 | 11.792 | CSDI 补值 → RDE-GPR, 全 36 站, trainlength=200 | `experiments_v2/pm25/rdegpr_modeB` |
| Track-B | **NeuralCDE** | 27.798 | 20.912 | 基线 + NaN mask (论文机制) | `experiments_v2/pm25/neuralcde_mask` |

## EEG

| Track | Method | RMSE | MAE | 说明 | 来源 |
|-------|--------|------|-----|------|------|
| Track-A | **NeuralCDE** | 17.042 | 12.273 | CSDI 补值输入 | `experiments_v2/eeg/neuralcde` |
| Track-A | **GRU-ODE-Bayes** | 6.236 | 5.188 | CSDI 补值输入 | `experiments_v2/eeg/gruodebayes` |
| Track-A | **SSSD_v1** | 87.573 | 73.149 | CSDI 补值输入 | `experiments_v2/eeg/sssd` |
| Track-A | **SSSD_v2** | 64.060 | 56.475 | CSDI 补值输入 | `experiments_v2/eeg/sssd_v2` |
| Track-A | **RDE-GPR (ours)** 🏆 | 61.471 | 53.770 | CSDI 补值 → RDE-GPR (空间版, 对照) | `experiments_v2/eeg/rdegpr_spatial_modeB` |
| Track-A | **RDE-Delay-GPR (ours)** 🏆 | 12.131 | 10.341 | CSDI 补值 → RDE-Delay-GPR | `experiments_v2/eeg/rde_delay_gpr_modeB_redo` |
| Track-B | **NeuralCDE** | 52.385 | 44.418 | 基线 + forward-fill 预处理 | `experiments_v2/eeg/neuralcde_naive` |
| Track-B | **GRU-ODE-Bayes** | 7.670 | 6.135 | 基线 + NaN mask (论文机制) | `experiments_v2/eeg/gruodebayes_mask` |

## 一页速览 Track-A (CSDI 补值, 所有方法, RMSE)

| Method | Lorenz63 | Lorenz96 | PM2.5 | EEG |
|--------|------|------|------|------|
| **NeuralCDE** | 6.048 | 9.939 | 15.064 | 17.042 |
| **GRU-ODE-Bayes** | 5.969 | 4.105 | 20.986 | 6.236 |
| **SSSD_v1** | 18.801 | 5.592 | 105.211 | 87.573 |
| **SSSD_v2** | 15.209 | 6.659 | — | 64.060 |
| **RDE-GPR (ours)** 🏆 | 0.573 | 0.284 | 17.205 | 61.471 |
| **RDE-Delay-GPR (ours)** 🏆 | 1.403 | 0.265 | — | 12.131 |

## 一页速览 Track-B (基线吃稀疏/缺失, RMSE)

| Method | Lorenz63 | Lorenz96 | PM2.5 | EEG |
|--------|------|------|------|------|
| **NeuralCDE** | 6.887 | 3.533 | 27.798 | 52.385 |
| **GRU-ODE-Bayes** | 2.692 | 1.140 | — | 7.670 |
