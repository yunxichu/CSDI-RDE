# 实验对比总结（Mode B 滚动 teacher-forcing 为主赛道）

生成时间: 2026-04-18 18:47

## 方法命名

- **RDE-GPR**: 随机嵌入 + 高斯过程回归（空间集成，同一时刻不同维度组合）
- **RDE-Delay-GPR**: 随机延迟嵌入 + 高斯过程回归（时间延迟特征）
- Lorenz63/96 的 eval_aligned.py 同时跑了上述两种方法；EEG 用延迟版；PM25 用空间版

## 对齐基线 Mode B 的主对比表

| 数据集 | NeuralCDE | GRU-ODE-Bayes | SSSD_v2 | RDE-GPR | RDE-Delay-GPR |
|--------|-----------|---------------|---------|---------|---------------|
| Lorenz63 | 6.05 | 5.97 | 15.21 | 0.57±0.14 | 1.40±0.41 |
| Lorenz96 | 9.94 | 4.10 | 6.66 | 0.28±0.10 | 0.26±0.11 |
| PM2.5 | 15.06 | 20.99 | — | 17.21 | — |
| EEG | 17.04 | 6.24 | 64.06 | 61.47 | 12.13 |

（数字是 RMSE；Lorenz63/96 是 dim 0 / 5 seeds 均值；PM2.5/EEG 是全 target 维度 overall）

## 对齐设置说明

- **Lorenz63/96**: trainlength=60, horizon=40, dim=0（5 seeds 均值）
- **PM2.5**: split_ratio=0.5, horizon=24, target=全 36 站, trainlength=500（RDE-GPR）/ horizon 长度窗 (基线)
- **EEG**: history=976, horizon=24, target=0,1,2, trainlength=300 (RDE-GPR)
- **预测模式**: 所有方法 **单步滚动 + teacher-forcing** — 每步预测 1 步，下一步窗口引入真值

## Mode A 前馈参考（不是主赛道）

| 实验 | RMSE | 备注 |
|------|------|------|
| EEG RDE-GPR | 91.06 | --multi_step --multi_step_mode direct |

## 运行中/待更新
- PM2.5 SSSD v2 Mode B 仍在 GPU 7 跑（~30% 进度）

完成后重新运行 `python experiments_v2/comparison_v3/build_comparison.py` 更新图表。