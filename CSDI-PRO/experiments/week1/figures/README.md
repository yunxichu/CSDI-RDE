# week1/figures — 图片清单

> 主索引见仓库根目录 [PAPER_FIGURES.md](../../../PAPER_FIGURES.md)。
> 此处仅做本地快速索引，便于在这层目录下快速找图。

---

## Paper-ready 主图

| 文件 | 类型 | Paper 位置 | 生成脚本 |
|---|---|:-:|---|
| **`pt_v2_with_panda_n5_small_paperfig.png`** | Phase Transition 主图（3 panel） | **Fig 1** | [summarize_phase_transition.py](../summarize_phase_transition.py) |
| **`trajectory_overlay_seed3_S0_S2_S3_S5.png`** | 观测空间轨迹叠加（4×3 网格） | **Fig 2** | [plot_trajectory_overlay.py](../plot_trajectory_overlay.py) |
| **`separatrix_ensemble_seed4_S0_K30_ic05.png`** ⭐ | SVGP ensemble rollout 分岔点行为 | **Fig 3** | [plot_separatrix_ensemble.py](../plot_separatrix_ensemble.py) |

## Phase Transition 系列（历史版本 + 当前）

| 文件 | 内容 | 备注 |
|---|---|---|
| `pt_v2_with_panda_n5_small.png` | 自动生成的 3-panel 版本 | 同数据，简版，paperfig 是升级版 |
| `pt_v2_with_panda_n5_small_paperfig.png` ⭐ | 带 parrot phase-transition 标记线的 paper 版 | **Fig 1** |
| `pt_v2_with_ours_n5_small.png` / `..._paperfig.png` | 4 methods 版（没有 Panda，ef7f505 之前） | 历史备份 |
| `pt_v2_multibase_n5_small.png` | 3 methods 版（chronos/parrot/persist） | 最早期，caab1e6 之前 |
| `pt_v2_panda_smoke.png` | Panda 单独 smoke 测试 | 验证 Panda 接入 |
| `phase_transition_{base,small,large}_dt025.png` | v1 的 3 个 Chronos 模型对比 | W1 原始 pilot |
| `phase_transition_smoke*.png` | smoke test | 历史 |

## Separatrix Ensemble 系列（Figure 3 候选 + 辅图）

| 文件 | seed | scenario | K | IC perturb | 用途 |
|---|:-:|---|:-:|:-:|---|
| `separatrix_ensemble_seed4_S0_K30_ic05.png` ⭐ | 4 | S0 clean | 30 | 0.5 | **Paper Fig 3**（两次 lobe switch 全中，std spike 清晰） |
| `separatrix_ensemble_seed3_S0_K30_ic05.png` | 3 | S0 clean | 30 | 0.5 | 辅图：单 lobe switch，3/30 样本正确早 split |
| `separatrix_ensemble_seed3_S2_K30_ic05.png` | 3 | S2（0.4/0.3） | 30 | 0.5 | 辅图：中噪声场景 |
| `separatrix_ensemble_seed3_S3_K30_ic05.png` | 3 | S3（0.6/0.5） | 30 | 0.5 | 辅图：高噪声 ensemble collapse（局限 demo） |
| `separatrix_ensemble_seed3_sp00_n00_K20.png` | 3 | S0 | 20 | 0（process noise=1.0）| 早期有问题的 process-only 版本 |
| `separatrix_ensemble_seed3_S0_K20_ICpert.png` | 3 | S0 | 20 | 0.15 | 早期 IC 太小全一致版本 |

## Trajectory Overlay 系列

| 文件 | seed | scenarios | 内容 |
|---|:-:|---|---|
| `trajectory_overlay_seed3_S0_S2_S3_S5.png` ⭐ | 3 | S0 / S2 / S3 / S5 | **Paper Fig 2** |

---

## 快速复现主图

```bash
cd /home/rhl/Github/CSDI-PRO

# Figure 1: Phase Transition（~8 min with n_seeds=5）
CUDA_VISIBLE_DEVICES=0 python -u -m experiments.week1.phase_transition_pilot_v2 \
    --n_seeds 5 --tag with_panda_n5_small
python -m experiments.week1.summarize_phase_transition \
    --json experiments/week1/results/pt_v2_with_panda_n5_small.json

# Figure 2: Trajectory overlay（~3 min）
python -u -m experiments.week1.plot_trajectory_overlay \
    --seed 3 --scenarios S0 S2 S3 S5 --tag seed3_S0_S2_S3_S5

# Figure 3: Separatrix ensemble（~30s）
python -u -m experiments.week1.plot_separatrix_ensemble \
    --seed 4 --sparsity 0 --noise 0 --K 30 --tag seed4_S0_K30_ic05
```
