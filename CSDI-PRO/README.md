# CSDI-PRO

基于 **CSDI-RDE-GPR** 原框架的方法创新工作空间。本目录是独立于原 `csdi/` / `rde_gpr/` / `lorenz_rde_delay/` 的干净起点，用于在此迭代出可投 **NeurIPS / ICLR / ICML / TPAMI** 的方法改进。

## 目录结构

```
CSDI-PRO/
├── csdi/              # CSDI 补值核心（Transformer score network + diffusion）
│   ├── main_model.py        # CSDI_base / CSDI_Forecasting 等主模型
│   ├── diff_models.py       # diff_CSDI score network
│   └── utils.py             # 训练/评估 utils
├── rde_spatial/       # RDE 空间维度集成（从 D 维随机选 L 维组合）
│   └── rde_spatial.py       # rde_predict + _rde_single_comb
├── rde_delay/         # RDE-Delay 延迟嵌入（(dim, τ) 无放回随机采样 + KDE 聚合）
│   └── rde_module.py        # RandomlyDelayEmbedding class
├── gpr/               # 自实现 GPR（RBF + Cholesky + L-BFGS-B 超参优化）
│   └── gpr_module.py        # GaussianProcessRegressor
├── experiments/       # 后续创新实验脚本放这里
└── data/              # 数据软链或本地数据（大文件被 .gitignore 跳过）
```

## 模块来源（便于回溯）

| 本目录文件 | 原始路径 |
|---|---|
| `csdi/main_model.py` | `../csdi/main_model.py` |
| `csdi/diff_models.py` | `../csdi/diff_models.py` |
| `csdi/utils.py` | `../csdi/utils.py` |
| `gpr/gpr_module.py` | `../csdi/gpr_module.py` |
| `rde_delay/rde_module.py` | `../lorenz_rde_delay/models/rde_module.py` |
| `rde_spatial/rde_spatial.py` | 提取自 `../rde_gpr/pm25_test_comb.py:71-147` |

## 下一步创新方向（见 `session_notes/2026-04-21_innovation_directions_survey.md`）

**推荐路径 1（NeurIPS / ICLR main）**：
1. **`gpr/` → SVGP-DKL**：把自实现 GPR 换成 GPyTorch `SVGP` + Deep Kernel，破 O(n³) 瓶颈 → 支持 EEG h=976
2. **`rde_delay/` → Gumbel-Softmax 可微延迟选择**：把"随机采样 (d, τ)"升级为可学 one-hot
3. **新增 `conformal/` 模块**：CT-SSF 式加权 non-conformity，替掉 KDE + PICP@2σ 启发式

**备选路径 2（TPAMI / ICLR long paper）**：端到端扩散预测（MG-TSD 式）+ 物理约束（invariant measure score matching），把 CSDI 从"补值器"升级为"混沌系统 physics-aware 扩散预测器"。

## 使用注意

- CSDI 模块使用**同目录相对导入**（例如 `from diff_models import diff_CSDI`），请从 `CSDI-PRO/csdi/` 目录运行或加 `sys.path.insert(0, 'CSDI-PRO/csdi')`。
- `rde_spatial/rde_spatial.py` 依赖 `gpr.gpr_module`，需在项目根目录（`CSDI-PRO/`）下运行或设置 `PYTHONPATH`。
