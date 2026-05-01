"""混沌系统专用评估指标。

  vpt()   Valid Prediction Time（有效预测时间）：预测误差超过阈值的时刻，
          以 Lyapunov 时间为单位。论文主指标，threshold=0.3 为 main figure。
  nrmse() 吸引子归一化 RMSE：用 attractor_std 归一化，跨系统可比。

与 experiments/week1/lorenz63_utils.py 中的 valid_prediction_time() 一致，
此处独立实现供 metrics/ 模块统一调用（lorenz96、KS 等也用此版本）。
"""
from __future__ import annotations

import numpy as np


def vpt(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dt: float,
    lyap: float,
    threshold: float = 0.3,
    attractor_std: float | None = None,
) -> float:
    """Valid Prediction Time in Lyapunov units.

    y_true, y_pred: (T, D). attractor_std: scalar normaliser; if None, uses std
    of y_true. Error: ``||y - y_hat|| / (sqrt(D) * attractor_std)``; exceeds
    threshold → failure.
    """
    T, D = y_true.shape
    if attractor_std is None:
        attractor_std = float(y_true.std())
    err = np.linalg.norm(y_true - y_pred, axis=1) / (np.sqrt(D) * attractor_std)
    bad = np.where(err > threshold)[0]
    t_fail = int(bad[0]) if bad.size else T
    return t_fail * dt * lyap


def nrmse(y_true: np.ndarray, y_pred: np.ndarray, attractor_std: float | None = None) -> float:
    if attractor_std is None:
        attractor_std = float(y_true.std())
    return float(np.sqrt(((y_true - y_pred) ** 2).mean()) / attractor_std)
