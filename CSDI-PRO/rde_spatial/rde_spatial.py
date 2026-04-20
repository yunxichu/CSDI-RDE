# rde_spatial.py - Random Dimension Embedding (RDE, 空间版)
#
# 从 D 维特征中随机采样 L 维子集组合 → 每组合独立 GPR 预测 → KDE 融合。
# 不含时间延迟（延迟版见 rde_delay/rde_module.py）。
#
# 源自 rde_gpr/pm25_test_comb.py:71-147，提取为独立模块以便复用与替换。

import numpy as np
import multiprocessing as mp
from functools import partial
from itertools import combinations
from scipy.stats import gaussian_kde
from sklearn.preprocessing import StandardScaler

from gpr.gpr_module import GaussianProcessRegressor


def rde_predict(traindata, target_idx, L=4, s=50, steps_ahead=1, n_jobs=4):
    """RDE 预测 - 空间维度组合嵌入

    Parameters
    ----------
    traindata : (T, D) array
        补值后的训练段。
    target_idx : int
        预测目标维度。
    L : int
        每个组合使用的维度数。
    s : int
        随机采样的组合数（集成规模）。
    steps_ahead : int
        预测步长。
    n_jobs : int
        multiprocessing 进程数。

    Returns
    -------
    pred : float
        KDE 加权均值点估计。
    std : float
        s 个 GP 均值的分散度（集成方差）。
    """
    trainlength = len(traindata)
    if trainlength - steps_ahead <= L:
        return np.nan, np.nan

    X = traindata[:trainlength - steps_ahead, :]
    y = traindata[steps_ahead:, target_idx]
    x_test = traindata[trainlength - steps_ahead, :].reshape(1, -1)

    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        return np.nan, np.nan

    combs = list(combinations(range(X.shape[1]), L))
    np.random.shuffle(combs)
    selected = combs[:min(s, len(combs))]

    pool = mp.Pool(processes=n_jobs)
    results = pool.map(
        partial(_rde_single_comb, X=X, y=y, x_test=x_test),
        selected
    )
    pool.close()
    pool.join()

    preds = np.array([r[0] for r in results])
    stds = np.array([r[1] for r in results])
    valid = ~np.isnan(preds) & ~np.isnan(stds)
    vp, vs = preds[valid], stds[valid]

    if len(vp) == 0:
        return np.nan, np.nan

    try:
        kde = gaussian_kde(vp)
        xi = np.linspace(vp.min(), vp.max(), 1000)
        density = kde(xi)
        pred = np.sum(xi * density) / np.sum(density)
    except Exception:
        pred = np.mean(vp)

    return pred, np.std(vp)


def _rde_single_comb(comb, X, y, x_test):
    """单个 RDE 组合的 GPR 训练 + 预测（multiprocessing worker）。"""
    try:
        X_c = X[:, list(comb)]
        x_test_c = x_test[:, list(comb)]

        sx = StandardScaler()
        sy = StandardScaler()

        X_all = np.vstack([X_c, x_test_c])
        X_all_s = sx.fit_transform(X_all)
        X_s = X_all_s[:-1]
        x_test_s = X_all_s[-1:]
        y_s = sy.fit_transform(y.reshape(-1, 1)).flatten()

        if np.std(y_s) < 1e-8:
            return np.nan, np.nan

        gp = GaussianProcessRegressor(noise=1e-6)
        gp.fit(X_s, y_s, init_params=(1.0, 0.1, 0.1), optimize=True)
        pred_s, std_s = gp.predict(x_test_s, return_std=True)

        pred = sy.inverse_transform(pred_s.reshape(-1, 1))[0, 0]
        return pred, std_s[0]
    except Exception:
        return np.nan, np.nan
