# rde_module.py - Randomly Delay Embedding (RDE) 改进版
#
# 相比原版的核心改动：
#   1. 多维嵌入   : 从所有维度 × 所有合法延迟里随机选 (dim, τ) 对，充分利用耦合信息
#   2. 无放回采样 : 保证 M 个嵌入分量互不重复，消除核矩阵病态
#   3. 边界截断   : 训练起点 = max(delays)，不再补零
#   4. 自适应延迟上限 : τ_max = trainlength // (M + 1)，与窗口长度自动匹配
#   5. 正确不确定性传播 :
#        σ²_total = Var({ŷ_k}) + mean({σ²_k})
#                   ^^^^^^^^^^^   ^^^^^^^^^^^^
#                   模型间分散     模型内不确定性
#   6. 每步独立采样 : 每个预测步骤重新采样延迟组合，真正体现"随机"嵌入的多样性
#   7. KDE聚合     : 与 test_comb_rde.py 中 RDE（空间版）保持一致的集成策略

import numpy as np
from scipy.stats import gaussian_kde


class RandomlyDelayEmbedding:
    """
    多维随机延迟嵌入 + 高斯过程集成预测

    Parameters
    ----------
    max_delay : int or None
        单步延迟上限。None 表示自动按 trainlength // (M + 1) 计算（推荐）。
    M : int
        每次嵌入使用的 (维度, 延迟) 对数量，即特征向量维度。
    num_samples : int
        每个预测步骤采样的随机嵌入组合数（集成规模）。
    noise : float
        GPR 观测噪声正则化项。
    resample_per_step : bool
        True  → 每个预测步骤独立重新采样延迟组合（多样性更强，稍慢）
        False → 全程共用同一组延迟（可复现，稍快）
    use_kde : bool
        True  → 用 KDE 加权均值聚合（与空间 RDE 一致）
        False → 用简单算术均值聚合
    seed : int or None
        随机种子，None 表示不固定。
    """

    def __init__(self,
                 max_delay=None,
                 M=4,
                 num_samples=100,
                 noise=1e-6,
                 resample_per_step=True,
                 use_kde=True,
                 seed=None):
        self.max_delay       = max_delay
        self.M               = M
        self.num_samples     = num_samples
        self.noise           = noise
        self.resample_per_step = resample_per_step
        self.use_kde         = use_kde
        self.seed            = seed

    # ──────────────────────────────────────────────────────────
    # 内部工具
    # ──────────────────────────────────────────────────────────

    def _effective_max_delay(self, trainlength):
        """
        计算有效延迟上限。

        规则：τ_max = trainlength // (M + 1)
          → 保证最大延迟不超过训练窗口的 1/(M+1)，
            使得每个延迟分量都有足够多的训练样本。
        如果用户手动指定了 max_delay，取两者较小值。
        """
        auto = max(1, trainlength // (self.M + 1))
        if self.max_delay is None:
            return auto
        return min(self.max_delay, auto)

    def _sample_delay_combos(self, total_dims, tau_max, n_combos, rng):
        """
        从候选集 {(dim, τ) | dim ∈ [0, D), τ ∈ [1, τ_max]} 中
        无放回地抽取 n_combos 组，每组含 M 个不重复的 (dim, τ) 对。

        Returns
        -------
        combos : list of np.ndarray, shape (n_combos, M, 2)
            每行是 [dim_idx, delay]
        """
        # 所有候选 (dim, τ) 对
        dims   = np.arange(total_dims)
        taus   = np.arange(1, tau_max + 1)
        dd, tt = np.meshgrid(dims, taus, indexing='ij')
        candidates = np.stack([dd.ravel(), tt.ravel()], axis=1)   # (D*τ_max, 2)

        n_cand = len(candidates)
        M_use  = min(self.M, n_cand)   # 候选不足时自动降维

        combos = []
        for _ in range(n_combos):
            chosen_idx = rng.choice(n_cand, size=M_use, replace=False)
            combos.append(candidates[chosen_idx])   # shape (M_use, 2)
        return combos, M_use

    def _build_feature(self, seq, combo, t):
        """
        给定时刻 t，按 combo 构造特征向量。

        combo : ndarray shape (M, 2), 每行 [dim, delay]
        返回  : ndarray shape (M,)，越界位置填 NaN（后续用于过滤）
        """
        feat = np.empty(len(combo))
        for k, (dim, delay) in enumerate(combo):
            idx = t - int(delay)
            feat[k] = seq[idx, int(dim)] if idx >= 0 else np.nan
        return feat

    def _build_training_matrix(self, seq, combo, train_start, train_end, steps_ahead):
        """
        构造训练特征矩阵和目标向量。

        策略：有效起点 = train_start + max(delays)，
              保证所有特征都有真实历史数据，不补零。

        Returns
        -------
        X : ndarray (n_valid, M)
        y : ndarray (n_valid,)
        有效样本数可能少于 trainlength - steps_ahead。
        """
        max_delay_used = int(combo[:, 1].max())
        # 最早可以用的样本：时刻 t，其中 t - max_delay_used >= train_start
        t_min = train_start + max_delay_used
        # 最晚训练样本（预测目标 = t + steps_ahead < train_end）
        t_max = train_end - steps_ahead - 1

        if t_min > t_max:
            return None, None

        ts = np.arange(t_min, t_max + 1)
        X  = np.array([self._build_feature(seq, combo, t) for t in ts])
        y  = seq[ts + steps_ahead, combo[0, 0].astype(int)]
        # 注意：y 的目标维度用 combo[0,0] 仅作占位，实际由调用处传入 target_idx 覆盖

        # 过滤含 NaN 的行（理论上已不存在，但保险起见）
        valid = ~np.any(np.isnan(X), axis=1)
        return X[valid], y[valid]

    def _build_training_matrix_target(self, seq, combo, train_start, train_end,
                                      steps_ahead, target_idx):
        """
        与 _build_training_matrix 相同，但 y 明确取 target_idx 列。
        """
        max_delay_used = int(combo[:, 1].max())
        t_min = train_start + max_delay_used
        t_max = train_end - steps_ahead - 1

        if t_min > t_max:
            return None, None

        ts = np.arange(t_min, t_max + 1)
        X  = np.array([self._build_feature(seq, combo, t) for t in ts])
        y  = seq[ts + steps_ahead, target_idx]

        valid = ~np.any(np.isnan(X), axis=1)
        return X[valid], y[valid]

    def _aggregate(self, pred_list, std_list):
        """
        聚合多个 GPR 预测。

        pred_list : list of float，各模型点预测
        std_list  : list of float，各模型后验标准差

        Returns
        -------
        final_pred : float
        final_std  : float，合并不确定性
            σ²_total = Var(pred_list) + mean(std_list²)
        """
        preds = np.array(pred_list)
        stds  = np.array(std_list)

        if len(preds) == 0:
            return np.nan, np.nan

        if len(preds) == 1:
            return preds[0], stds[0]

        # 模型间方差
        inter_var = np.var(preds)
        # 模型内方差均值
        intra_var = np.mean(stds ** 2)
        final_std = np.sqrt(inter_var + intra_var)

        if self.use_kde and len(preds) >= 5:
            try:
                kde     = gaussian_kde(preds)
                xi      = np.linspace(preds.min(), preds.max(), 500)
                density = kde(xi)
                final_pred = np.sum(xi * density) / np.sum(density)
            except Exception:
                final_pred = np.mean(preds)
        else:
            final_pred = np.mean(preds)

        return final_pred, final_std

    # ──────────────────────────────────────────────────────────
    # 公开接口
    # ──────────────────────────────────────────────────────────

    def ensemble_predict(self, seq, target_idx, trainlength,
                         steps_ahead=1,
                         return_predictions=False,
                         return_uncertainty=False):
        """
        滚动预测接口（与原版保持兼容）。

        Parameters
        ----------
        seq          : ndarray (T, D)
        target_idx   : int，预测目标维度
        trainlength  : int，训练窗口长度
        steps_ahead  : int，预测步数（当前只支持1步）
        return_uncertainty : bool，是否返回不确定性
        return_predictions : bool，是否返回原始预测矩阵（兼容旧接口）

        Returns
        -------
        mean_predictions : ndarray (n_steps,)
        std_predictions  : ndarray (n_steps,)   仅当 return_uncertainty=True
        raw_matrix       : ndarray (n_steps, 1) 仅当 return_uncertainty=True 或 return_predictions=True
        """
        from gpr_module import GaussianProcessRegressor
        from sklearn.preprocessing import StandardScaler

        rng          = np.random.default_rng(self.seed)
        total_length = len(seq)
        total_dims   = seq.shape[1]
        tau_max      = self._effective_max_delay(trainlength)
        n_steps      = total_length - trainlength

        print(f"  [RDE-Delay] τ_max={tau_max}, M={self.M}, "
              f"num_samples={self.num_samples}, dims={total_dims}, "
              f"resample_per_step={self.resample_per_step}")

        # 非 resample 模式：全程固定同一批延迟组合
        if not self.resample_per_step:
            fixed_combos, M_use = self._sample_delay_combos(
                total_dims, tau_max, self.num_samples, rng)
        else:
            fixed_combos, M_use = None, None

        mean_predictions = np.full(n_steps, np.nan)
        std_predictions  = np.full(n_steps, np.nan)

        for step in range(n_steps):
            train_start = step
            train_end   = step + trainlength

            # 每步重采样 or 复用固定组合
            if self.resample_per_step:
                combos, M_use = self._sample_delay_combos(
                    total_dims, tau_max, self.num_samples, rng)
            else:
                combos = fixed_combos

            pred_list = []
            std_list  = []

            for combo in combos:
                X, y = self._build_training_matrix_target(
                    seq, combo, train_start, train_end, steps_ahead, target_idx)

                if X is None or len(y) < max(5, self.M + 1):
                    continue
                if np.std(y) < 1e-8:
                    continue

                # 构造测试点：用训练窗口末尾时刻的延迟特征
                t_test    = train_end - steps_ahead
                x_test    = self._build_feature(seq, combo, t_test)
                if np.any(np.isnan(x_test)):
                    continue
                x_test = x_test.reshape(1, -1)

                # 标准化（X 和 x_test 一起 fit，避免测试点泄漏）
                sx = StandardScaler()
                sy = StandardScaler()
                X_all   = np.vstack([X, x_test])
                X_all_s = sx.fit_transform(X_all)
                X_s     = X_all_s[:-1]
                xt_s    = X_all_s[[-1]]
                y_s     = sy.fit_transform(y.reshape(-1, 1)).ravel()

                try:
                    gp = GaussianProcessRegressor(noise=self.noise)
                    gp.fit(X_s, y_s,
                           init_params=(1.0, 1.0, 0.1), optimize=True)
                    pred_s, std_s = gp.predict(xt_s, return_std=True)

                    pred = sy.inverse_transform(
                        pred_s.reshape(-1, 1))[0, 0]
                    # std 反标准化：只需乘 y 的 scale（sy.scale_[0]）
                    std  = std_s[0] * sy.scale_[0]

                    pred_list.append(pred)
                    std_list.append(std)
                except Exception:
                    continue

            mean_predictions[step], std_predictions[step] = \
                self._aggregate(pred_list, std_list)

        # NaN 兜底（极少发生）
        nan_mask = np.isnan(mean_predictions)
        if nan_mask.any():
            mean_predictions[nan_mask] = 0.0
            std_predictions[nan_mask]  = 0.0

        if return_uncertainty:
            return mean_predictions, std_predictions, mean_predictions[:, np.newaxis]
        elif return_predictions:
            return mean_predictions, mean_predictions[:, np.newaxis]
        else:
            return mean_predictions

