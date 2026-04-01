# 高斯过程
import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import minimize

class GaussianProcessRegressor:
    def __init__(self, kernel='rbf', noise=1e-6):
        """
        高斯过程回归模型初始化
        :param kernel: 核函数类型，默认为RBF核
        :param noise: 噪声项（数值稳定性）
        """
        self.kernel = kernel
        self.noise = noise
        self.X_train = None
        self.y_train = None
        self.L = None          # Cholesky分解下三角矩阵
        self.alpha = None       # 后验均值参数
        self.params = None      # 超参数 [sigma_f, l, sigma_n]
        self.mu_X, self.sigma_X = None, None  # 数据标准化参数
        self.mu_y, self.sigma_y = None, None

    def _rbf_kernel(self, X1, X2, sigma_f, l):
        """RBF核函数实现"""
        sqdist = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1) - 2 * X1 @ X2.T
        return sigma_f**2 * np.exp(-sqdist / (2 * l**2))

    def _kernel_matrix(self, X1, X2):
        """根据核类型生成协方差矩阵"""
        sigma_f, l, sigma_n = self.params
        K = self._rbf_kernel(X1, X2, sigma_f, l)
        if X1 is X2:
            K += (sigma_n**2 + self.noise) * np.eye(X1.shape[0])
        return K

    def fit(self, X_train, y_train, init_params=(1.0, 1.0, 0.1), optimize=False):
        """
        训练GPR模型
        :param X_train: 训练数据 (n_samples, n_features)
        :param y_train: 训练标签 (n_samples,)
        :param init_params: 初始超参数 [sigma_f, l, sigma_n]
        :param optimize: 是否优化超参数
        """
        # 数据标准化
        X_train, self.mu_X, self.sigma_X = self._normalize(X_train)
        y_train, self.mu_y, self.sigma_y = self._normalize(y_train)
        self.X_train = X_train
        self.y_train = y_train

        # 超参数优化
        if optimize:
            self.params = self._optimize_hyperparams(init_params)
        else:
            self.params = np.array(init_params)

        # 计算核矩阵并进行Cholesky分解
        K = self._kernel_matrix(X_train, X_train)
        try:
            self.L = cholesky(K, lower=True)
        except np.linalg.LinAlgError:
            K += self.noise * np.eye(K.shape[0])
            self.L = cholesky(K, lower=True)
        self.alpha = solve_triangular(self.L.T, solve_triangular(self.L, y_train, lower=True))

    def predict(self, X_test, return_std=False):
        """对新数据进行预测"""
        X_test = (X_test - self.mu_X) / self.sigma_X  # 标准化
        K_star = self._kernel_matrix(self.X_train, X_test)
        y_mean = K_star.T @ self.alpha
        y_mean = y_mean * self.sigma_y + self.mu_y    # 逆标准化

        if return_std:
            v = solve_triangular(self.L, K_star, lower=True)
            K_starstar = self._kernel_matrix(X_test, X_test)
            y_cov = K_starstar - v.T @ v
            y_std = np.sqrt(np.diag(y_cov)) * self.sigma_y
            return y_mean, y_std
        else:
            return y_mean

    def _optimize_hyperparams(self, init_params):
        def neg_log_likelihood(params):
            sigma_f, l, sigma_n = params
            # 增加噪声项到协方差矩阵
            K = self._rbf_kernel(self.X_train, self.X_train, sigma_f, l) + (sigma_n**2 + 1e-5) * np.eye(len(self.X_train))
            try:
                L = cholesky(K, lower=True)
            except np.linalg.LinAlgError:
                return np.inf
            alpha = solve_triangular(L.T, solve_triangular(L, self.y_train, lower=True))
            return 0.5 * self.y_train.T @ alpha + np.sum(np.log(np.diag(L))) + 0.5 * len(self.y_train) * np.log(2*np.pi)

        # 调整优化范围，避免参数过小
        # 在 _optimize_hyperparams 方法中修改 bounds
        bounds = [(1e-5, 1e2), (1e-5, 1e2), (1e-5, 1e2)]  # 放宽参数范围
        #bounds = [(0.1, 10.0), (0.1, 10.0), (0.01, 1.0)]  # 修改此行
        res = minimize(neg_log_likelihood, init_params, method='L-BFGS-B', bounds=bounds)
        return res.x

    @staticmethod
    def _normalize(X):
        """数据标准化"""
        mu = np.mean(X, axis=0)
        sigma = np.std(X, axis=0)
        return (X - mu) / sigma, mu, sigma
