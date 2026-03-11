# 改进的高斯过程回归模块 - 针对EEG信号优化（优化归一化）
import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import minimize

class GaussianProcessRegressor:
    def __init__(self, kernel='rbf', noise=1e-6, normalize_X=True, normalize_y=True):
        """
        高斯过程回归模型初始化
        :param kernel: 核函数类型 ('rbf', 'matern', 'periodic', 'rbf_periodic')
        :param noise: 噪声项（数值稳定性）
        :param normalize_X: 是否对输入X进行标准化
        :param normalize_y: 是否对输出y进行标准化
        """
        self.kernel = kernel
        self.noise = noise
        self.normalize_X = normalize_X
        self.normalize_y = normalize_y
        
        self.X_train = None
        self.y_train = None
        self.L = None          # Cholesky分解下三角矩阵
        self.alpha = None       # 后验均值参数
        self.params = None      # 超参数
        
        # 标准化参数
        self.X_scaler = {'mean': None, 'std': None}
        self.y_scaler = {'mean': None, 'std': None}
        self.is_fitted = False

    def _rbf_kernel(self, X1, X2, sigma_f, l):
        """RBF核函数实现"""
        sqdist = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1) - 2 * X1 @ X2.T
        return sigma_f**2 * np.exp(-sqdist / (2 * l**2))

    def _matern_kernel(self, X1, X2, sigma_f, l, nu=1.5):
        """Matérn核函数 - 更适合非光滑信号如EEG"""
        dists = np.sqrt(np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1) - 2 * X1 @ X2.T)
        dists = np.maximum(dists, 1e-10)  # 避免数值问题
        
        if nu == 0.5:
            # Exponential kernel
            K = sigma_f**2 * np.exp(-dists / l)
        elif nu == 1.5:
            # Matérn 3/2
            sqrt3_d_over_l = np.sqrt(3) * dists / l
            K = sigma_f**2 * (1 + sqrt3_d_over_l) * np.exp(-sqrt3_d_over_l)
        elif nu == 2.5:
            # Matérn 5/2
            sqrt5_d_over_l = np.sqrt(5) * dists / l
            K = sigma_f**2 * (1 + sqrt5_d_over_l + 5 * dists**2 / (3 * l**2)) * np.exp(-sqrt5_d_over_l)
        else:
            # 通用Matérn核（计算复杂）
            try:
                from scipy.special import kv, gamma
                sqrt_2nu_d_over_l = np.sqrt(2 * nu) * dists / l
                K = sigma_f**2 * (2**(1-nu) / gamma(nu)) * (sqrt_2nu_d_over_l**nu) * kv(nu, sqrt_2nu_d_over_l)
                K[dists == 0] = sigma_f**2
            except:
                # 如果计算失败，回退到Matérn 3/2
                sqrt3_d_over_l = np.sqrt(3) * dists / l
                K = sigma_f**2 * (1 + sqrt3_d_over_l) * np.exp(-sqrt3_d_over_l)
        
        return K

    def _periodic_kernel(self, X1, X2, sigma_f, l, period):
        """周期性核函数 - 捕捉EEG的节律性"""
        dists = np.sqrt(np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1) - 2 * X1 @ X2.T)
        return sigma_f**2 * np.exp(-2 * np.sin(np.pi * dists / period)**2 / l**2)

    def _kernel_matrix(self, X1, X2):
        """根据核类型生成协方差矩阵"""
        if self.kernel == 'rbf':
            sigma_f, l, sigma_n = self.params
            K = self._rbf_kernel(X1, X2, sigma_f, l)
        elif self.kernel == 'matern':
            sigma_f, l, sigma_n = self.params
            K = self._matern_kernel(X1, X2, sigma_f, l, nu=1.5)
        elif self.kernel == 'periodic':
            sigma_f, l, period, sigma_n = self.params
            K = self._periodic_kernel(X1, X2, sigma_f, l, period)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
            
        # 添加噪声项（仅对角线）
        if X1 is X2:
            K += (sigma_n**2 + self.noise) * np.eye(X1.shape[0])
        return K

    def _standardize_X(self, X, fit=False):
        """标准化输入数据X"""
        if not self.normalize_X:
            return X
            
        # 确保X是2D数组
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        if fit:  # 训练时计算并保存标准化参数
            self.X_scaler['mean'] = np.mean(X, axis=0, keepdims=True)
            self.X_scaler['std'] = np.std(X, axis=0, keepdims=True)
            # 防止标准差为0
            self.X_scaler['std'] = np.where(self.X_scaler['std'] < 1e-8, 1.0, self.X_scaler['std'])
        
        # 标准化
        X_normalized = (X - self.X_scaler['mean']) / self.X_scaler['std']
        return X_normalized

    def _standardize_y(self, y, fit=False):
        """标准化输出数据y"""
        if not self.normalize_y:
            return y
            
        # 确保y是1D数组
        if y.ndim > 1:
            y = y.ravel()
            
        if fit:  # 训练时计算并保存标准化参数
            self.y_scaler['mean'] = np.mean(y)
            self.y_scaler['std'] = np.std(y)
            # 防止标准差为0
            if self.y_scaler['std'] < 1e-8:
                self.y_scaler['std'] = 1.0
        
        # 标准化
        y_normalized = (y - self.y_scaler['mean']) / self.y_scaler['std']
        return y_normalized

    def _unstandardize_y(self, y):
        """反标准化输出数据y"""
        if not self.normalize_y:
            return y
        return y * self.y_scaler['std'] + self.y_scaler['mean']

    def fit(self, X_train, y_train, init_params=None, optimize=True):
        """
        训练GPR模型
        :param X_train: 训练数据 (n_samples, n_features)
        :param y_train: 训练标签 (n_samples,)
        :param init_params: 初始超参数
        :param optimize: 是否优化超参数
        """
        # 输入验证
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        
        if X_train.ndim == 1:
            X_train = X_train.reshape(-1, 1)
        if y_train.ndim > 1:
            y_train = y_train.ravel()
            
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("X_train and y_train must have the same number of samples")

        # 数据标准化
        X_train_normalized = self._standardize_X(X_train, fit=True)
        y_train_normalized = self._standardize_y(y_train, fit=True)
        
        self.X_train = X_train_normalized
        self.y_train = y_train_normalized

        # 设置初始参数
        if init_params is None:
            if self.kernel == 'rbf' or self.kernel == 'matern':
                # 对于EEG信号，使用较小的长度尺度
                init_params = [1.0, 0.5, 0.1]  # [sigma_f, l, sigma_n]
            elif self.kernel == 'periodic':
                init_params = [1.0, 0.5, 10.0, 0.1]  # [sigma_f, l, period, sigma_n]
            elif self.kernel == 'rbf_periodic':
                init_params = [0.5, 0.5, 0.5, 0.5, 10.0, 0.1]
        
        self.params = np.array(init_params)

        # 超参数优化
        if optimize:
            optimized_params = self._optimize_hyperparams(self.params)
            if optimized_params is not None:
                self.params = optimized_params

        # 计算核矩阵并进行Cholesky分解
        K = self._kernel_matrix(self.X_train, self.X_train)
        
        try:
            self.L = cholesky(K, lower=True)
        except np.linalg.LinAlgError:
            print("Warning: Cholesky decomposition failed, adding jitter...")
            K += self.noise * np.eye(K.shape[0])
            self.L = cholesky(K, lower=True)
            
        self.alpha = solve_triangular(self.L.T, solve_triangular(self.L, self.y_train, lower=True))
        self.is_fitted = True
        
        print(f"GPR训练完成 - 核函数: {self.kernel}, 样本数: {X_train.shape[0]}")
        print(f"优化后超参数: {self.params}")

    def predict(self, X_test, return_std=False):
        """对新数据进行预测"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        X_test = np.asarray(X_test)
        if X_test.ndim == 1:
            X_test = X_test.reshape(-1, 1)
            
        # 标准化测试数据
        X_test_normalized = self._standardize_X(X_test, fit=False)
        
        # 计算预测
        K_star = self._kernel_matrix(self.X_train, X_test_normalized)
        y_mean = K_star.T @ self.alpha
        
        # 反标准化预测结果
        y_mean = self._unstandardize_y(y_mean)

        if return_std:
            v = solve_triangular(self.L, K_star, lower=True)
            K_starstar = self._kernel_matrix(X_test_normalized, X_test_normalized)
            y_cov = K_starstar - v.T @ v
            y_std = np.sqrt(np.maximum(np.diag(y_cov), 0))  # 防止负方差
            
            # 反标准化标准差
            if self.normalize_y:
                y_std = y_std * self.y_scaler['std']
                
            return y_mean, y_std
        else:
            return y_mean

    def _optimize_hyperparams(self, init_params):
        """优化超参数"""
        def neg_log_likelihood(params):
            try:
                # 参数边界检查
                if np.any(params <= 0):
                    return np.inf
                    
                old_params = self.params.copy()
                self.params = params
                K = self._kernel_matrix(self.X_train, self.X_train)
                
                # 确保矩阵正定
                eigvals = np.linalg.eigvals(K)
                if np.min(eigvals) <= 1e-10:
                    self.params = old_params
                    return np.inf
                    
                L = cholesky(K, lower=True)
                alpha = solve_triangular(L.T, solve_triangular(L, self.y_train, lower=True))
                
                # 负对数边际似然
                nll = (0.5 * self.y_train.T @ alpha + 
                       np.sum(np.log(np.diag(L))) + 
                       0.5 * len(self.y_train) * np.log(2*np.pi))
                
                self.params = old_params
                return float(nll)
            except Exception as e:
                return np.inf

        # 根据核函数类型设置bounds
        if self.kernel == 'rbf' or self.kernel == 'matern':
            bounds = [(0.01, 10.0), (0.01, 5.0), (1e-6, 1.0)]  # [sigma_f, l, sigma_n]
        elif self.kernel == 'periodic':
            bounds = [(0.01, 10.0), (0.01, 5.0), (1.0, 100.0), (1e-6, 1.0)]  # [sigma_f, l, period, sigma_n]
        elif self.kernel == 'rbf_periodic':
            bounds = [(0.01, 5.0), (0.01, 2.0), (0.01, 5.0), (0.01, 2.0), (1.0, 100.0), (1e-6, 1.0)]
        
        try:
            # 多次尝试不同的初始点
            best_result = None
            best_score = np.inf
            
            for i in range(3):  # 尝试3次
                if i == 0:
                    x0 = init_params
                else:
                    # 随机扰动初始参数
                    x0 = init_params * (1 + 0.1 * np.random.randn(len(init_params)))
                    x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])
                
                res = minimize(neg_log_likelihood, x0, method='L-BFGS-B', bounds=bounds)
                
                if res.success and res.fun < best_score:
                    best_result = res
                    best_score = res.fun
            
            if best_result is not None and best_result.success:
                return best_result.x
            else:
                print("Warning: Hyperparameter optimization failed, using initial parameters")
                return init_params
                
        except Exception as e:
            print(f"Warning: Hyperparameter optimization error: {e}")
            return init_params

    def get_params(self):
        """获取当前超参数"""
        if not self.is_fitted:
            return None
        return {
            'kernel': self.kernel,
            'params': self.params,
            'normalize_X': self.normalize_X,
            'normalize_y': self.normalize_y
        }