# dataset_lorenz.py
import os
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader

def NWnetwork(N, m, p):
    """小世界网络生成器 (Newman-Watts 模型)"""
    matrix = np.zeros((N, N), dtype=bool)
    
    # 生成环形规则网络
    for i in range(N):
        # 计算邻居索引（含环形处理）
        neighbors = [(i + k) % N for k in range(-m, m+1) if k != 0]
        matrix[i, neighbors] = True
    
    # 随机添加长程连接 (排除自环)
    rand_mask = np.random.rand(N, N) < p
    np.fill_diagonal(rand_mask, False)  # 移除自环
    matrix = matrix | rand_mask
    
    # 确保对称性（无向图）
    matrix = matrix | matrix.T
    np.fill_diagonal(matrix, False)  # 最终移除自环
    
    return matrix.astype(float), N

def generate_coupled_lorenz(N=5, L=100, stepsize=1, C=0.1, m=1, p=0.1, delta=0.01):
    """完全对齐MATLAB行为的耦合洛伦兹系统生成"""
    # 初始化网络和状态变量
    adjmat, M = NWnetwork(N, m, p)    # 阶段3：同阶段2
    sigma = 10.0
    total_l = L * stepsize

    # 初始化状态变量 (MATLAB风格列优先)
    x = np.zeros((M, total_l))
    y = np.zeros((M, total_l))
    z = np.zeros((M, total_l))
    x[:, 0] = np.random.rand(M)
    y[:, 0] = np.random.rand(M)
    z[:, 0] = np.random.rand(M)

    # 分阶段模拟
    for i in range(total_l-1):
        for j in range(M):
            # 计算耦合项 (保持MATLAB的矩阵乘法顺序)
            coupling = C * np.dot(adjmat[j], x[:, i])

            # 更新方程 (严格对齐MATLAB实现)
            dx = delta * (sigma * (y[j,i] - x[j,i]) + coupling)
            dy = delta * (28 * x[j,i] - y[j,i] - x[j,i] * z[j,i])
            dz = delta * (-8/3 * z[j,i] + x[j,i] * y[j,i])

            x[j,i+1] = x[j,i] + dx
            y[j,i+1] = y[j,i] + dy
            z[j,i+1] = z[j,i] + dz

    # 构建与MATLAB完全相同的输出结构
    X = np.zeros((3*M, total_l))
    for j in range(M):
        X[3*j] = x[j]
        X[3*j+1] = y[j]
        X[3*j+2] = z[j]

    X = X.T
    ret = X[::stepsize]

    return ret, X  # 保持(L, 3*M)的输出格式 (4000, 15)

class Lorenz_Dataset(Dataset):
    """
    Dataset for Lorenz time series data.
    Returned sample shapes:
      observed_data: (eval_length, num_features)
      observed_mask: (eval_length, num_features)
      gt_mask:       (eval_length, num_features)
      timepoints:    np.arange(eval_length)
    """
    def __init__(self, data_array, eval_length=100):
        """
        data_array: numpy array shape (num_sequences, T, D).
        eval_length: sequence length to return
        """
        self.eval_length = eval_length
        self.raw_data = data_array
        num_sequences, T, D = self.raw_data.shape

        # Calculate mean and std from the data
        mean = np.mean(self.raw_data, axis=(0, 1))
        std = np.std(self.raw_data, axis=(0, 1)) + 1e-8
        self.data_mean = mean
        self.data_std = std

        self.observed_mask = np.ones_like(self.raw_data, dtype=np.float32)
        self.gt_mask = np.zeros_like(self.raw_data, dtype=np.float32)
        self.gt_mask[:, -1::-2, :] = 1.0

    def __len__(self):
        return self.raw_data.shape[0]

    def __getitem__(self, idx):
        return {
            "observed_data": self.raw_data[idx],
            "observed_mask": self.observed_mask[idx],
            "gt_mask": self.gt_mask[idx],
            "timepoints": np.arange(self.eval_length).astype(np.float32),
        }

def get_dataloader(batch_size=16, seq_len=100, seq_count=50, stepsize=1, cache_dir="./cache_lorenz", N=5):
    all_data = np.array([generate_coupled_lorenz(N=N, L=seq_len, stepsize=stepsize)[0] for _ in range(seq_count)]).astype(np.float32)

    # Split data into train, validation, and test sets
    total_size = len(all_data)
    train_size = int(total_size * 0.7)
    valid_size = int(total_size * 0.15)

    indices = np.random.permutation(total_size)
    train_indices = indices[:train_size]
    valid_indices = indices[train_size:train_size + valid_size]
    test_indices = indices[train_size + valid_size:]

    train_data = all_data[train_indices]
    valid_data = all_data[valid_indices]
    test_data = all_data[test_indices]

    # Create datasets and dataloaders
    train_dataset = Lorenz_Dataset(train_data, eval_length=seq_len)
    valid_dataset = Lorenz_Dataset(valid_data, eval_length=seq_len)
    test_dataset = Lorenz_Dataset(test_data, eval_length=seq_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader
