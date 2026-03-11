import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# 从您的原始代码中导入洛伦兹生成函数
from example_usage import generate_coupled_lorenz  # 替换your_main_code_file为实际文件名

class LorenzDataset(Dataset):
    def __init__(self, raw_data, seq_len=100, missing_rate=0.1, mode="train"):
        """
        raw_data: 原始数据 [总时间步长, 特征数]
        seq_len: 时序片段长度
        missing_rate: 缺失率
        mode: 数据模式（train/valid/test）
        """
        self.seq_len = seq_len
        self.missing_rate = missing_rate
        
        # 数据标准化
        self.data_mean = np.mean(raw_data, axis=0)
        self.data_std = np.std(raw_data, axis=0)
        self.norm_data = (raw_data - self.data_mean) / (self.data_std + 1e-8)
        
        # 生成时序片段
        self.samples = []
        total_length = len(self.norm_data)
        
        for i in range(total_length - seq_len):
            sample = self.norm_data[i:i+seq_len]
            mask = self._generate_mask(sample.shape)
            self.samples.append((sample, mask))

    def _generate_mask(self, shape):
        """生成随机缺失掩码"""
        mask = np.random.rand(*shape) > self.missing_rate
        return mask.astype(np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample, mask = self.samples[idx]
        return {
            "observed_data": torch.FloatTensor(sample),
            "observed_mask": torch.FloatTensor(mask),
            "timepoints": torch.arange(self.seq_len)
        }

def get_dataloader(batch_size=32, seq_len=100, missing_rate=0.1):
    # 生成原始数据（直接调用您已有的函数）
    raw_data = generate_coupled_lorenz(N=5, L=4000).astype(np.float32)  # shape (4000, 15)
    
    # 划分数据集
    train_size = int(4000 * 0.7)
    valid_size = int(4000 * 0.15)
    
    # 创建数据集实例并获取标准化参数
    train_dataset = LorenzDataset(raw_data[:train_size], seq_len, missing_rate, "train")
    valid_dataset = LorenzDataset(raw_data[train_size:train_size+valid_size], seq_len, missing_rate, "valid")
    test_dataset = LorenzDataset(raw_data[train_size+valid_size:], seq_len, missing_rate, "test")
    
    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # 返回标准化参数（从任意一个数据集实例获取）
    return train_loader, valid_loader, test_loader, (train_dataset.data_mean, train_dataset.data_std)