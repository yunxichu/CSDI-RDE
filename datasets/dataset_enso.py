# dataset_enso.py
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

class ENSO_Dataset(Dataset):
    """
    Dataset for ENSO time series data.
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

        # Normalize data
        self.normalized_data = (self.raw_data - mean) / std

        self.observed_mask = np.ones_like(self.normalized_data, dtype=np.float32)
        self.gt_mask = np.zeros_like(self.normalized_data, dtype=np.float32)
        self.gt_mask[:, -1::-2, :] = 1.0  # Every other point is treated as ground truth

    def __len__(self):
        return self.raw_data.shape[0]

    def __getitem__(self, idx):
        return {
            "observed_data": self.normalized_data[idx],
            "observed_mask": self.observed_mask[idx],
            "gt_mask": self.gt_mask[idx],
            "timepoints": np.arange(self.eval_length).astype(np.float32),
        }

    def inverse_transform(self, data):
        """Convert normalized data back to original scale"""
        return data * self.data_std + self.data_mean

def load_enso_data(file_path, seq_len=100, seq_stride=50):
    """
    Load ENSO data from Excel file and create sequences
    """
    # 读取 Excel 文件
    df = pd.read_excel(file_path, sheet_name='Sheet1')

    # 从输出可知，真实数据从第 4 行(index=3)开始，重新加载数据
    df = df[3:]

    # 重置索引
    df = df.reset_index(drop=True)

    # 通过列索引提取 SST 列 (Nino1+2, Nino3, Nino3.4, Nino4)
    # 假设参与分析的列是第 2、4、6、8 列（索引从 0 开始）
    sst_data = df.iloc[:, [1, 3, 5, 7]].values.astype(np.float32)

    # 移除包含 NaN 值的行
    valid_mask = ~np.isnan(sst_data).any(axis=1)
    sst_data = sst_data[valid_mask]

    print(f"Loaded ENSO data shape: {sst_data.shape}")
    print(f"Data range: [{sst_data.min():.2f}, {sst_data.max():.2f}]")

    # 使用滑动窗口创建序列
    sequences = []
    for i in range(0, len(sst_data) - seq_len + 1, seq_stride):
        sequence = sst_data[i:i + seq_len]
        sequences.append(sequence)

    sequences = np.array(sequences)
    print(f"Created {len(sequences)} sequences of length {seq_len}")

    return sequences

def get_enso_dataloader(file_path, batch_size=16, seq_len=100, seq_stride=50):
    """
    Create dataloaders for ENSO data
    """
    # Load and preprocess data
    sequences = load_enso_data(file_path, seq_len, seq_stride)
    
    # Split data into train, validation, and test sets
    total_size = len(sequences)
    train_size = int(total_size * 0.7)
    valid_size = int(total_size * 0.15)

    indices = np.random.permutation(total_size)
    train_indices = indices[:train_size]
    valid_indices = indices[train_size:train_size + valid_size]
    test_indices = indices[train_size + valid_size:]

    train_data = sequences[train_indices]
    valid_data = sequences[valid_indices]
    test_data = sequences[test_indices]

    # Create datasets and dataloaders
    train_dataset = ENSO_Dataset(train_data, eval_length=seq_len)
    valid_dataset = ENSO_Dataset(valid_data, eval_length=seq_len)
    test_dataset = ENSO_Dataset(test_data, eval_length=seq_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_dataset