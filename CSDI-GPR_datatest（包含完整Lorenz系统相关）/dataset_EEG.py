#dataset_EEG.py
import os
import pickle
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch

class EEG_Dataset(Dataset):
    """
    EEG 时间序列数据数据集类
    返回的样本形状：
      observed_data: (序列长度, 64) - 标准化后的EEG数据
      observed_mask: (序列长度, 64) - 观测掩码（1表示观测到，0表示缺失）
      gt_mask: (序列长度, 64) - 真实值掩码（用于训练和评估）
      timepoints: np.arange(序列长度) - 时间点数组
    """
    def __init__(self, data_path, eval_length=100, mode="train", valid_ratio=0.2):
        """
        初始化EEG数据集
        
        参数:
            data_path: EEG数据Excel文件路径
            eval_length: 每个序列的时间点数量
            mode: 模式，可选"train"、"valid"或"test"
            valid_ratio: 验证集和测试集占总数据的比例
        """
        self.eval_length = eval_length  # 序列长度
        self.mode = mode  # 数据集模式
        
        # 从Excel文件读取EEG数据
        # 注意：数据没有时间戳列，只有数值
        df = pd.read_excel(data_path, header=None)  # 不设置列名
        
        # 确保数据有64个通道（电极）
        # 如果第一列是索引，则从第二列开始取64列

        self.raw_data = df.values.astype(np.float32)  # 直接使用所有列
        
        # 确保数据维度正确
        assert self.raw_data.shape[1] == 64, f"EEG数据应有64个通道，但得到{self.raw_data.shape[1]}个"
        
        # 计算数据的均值和标准差（用于标准化）
        self.data_mean = np.mean(self.raw_data, axis=0)  # 沿时间轴计算均值
        self.data_std = np.std(self.raw_data, axis=0) + 1e-8  # 沿时间轴计算标准差，添加小值避免除零
        
        # 标准化数据：(原始数据 - 均值) / 标准差
        self.normalized_data = (self.raw_data - self.data_mean) / self.data_std
        
        # 将数据分割成固定长度的序列
        total_length = len(self.normalized_data)  # 总时间点数
        num_sequences = total_length // eval_length  # 可以创建的序列数量
        
        # 将数据重塑为多个序列
        sequences = []
        for i in range(num_sequences):
            start_idx = i * eval_length  # 序列起始索引
            end_idx = start_idx + eval_length  # 序列结束索引
            sequences.append(self.normalized_data[start_idx:end_idx])
        
        # 转换为numpy数组
        sequences_array = np.array(sequences)
        
        # 将序列划分为训练集、验证集和测试集
        total_sequences = len(sequences_array)  # 总序列数
        train_size = int(total_sequences * (1 - valid_ratio))  # 训练集大小
        valid_size = int(total_sequences * valid_ratio / 2)  # 验证集大小
        test_size = total_sequences - train_size - valid_size  # 测试集大小
        
        # 根据模式选择数据子集（确保是NumPy数组）
        if mode == "train":
            self.data = sequences_array[:train_size]
        elif mode == "valid":
            self.data = sequences_array[train_size:train_size + valid_size]
        elif mode == "test":
            self.data = sequences_array[train_size + valid_size:]
        
        # 创建观测掩码（假设所有数据都被观测到）
        self.observed_mask = np.ones_like(self.data, dtype=np.float32)
        
        # 创建真实值掩码（用于模拟缺失数据模式）
        if mode == "train":
            # 训练模式：随机掩盖20%的数据
            mask_shape = self.data.shape  # 获取掩码形状
            random_mask = np.random.rand(*mask_shape) > 0.2  # 生成随机掩码
            self.gt_mask = random_mask.astype(np.float32)  # 转换为浮点型
        else:
            # 验证和测试模式：使用固定缺失模式（每5个时间点缺失一次）
            mask_shape = self.data.shape
            self.gt_mask = np.ones(mask_shape, dtype=np.float32)  # 初始化为全1
            self.gt_mask[:, ::5, :] = 0.0  # 每5个时间点设置为0（缺失）

    def __len__(self):
        """返回数据集中的样本数量"""
        return len(self.data)

    def __getitem__(self, idx):
        """获取指定索引的样本"""
        return {
            "observed_data": self.data[idx],  # 标准化后的EEG数据
            "observed_mask": self.observed_mask[idx],  # 观测掩码
            "gt_mask": self.gt_mask[idx],  # 真实值掩码
            "timepoints": np.arange(self.eval_length).astype(np.float32),  # 时间点数组
        }

def get_dataloader(data_path, batch_size=16, eval_length=100, valid_ratio=0.2):
    """
    创建EEG数据的数据加载器
    
    参数:
        data_path: EEG数据文件路径
        batch_size: 批量大小
        eval_length: 序列长度
        valid_ratio: 验证集和测试集比例
    
    返回:
        train_loader: 训练数据加载器
        valid_loader: 验证数据加载器
        test_loader: 测试数据加载器
        scaler: 标准化参数（标准差）
        mean_scaler: 标准化参数（均值）
    """
    # 创建训练数据集
    train_dataset = EEG_Dataset(
        data_path, 
        eval_length=eval_length, 
        mode="train", 
        valid_ratio=valid_ratio
    )
    
    # 创建验证数据集
    valid_dataset = EEG_Dataset(
        data_path, 
        eval_length=eval_length, 
        mode="valid", 
        valid_ratio=valid_ratio
    )
    
    # 创建测试数据集
    test_dataset = EEG_Dataset(
        data_path, 
        eval_length=eval_length, 
        mode="test", 
        valid_ratio=valid_ratio
    )
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 获取标准化参数并转换为PyTorch张量
    scaler = torch.from_numpy(train_dataset.data_std).float()
    mean_scaler = torch.from_numpy(train_dataset.data_mean).float()
    
    return train_loader, valid_loader, test_loader, scaler, mean_scaler

# 示例用法
if __name__ == "__main__":
    # 指定EEG数据Excel文件路径
    eeg_data_path = "/home/rhl/CSDI-GPR_project/CSDI-GPR_datatest/data/Dataset_3-EEG.xlsx"
    
    # 获取数据加载器
    train_loader, valid_loader, test_loader, scaler, mean_scaler = get_dataloader(
        data_path=eeg_data_path,
        batch_size=32,  # 批量大小
        eval_length=100,  # 每个序列100个时间点（秒）
        valid_ratio=0.2  # 20%的数据用于验证和测试
    )
    
    # 打印数据集大小
    print(f"训练样本数: {len(train_loader.dataset)}")
    print(f"验证样本数: {len(valid_loader.dataset)}")
    print(f"测试样本数: {len(test_loader.dataset)}")
    
    # 检查一个样本的形状
    sample = next(iter(train_loader))
    print(f"样本数据形状: {sample['observed_data'].shape}")
    print(f"样本掩码形状: {sample['observed_mask'].shape}")