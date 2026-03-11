# -*- coding: utf-8 -*-
"""
PM2.5数据集 - 补值版本（随机时间划分）

数据处理流程：
1. 按时间对半分：前50%用于补值，后50%用于预测（不使用）
2. 在前50%数据中随机划分train/valid/test（不按月份）
3. 使用CSDI方法进行补值训练

说明（新增但不影响原功能）：
- 为了让“每次随机选取都固定到一个随机种子”，本版本把所有随机操作都绑定到 seed：
  1) 数据划分（train/valid/test）使用独立的 numpy RNG（不会污染全局 np.random）
  2) __getitem__ 内的随机（hist_mask 选择、人工 mask 位置选择）使用“按样本确定的 RNG”
     -> 只由 (seed + start_idx) 决定，因此即使 DataLoader shuffle、多 worker、访问顺序变化也完全可复现
  3) DataLoader shuffle 使用固定 torch.Generator，并提供 worker_init_fn（未来 num_workers>1 也不漂）
"""

import pickle
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import torch
import random


def set_global_seed(seed: int):
    """
    设置全局随机种子（新增，不减少原功能）
    目的：确保 random / numpy / torch 的随机行为在同一 seed 下可复现
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # （可选）更强确定性设置：不影响你的数据逻辑，只影响训练时某些算子行为
    # 如果你发现某些环境/算子不兼容，可注释掉这些行
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)


def seed_worker_factory(base_seed: int):
    """
    返回一个 worker_init_fn（新增，不减少原功能）
    目的：DataLoader 多 worker 时，每个 worker 的随机种子也固定
    """
    base_seed = int(base_seed)

    def seed_worker(worker_id: int):
        worker_seed = base_seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return seed_worker


class PM25_Imputation_Dataset(Dataset):
    """
    PM2.5数据集的补值版本（随机时间划分）
    """

    def __init__(self, eval_length=36, target_dim=36, mode="train",
                 missing_ratio=0.1, split_ratio=0.5,
                 train_ratio=0.7, valid_ratio=0.15, seed=42):
        """
        参数:
            eval_length: 评估序列长度（时间步数）
            target_dim: 目标维度（站点数量）
            mode: 数据集模式 ("train", "valid", "test")
            missing_ratio: 训练时人工mask的缺失比例（CSDI方法）
            split_ratio: 时间切分比例（前split_ratio用于补值，后面用于预测）
            train_ratio: 训练集比例（在前半部分数据中）
            valid_ratio: 验证集比例（在前半部分数据中）
            seed: 随机种子（用于可复现的随机划分）
        """
        self.eval_length = eval_length
        self.target_dim = target_dim
        self.missing_ratio = missing_ratio
        self.split_ratio = split_ratio
        self.mode = mode
        self.seed = int(seed)

        # 加载训练数据的均值和标准差
        path = "./data/pm25/pm25_meanstd.pk"
        with open(path, "rb") as f:
            self.train_mean, self.train_std = pickle.load(f)

        # ========== 步骤1：读取完整数据集 ==========
        print(f"正在读取数据...")
        df_full = pd.read_csv(
            "./data/pm25/Code/STMVL/SampleData/pm25_ground.txt",
            index_col="datetime",
            parse_dates=True,
        )
        df_gt_full = pd.read_csv(
            "./data/pm25/Code/STMVL/SampleData/pm25_missing.txt",
            index_col="datetime",
            parse_dates=True,
        )

        # 确保数据按时间排序
        df_full = df_full.sort_index()
        df_gt_full = df_gt_full.sort_index()

        print(f"完整数据集形状: {df_full.shape}")
        print(f"时间范围: {df_full.index.min()} 至 {df_full.index.max()}")

        # ========== 步骤2：按时间切分为两部分 ==========
        total_length = len(df_full)
        split_point = int(total_length * self.split_ratio)

        # 前半部分：用于补值任务
        df_imputation = df_full.iloc[:split_point].copy()
        df_gt_imputation = df_gt_full.iloc[:split_point].copy()

        # 后半部分：留给预测任务（当前不使用）
        # df_prediction = df_full.iloc[split_point:].copy()

        print(f"\n数据切分（split_ratio={split_ratio}）:")
        print(f"  前半部分（补值）: {len(df_imputation)} 条记录")
        print(f"    时间范围: {df_imputation.index.min()} 至 {df_imputation.index.max()}")
        print(f"  后半部分（预测）: {total_length - split_point} 条记录")
        print(f"    时间范围: {df_full.iloc[split_point:].index.min()} 至 {df_full.iloc[split_point:].index.max()}")

        # ========== 步骤3：在前半部分随机划分train/valid/test ==========
        imputation_length = len(df_imputation)

        # 创建所有可能的样本索引（考虑滑动窗口）
        # 每个样本需要eval_length个连续时间点
        max_start_index = imputation_length - eval_length + 1
        all_indices = np.arange(max_start_index)

        # 设置随机种子以确保可复现（改进：使用独立 RNG，避免污染全局 np.random）
        split_rng = np.random.default_rng(self.seed)

        # 随机打乱索引（可复现）
        all_indices = split_rng.permutation(all_indices)

        # 按比例划分
        train_end = int(len(all_indices) * train_ratio)
        valid_end = train_end + int(len(all_indices) * valid_ratio)

        if mode == "train":
            self.use_indices = all_indices[:train_end]
            print(f"\n训练集: {len(self.use_indices)} 个样本")
        elif mode == "valid":
            self.use_indices = all_indices[train_end:valid_end]
            print(f"\n验证集: {len(self.use_indices)} 个样本")
        elif mode == "test":
            self.use_indices = all_indices[valid_end:]
            print(f"\n测试集: {len(self.use_indices)} 个样本")
        else:
            raise ValueError(f"mode must be one of ['train','valid','test'], got: {mode}")

        # 将索引排序以便按时间顺序访问（可选）
        self.use_indices = np.sort(self.use_indices)

        # ========== 步骤4：准备数据 ==========
        # 创建观测掩码和标准化数据
        self.observed_mask = 1 - df_imputation.isnull().values  # 1=观测到，0=缺失
        self.gt_mask = 1 - df_gt_imputation.isnull().values

        # 标准化数据
        self.observed_data = (
            (df_imputation.fillna(0).values - self.train_mean) / self.train_std
        ) * self.observed_mask

        print(f"数据形状: {self.observed_data.shape}")
        print(f"观测掩码形状: {self.observed_mask.shape}")

    def __getitem__(self, index):
        """获取单个样本"""
        # 获取实际的起始索引
        start_idx = int(self.use_indices[index])
        end_idx = start_idx + self.eval_length

        # 提取数据切片
        observed_data = self.observed_data[start_idx:end_idx]
        observed_mask = self.observed_mask[start_idx:end_idx]
        gt_mask = self.gt_mask[start_idx:end_idx]

        # ========== 固定 __getitem__ 内所有随机选取 ==========
        # 关键：为每个样本构建“确定性的 RNG”，只由 (seed + start_idx) 决定
        # 这样无论 DataLoader 是否 shuffle、是否多 worker、访问顺序如何变化，都完全可复现
        rng = np.random.default_rng(self.seed + start_idx)

        # ========== 获取hist_mask（历史缺失模式，用于CSDI） ==========
        # 随机选择另一个时间窗口的observed_mask作为历史模式
        # 这样可以让模型学习不同时间段的缺失模式
        max_hist_start = len(self.observed_data) - self.eval_length + 1
        hist_start_idx = int(rng.integers(0, max_hist_start))
        hist_mask = self.observed_mask[hist_start_idx:hist_start_idx + self.eval_length]

        # ========== CSDI补值方法：在已观测数据上随机创建缺失 ==========
        # target_mask: 用于训练的掩码（在observed_mask基础上人工mask）
        target_mask = observed_mask.copy()

        # 只在训练模式下进行人工mask
        if self.mode == "train" and self.missing_ratio > 0:
            # 找到所有已观测的位置
            observed_positions = np.where(observed_mask == 1)

            if len(observed_positions[0]) > 0:
                # 随机选择 missing_ratio 比例的观测点进行mask
                n_mask = int(len(observed_positions[0]) * self.missing_ratio)
                if n_mask > 0:
                    # 随机选择要mask的位置（可复现）
                    mask_indices = rng.choice(
                        len(observed_positions[0]),
                        size=n_mask,
                        replace=False
                    )
                    # 将选中的位置设为0（mask掉）
                    #（向量化赋值，功能不变）
                    t_idx = observed_positions[0][mask_indices]
                    d_idx = observed_positions[1][mask_indices]
                    target_mask[t_idx, d_idx] = 0

        # 构建样本字典
        s = {
            "observed_data": observed_data.astype(np.float32),
            "observed_mask": observed_mask.astype(np.float32),
            "target_mask": target_mask.astype(np.float32),
            "gt_mask": gt_mask.astype(np.float32),
            "hist_mask": hist_mask.astype(np.float32),  # 添加hist_mask字段
            "timepoints": np.arange(self.eval_length).astype(np.float32),
            "cut_length": 0,  # 不再需要cut_length
        }

        return s

    def __len__(self):
        return len(self.use_indices)


def get_dataloader(batch_size, device, missing_ratio=0.1, split_ratio=0.5,
                   train_ratio=0.7, valid_ratio=0.15, seed=42):
    """
    获取数据加载器

    参数:
        batch_size: 批次大小
        device: 设备（CPU或GPU）
        missing_ratio: 训练时人工mask的缺失比例（CSDI方法）
        split_ratio: 数据集分割比例（前split_ratio用于补值）
        train_ratio: 训练集比例（在前半部分数据中）
        valid_ratio: 验证集比例（在前半部分数据中）
        seed: 随机种子

    返回:
        train_loader: 训练数据加载器
        valid_loader: 验证数据加载器
        test_loader: 测试数据加载器
        scaler: 标准差（用于反标准化）
        mean_scaler: 均值（用于反标准化）
    """
    print("=" * 80)
    print("创建数据加载器")
    print("=" * 80)

    seed = int(seed)

    # 设置全局随机种子（新增，不减少原功能）
    set_global_seed(seed)

    # 为 DataLoader 的 shuffle 提供固定 generator（新增，不减少原功能）
    g = torch.Generator()
    g.manual_seed(seed)

    # worker 初始化函数（新增，不减少原功能）
    worker_init_fn = seed_worker_factory(seed)

    # 创建训练数据集（使用人工mask）
    dataset = PM25_Imputation_Dataset(
        mode="train",
        missing_ratio=missing_ratio,
        split_ratio=split_ratio,
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
        seed=seed
    )
    train_loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=1, shuffle=True,
        generator=g, worker_init_fn=worker_init_fn
    )

    # 创建验证数据集（不使用人工mask）
    dataset_valid = PM25_Imputation_Dataset(
        mode="valid",
        missing_ratio=0.0,  # 验证时不额外mask
        split_ratio=split_ratio,
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
        seed=seed
    )
    valid_loader = DataLoader(
        dataset_valid, batch_size=batch_size, num_workers=1, shuffle=False,
        generator=g, worker_init_fn=worker_init_fn
    )

    # 创建测试数据集（不使用人工mask）
    dataset_test = PM25_Imputation_Dataset(
        mode="test",
        missing_ratio=0.0,  # 测试时不额外mask
        split_ratio=split_ratio,
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
        seed=seed
    )
    test_loader = DataLoader(
        dataset_test, batch_size=batch_size, num_workers=1, shuffle=False,
        generator=g, worker_init_fn=worker_init_fn
    )

    # 获取标准化参数
    scaler = torch.from_numpy(dataset.train_std).to(device).float()
    mean_scaler = torch.from_numpy(dataset.train_mean).to(device).float()

    print("\n" + "=" * 80)
    print("数据加载器创建完成")
    print("=" * 80)

    return train_loader, valid_loader, test_loader, scaler, mean_scaler


# ========== 使用示例 ==========
if __name__ == "__main__":
    import torch

    print("\n" + "=" * 80)
    print("PM2.5 补值数据集测试")
    print("=" * 80 + "\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}\n")

    # 创建数据加载器
    train_loader, valid_loader, test_loader, scaler, mean_scaler = get_dataloader(
        batch_size=16,
        device=device,
        missing_ratio=0.1,    # 训练时人工mask 10%的观测值
        split_ratio=0.5,      # 前50%用于补值
        train_ratio=0.7,      # 训练集占前半部分的70%
        valid_ratio=0.15,     # 验证集占前半部分的15%
        seed=42               # 随机种子
    )

    print(f"\n数据集大小:")
    print(f"  训练集: {len(train_loader.dataset)} 个样本")
    print(f"  验证集: {len(valid_loader.dataset)} 个样本")
    print(f"  测试集: {len(test_loader.dataset)} 个样本")
    print(f"  总计:   {len(train_loader.dataset) + len(valid_loader.dataset) + len(test_loader.dataset)} 个样本")

    # 查看一个batch的数据
    print("\n" + "=" * 80)
    print("训练集样本示例")
    print("=" * 80)

    for batch in train_loader:
        print(f"\n数据形状:")
        print(f"  observed_data: {batch['observed_data'].shape}")
        print(f"  observed_mask: {batch['observed_mask'].shape}")
        print(f"  target_mask:   {batch['target_mask'].shape}")
        print(f"  gt_mask:       {batch['gt_mask'].shape}")

        # 统计mask的差异
        obs_sum = batch['observed_mask'].sum()
        target_sum = batch['target_mask'].sum()
        print(f"\nMask统计:")
        print(f"  observed_mask中观测点数: {obs_sum.item():.0f}")
        print(f"  target_mask中观测点数:   {target_sum.item():.0f}")
        print(f"  被人工mask的点数:        {(obs_sum - target_sum).item():.0f}")
        print(f"  人工mask比例:            {(obs_sum - target_sum).item() / obs_sum.item() * 100:.2f}%")

        break

    print("\n" + "=" * 80)
    print("测试完成")
    print("=" * 80)
