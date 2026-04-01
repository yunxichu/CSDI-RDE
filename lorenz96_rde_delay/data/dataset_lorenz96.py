import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def lorenz96_generate(N=100, T=400, dt=0.01, forcing=8.0, burn_in=1000, seed=None):
    if seed is not None:
        np.random.seed(seed)
    # 初始条件：forcing + 小随机扰动打破平衡
    x = np.ones(N) * forcing + np.random.randn(N) * 0.01
    for _ in range(burn_in):
        x_dot = np.roll(x, -1) * (np.roll(x, -2) - np.roll(x, 1)) - x + forcing
        x = x + dt * x_dot

    history = [x.copy()]
    for _ in range(T - 1):
        x_dot = np.roll(x, -1) * (np.roll(x, -2) - np.roll(x, 1)) - x + forcing
        x = x + dt * x_dot
        history.append(x.copy())

    return np.array(history)


def lorenz96_sparse_sample(data, sample_step=8):
    sparse_data = data[::sample_step]
    mask = np.zeros_like(data)
    mask[::sample_step] = 1
    return sparse_data, mask


class Lorenz96Dataset(Dataset):
    def __init__(self, N=100, T=400, sample_step=8, mode='train', seed=42):
        np.random.seed(seed)
        self.N = N
        self.T = T
        self.sample_step = sample_step
        self.eval_length = 100
        
        # 生成多个序列，每个序列使用不同的seed以增加多样性
        # 训练序列长度=100，与推理时一致
        self.all_data = []
        for i in range(200):
            # 生成足够长的数据：100个稀疏点需要 100*8=800 时间点
            data = lorenz96_generate(N=N, T=850, forcing=8.0, seed=seed + i)
            # 与inference一致：从t=4开始采样，每8步取一个点
            sparse = data[4::sample_step][:self.eval_length]
            self.all_data.append(sparse)
        self.all_data = np.array(self.all_data)
        
        n_train = int(0.7 * len(self.all_data))
        n_valid = int(0.15 * len(self.all_data))
        
        if mode == 'train':
            self.data = self.all_data[:n_train]
        elif mode == 'valid':
            self.data = self.all_data[n_train:n_train+n_valid]
        else:
            self.data = self.all_data[n_train+n_valid:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        observed_data = self.data[idx]
        
        # CSDI期望: gt_mask标记已知位置(用于条件), target_mask = observed_mask - gt_mask标记要预测的位置
        # 与原版Lorenz一致：从末尾向前标记奇数位置
        # 序列长度=100，标记索引99,97,95,...,1（共50个已知位置）
        observed_mask = np.ones((self.eval_length, self.N))
        gt_mask = np.zeros((self.eval_length, self.N))
        gt_mask[-1::-2, :] = 1.0  # 从末尾向前标记（索引99,97,95...）
        # 偶数位会被预测 (target_mask = observed_mask - gt_mask)
        
        return {
            'observed_data': torch.FloatTensor(observed_data),
            'observed_mask': torch.FloatTensor(observed_mask),
            'gt_mask': torch.FloatTensor(gt_mask),
            'timepoints': torch.arange(self.eval_length).float()
        }


def get_dataloader(N=100, T=400, sample_step=8, batch_size=32, seed=42):
    train_dataset = Lorenz96Dataset(N=N, T=T, sample_step=sample_step, mode='train', seed=seed)
    valid_dataset = Lorenz96Dataset(N=N, T=T, sample_step=sample_step, mode='valid', seed=seed)
    test_dataset = Lorenz96Dataset(N=N, T=T, sample_step=sample_step, mode='test', seed=seed)

    def collate_fn(batch):
        return {
            'observed_data': torch.stack([b['observed_data'].squeeze(0) for b in batch]),
            'observed_mask': torch.stack([b['observed_mask'].squeeze(0) for b in batch]),
            'gt_mask': torch.stack([b['gt_mask'].squeeze(0) for b in batch]),
            'timepoints': torch.stack([b['timepoints'] for b in batch])
        }

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, valid_loader, test_loader


if __name__ == '__main__':
    data = lorenz96_generate(N=100, T=400)
    print(f"Generated data shape: {data.shape}")

    sparse, mask = lorenz96_sparse_sample(data, sample_step=8)
    print(f"Sparse data shape: {sparse.shape}")

    dataset = Lorenz96Dataset(N=100, T=400, sample_step=8, mode='train')
    print(f"Dataset length: {len(dataset)}")

    sample = dataset[0]
    print(f"Sample observed_data shape: {sample['observed_data'].shape}")
    print(f"Sample gt_data shape: {sample['gt_data'].shape}")