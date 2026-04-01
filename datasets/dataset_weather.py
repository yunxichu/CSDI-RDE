import pickle
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class Weather_Dataset(Dataset):
    def __init__(self, eval_length=36, target_dim=21, mode="train", missing_ratio=0.1, 
                 seed=0, missing_mode="random", use_generated_missing=True):
        """
        Args:
            eval_length: 时间窗口长度
            target_dim: 特征数量
            mode: train/valid/test
            missing_ratio: 缺失率
            seed: 随机种子
            missing_mode: 缺失模式 (uniform/random/temporal/feature)
            use_generated_missing: 是否使用预生成的缺失数据文件
        """
        self.eval_length = eval_length
        self.target_dim = target_dim
        self.missing_mode = missing_mode
        np.random.seed(seed)
        
        ground_path = "./data/weather/weather_ground.npy"
        if os.path.exists(ground_path):
            ground_data = np.load(ground_path)
        else:
            ground_data = np.load("./data/weather/weather.npy")
        
        missing_value = -9999.0
        ground_masks = (ground_data != missing_value).astype(np.float32)
        ground_values = np.where(ground_data == missing_value, 0, ground_data).astype(np.float32)
        
        if use_generated_missing:
            missing_filename = f"./data/weather/weather_missing_{missing_mode}_ratio{missing_ratio}_seed{seed}.npy"
            mask_filename = f"./data/weather/weather_mask_{missing_mode}_ratio{missing_ratio}_seed{seed}.npy"
            
            if os.path.exists(missing_filename):
                missing_data = np.load(missing_filename)
                observed_masks = (missing_data != missing_value).astype(np.float32)
                observed_values = np.where(missing_data == missing_value, 0, missing_data).astype(np.float32)
                
                if os.path.exists(mask_filename):
                    gt_masks = np.load(mask_filename)
                else:
                    gt_masks = observed_masks.copy()
            else:
                print(f"警告: 缺失数据文件不存在 {missing_filename}，使用原始数据")
                observed_masks = ground_masks.copy()
                observed_values = ground_values.copy()
                gt_masks = ground_masks.copy()
        else:
            observed_masks = ground_masks.copy()
            observed_values = ground_values.copy()
            gt_masks = ground_masks.copy()
            
            if missing_ratio > 0:
                masks = observed_masks.reshape(-1).copy()
                obs_indices = np.where(masks == 1)[0]
                n_missing = int(len(obs_indices) * missing_ratio)
                miss_indices = np.random.choice(obs_indices, n_missing, replace=False)
                masks[miss_indices] = 0
                gt_masks = masks.reshape(observed_masks.shape).astype(np.float32)
        
        mean_path = "./data/weather/weather_meanstd.pk"
        if not os.path.isfile(mean_path):
            tmp_values = ground_values.reshape(-1, target_dim)
            tmp_masks = ground_masks.reshape(-1, target_dim)
            mean = np.zeros(target_dim)
            std = np.zeros(target_dim)
            for k in range(target_dim):
                c_data = tmp_values[:, k][tmp_masks[:, k] == 1]
                if len(c_data) > 0:
                    mean[k] = c_data.mean()
                    std[k] = c_data.std()
                else:
                    mean[k] = 0.0
                    std[k] = 1.0
            std = np.where(std == 0, 1.0, std)
            with open(mean_path, "wb") as f:
                pickle.dump([mean, std], f)
        else:
            with open(mean_path, "rb") as f:
                mean, std = pickle.load(f)
        
        self.train_mean = mean.astype(np.float32)
        self.train_std = std.astype(np.float32)
        
        normalized_values = ((observed_values - mean) / std) * observed_masks
        
        total_len = len(ground_data)
        if mode == "train":
            train_end = int(total_len * 0.7)
            self.observed_data = normalized_values[:train_end]
            self.observed_masks = observed_masks[:train_end]
            self.gt_masks = gt_masks[:train_end]
            self.gt_values = ground_values[:train_end]
        elif mode == "valid":
            train_end = int(total_len * 0.7)
            valid_end = int(total_len * 0.85)
            self.observed_data = normalized_values[train_end:valid_end]
            self.observed_masks = observed_masks[train_end:valid_end]
            self.gt_masks = gt_masks[train_end:valid_end]
            self.gt_values = ground_values[train_end:valid_end]
        else:
            valid_end = int(total_len * 0.85)
            self.observed_data = normalized_values[valid_end:]
            self.observed_masks = observed_masks[valid_end:]
            self.gt_masks = gt_masks[valid_end:]
            self.gt_values = ground_values[valid_end:]
        
        self.use_index = np.arange(len(self.observed_data) - eval_length + 1)
    
    def __getitem__(self, org_index):
        index = self.use_index[org_index]
        s = {
            "observed_data": self.observed_data[index:index + self.eval_length],
            "observed_mask": self.observed_masks[index:index + self.eval_length],
            "gt_mask": self.gt_masks[index:index + self.eval_length],
            "gt_data": self.gt_values[index:index + self.eval_length],
            "timepoints": np.arange(self.eval_length),
        }
        return s
    
    def __len__(self):
        return len(self.use_index)


def get_dataloader(batch_size=16, device='cpu', missing_ratio=0.1, seed=0, 
                   missing_mode="random", use_generated_missing=True):
    train_dataset = Weather_Dataset(
        mode="train", 
        missing_ratio=missing_ratio, 
        seed=seed,
        missing_mode=missing_mode,
        use_generated_missing=use_generated_missing
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    
    valid_dataset = Weather_Dataset(
        mode="valid", 
        missing_ratio=missing_ratio, 
        seed=seed,
        missing_mode=missing_mode,
        use_generated_missing=use_generated_missing
    )
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    
    test_dataset = Weather_Dataset(
        mode="test", 
        missing_ratio=missing_ratio, 
        seed=seed,
        missing_mode=missing_mode,
        use_generated_missing=use_generated_missing
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    
    scaler = torch.from_numpy(train_dataset.train_std).to(device).float()
    mean_scaler = torch.from_numpy(train_dataset.train_mean).to(device).float()
    
    return train_loader, valid_loader, test_loader, scaler, mean_scaler
