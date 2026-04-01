"""
EEG Dataset Loader for CSDI-RDE-GPR
"""
import os
import numpy as np
import pandas as pd
import json

class EEG_Dataset:
    def __init__(self, data_path, missing_ratio=0.1, seed=42, missing_mode='random', use_generated_missing=False):
        self.seed = seed
        self.missing_ratio = missing_ratio
        np.random.seed(seed)

        df = pd.read_excel(data_path, header=None)
        self.data = df.values.astype(np.float32)
        self.n_timesteps, self.n_features = self.data.shape

        print(f"EEG data shape: {self.data.shape}")
        print(f"Data range: [{self.data.min():.2f}, {self.data.max():.2f}]")

        self.train_mean = self.data[:int(len(self.data)*0.5)].mean(axis=0)
        self.train_std = self.data[:int(len(self.data)*0.5)].std(axis=0) + 1e-6

        self.normalized_data = ((self.data - self.train_mean) / self.train_std).astype(np.float32)

        self.gt_values = self.data.copy()

        if use_generated_missing:
            missing_file = f"./data/eeg/eeg_random_ratio{missing_ratio}_seed{seed}.npy"
            mask_file = f"./data/eeg/eeg_random_ratio{missing_ratio}_seed{seed}_mask.npy"
            if os.path.exists(missing_file):
                raw_observed = np.load(missing_file)
                masks = (raw_observed != -9999).astype(float)
                raw_observed_clean = raw_observed.copy()
                raw_observed_clean[raw_observed == -9999.0] = 0
                normalized = ((raw_observed_clean - self.train_mean) / self.train_std).astype(np.float32)
                self.observed_data = normalized * masks
                self.observed_masks = masks
            else:
                self.observed_data = self.normalized_data.copy()
                self.observed_masks = np.ones_like(self.data)
        else:
            obs_data, obs_masks = self._generate_missing(missing_ratio, missing_mode)
            obs_data_norm = ((obs_data - self.train_mean) / self.train_std).astype(np.float32)
            obs_data_norm = obs_data_norm * obs_masks
            self.observed_data = obs_data_norm
            self.observed_masks = obs_masks

        self.gt_masks = self.observed_masks.copy()

        missing_mask = (self.observed_masks == 0)
        print(f"Missing ratio: {missing_mask.mean():.4f}")

        self.eval_length = 100
        self.use_index = np.arange(len(self.observed_data) - self.eval_length + 1)

    def __len__(self):
        return len(self.use_index)

    def __getitem__(self, org_index):
        index = self.use_index[org_index]
        s = {
            "observed_data": self.observed_data[index:index + self.eval_length],
            "observed_mask": self.observed_masks[index:index + self.eval_length],
            "gt_mask": self.gt_masks[index:index + self.eval_length],
            "gt_data": self.normalized_data[index:index + self.eval_length],
            "timepoints": np.arange(self.eval_length),
        }
        return s

    def _generate_random_missing(self, missing_ratio):
        observed = self.data.copy()
        masks = np.ones_like(observed, dtype=float)

        n_total = observed.size
        n_missing = int(n_total * missing_ratio)

        all_indices = np.arange(n_total)
        np.random.seed(self.seed)
        np.random.shuffle(all_indices)
        missing_indices = all_indices[:n_missing]

        rows = missing_indices // self.n_features
        cols = missing_indices % self.n_features

        observed[rows, cols] = -9999.0
        masks[rows, cols] = 0

        return observed, masks

    def _generate_uniform_missing(self, missing_ratio):
        observed = self.data.copy()
        masks = np.ones_like(observed, dtype=float)
        interval = max(1, int(1.0 / missing_ratio))

        for i in range(0, self.n_timesteps, interval):
            if i < self.n_timesteps:
                masks[i, :] = 0
                observed[i, :] = -9999.0

        return observed, masks

    def _generate_missing(self, missing_ratio, mode):
        if mode == 'random':
            return self._generate_random_missing(missing_ratio)
        elif mode == 'uniform':
            return self._generate_uniform_missing(missing_ratio)
        else:
            raise ValueError(f"Unknown missing mode: {mode}")