import numpy as np
import torch
from torch.utils.data import Dataset

class AccDenoiseDataset(Dataset):
    def __init__(self, data_array, noise_std=0.1, fixed_noise=False, seed=123):
        """
        data_array: np.ndarray of shape (n_samples, seq_len), already loaded
        noise_std: std of Gaussian noise added to input
        fixed_noise: if True, generate noisy samples once and reuse them
        seed: RNG seed for fixed-noise mode
        """
        self.data = data_array.astype(np.float32)

        # Normalize once per dataset
        mean = self.data.mean()
        std = self.data.std()
        self.data = (self.data - mean) / std
        self.mean = mean
        self.std = std

        self.noise_std = noise_std
        self.fixed_noise = fixed_noise

        if fixed_noise:
            rng = np.random.default_rng(seed)
            noise = rng.normal(0, self.noise_std, self.data.shape)
            self.noisy_data = (self.data + noise).astype(np.float32)
        else:
            self.noisy_data = None

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        clean = self.data[idx]  # (seq_len,)

        if self.fixed_noise:
            noisy = self.noisy_data[idx]
        else:
            # maybe add astype(np.float32)
            noise = np.random.normal(0, self.noise_std, clean.shape)
            noisy = clean + noise

        # maybe add dtype=torch.float32
        clean = torch.tensor(clean).unsqueeze(-1)  # (seq_len, 1)
        noisy = torch.tensor(noisy).unsqueeze(-1)  # (seq_len, 1)
        return noisy, clean