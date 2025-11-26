import numpy as np
import torch
from torch.utils.data import Dataset

class AccDenoiseDataset(Dataset):
    def __init__(self, path, noise_std=0.1):
        """
        path: path to the TXT file (e.g. body_acc_x_train.txt)
        noise_std: standard deviation of Gaussian noise added to input
        """
        data = np.loadtxt(path)  # shape: (n_samples, seq_len)
        self.data = data.astype(np.float32)

        # Normalize per entire file (simple and effective)
        mean = self.data.mean()
        std = self.data.std()
        self.data = (self.data - mean) / std
        self.mean = mean
        self.std = std

        self.noise_std = noise_std

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        clean = self.data[idx]                         # shape (seq_len,)
        noise = np.random.normal(0, self.noise_std, clean.shape)
        noisy = clean + noise

        # Convert to PyTorch tensors, shape (seq_len, 1)
        clean = torch.tensor(clean, dtype=torch.float32).unsqueeze(-1)
        noisy = torch.tensor(noisy, dtype=torch.float32).unsqueeze(-1)

        return noisy, clean