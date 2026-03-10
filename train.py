import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from dataset import AccDenoiseDataset
from model import LSTMDenoiser

def get_best_device():
    """Get the best available device for PyTorch computation."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    else:
        return torch.device("cpu")



def main():
    # --- Config ---
    file_path = Path("UCI HAR Dataset/train/Inertial Signals/body_acc_x_train.txt")
    noise_std = 0.2
    batch_size = 64
    num_epochs = 20
    lr = 1e-3
    hidden_size = 64
    val_fraction = 0.2

    device = get_best_device()      
    print("Using device:", device)

    # --- Load raw data once ---
    raw_data = np.loadtxt(file_path)  # (n_samples, seq_len)
    n_samples = raw_data.shape[0]
    n_val = int(n_samples * val_fraction)

    # Simple split: last n_val samples as validation
    train_data = raw_data[:-n_val]
    val_data = raw_data[-n_val:]

    # --- Datasets ---
    train_dataset = AccDenoiseDataset(
        train_data,
        noise_std=noise_std,
        fixed_noise=False,   # random noise each epoch
    )
    val_dataset = AccDenoiseDataset(
        val_data,
        noise_std=noise_std,
        fixed_noise=True,    # fixed noisy samples
        seed=42,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # --- Load raw data once ---
    raw_data = np.loadtxt(file_path)  # (n_samples, seq_len)
    n_samples = raw_data.shape[0]
    n_val = int(n_samples * val_fraction)

    # Simple split: last n_val samples as validation
    train_data = raw_data[:-n_val]
    val_data = raw_data[-n_val:]

    # --- Datasets ---
    train_dataset = AccDenoiseDataset(
        train_data,
        noise_std=noise_std,
        fixed_noise=False,   # random noise each epoch
    )
    val_dataset = AccDenoiseDataset(
        val_data,
        noise_std=noise_std,
        fixed_noise=True,    # fixed noisy samples
        seed=42,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

  # --- Model, loss, optimizer ---
    model = LSTMDenoiser(input_size=1, hidden_size=hidden_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []

    # --- Training loop ---
    for epoch in range(num_epochs):
        # Train
        model.train()
        running_train_loss = 0.0
        for noisy, clean in train_loader:
            noisy = noisy.to(device)
            clean = clean.to(device)

            optimizer.zero_grad()
            output = model(noisy)
            loss = criterion(output, clean)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * noisy.size(0)

        epoch_train_loss = running_train_loss / len(train_dataset)

        # Validation
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy = noisy.to(device)
                clean = clean.to(device)
                output = model(noisy)
                loss = criterion(output, clean)
                running_val_loss += loss.item() * noisy.size(0)

        epoch_val_loss = running_val_loss / len(val_dataset)

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)

        print(
            f"Epoch {epoch+1}/{num_epochs} "
            f"- train_loss: {epoch_train_loss:.6f} "
            f"- val_loss: {epoch_val_loss:.6f}"
        )

    # --- Plot loss curves ---
    plt.figure()
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Visual check on a validation sample ---
    model.eval()
    with torch.no_grad():
        noisy, clean = val_dataset[0]                  # CPU tensors
        noisy_batch = noisy.unsqueeze(0).to(device)    # (1, seq_len, 1)
        denoised_batch = model(noisy_batch)
        denoised = denoised_batch.squeeze(0).cpu()     # (seq_len, 1)

    clean_np = clean.squeeze().numpy()
    noisy_np = noisy.squeeze().numpy()
    denoised_np = denoised.squeeze().numpy()

    plt.figure(figsize=(10, 5))
    plt.plot(clean_np, label="clean")
    plt.plot(noisy_np, label="noisy", alpha=0.5)
    plt.plot(denoised_np, label="denoised", alpha=0.8)
    plt.legend()
    plt.title("LSTM Denoiser - Validation Example")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()    
