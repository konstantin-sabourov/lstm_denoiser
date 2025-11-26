import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path

from dataset import AccDenoiseDataset
from model import LSTMDenoiser

def main():
    # --- Config ---
    data_path = Path("UCI HAR Dataset/train/Inertial Signals/body_acc_x_train.txt")
    noise_std = 0.2
    batch_size = 64
    num_epochs = 5
    lr = 1e-3
    hidden_size = 64

    # NOTE: No CUDA option for HW acceleration on macOS, using MPS backend.
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)

    # --- Dataset & loader ---
    dataset = AccDenoiseDataset(data_path, noise_std=noise_std)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # --- Model, loss, optimizer ---
    model = LSTMDenoiser(input_size=1, hidden_size=hidden_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # --- Training loop ---
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for noisy, clean in dataloader:
            noisy = noisy.to(device)   # (batch, seq_len, 1)
            clean = clean.to(device)   # (batch, seq_len, 1)

            optimizer.zero_grad()

            output = model(noisy)      # (batch, seq_len, 1)
            loss = criterion(output, clean)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * noisy.size(0)

        epoch_loss = running_loss / len(dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.6f}")

    # --- Quick visual sanity check on a single sequence ---
    model.eval()
    with torch.no_grad():
        noisy, clean = dataset[0]                    # one example, on CPU
        noisy_batch = noisy.unsqueeze(0).to(device)  # (1, seq_len, 1)
        denoised_batch = model(noisy_batch)
        denoised = denoised_batch.squeeze(0).cpu()   # (seq_len, 1) -> (seq_len, 1)

    clean_np = clean.squeeze().numpy()
    noisy_np = noisy.squeeze().numpy()
    denoised_np = denoised.squeeze().numpy()

    plt.figure(figsize=(10,5))
    plt.plot(clean_np, label="clean")
    plt.plot(noisy_np, label="noisy", alpha=0.6)
    plt.plot(denoised_np, label="denoised", alpha=0.8)
    plt.legend()
    plt.title("LSTM Denoiser - Example Sequence")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()