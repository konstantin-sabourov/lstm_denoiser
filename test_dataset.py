import numpy as np
import matplotlib.pyplot as plt
from dataset import AccDenoiseDataset
from pathlib import Path

path = Path("UCI HAR Dataset/train/Inertial Signals/body_acc_x_train.txt")
ds = AccDenoiseDataset(data_array=np.loadtxt(path), noise_std=0.2, fixed_noise=True, seed=42)

noisy, clean = ds[0]
print("Shapes:", noisy.shape, clean.shape)

print(f"Clean: mean={clean.mean()}, std={clean.std()}")
print(f"Noisy: mean={noisy.mean()}, std={noisy.std()}")

plt.figure(figsize=(8,4))
plt.plot(clean.squeeze().numpy(), label="clean")
plt.plot(noisy.squeeze().numpy(), label="noisy", alpha=0.7)
plt.legend()
plt.title("Example Noisy vs Clean Sequence")
plt.tight_layout()
plt.show()