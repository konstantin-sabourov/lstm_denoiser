import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

data_dir = Path("UCI HAR Dataset") / "train" / "Inertial Signals"

# We'll start with body_acc_x_train: X-axis body acceleration, training set
acc_x_path = data_dir / "body_acc_x_train.txt"

acc_x = np.loadtxt(acc_x_path)
print("acc_x shape:", acc_x.shape)

# acc_x should be (n_samples, sequence_length), typically (7352, 128)
n_samples, seq_len = acc_x.shape

# Let's look at the first sequence
seq0 = acc_x[0]

print("First sequence length:", len(seq0))
print("First 10 samples:", seq0[:10])

# Plot the first sequence to see what it looks like
plt.plot(seq0)
plt.title("Body Acc X - Sample 0")
plt.xlabel("Time step")
plt.ylabel("Acceleration (normalized-ish)")
plt.tight_layout()
plt.show()