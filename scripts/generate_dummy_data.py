import numpy as np
import os

# Load clean spectrogram
clean_spec = np.load("data/test_spec.npy")

# Create noisy version (add Gaussian noise)
noise = np.random.normal(0, 5.0, clean_spec.shape)
noisy_spec = clean_spec + noise

# Save both
np.save("data/train_clean.npy", clean_spec)
np.save("data/train_noisy.npy", noisy_spec)

print("✅ Dummy training data created: train_clean.npy and train_noisy.npy")