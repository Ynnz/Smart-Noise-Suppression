import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import sys

# Add project root to import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.basic_denoiser import BasicDenoiser

# Load test data
noisy = np.load("data/train_noisy.npy")
clean = np.load("data/train_clean.npy")

noisy_tensor = torch.tensor(noisy, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

# Load model
model = BasicDenoiser()
model.load_state_dict(torch.load("models/denoiser.pth"))
model.eval()

# Inference
with torch.no_grad():
    denoised_tensor = model(noisy_tensor)

denoised = denoised_tensor.squeeze().numpy()

# Plot comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
titles = ["Noisy Input", "Denoised Output", "Clean Target"]
spectrograms = [noisy, denoised, clean]

for ax, title, spec in zip(axes, titles, spectrograms):
    im = ax.imshow(spec, aspect="auto", origin="lower", cmap="magma")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)

plt.tight_layout()
plt.show()