import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.basic_denoiser import BasicDenoiser

# Load data
clean = np.load("data/train_clean.npy")
noisy = np.load("data/train_noisy.npy")

# Reshape to (batch, channel, height, width)
clean_tensor = torch.tensor(clean, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
noisy_tensor = torch.tensor(noisy, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

# Model
model = BasicDenoiser()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(500):
    optimizer.zero_grad()
    output = model(noisy_tensor)
    loss = criterion(output, clean_tensor)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Save model
torch.save(model.state_dict(), "models/denoiser.pth")
print("✅ Model trained and saved.")