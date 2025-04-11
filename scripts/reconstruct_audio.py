import numpy as np
import librosa
import soundfile as sf
import torch
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.basic_denoiser import BasicDenoiser

# Load noisy input
noisy = np.load("data/train_noisy.npy")
noisy_tensor = torch.tensor(noisy, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

# Load model
model = BasicDenoiser()
model.load_state_dict(torch.load("models/denoiser.pth"))
model.eval()

# Predict denoised spectrogram
with torch.no_grad():
    denoised_tensor = model(noisy_tensor)

denoised = denoised_tensor.squeeze().numpy()

# Convert from dB to amplitude
mag = librosa.db_to_amplitude(denoised)

# Reconstruct waveform using Griffin-Lim
wav = librosa.griffinlim(mag, n_iter=32, hop_length=128, win_length=512)

# Save
os.makedirs("output", exist_ok=True)
sf.write("output/denoised.wav", wav, samplerate=16000)

print("✅ Denoised audio saved to output/denoised.wav")