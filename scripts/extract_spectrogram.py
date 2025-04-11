import librosa
import numpy as np
import matplotlib.pyplot as plt
import os

AUDIO_PATH = "data/test.wav"
SPECTROGRAM_PATH = "data/test_spec.npy"

# Load audio
y, sr = librosa.load(AUDIO_PATH, sr=16000)
print(f"Loaded audio: {y.shape}, Sample rate: {sr}")

# Compute STFT
stft = librosa.stft(y, n_fft=512, hop_length=128, win_length=512)
magnitude = np.abs(stft)

# Convert to log scale (dB)
log_mag = librosa.amplitude_to_db(magnitude)

# Save the spectrogram
np.save(SPECTROGRAM_PATH, log_mag)
print(f"Saved spectrogram: {SPECTROGRAM_PATH}, shape: {log_mag.shape}")

# Plot it
plt.figure(figsize=(10, 4))
librosa.display.specshow(log_mag, sr=sr, hop_length=128, x_axis='time', y_axis='hz')
plt.colorbar(format="%+2.0f dB")
plt.title("Log-magnitude Spectrogram")
plt.tight_layout()
plt.show()