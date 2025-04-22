import numpy as np
import librosa
import soundfile as sf
import os

INPUT_PATH = "data/train_noisy.npy"
OUTPUT_PATH = "output/train_noisy.wav"
SAMPLE_RATE = 16000

# Load log-magnitude spectrogram
log_mag = np.load(INPUT_PATH)

# Convert from dB to linear magnitude
mag = librosa.db_to_amplitude(log_mag)

# Reconstruct waveform using Griffin-Lim
wav = librosa.griffinlim(mag, n_iter=32, hop_length=128, win_length=512)

# Ensure output directory exists
os.makedirs("output", exist_ok=True)

# Save as WAV
sf.write(OUTPUT_PATH, wav, SAMPLE_RATE)
print(f"✅ Saved reconstructed waveform to {OUTPUT_PATH}")