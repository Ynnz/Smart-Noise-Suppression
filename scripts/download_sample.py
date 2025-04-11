import torchaudio
import os

os.makedirs("data", exist_ok=True)

waveform, sample_rate = torchaudio.load(
    torchaudio.utils.download_asset("tutorial-assets/steam-train-whistle-daniel_simon.wav")
)

print(f"Sample loaded: shape = {waveform.shape}, sr = {sample_rate}")

# Save the waveform using torchaudio instead of soundfile
torchaudio.save("data/test.wav", waveform, sample_rate)

print("Saved as data/test.wav ✅ (via torchaudio)")