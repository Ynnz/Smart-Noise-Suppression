# 🎧 Smart Noise Suppression

A lightweight AI model that denoises audio spectrograms — simulating a step toward real-world applications in hearing aids and headsets.

---

## 🧠 What It Does

- Trains a simple CNN-based model to suppress noise in audio
- Converts audio into spectrograms (log-magnitude STFT)
- Denoises the spectrogram using a neural network
- Reconstructs the waveform using Griffin-Lim
- Deploys inference using fixed-point–friendly PyTorch

---

## 🛠 Tech Stack

- Python 3.10
- PyTorch, Torchaudio
- Librosa, Soundfile, Matplotlib
- Spectrogram-based audio processing
- Easy to port to C

---

## 📁 Folder Structure

```
smart-noise-suppression/
├── data/            # Audio and spectrogram data
├── models/          # Model definitions
├── output/          # Generated WAV files
├── scripts/         # Preprocessing, training, evaluation
├── export/          # C-compatible exported weights and inputs
├── c_inference/     # C fixed-point model implementation
└── README.md
```

---

## 🚀 How to Run

```bash
# 1. Create environment
conda env create -f environment.yml
conda activate smart-audio-env

# 2. Download a sample and generate spectrogram
python scripts/download_sample.py
python scripts/extract_spectrogram.py

# 3. Generate training data and train
python scripts/generate_dummy_data.py
python scripts/train_model.py

# 4. Evaluate and listen
python scripts/eval_model.py
python scripts/reconstruct_audio.py
```

---

## 🔊 Demo Audio

| Type        | File                       |
|-------------|----------------------------|
| Noisy Input | `data/test.wav`            |
| Denoised    | `output/denoised.wav`      |
| Clean Ref   | `output/reconstructed.wav` |

---

## 💻 C Inference (Fixed-Point)

This project includes a plain C implementation of model inference using fixed-point math (Q7.8 format). The goal is to simulate embedded inference as used in real-time DSP systems like hearing aids or headsets.

### 📂 Folder: `c_inference/`

- `model_infer.c`: Fixed-point conv2D, ReLU
- `model_weights.h`: Exported model weights from PyTorch (Q7.8)