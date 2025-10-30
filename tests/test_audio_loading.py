import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np

# Load one audio file
audio_path = "dataset/LJSpeech-1.1/wavs/LJ001-0001.wav"

# Load audio
waveform, sample_rate = torchaudio.load(audio_path)

print("=" * 50)
print("AUDIO FILE INFO:")
print("=" * 50)
print(f"Sample rate: {sample_rate} Hz")
print(f"Waveform shape: {waveform.shape}")  # Should be [channels, samples]
print(f"Number of channels: {waveform.shape[0]}")
print(f"Number of samples: {waveform.shape[1]}")
print(f"Duration: {waveform.shape[1] / sample_rate:.2f} seconds")
print(f"Min value: {waveform.min():.4f}")
print(f"Max value: {waveform.max():.4f}")
print("=" * 50)

# If stereo, convert to mono (LJSpeech should already be mono)
if waveform.shape[0] > 1:
    waveform = torch.mean(waveform, dim=0, keepdim=True)
    print("Converted stereo to mono")

# Plot the waveform
plt.figure(figsize=(12, 4))
plt.plot(waveform.squeeze().numpy())
plt.title(f"Audio Waveform: {audio_path}")
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.grid(True)
plt.savefig("spectrogram_outputs/test_waveform.png")
print(f"\nSaved waveform plot to: spectrogram_outputs/test_waveform.png")