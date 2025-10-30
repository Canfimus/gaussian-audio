import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs("spectrogram_outputs", exist_ok=True)

# Configuration
audio_path = "dataset/LJSpeech-1.1/wavs/LJ001-0001.wav"
n_fft = 2048
win_length = n_fft # Typically equal to n_fft
hop_length = 256

# Load audio
waveform, sample_rate = torchaudio.load(audio_path)
print(f"Loaded audio: {waveform.shape}, {sample_rate} Hz")

# Compute STFT (Short-Time Fourier Transform)
# This gives us complex numbers (real + imaginary)
stft = torch.stft(
    waveform.squeeze(0),
    n_fft=n_fft,
    hop_length=hop_length,
    win_length=win_length,
    window=torch.hann_window(win_length),
    return_complex=True
)

print(f"\nSTFT shape: {stft.shape}")  # Should be [freq_bins, time_frames]
print(f"STFT dtype: {stft.dtype}")    # Should be complex

# Extract amplitude and phase
amplitude = torch.abs(stft)  # Magnitude
phase = torch.angle(stft)     # Phase in radians [-π, π]

print(f"\nAmplitude shape: {amplitude.shape}")
print(f"Amplitude range: [{amplitude.min():.4f}, {amplitude.max():.4f}]")
print(f"\nPhase shape: {phase.shape}")
print(f"Phase range: [{phase.min():.4f}, {phase.max():.4f}] radians")

# Visualize
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Plot amplitude (in dB scale - more interpretable)
amplitude_db = 20 * torch.log10(amplitude + 1e-8)  # Add small value to avoid log(0)
im1 = axes[0].imshow(amplitude_db.numpy(), aspect='auto', origin='lower', cmap='viridis')
axes[0].set_title('Amplitude Spectrogram (dB)')
axes[0].set_ylabel('Frequency Bin')
axes[0].set_xlabel('Time Frame')
plt.colorbar(im1, ax=axes[0])

# Plot phase
im2 = axes[1].imshow(phase.numpy(), aspect='auto', origin='lower', cmap='twilight')
axes[1].set_title('Phase Spectrogram (radians)')
axes[1].set_ylabel('Frequency Bin')
axes[1].set_xlabel('Time Frame')
plt.colorbar(im2, ax=axes[1])

plt.tight_layout()
plt.savefig("spectrogram_outputs/test_spectrogram.png")
print(f"\nSaved spectrogram visualization!")