import torch
import torchaudio
import matplotlib.pyplot as plt
import os

os.makedirs("spectrogram_outputs", exist_ok=True)

# Configuration
audio_path = "dataset/LJSpeech-1.1/wavs/LJ001-0001.wav"
n_fft = 1024
hop_length = 256
win_length = 1024

# Load original audio
waveform_original, sample_rate = torchaudio.load(audio_path)
print(f"Original audio shape: {waveform_original.shape}")

# Forward: Audio → Spectrogram (Amplitude + Phase)
stft = torch.stft(
    waveform_original.squeeze(0),
    n_fft=n_fft,
    hop_length=hop_length,
    win_length=win_length,
    window=torch.hann_window(win_length),
    return_complex=True
)

amplitude = torch.abs(stft)
phase = torch.angle(stft)

print(f"Amplitude shape: {amplitude.shape}")
print(f"Phase shape: {phase.shape}")

# Backward: Spectrogram (Amplitude + Phase) → Audio
# Reconstruct complex STFT from amplitude and phase
stft_reconstructed = amplitude * torch.exp(1j * phase)

waveform_reconstructed = torch.istft(
    stft_reconstructed,
    n_fft=n_fft,
    hop_length=hop_length,
    win_length=win_length,
    window=torch.hann_window(win_length),
    length=waveform_original.shape[1]  # Match original length
)

print(f"Reconstructed audio shape: {waveform_reconstructed.shape}")

# Calculate reconstruction error
mse = torch.mean((waveform_original.squeeze() - waveform_reconstructed) ** 2)
print(f"\nReconstruction MSE: {mse.item():.10f}")
print(f"This should be VERY close to 0!")

# Save reconstructed audio
torchaudio.save(
    "spectrogram_outputs/reconstructed.wav",
    waveform_reconstructed.unsqueeze(0),
    sample_rate
)
print(f"\nSaved reconstructed audio to: spectrogram_outputs/reconstructed.wav")

# Plot comparison
fig, axes = plt.subplots(3, 1, figsize=(12, 8))

# Original waveform
axes[0].plot(waveform_original.squeeze().numpy())
axes[0].set_title("Original Audio")
axes[0].set_ylabel("Amplitude")
axes[0].grid(True)

# Reconstructed waveform
axes[1].plot(waveform_reconstructed.numpy())
axes[1].set_title("Reconstructed Audio")
axes[1].set_ylabel("Amplitude")
axes[1].grid(True)

# Difference (error)
difference = waveform_original.squeeze() - waveform_reconstructed
axes[2].plot(difference.numpy())
axes[2].set_title("Difference (Error)")
axes[2].set_ylabel("Amplitude")
axes[2].set_xlabel("Sample")
axes[2].grid(True)

plt.tight_layout()
plt.savefig("spectrogram_outputs/reconstruction_comparison.png")
print("Saved comparison plot!")