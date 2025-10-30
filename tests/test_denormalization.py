import torch
import torchaudio
import os

os.makedirs("spectrogram_outputs", exist_ok=True)

# Configuration
n_fft = 1024
hop_length = 256
win_length = 1024

# Load the saved .pt file
data = torch.load('dataset/spectrograms/LJ001-0001.pt')

print("=" * 60)
print("TESTING DENORMALIZATION & RECONSTRUCTION")
print("=" * 60)

# Get normalized data
amp_normalized = data['amplitude_normalized']
phase_normalized = data['phase_normalized']
amp_original = data['amplitude_original']
phase_original = data['phase_original']

print(f"\nNormalized amplitude range: [{amp_normalized.min():.4f}, {amp_normalized.max():.4f}]")
print(f"Normalized phase range: [{phase_normalized.min():.4f}, {phase_normalized.max():.4f}]")

# Denormalize amplitude
# We need to reverse: amp_norm = (log(amp) - min) / (max - min)
amp_log = torch.log(amp_original + 1e-8)
amp_min = amp_log.min()
amp_max = amp_log.max()

# Reverse normalization
amp_log_reconstructed = amp_normalized * (amp_max - amp_min) + amp_min
amp_reconstructed = torch.exp(amp_log_reconstructed) - 1e-8

# Denormalize phase: phase_norm = (phase + π) / (2π)
phase_reconstructed = phase_normalized * (2 * torch.pi) - torch.pi

print(f"\nReconstructed amplitude range: [{amp_reconstructed.min():.4f}, {amp_reconstructed.max():.4f}]")
print(f"Original amplitude range: [{amp_original.min():.4f}, {amp_original.max():.4f}]")

print(f"\nReconstructed phase range: [{phase_reconstructed.min():.4f}, {phase_reconstructed.max():.4f}]")
print(f"Original phase range: [{phase_original.min():.4f}, {phase_original.max():.4f}]")

# Check denormalization accuracy
amp_error = torch.mean((amp_reconstructed - amp_original) ** 2)
phase_error = torch.mean((phase_reconstructed - phase_original) ** 2)

print(f"\nAmplitude denormalization MSE: {amp_error.item():.10f}")
print(f"Phase denormalization MSE: {phase_error.item():.10f}")
print("Both should be very close to 0!")

# Reconstruct audio from denormalized spectrogram
stft_reconstructed = amp_reconstructed * torch.exp(1j * phase_reconstructed)

waveform_reconstructed = torch.istft(
    stft_reconstructed,
    n_fft=n_fft,
    hop_length=hop_length,
    win_length=win_length,
    window=torch.hann_window(win_length)
)

# Load original audio for comparison
waveform_original, sample_rate = torchaudio.load(data['audio_path'])

# Match lengths
min_len = min(waveform_original.shape[1], waveform_reconstructed.shape[0])
waveform_original = waveform_original.squeeze()[:min_len]
waveform_reconstructed = waveform_reconstructed[:min_len]

# Calculate final reconstruction error
audio_mse = torch.mean((waveform_original - waveform_reconstructed) ** 2)
print(f"\nFinal audio reconstruction MSE: {audio_mse.item():.10f}")
print("This should be very close to 0!")

# Save reconstructed audio
torchaudio.save(
    "spectrogram_outputs/reconstructed_from_normalized.wav",
    waveform_reconstructed.unsqueeze(0),
    sample_rate
)

print(f"\n✅ Saved reconstructed audio!")
print("\nCompare these files:")
print(f"  Original: {data['audio_path']}")
print(f"  Reconstructed: spectrogram_outputs/reconstructed_from_normalized.wav")