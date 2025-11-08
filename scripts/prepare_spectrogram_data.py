import torch
import torchaudio
import os
import matplotlib.pyplot as plt

os.makedirs("dataset/spectrograms", exist_ok=True)

# Configuration
n_fft = 1024
hop_length = 256
win_length = 1024

def audio_to_spectrogram(audio_path):
    """Convert audio to normalized amplitude and phase spectrograms."""
    # Load audio
    waveform, sample_rate = torchaudio.load(audio_path)
    
    # Compute STFT
    stft = torch.stft(
        waveform.squeeze(0),
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=torch.hann_window(win_length),
        return_complex=True
    )
    
    # Extract amplitude and phase
    amplitude = torch.abs(stft)
    phase = torch.angle(stft)
    
    # Normalize amplitude to [0, 1]
    # Use log scale for better dynamic range (???)
    amplitude_log = torch.log(amplitude + 1e-8)  # Add small value to avoid log(0)
    amplitude_normalized = (amplitude_log - amplitude_log.min()) / (amplitude_log.max() - amplitude_log.min())
    
    # Normalize phase from [-π, π] to [0, 1]
    phase_normalized = (phase + torch.pi) / (2 * torch.pi)
    
    return amplitude_normalized, phase_normalized, amplitude, phase

# Test on one file
audio_path = "dataset/LJSpeech-1.1/wavs/LJ001-0001.wav"
amp_norm, phase_norm, amp_orig, phase_orig = audio_to_spectrogram(audio_path)

print("=" * 60)
print("NORMALIZED SPECTROGRAMS:")
print("=" * 60)
print(f"Amplitude normalized shape: {amp_norm.shape}")
print(f"Amplitude normalized range: [{amp_norm.min():.4f}, {amp_norm.max():.4f}]")
print(f"\nPhase normalized shape: {phase_norm.shape}")
print(f"Phase normalized range: [{phase_norm.min():.4f}, {phase_norm.max():.4f}]")
print("=" * 60)

# Stack as 2-channel "image" [2, H, W]
spectrogram_2ch = torch.stack([amp_norm, phase_norm], dim=0)
print(f"\n2-channel spectrogram shape: {spectrogram_2ch.shape}")
print("This is like an 'image' with 2 channels instead of RGB's 3!")

# Visualize
fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# Original amplitude (dB)
amp_db = 20 * torch.log10(amp_orig + 1e-8)
im1 = axes[0, 0].imshow(amp_db.numpy(), aspect='auto', origin='lower', cmap='viridis')
axes[0, 0].set_title('Original Amplitude (dB)')
axes[0, 0].set_ylabel('Frequency Bin')
plt.colorbar(im1, ax=axes[0, 0])

# Normalized amplitude [0,1]
im2 = axes[0, 1].imshow(amp_norm.numpy(), aspect='auto', origin='lower', cmap='viridis')
axes[0, 1].set_title('Normalized Amplitude [0,1]')
plt.colorbar(im2, ax=axes[0, 1])

# Original phase
im3 = axes[1, 0].imshow(phase_orig.numpy(), aspect='auto', origin='lower', cmap='twilight')
axes[1, 0].set_title('Original Phase [-π, π]')
axes[1, 0].set_ylabel('Frequency Bin')
axes[1, 0].set_xlabel('Time Frame')
plt.colorbar(im3, ax=axes[1, 0])

# Normalized phase [0,1]
im4 = axes[1, 1].imshow(phase_norm.numpy(), aspect='auto', origin='lower', cmap='twilight')
axes[1, 1].set_title('Normalized Phase [0,1]')
axes[1, 1].set_xlabel('Time Frame')
plt.colorbar(im4, ax=axes[1, 1])

plt.tight_layout()
plt.savefig("spectrogram_outputs/normalized_spectrogram.png", dpi=150)
print(f"\nSaved visualization to: spectrogram_outputs/normalized_spectrogram.png")

# Save the 2-channel spectrogram
torch.save({
    'spectrogram_2ch': spectrogram_2ch,
    'amplitude_normalized': amp_norm,
    'phase_normalized': phase_norm,
    'amplitude_original': amp_orig,
    'phase_original': phase_orig,
    'n_fft': n_fft,
    'hop_length': hop_length,
    'win_length': win_length,
    'audio_path': audio_path
}, 'dataset/spectrograms/LJ001-0001.pt')
print(f"Saved 2-channel spectrogram to: dataset/spectrograms/LJ001-0001.pt")