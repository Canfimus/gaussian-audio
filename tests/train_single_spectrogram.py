import sys
sys.path.append('..')  # Add parent directory to path

import torch
from gaussianimage_audio import GaussianImage_Cholesky
import matplotlib.pyplot as plt
import os

os.makedirs("../training_outputs", exist_ok=True)

print("=" * 60)
print("TRAINING ON SINGLE SPECTROGRAM")
print("=" * 60)

# Load the spectrogram we prepared
data = torch.load('../dataset/spectrograms/LJ001-0001.pt')
gt_spectrogram = data['spectrogram_2ch']  # [2, 513, 832]

print(f"\nLoaded spectrogram:")
print(f"  Shape: {gt_spectrogram.shape}")
print(f"  Channel 0 (amplitude) range: [{gt_spectrogram[0].min():.4f}, {gt_spectrogram[0].max():.4f}]")
print(f"  Channel 1 (phase) range: [{gt_spectrogram[1].min():.4f}, {gt_spectrogram[1].max():.4f}]")

# Prepare ground truth: add batch dimension [1, 2, 513, 832]
gt_spectrogram = gt_spectrogram.unsqueeze(0)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
gt_spectrogram = gt_spectrogram.to(device)

# Initialize model
H, W = 513, 832  # Spectrogram dimensions
model = GaussianImage_Cholesky(
    loss_type="L2",
    num_points=1000,  # Start with 1000 Gaussians
    H=H,
    W=W,
    BLOCK_W=16,
    BLOCK_H=16,
    device=device,
    quantize=False,
    opt_type='adan',
    lr=1e-3
).to(device)

print(f"\n✅ Model initialized on {device}")
print(f"   Number of Gaussians: {model.init_num_points}")
print(f"   Image size: {H}x{W}")

# Training loop (just a few iterations to test)
num_iterations = 100
print(f"\n🚀 Starting training for {num_iterations} iterations...")

for i in range(num_iterations):
    loss, psnr = model.train_iter(gt_spectrogram)
    
    if i % 10 == 0:
        print(f"Iter {i:3d}: Loss = {loss.item():.6f}, PSNR = {psnr:.2f} dB")

# Get final reconstruction
with torch.no_grad():
    output = model.forward()
    reconstructed = output['render']  # [1, 2, 513, 832]

print(f"\n✅ Training complete!")
print(f"   Final PSNR: {psnr:.2f} dB")

# Visualize results
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# Ground truth amplitude
axes[0, 0].imshow(gt_spectrogram[0, 0].cpu().numpy(), aspect='auto', origin='lower', cmap='viridis')
axes[0, 0].set_title('GT Amplitude')
axes[0, 0].set_ylabel('Frequency')

# Reconstructed amplitude
axes[0, 1].imshow(reconstructed[0, 0].cpu().numpy(), aspect='auto', origin='lower', cmap='viridis')
axes[0, 1].set_title('Reconstructed Amplitude')

# Amplitude error
amp_error = torch.abs(gt_spectrogram[0, 0] - reconstructed[0, 0])
axes[0, 2].imshow(amp_error.cpu().numpy(), aspect='auto', origin='lower', cmap='hot')
axes[0, 2].set_title(f'Amplitude Error (mean: {amp_error.mean():.4f})')

# Ground truth phase
axes[1, 0].imshow(gt_spectrogram[0, 1].cpu().numpy(), aspect='auto', origin='lower', cmap='twilight')
axes[1, 0].set_title('GT Phase')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_xlabel('Time')

# Reconstructed phase
axes[1, 1].imshow(reconstructed[0, 1].cpu().numpy(), aspect='auto', origin='lower', cmap='twilight')
axes[1, 1].set_title('Reconstructed Phase')
axes[1, 1].set_xlabel('Time')

# Phase error
phase_error = torch.abs(gt_spectrogram[0, 1] - reconstructed[0, 1])
axes[1, 2].imshow(phase_error.cpu().numpy(), aspect='auto', origin='lower', cmap='hot')
axes[1, 2].set_title(f'Phase Error (mean: {phase_error.mean():.4f})')
axes[1, 2].set_xlabel('Time')

plt.tight_layout()
plt.savefig('../training_outputs/training_result_100iters.png', dpi=150)
print(f"\n💾 Saved visualization to: training_outputs/training_result_100iters.png")

# Save the model
torch.save({
    'model_state_dict': model.state_dict(),
    'num_points': model.init_num_points,
    'H': H,
    'W': W,
    'final_psnr': psnr,
}, '../training_outputs/model_100iters.pt')
print(f"💾 Saved model to: training_outputs/model_100iters.pt")

print("\n" + "=" * 60)
print("✅ SINGLE SPECTROGRAM TRAINING TEST COMPLETE!")
print("=" * 60)