import torch
import sys
sys.path.insert(0, '.')  # Add current directory to path

from gaussianimage_audio import GaussianImage_Cholesky

# Load our saved spectrogram
data = torch.load('dataset/spectrograms/LJ001-0001.pt')
spectrogram_2ch = data['spectrogram_2ch']  # [2, 513, 832]

print("=" * 60)
print("TESTING GAUSSIANIMAGE WITH 2-CHANNEL AUDIO")
print("=" * 60)
print(f"\nInput spectrogram shape: {spectrogram_2ch.shape}")
print(f"Input range: [{spectrogram_2ch.min():.4f}, {spectrogram_2ch.max():.4f}]")

# Create GaussianImage model
H, W = spectrogram_2ch.shape[1], spectrogram_2ch.shape[2]  # 513, 832
model = GaussianImage_Cholesky(
    loss_type="L2",
    num_points=5000,  # Start with 5000 Gaussians
    H=H,
    W=W,
    BLOCK_W=16,
    BLOCK_H=16,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    quantize=False,
    opt_type="adan",
    lr=1e-3
)

print(f"\nModel created successfully!")
print(f"Device: {model.device}")
print(f"Number of Gaussians: {model.init_num_points}")
print(f"Image size: {H} x {W}")
print(f"Parameters per Gaussian: 2 (pos) + 3 (cov) + 2 (color) = 7")

# Move to device
device = model.device
model = model.to(device)
spectrogram_2ch = spectrogram_2ch.to(device)

# Add batch dimension: [2, H, W] → [1, 2, H, W]
gt_image = spectrogram_2ch.unsqueeze(0)
print(f"\nGround truth shape: {gt_image.shape}")

# Test forward pass
print("\nTesting forward pass...")
try:
    with torch.no_grad():
        output = model.forward()
        rendered = output['render']
    
    print(f"✅ Forward pass successful!")
    print(f"Output shape: {rendered.shape}")
    print(f"Output range: [{rendered.min():.4f}, {rendered.max():.4f}]")
    
    # Check shape matches
    if rendered.shape == gt_image.shape:
        print(f"✅ Output shape matches input shape!")
    else:
        print(f"❌ Shape mismatch! Expected {gt_image.shape}, got {rendered.shape}")
    
except Exception as e:
    print(f"❌ Forward pass failed!")
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

# Test one training iteration
print("\nTesting one training iteration...")
try:
    loss, psnr = model.train_iter(gt_image)
    print(f"✅ Training iteration successful!")
    print(f"Loss: {loss.item():.6f}")
    print(f"PSNR: {psnr:.2f} dB")
except Exception as e:
    print(f"❌ Training failed!")
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("TEST COMPLETE!")
print("=" * 60)