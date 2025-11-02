import torch
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np

# Import the models
from gaussianimage_cholesky import GaussianImage_Cholesky  # Original 3-channel
from gaussianimage_audio import GaussianImage_Cholesky as GaussianImage_Audio  # Your 2-channel

os.makedirs("visualization_outputs", exist_ok=True)

def visualize_rgb_fitting():
    """Test on RGB image using original code"""
    print("=" * 60)
    print("TESTING RGB IMAGE (3 channels)")
    print("=" * 60)
    
    # Load a Kodak image
    img_path = "dataset/kodak/kodim01.png"
    img = Image.open(img_path).convert('RGB')
    img_tensor = torch.from_numpy(np.array(img)).float() / 255.0  # [H, W, 3]
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
    
    H, W = img_tensor.shape[2], img_tensor.shape[3]
    print(f"Image shape: {img_tensor.shape}")
    
    # Create model
    model = GaussianImage_Cholesky(
        loss_type="L2",
        num_points=5000,  # Start with fewer Gaussians for visualization
        H=H,
        W=W,
        BLOCK_H=16,
        BLOCK_W=16,
        device="cuda",
        quantize=False,
        opt_type="adan",
        lr=1e-3
    ).cuda()
    
    img_tensor = img_tensor.cuda()
    
    # Train for a few iterations and save visualizations
    iterations = [0, 100, 500, 1000, 2000, 5000]
    results = {}
    
    for i in range(max(iterations) + 1):
        loss, psnr = model.train_iter(img_tensor)
        
        if i in iterations:
            with torch.no_grad():
                output = model.forward()
                reconstructed = output["render"].cpu()
                
                results[i] = {
                    'reconstructed': reconstructed,
                    'psnr': psnr,
                    'loss': loss.item(),
                    'gaussian_positions': model.get_xyz.cpu().numpy()
                }
                print(f"Iter {i}: Loss={loss.item():.6f}, PSNR={psnr:.2f}dB")
    
    # Create visualization
    fig, axes = plt.subplots(2, len(iterations), figsize=(20, 7))
    
    for idx, iter_num in enumerate(iterations):
        # Show reconstructed image
        recon = results[iter_num]['reconstructed'][0].permute(1, 2, 0).numpy()
        axes[0, idx].imshow(np.clip(recon, 0, 1))
        axes[0, idx].set_title(f"Iter {iter_num}\nPSNR: {results[iter_num]['psnr']:.2f}dB")
        axes[0, idx].axis('off')
        
        # Show Gaussian positions overlaid on original
        axes[1, idx].imshow(img_tensor[0].cpu().permute(1, 2, 0).numpy())
        positions = results[iter_num]['gaussian_positions']
        # Convert from [-1, 1] to pixel coordinates
        x_coords = (positions[:, 0] + 1) * W / 2
        y_coords = (positions[:, 1] + 1) * H / 2
        axes[1, idx].scatter(x_coords, y_coords, c='red', s=1, alpha=0.3)
        axes[1, idx].set_title(f"{len(positions)} Gaussians")
        axes[1, idx].axis('off')
    
    plt.tight_layout()
    plt.savefig("visualization_outputs/rgb_fitting_process.png", dpi=150)
    print(f"\n✅ Saved RGB visualization to: visualization_outputs/rgb_fitting_process.png")

def visualize_audio_fitting():
    """Test on spectrogram using your modified code"""
    print("\n" + "=" * 60)
    print("TESTING SPECTROGRAM (2 channels)")
    print("=" * 60)
    
    # Load spectrogram
    data = torch.load('dataset/spectrograms/LJ001-0001.pt')
    spec_tensor = data['spectrogram_2ch'].unsqueeze(0)  # [1, 2, H, W]
    
    H, W = spec_tensor.shape[2], spec_tensor.shape[3]
    print(f"Spectrogram shape: {spec_tensor.shape}")
    
    # Create model
    model = GaussianImage_Audio(
        loss_type="L2",
        num_points=5000,
        H=H,
        W=W,
        BLOCK_H=16,
        BLOCK_W=16,
        device="cuda",
        quantize=False,
        opt_type="adan",
        lr=1e-3
    ).cuda()
    
    spec_tensor = spec_tensor.cuda()
    
    # Train
    iterations = [0, 100, 500, 1000, 2000, 5000]
    results = {}
    
    for i in range(max(iterations) + 1):
        loss, psnr = model.train_iter(spec_tensor)
        
        if i in iterations:
            with torch.no_grad():
                output = model.forward()
                reconstructed = output["render"].cpu()
                
                results[i] = {
                    'reconstructed': reconstructed,
                    'psnr': psnr,
                    'loss': loss.item(),
                    'gaussian_positions': model.get_xyz.cpu().numpy()
                }
                print(f"Iter {i}: Loss={loss.item():.6f}, PSNR={psnr:.2f}dB")
    
    # Create visualization
    fig, axes = plt.subplots(3, len(iterations), figsize=(20, 10))
    
    for idx, iter_num in enumerate(iterations):
        recon = results[iter_num]['reconstructed'][0]  # [2, H, W]
        
        # Show amplitude channel
        axes[0, idx].imshow(recon[0].numpy(), aspect='auto', origin='lower', cmap='viridis')
        axes[0, idx].set_title(f"Iter {iter_num} - Amplitude\nPSNR: {results[iter_num]['psnr']:.2f}dB")
        axes[0, idx].axis('off')
        
        # Show phase channel
        axes[1, idx].imshow(recon[1].numpy(), aspect='auto', origin='lower', cmap='twilight')
        axes[1, idx].set_title(f"Phase")
        axes[1, idx].axis('off')
        
        # Show Gaussian positions
        axes[2, idx].imshow(spec_tensor[0, 0].cpu().numpy(), aspect='auto', origin='lower', cmap='gray', alpha=0.5)
        positions = results[iter_num]['gaussian_positions']
        x_coords = (positions[:, 0] + 1) * W / 2
        y_coords = (positions[:, 1] + 1) * H / 2
        axes[2, idx].scatter(x_coords, y_coords, c='red', s=1, alpha=0.3)
        axes[2, idx].set_title(f"{len(positions)} Gaussians")
        axes[2, idx].axis('off')
    
    plt.tight_layout()
    plt.savefig("visualization_outputs/spectrogram_fitting_process.png", dpi=150)
    print(f"\n✅ Saved spectrogram visualization to: visualization_outputs/spectrogram_fitting_process.png")

if __name__ == "__main__":
    # Test RGB first
    visualize_rgb_fitting()
    
    # Then test audio
    visualize_audio_fitting()
    
    print("\n" + "=" * 60)
    print("DONE! Check visualization_outputs/ folder")
    print("=" * 60)