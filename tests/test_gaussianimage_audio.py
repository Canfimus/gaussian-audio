import torch
from gaussianimage_audio import GaussianImage_Cholesky

print("=" * 60)
print("TESTING GaussianImage_Audio (2-channel)")
print("=" * 60)

# Test initialization
model = GaussianImage_Cholesky(
    loss_type="L2",
    num_points=500,
    H=513,  # Spectrogram height
    W=832,  # Spectrogram width
    BLOCK_W=16,
    BLOCK_H=16,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    quantize=False,
    opt_type='adan',
    lr=1e-3
)

print(f"\n✅ Model initialized successfully!")
print(f"   Device: {model.device}")
print(f"   Features shape: {model._features_dc.shape}")
print(f"   Background shape: {model.background.shape}")
print(f"   Expected: [500, 2] and [2]")

# Test forward pass
try:
    output = model.forward()
    print(f"\n✅ Forward pass successful!")
    print(f"   Output shape: {output['render'].shape}")
    print(f"   Expected: [1, 2, 513, 832]")
    
    assert output['render'].shape == torch.Size([1, 2, 513, 832])
    print(f"\n🎉 ALL TESTS PASSED! GaussianImage_Audio is ready for 2-channel spectrograms!")
    
except Exception as e:
    print(f"\n❌ Error during forward pass:")
    print(f"   {e}")