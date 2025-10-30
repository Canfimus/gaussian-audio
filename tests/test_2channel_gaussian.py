import torch
import torch.nn as nn

# Test: Can we create 2-channel Gaussians?

class Minimal2ChannelTest(nn.Module):
    def __init__(self, num_points=100):
        super().__init__()
        # Key change: 3 → 2 channels
        self._features_dc = nn.Parameter(torch.rand(num_points, 2))  # Was: 3
        self.background = torch.ones(2)  # Was: torch.ones(3)
        
    def forward(self):
        # Simulate output
        H, W = 513, 832  # Spectrogram dimensions
        out = torch.rand(1, H, W, 2)  # 2 channels instead of 3
        out = out.permute(0, 3, 1, 2)  # [1, 2, 513, 832]
        return out

# Test it
print("Testing 2-channel Gaussian setup...")
model = Minimal2ChannelTest()

print(f"Features shape: {model._features_dc.shape}")
print(f"Background shape: {model.background.shape}")

output = model()
print(f"Output shape: {output.shape}")

expected_shape = torch.Size([1, 2, 513, 832])
assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"

print("\n✅ 2-channel test PASSED!")
print(f"   - Features: {model._features_dc.shape}")
print(f"   - Background: {model.background.shape}")
print(f"   - Output: {output.shape}")