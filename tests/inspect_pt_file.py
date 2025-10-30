import torch

# Load the .pt file
data = torch.load('dataset/spectrograms/LJ001-0001.pt')

print("=" * 60)
print("CONTENTS OF .pt FILE:")
print("=" * 60)

# Show what keys are in the dictionary
print("\nKeys in the file:")
for key in data.keys():
    print(f"  - {key}")

print("\n" + "=" * 60)
print("DETAILED INFO:")
print("=" * 60)

# Print details about each item
for key, value in data.items():
    print(f"\n{key}:")
    if isinstance(value, torch.Tensor):
        print(f"  Type: Tensor")
        print(f"  Shape: {value.shape}")
        print(f"  Dtype: {value.dtype}")
        print(f"  Range: [{value.min():.4f}, {value.max():.4f}]")
    else:
        print(f"  Type: {type(value)}")
        print(f"  Value: {value}")

# Verify the 2-channel spectrogram
print("\n" + "=" * 60)
print("VERIFYING 2-CHANNEL SPECTROGRAM:")
print("=" * 60)
spec_2ch = data['spectrogram_2ch']
print(f"Shape: {spec_2ch.shape}")
print(f"  - Channel 0 (amplitude): {spec_2ch[0].shape}")
print(f"  - Channel 1 (phase): {spec_2ch[1].shape}")
print(f"\nChannel 0 range: [{spec_2ch[0].min():.4f}, {spec_2ch[0].max():.4f}]")
print(f"Channel 1 range: [{spec_2ch[1].min():.4f}, {spec_2ch[1].max():.4f}]")