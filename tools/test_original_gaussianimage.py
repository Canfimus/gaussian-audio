#!/usr/bin/env python3
"""
Diagnostic Test for GaussianImage - Understanding Original 3-Channel Code
==========================================================================

This test will help us understand:
1. How the original 3-channel (RGB) version works
2. What functions are being called in gsplat
3. What needs to change for 2-channel (spectrogram) inputs
4. The exact error points when switching to 2 channels

Run this BEFORE making any changes to understand the baseline.
"""

import torch
import torch.nn as nn
import sys
import os

print("="*80)
print("DIAGNOSTIC TEST: GaussianImage Original Code Analysis")
print("="*80)

# ============================================================================
# PART 1: Check gsplat installation and available functions
# ============================================================================
print("\n[PART 1] Checking gsplat installation...\n")

try:
    import gsplat
    print(f"✅ gsplat imported successfully")
    print(f"   Location: {gsplat.__file__}")
except ImportError as e:
    print(f"❌ Failed to import gsplat: {e}")
    sys.exit(1)

# Check what's available in gsplat.cuda (this is what the code imports)
try:
    import gsplat.cuda as _C
    print(f"✅ gsplat.cuda imported as _C")
    
    # List all rasterize functions
    rasterize_funcs = [x for x in dir(_C) if 'rasterize' in x.lower()]
    print(f"\n📋 Available rasterize functions in gsplat.cuda:")
    for func in sorted(rasterize_funcs):
        print(f"   - {func}")
    
    # Check for the specific functions we need
    print(f"\n🔍 Checking for required functions:")
    required_funcs = [
        'rasterize_sum_forward',
        'rasterize_sum_backward', 
        'nd_rasterize_sum_forward',  # This is what the code tries to use
        'nd_rasterize_sum_backward',
    ]
    
    for func in required_funcs:
        if hasattr(_C, func):
            print(f"   ✅ {func} - EXISTS")
        else:
            print(f"   ❌ {func} - MISSING")
            if 'nd_rasterize_sum' in func:
                print(f"      → This is the problem! Code looks for 'nd_rasterize_sum_*' but only 'rasterize_sum_*' exists")
    
except Exception as e:
    print(f"❌ Error checking gsplat.cuda: {e}")

# ============================================================================
# PART 2: Check if we can import the GaussianImage functions
# ============================================================================
print("\n" + "="*80)
print("[PART 2] Checking GaussianImage imports...")
print("="*80 + "\n")

try:
    from gsplat.project_gaussians_2d import project_gaussians_2d
    print("✅ project_gaussians_2d imported")
except ImportError as e:
    print(f"❌ Failed to import project_gaussians_2d: {e}")

try:
    from gsplat.rasterize_sum import rasterize_gaussians_sum
    print("✅ rasterize_gaussians_sum imported")
    
    # Check what this function expects
    import inspect
    sig = inspect.signature(rasterize_gaussians_sum)
    print(f"\n📋 Function signature:")
    print(f"   {sig}")
    
except ImportError as e:
    print(f"❌ Failed to import rasterize_gaussians_sum: {e}")

# ============================================================================
# PART 3: Create a minimal test case with 3 channels (ORIGINAL)
# ============================================================================
print("\n" + "="*80)
print("[PART 3] Testing ORIGINAL 3-channel version...")
print("="*80 + "\n")

def test_3channel_forward():
    """Test the forward pass with 3 channels (RGB) - this is the original"""
    print("🧪 Creating minimal 3-channel test case...\n")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Device: {device}")
    
    # Small test image - 3 channels (RGB)
    H, W = 64, 64
    num_points = 100
    
    print(f"   Image size: {H}x{W}")
    print(f"   Channels: 3 (RGB)")
    print(f"   Num Gaussians: {num_points}")
    
    # Create fake Gaussian parameters (same as original code)
    xyz = torch.rand(num_points, 2, device=device) * 0.8 + 0.1  # positions in [0.1, 0.9]
    cholesky = torch.rand(num_points, 3, device=device) * 0.5  # covariance params
    colors = torch.rand(num_points, 3, device=device)  # RGB colors
    opacity = torch.ones(num_points, 1, device=device)
    
    print(f"\n   Parameter shapes:")
    print(f"   - xyz: {xyz.shape}")
    print(f"   - cholesky: {cholesky.shape}")
    print(f"   - colors: {colors.shape} ← 3 channels (RGB)")
    print(f"   - opacity: {opacity.shape}")
    
    try:
        print(f"\n   Step 1: project_gaussians_2d...")
        BLOCK_W, BLOCK_H = 16, 16
        tile_bounds = (
            (W + BLOCK_W - 1) // BLOCK_W,
            (H + BLOCK_H - 1) // BLOCK_H,
            1,
        )
        
        xys, depths, radii, conics, num_tiles_hit = project_gaussians_2d(
            xyz, cholesky, H, W, tile_bounds
        )
        print(f"   ✅ Projection successful")
        print(f"      - xys: {xys.shape}")
        print(f"      - depths: {depths.shape}")
        print(f"      - radii: {radii.shape}")
        
        print(f"\n   Step 2: rasterize_gaussians_sum...")
        background = torch.ones(3, device=device)
        
        out_img = rasterize_gaussians_sum(
            xys, depths, radii, conics, num_tiles_hit,
            colors, opacity, H, W, BLOCK_H, BLOCK_W, 
            background=background, return_alpha=False
        )
        
        print(f"   ✅ Rasterization successful")
        print(f"      Output shape: {out_img.shape}")
        print(f"      Expected: [H={H}, W={W}, C=3]")
        
        if out_img.shape == (H, W, 3):
            print(f"   ✅✅ PERFECT! Shape matches expected 3-channel output")
            return True
        else:
            print(f"   ⚠️  Shape mismatch!")
            return False
            
    except Exception as e:
        print(f"   ❌ Error during 3-channel forward pass:")
        print(f"      {type(e).__name__}: {e}")
        import traceback
        print("\n   Full traceback:")
        traceback.print_exc()
        return False

# ============================================================================
# PART 4: Test what happens with 2 channels (SPECTROGRAM)
# ============================================================================
print("\n" + "="*80)
print("[PART 4] Testing 2-channel version (for spectrograms)...")
print("="*80 + "\n")

def test_2channel_forward():
    """Test the forward pass with 2 channels - this is what we WANT for spectrograms"""
    print("🧪 Creating minimal 2-channel test case...\n")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Device: {device}")
    
    # Small test image - 2 channels (amplitude + phase)
    H, W = 64, 64
    num_points = 100
    
    print(f"   Image size: {H}x{W}")
    print(f"   Channels: 2 (amplitude + phase)")
    print(f"   Num Gaussians: {num_points}")
    
    # Create fake Gaussian parameters
    xyz = torch.rand(num_points, 2, device=device) * 0.8 + 0.1
    cholesky = torch.rand(num_points, 3, device=device) * 0.5
    colors = torch.rand(num_points, 2, device=device)  # ← ONLY 2 channels!
    opacity = torch.ones(num_points, 1, device=device)
    
    print(f"\n   Parameter shapes:")
    print(f"   - xyz: {xyz.shape}")
    print(f"   - cholesky: {cholesky.shape}")
    print(f"   - colors: {colors.shape} ← 2 channels (amplitude + phase)")
    print(f"   - opacity: {opacity.shape}")
    
    try:
        print(f"\n   Step 1: project_gaussians_2d...")
        BLOCK_W, BLOCK_H = 16, 16
        tile_bounds = (
            (W + BLOCK_W - 1) // BLOCK_W,
            (H + BLOCK_H - 1) // BLOCK_H,
            1,
        )
        
        xys, depths, radii, conics, num_tiles_hit = project_gaussians_2d(
            xyz, cholesky, H, W, tile_bounds
        )
        print(f"   ✅ Projection successful (same as 3-channel)")
        
        print(f"\n   Step 2: rasterize_gaussians_sum with 2 channels...")
        background = torch.ones(2, device=device)  # ← 2-channel background
        
        out_img = rasterize_gaussians_sum(
            xys, depths, radii, conics, num_tiles_hit,
            colors, opacity, H, W, BLOCK_H, BLOCK_W, 
            background=background, return_alpha=False
        )
        
        print(f"   ✅ Rasterization successful!")
        print(f"      Output shape: {out_img.shape}")
        print(f"      Expected: [H={H}, W={W}, C=2]")
        
        if out_img.shape == (H, W, 2):
            print(f"   ✅✅ PERFECT! 2-channel output works!")
            return True
        else:
            print(f"   ⚠️  Shape mismatch!")
            return False
            
    except Exception as e:
        print(f"   ❌ Error during 2-channel forward pass:")
        print(f"      {type(e).__name__}: {e}")
        print(f"\n      This tells us what needs to be fixed!")
        import traceback
        print("\n   Full traceback:")
        traceback.print_exc()
        return False

# ============================================================================
# Run all tests
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("RUNNING ALL DIAGNOSTIC TESTS")
    print("="*80)
    
    success_3ch = test_3channel_forward()
    success_2ch = test_2channel_forward()
    
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"\n3-channel (original): {'✅ PASS' if success_3ch else '❌ FAIL'}")
    print(f"2-channel (spectrograms): {'✅ PASS' if success_2ch else '❌ FAIL'}")
    
    if success_3ch and not success_2ch:
        print(f"\n💡 INSIGHT: The original 3-channel code works, but 2-channel fails.")
        print(f"   This tells us EXACTLY what needs to be modified in gsplat.")
    elif success_3ch and success_2ch:
        print(f"\n🎉 GREAT NEWS: Both 2-channel and 3-channel work!")
        print(f"   The gsplat library already supports variable channels.")
    elif not success_3ch:
        print(f"\n⚠️  WARNING: Even the original 3-channel code fails.")
        print(f"   This suggests a gsplat installation issue (like the nd_rasterize_sum problem).")
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    
    if not success_3ch:
        print("\n1. Fix the gsplat function naming issue:")
        print("   - Check gsplat/gsplat/rasterize_sum.py")
        print("   - Replace 'nd_rasterize_sum_*' with 'rasterize_sum_*'")
        print("   - This is the issue from our previous chat!")
    
    if success_3ch and not success_2ch:
        print("\n1. The gsplat library needs modification to support 2 channels")
        print("2. We need to identify which C++/CUDA files handle channel dimensions")
        print("3. Or find a workaround (e.g., pad to 3 channels)")
    
    print("\n" + "="*80)
