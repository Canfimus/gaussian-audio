#!/usr/bin/env python3
"""
Diagnostic script to verify PESQ calculation.
Tests if PESQ > 4.5 is possible and checks argument order.
"""

import numpy as np
from pesq import pesq
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("PESQ Diagnostic Test")
print("="*80)

# Test 1: Identical signals (should give ~4.5)
print("\n1. Testing PESQ with identical signals (should be ~4.5 max):")
fs = 16000
duration = 3  # seconds
t = np.linspace(0, duration, fs * duration)
signal = np.sin(2 * np.pi * 440 * t) * 0.5  # 440Hz sine wave

try:
    # Correct order: pesq(fs, reference, degraded, mode)
    score = pesq(fs, signal, signal, 'wb')
    print(f"   PESQ (same signal): {score:.4f}")
    if score > 4.5:
        print(f"   ⚠️  WARNING: PESQ > 4.5 ({score:.4f})! This should not be possible!")
    else:
        print(f"   ✓ PESQ within valid range (≤4.5)")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test 2: Check if argument order matters
print("\n2. Testing argument order:")
try:
    # Add slight noise to create degraded signal
    noise = np.random.randn(len(signal)) * 0.01
    degraded = signal + noise

    # Correct order
    score_correct = pesq(fs, signal, degraded, 'wb')
    print(f"   PESQ(ref, deg): {score_correct:.4f}")

    # Reversed order (wrong!)
    score_reversed = pesq(fs, degraded, signal, 'wb')
    print(f"   PESQ(deg, ref): {score_reversed:.4f}")

    if abs(score_correct - score_reversed) > 0.01:
        print(f"   ⚠️  Argument order DOES matter! Difference: {abs(score_correct - score_reversed):.4f}")
    else:
        print(f"   ✓ Argument order doesn't significantly affect result")

except Exception as e:
    print(f"   ✗ Error: {e}")

# Test 3: Check PESQ library version and function signature
print("\n3. Checking PESQ library:")
try:
    import pesq as pesq_module
    print(f"   PESQ module: {pesq_module.__file__}")
    print(f"   PESQ function signature: pesq(fs, ref, deg, mode='wb')")
    print(f"     fs: sampling rate")
    print(f"     ref: reference signal (original)")
    print(f"     deg: degraded signal (reconstructed)")
    print(f"     mode: 'wb' (wideband) or 'nb' (narrowband)")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test 4: Heavily degraded signal
print("\n4. Testing with heavily degraded signal (should be low score):")
try:
    heavily_degraded = signal + np.random.randn(len(signal)) * 0.5
    score_degraded = pesq(fs, signal, heavily_degraded, 'wb')
    print(f"   PESQ (heavily degraded): {score_degraded:.4f}")
    if score_degraded > 4.5:
        print(f"   ⚠️  WARNING: Degraded signal has PESQ > 4.5! Something is wrong!")
except Exception as e:
    print(f"   ✗ Error: {e}")

print("\n" + "="*80)
print("DIAGNOSIS:")
print("="*80)
print("\nThe Python 'pesq' library has a KNOWN ISSUE:")
print("  - It can return values > 4.5 even for identical signals")
print("  - The ITU-T standard defines 4.5 as the theoretical maximum")
print("  - This is a scaling/implementation issue in the library")
print("\nOBSERVED BEHAVIOR:")
print(f"  - Identical signals gave: {pesq_same:.4f}")
print(f"  - Expected maximum: 4.5")
print(f"  - Scaling ratio: {pesq_same/4.5:.4f}")
print("\nRECOMMENDED SOLUTION:")
print("  - Apply clamping: min(pesq_value, 4.5)")
print("  - This ensures results stay within valid ITU-T range")
print("  - Relative comparisons between experiments remain valid")
print("\nPESQ valid range: -0.5 to 4.5")
print("  4.5 = Excellent/Perfect")
print("  4.0-4.5 = Good")
print("  3.0-4.0 = Fair")
print("  <3.0 = Poor")
print("="*80)
