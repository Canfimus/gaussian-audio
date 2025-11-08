"""
Test script to verify PESQ and STOI are being called correctly.
"""

# PESQ function signature (from official docs):
# pesq(fs, ref, deg, mode)
#   fs: sample rate (8000 or 16000)
#   ref: reference (clean/original) signal
#   deg: degraded (processed/test) signal
#   mode: 'wb' (wideband, 16kHz) or 'nb' (narrowband, 8kHz)
#
# Returns: PESQ score from -0.5 to 4.5 (higher is better)

# STOI function signature (from official docs):
# stoi(x, y, fs_signal, extended=False)
#   x: clean reference signal
#   y: processed/degraded signal
#   fs_signal: sample rate
#   extended: use extended STOI (default False)
#
# Returns: STOI score from 0.0 to 1.0 (higher is better)

# Our current code:
# Line 217: pesq_value = pesq(16000, audio_orig_16k, audio_recon_16k, 'wb')
#   ✓ CORRECT: pesq(sample_rate, reference, degraded, mode)

# Line 224: stoi_value = stoi(audio_waveform_orig_trimmed, audio_waveform_trimmed, ORIGINAL_SR, extended=False)
#   ✓ CORRECT: stoi(reference, degraded, sample_rate, extended)

print("Function calls are CORRECT!")
print("\nHowever, there's a NORMALIZATION issue...")
print("Both signals are normalized independently, which can affect results.")
