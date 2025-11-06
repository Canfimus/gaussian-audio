"""
Audio Quality Evaluation Script
Compares original and reconstructed audio files using multiple metrics:
- PESQ: Perceptual quality (telephony standard)
- STOI: Speech intelligibility
- UTMOS: Predicted Mean Opinion Score
- V/UV F1: Voiced/Unvoiced detection accuracy
"""

import librosa
import numpy as np
import soundfile as sf
import os
import glob
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Try to import metrics (we'll install these)
try:
    from pesq import pesq
    PESQ_AVAILABLE = True
except ImportError:
    PESQ_AVAILABLE = False
    print("⚠️  PESQ not available. Install with: pip install pesq")

try:
    from pystoi import stoi
    STOI_AVAILABLE = True
except ImportError:
    STOI_AVAILABLE = False
    print("⚠️  STOI not available. Install with: pip install pystoi")

try:
    import torch
    from speechmos import dnsmos  # or utmos
    UTMOS_AVAILABLE = True
except ImportError:
    UTMOS_AVAILABLE = False
    print("⚠️  UTMOS not available. We'll implement a simpler version or skip")

# === Voiced/Unvoiced Detection ===
def detect_voiced_unvoiced(audio, sr, frame_length=2048, hop_length=512):
    """
    Detect voiced/unvoiced regions using zero-crossing rate and energy.
    Returns binary mask: 1 = voiced, 0 = unvoiced
    """
    # Calculate features
    zcr = librosa.feature.zero_crossing_rate(audio, frame_length=frame_length, hop_length=hop_length)[0]
    energy = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    
    # Normalize
    zcr = (zcr - zcr.mean()) / (zcr.std() + 1e-8)
    energy = (energy - energy.mean()) / (energy.std() + 1e-8)
    
    # Simple threshold-based classification
    # Voiced: low ZCR, high energy
    # Unvoiced: high ZCR, moderate energy
    voiced = (zcr < 0) & (energy > -0.5)
    
    return voiced.astype(int)

def calculate_vuv_f1(original_audio, reconstructed_audio, sr):
    """
    Calculate F1 score for voiced/unvoiced detection agreement
    """
    # Ensure same length
    min_len = min(len(original_audio), len(reconstructed_audio))
    original_audio = original_audio[:min_len]
    reconstructed_audio = reconstructed_audio[:min_len]
    
    # Detect V/UV for both
    original_vuv = detect_voiced_unvoiced(original_audio, sr)
    reconstructed_vuv = detect_voiced_unvoiced(reconstructed_audio, sr)
    
    # Ensure same length (frames might differ by 1)
    min_frames = min(len(original_vuv), len(reconstructed_vuv))
    original_vuv = original_vuv[:min_frames]
    reconstructed_vuv = reconstructed_vuv[:min_frames]
    
    # Calculate F1
    tp = np.sum((original_vuv == 1) & (reconstructed_vuv == 1))
    fp = np.sum((original_vuv == 0) & (reconstructed_vuv == 1))
    fn = np.sum((original_vuv == 1) & (reconstructed_vuv == 0))
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    return f1, precision, recall

# === Main Evaluation Function ===
def evaluate_audio_pair(original_path, reconstructed_path, sr=22050):
    """
    Evaluate a pair of audio files using all available metrics.
    """
    results = {}
    
    # Load audio files
    try:
        original_audio, _ = librosa.load(original_path, sr=sr)
        reconstructed_audio, _ = librosa.load(reconstructed_path, sr=sr)
    except Exception as e:
        print(f"Error loading audio: {e}")
        return None
    
    # Ensure same length
    min_len = min(len(original_audio), len(reconstructed_audio))
    original_audio = original_audio[:min_len]
    reconstructed_audio = reconstructed_audio[:min_len]
    
    # === PESQ ===
    if PESQ_AVAILABLE:
        try:
            # PESQ requires specific sample rates (8000 or 16000)
            if sr not in [8000, 16000]:
                # Resample to 16000 for PESQ
                orig_16k = librosa.resample(original_audio, orig_sr=sr, target_sr=16000)
                recon_16k = librosa.resample(reconstructed_audio, orig_sr=sr, target_sr=16000)
                pesq_score = pesq(16000, orig_16k, recon_16k, 'wb')  # 'wb' = wideband
            else:
                pesq_score = pesq(sr, original_audio, reconstructed_audio, 'wb')
            results['PESQ'] = pesq_score
        except Exception as e:
            print(f"  PESQ calculation failed: {e}")
            results['PESQ'] = None
    else:
        results['PESQ'] = None
    
    # === STOI ===
    if STOI_AVAILABLE:
        try:
            stoi_score = stoi(original_audio, reconstructed_audio, sr, extended=False)
            results['STOI'] = stoi_score
        except Exception as e:
            print(f"  STOI calculation failed: {e}")
            results['STOI'] = None
    else:
        results['STOI'] = None
    
    # === V/UV F1 ===
    try:
        vuv_f1, vuv_precision, vuv_recall = calculate_vuv_f1(original_audio, reconstructed_audio, sr)
        results['VUV_F1'] = vuv_f1
        results['VUV_Precision'] = vuv_precision
        results['VUV_Recall'] = vuv_recall
    except Exception as e:
        print(f"  V/UV calculation failed: {e}")
        results['VUV_F1'] = None
    
    # === Simple SNR (bonus metric) ===
    try:
        noise = original_audio - reconstructed_audio
        signal_power = np.mean(original_audio ** 2)
        noise_power = np.mean(noise ** 2)
        snr = 10 * np.log10(signal_power / (noise_power + 1e-8))
        results['SNR_dB'] = snr
    except Exception as e:
        results['SNR_dB'] = None
    
    return results

# === Batch Evaluation ===
def evaluate_directory(original_dir, reconstructed_dir, output_file='results.txt'):
    """
    Evaluate all audio pairs in two directories.
    """
    print("="*60)
    print("Audio Quality Evaluation")
    print("="*60)
    print(f"Original audio directory: {original_dir}")
    print(f"Reconstructed audio directory: {reconstructed_dir}")
    print()
    
    # Find all reconstructed files
    recon_files = sorted(glob.glob(os.path.join(reconstructed_dir, '*_reconstructed.wav')))
    
    if not recon_files:
        print(f"❌ No '*_reconstructed.wav' files found in {reconstructed_dir}")
        return
    
    print(f"Found {len(recon_files)} reconstructed files to evaluate.\n")
    
    all_results = []
    
    for recon_path in recon_files:
        file_id = os.path.basename(recon_path).replace('_reconstructed.wav', '')
        original_path = os.path.join(original_dir, f"{file_id}_original.wav")
        
        if not os.path.exists(original_path):
            print(f"⚠️  Original file not found for {file_id}, skipping.")
            continue
        
        print(f"Evaluating: {file_id}")
        results = evaluate_audio_pair(original_path, recon_path)
        
        if results:
            results['file_id'] = file_id
            all_results.append(results)
            
            # Print results
            print(f"  PESQ: {results['PESQ']:.3f}" if results['PESQ'] else "  PESQ: N/A")
            print(f"  STOI: {results['STOI']:.3f}" if results['STOI'] else "  STOI: N/A")
            print(f"  V/UV F1: {results['VUV_F1']:.3f}" if results['VUV_F1'] else "  V/UV F1: N/A")
            print(f"  SNR: {results['SNR_dB']:.2f} dB" if results['SNR_dB'] else "  SNR: N/A")
            print()
    
    # Calculate averages
    if all_results:
        print("="*60)
        print("AVERAGE RESULTS")
        print("="*60)
        
        metrics = ['PESQ', 'STOI', 'VUV_F1', 'SNR_dB']
        for metric in metrics:
            values = [r[metric] for r in all_results if r[metric] is not None]
            if values:
                avg = np.mean(values)
                std = np.std(values)
                print(f"{metric}: {avg:.3f} ± {std:.3f}")
        
        # Save to file
        with open(output_file, 'w') as f:
            f.write("="*60 + "\n")
            f.write("Audio Quality Evaluation Results\n")
            f.write("="*60 + "\n\n")
            
            for result in all_results:
                f.write(f"File: {result['file_id']}\n")
                for key, value in result.items():
                    if key != 'file_id' and value is not None:
                        f.write(f"  {key}: {value:.4f}\n")
                f.write("\n")
            
            f.write("="*60 + "\n")
            f.write("AVERAGE RESULTS\n")
            f.write("="*60 + "\n")
            for metric in metrics:
                values = [r[metric] for r in all_results if r[metric] is not None]
                if values:
                    avg = np.mean(values)
                    std = np.std(values)
                    f.write(f"{metric}: {avg:.4f} ± {std:.4f}\n")
        
        print(f"\n✅ Results saved to: {output_file}")
    else:
        print("❌ No results to save.")

# === Main ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate audio quality metrics")
    
    parser.add_argument(
        "-orig", "--original_dir",
        type=str,
        required=True,
        help="Directory containing original audio files (*_original.wav)"
    )
    
    parser.add_argument(
        "-recon", "--reconstructed_dir",
        type=str,
        required=True,
        help="Directory containing reconstructed audio files (*_reconstructed.wav)"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="evaluation_results.txt",
        help="Output file for results"
    )
    
    args = parser.parse_args()
    
    evaluate_directory(args.original_dir, args.reconstructed_dir, args.output)