"""
Comprehensive Audio Quality + Compression Analysis

Analyzes both quality metrics AND compression efficiency:
- Quality: PESQ, STOI, V/UV F1, SNR
- Compression: File sizes, compression ratio, bitrate (kbps)
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

# Import metrics
try:
    from pesq import pesq
    PESQ_AVAILABLE = True
except ImportError:
    PESQ_AVAILABLE = False

try:
    from pystoi import stoi
    STOI_AVAILABLE = True
except ImportError:
    STOI_AVAILABLE = False

# === V/UV Detection (simplified) ===
def detect_voiced_unvoiced(audio, sr, frame_length=2048, hop_length=512):
    zcr = librosa.feature.zero_crossing_rate(audio, frame_length=frame_length, hop_length=hop_length)[0]
    energy = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, n_fft=frame_length, hop_length=hop_length)[0]
    
    zcr_norm = (zcr - zcr.mean()) / (zcr.std() + 1e-8)
    energy_norm = (energy - energy.mean()) / (energy.std() + 1e-8)
    centroid_norm = (spectral_centroid - spectral_centroid.mean()) / (spectral_centroid.std() + 1e-8)
    
    voiced = (zcr_norm < 0) & (energy_norm > -0.5) & (centroid_norm < 0.5)
    return voiced.astype(int)

def calculate_vuv_f1(original_audio, reconstructed_audio, sr):
    min_len = min(len(original_audio), len(reconstructed_audio))
    original_audio = original_audio[:min_len]
    reconstructed_audio = reconstructed_audio[:min_len]
    
    original_vuv = detect_voiced_unvoiced(original_audio, sr)
    reconstructed_vuv = detect_voiced_unvoiced(reconstructed_audio, sr)
    
    min_frames = min(len(original_vuv), len(reconstructed_vuv))
    original_vuv = original_vuv[:min_frames]
    reconstructed_vuv = reconstructed_vuv[:min_frames]
    
    tp = np.sum((original_vuv == 1) & (reconstructed_vuv == 1))
    fp = np.sum((original_vuv == 0) & (reconstructed_vuv == 1))
    fn = np.sum((original_vuv == 1) & (reconstructed_vuv == 0))
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    return f1, precision, recall

# === File Size Analysis ===
def get_file_size(filepath):
    """Get file size in bytes."""
    if os.path.exists(filepath):
        return os.path.getsize(filepath)
    return None

def calculate_bitrate(file_size_bytes, duration_seconds):
    """Calculate bitrate in kbps."""
    if duration_seconds > 0:
        return (file_size_bytes * 8) / duration_seconds / 1000  # kbps
    return None

# === Main Evaluation ===
def evaluate_audio_with_compression(original_wav, reconstructed_wav, spectrogram_npy, sr=22050):
    """
    Comprehensive evaluation including quality AND compression metrics.
    """
    results = {}
    
    # Load audio
    try:
        original_audio, _ = librosa.load(original_wav, sr=sr)
        reconstructed_audio, _ = librosa.load(reconstructed_wav, sr=sr)
    except Exception as e:
        print(f"Error loading audio: {e}")
        return None
    
    # Calculate duration
    min_len = min(len(original_audio), len(reconstructed_audio))
    original_audio = original_audio[:min_len]
    reconstructed_audio = reconstructed_audio[:min_len]
    duration = len(original_audio) / sr
    results['duration_sec'] = duration
    
    # === FILE SIZE ANALYSIS ===
    
    # Original .wav size
    wav_size = get_file_size(original_wav)
    results['original_wav_bytes'] = wav_size
    results['original_wav_kb'] = wav_size / 1024 if wav_size else None
    results['original_wav_kbps'] = calculate_bitrate(wav_size, duration) if wav_size else None
    
    # Compressed .npy size (your method)
    npy_size = get_file_size(spectrogram_npy)
    results['compressed_npy_bytes'] = npy_size
    results['compressed_npy_kb'] = npy_size / 1024 if npy_size else None
    results['compressed_npy_kbps'] = calculate_bitrate(npy_size, duration) if npy_size else None
    
    # Compression ratio
    if wav_size and npy_size:
        results['compression_ratio'] = wav_size / npy_size
    else:
        results['compression_ratio'] = None
    
    # === QUALITY METRICS ===
    
    # PESQ
    if PESQ_AVAILABLE:
        try:
            if sr not in [8000, 16000]:
                orig_16k = librosa.resample(original_audio, orig_sr=sr, target_sr=16000)
                recon_16k = librosa.resample(reconstructed_audio, orig_sr=sr, target_sr=16000)
                pesq_score = pesq(16000, orig_16k, recon_16k, 'wb')
            else:
                pesq_score = pesq(sr, original_audio, reconstructed_audio, 'wb')
            results['PESQ'] = pesq_score
        except Exception as e:
            results['PESQ'] = None
    else:
        results['PESQ'] = None
    
    # STOI
    if STOI_AVAILABLE:
        try:
            stoi_score = stoi(original_audio, reconstructed_audio, sr, extended=False)
            results['STOI'] = stoi_score
        except Exception as e:
            results['STOI'] = None
    else:
        results['STOI'] = None
    
    # V/UV F1
    try:
        vuv_f1, _, _ = calculate_vuv_f1(original_audio, reconstructed_audio, sr)
        results['VUV_F1'] = vuv_f1
    except Exception as e:
        results['VUV_F1'] = None
    
    # SNR
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
def evaluate_directory_with_compression(audio_dir, spectrogram_dir, output_file='compression_results.txt'):
    """
    Evaluate all files with both quality and compression metrics.
    """
    print("="*80)
    print("Comprehensive Audio Quality + Compression Analysis")
    print("="*80)
    print(f"Audio directory: {audio_dir}")
    print(f"Spectrogram directory: {spectrogram_dir}")
    print()
    
    # Find all reconstructed audio files
    recon_files = sorted(glob.glob(os.path.join(audio_dir, '*_reconstructed.wav')))
    
    if not recon_files:
        print(f"❌ No '*_reconstructed.wav' files found in {audio_dir}")
        return
    
    print(f"Found {len(recon_files)} audio files to evaluate.")
    
    # Build a mapping of file_id to .npy path by searching recursively
    print(f"Searching for .npy files in {spectrogram_dir}...")
    npy_mapping = {}
    for npy_path in glob.glob(os.path.join(spectrogram_dir, '**', '*_fitting.npy'), recursive=True):
        file_id = os.path.basename(npy_path).replace('_fitting.npy', '')
        npy_mapping[file_id] = npy_path
    
    print(f"Found {len(npy_mapping)} .npy files.\n")
    
    if not npy_mapping:
        print(f"❌ No '*_fitting.npy' files found in {spectrogram_dir}")
        print(f"   Make sure you ran train_subset.py with --save_imgs flag!")
        return
    
    all_results = []
    
    for recon_path in recon_files:
        file_id = os.path.basename(recon_path).replace('_reconstructed.wav', '')
        original_wav = os.path.join(audio_dir, f"{file_id}_original.wav")
        
        if not os.path.exists(original_wav):
            print(f"⚠️  Original WAV not found for {file_id}, skipping.")
            continue
        
        # Look up the .npy file from our mapping
        if file_id not in npy_mapping:
            print(f"⚠️  Spectrogram NPY not found for {file_id}, skipping.")
            continue
        
        spectrogram_npy = npy_mapping[file_id]
        
        print(f"Evaluating: {file_id}")
        results = evaluate_audio_with_compression(original_wav, recon_path, spectrogram_npy)
        
        if results:
            results['file_id'] = file_id
            all_results.append(results)
            
            # Print results
            print(f"  Duration: {results['duration_sec']:.2f}s")
            print(f"  Original WAV: {results['original_wav_kb']:.1f} KB ({results['original_wav_kbps']:.1f} kbps)")
            print(f"  Compressed NPY: {results['compressed_npy_kb']:.1f} KB ({results['compressed_npy_kbps']:.1f} kbps)")
            print(f"  Compression: {results['compression_ratio']:.2f}x")
            print(f"  PESQ: {results['PESQ']:.3f}" if results['PESQ'] else "  PESQ: N/A")
            print(f"  STOI: {results['STOI']:.3f}" if results['STOI'] else "  STOI: N/A")
            print()
    
    # Calculate averages
    if all_results:
        print("="*80)
        print("AVERAGE RESULTS")
        print("="*80)
        
        metrics = ['duration_sec', 'original_wav_kbps', 'compressed_npy_kbps', 'compression_ratio', 
                   'PESQ', 'STOI', 'VUV_F1', 'SNR_dB']
        
        for metric in metrics:
            values = [r[metric] for r in all_results if r.get(metric) is not None]
            if values:
                avg = np.mean(values)
                std = np.std(values)
                
                if metric == 'duration_sec':
                    print(f"Average duration: {avg:.2f} ± {std:.2f} seconds")
                elif metric == 'original_wav_kbps':
                    print(f"Original WAV bitrate: {avg:.1f} ± {std:.1f} kbps")
                elif metric == 'compressed_npy_kbps':
                    print(f"Compressed bitrate: {avg:.1f} ± {std:.1f} kbps")
                elif metric == 'compression_ratio':
                    print(f"Compression ratio: {avg:.2f}x ± {std:.2f}x")
                else:
                    print(f"{metric}: {avg:.3f} ± {std:.3f}")
        
        # Save detailed results
        with open(output_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("Comprehensive Audio Quality + Compression Analysis\n")
            f.write("="*80 + "\n\n")
            
            # Per-file results
            for result in all_results:
                f.write(f"File: {result['file_id']}\n")
                f.write(f"  Duration: {result['duration_sec']:.2f} seconds\n")
                f.write(f"  Original WAV: {result['original_wav_kb']:.1f} KB ({result['original_wav_kbps']:.1f} kbps)\n")
                f.write(f"  Compressed NPY: {result['compressed_npy_kb']:.1f} KB ({result['compressed_npy_kbps']:.1f} kbps)\n")
                f.write(f"  Compression Ratio: {result['compression_ratio']:.2f}x\n")
                
                for key in ['PESQ', 'STOI', 'VUV_F1', 'SNR_dB']:
                    if result.get(key) is not None:
                        f.write(f"  {key}: {result[key]:.4f}\n")
                f.write("\n")
            
            # Averages
            f.write("="*80 + "\n")
            f.write("AVERAGE RESULTS\n")
            f.write("="*80 + "\n")
            
            for metric in metrics:
                values = [r[metric] for r in all_results if r.get(metric) is not None]
                if values:
                    avg = np.mean(values)
                    std = np.std(values)
                    f.write(f"{metric}: {avg:.4f} ± {std:.4f}\n")
        
        print(f"\n✅ Results saved to: {output_file}")
        
        # Summary comparison table
        print("\n" + "="*80)
        print("QUALITY vs COMPRESSION SUMMARY")
        print("="*80)
        avg_kbps = np.mean([r['compressed_npy_kbps'] for r in all_results if r.get('compressed_npy_kbps')])
        avg_pesq = np.mean([r['PESQ'] for r in all_results if r.get('PESQ')])
        avg_stoi = np.mean([r['STOI'] for r in all_results if r.get('STOI')])
        avg_comp = np.mean([r['compression_ratio'] for r in all_results if r.get('compression_ratio')])
        
        print(f"Your GaussianImage method achieves:")
        print(f"  • Bitrate: {avg_kbps:.1f} kbps (compression: {avg_comp:.1f}x)")
        print(f"  • PESQ: {avg_pesq:.2f}")
        print(f"  • STOI: {avg_stoi:.3f}")
        print()
        print("Compare this to baselines like:")
        print("  • Opus @ 32 kbps: PESQ ~3.4, STOI ~0.90")
        print("  • MP3 @ 128 kbps: PESQ ~3.8, STOI ~0.92")
        print("="*80)
    else:
        print("❌ No results to save.")

# === Main ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate audio quality with compression analysis")
    
    parser.add_argument(
        "-a", "--audio_dir",
        type=str,
        default="./outputs/audio",
        help="Directory containing original and reconstructed audio files"
    )
    
    parser.add_argument(
        "-s", "--spectrogram_dir",
        type=str,
        required=True,
        help="Directory containing compressed spectrogram .npy files (e.g., checkpoints/.../LJ001-0001/)"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="compression_results.txt",
        help="Output file for results"
    )
    
    args = parser.parse_args()
    
    evaluate_directory_with_compression(args.audio_dir, args.spectrogram_dir, args.output)