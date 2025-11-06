import librosa
import numpy as np
import soundfile as sf
import os
import glob
import argparse
import matplotlib
matplotlib.use('Agg')  # Use 'Agg' backend for saving plots on servers (no GUI)
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from pystoi import stoi
from pesq import pesq

# --- 1. Spectrogram Parameters (Must match preprocess.py) ---
N_FFT = 1024
HOP_LENGTH = 256
ORIGINAL_SR = 22050

def load_and_convert_to_db(npy_path):
    """Loads a 2-ch (Real, Imag) .npy file and converts to log-magnitude (dB)."""
    try:
        spec_ri = np.load(npy_path)
    except Exception as e:
        print(f"  Error loading file: {npy_path} | {e}")
        return None, None
    
    # Reconstruct the complex spectrogram
    S_complex = spec_ri[..., 0] + 1j * spec_ri[..., 1]
    
    # Convert to magnitude
    S_magnitude = np.abs(S_complex)
    
    # Apply epsilon to avoid log(0)
    S_magnitude = np.maximum(S_magnitude, 1e-10)
    
    # Convert to log-magnitude (decibels) for visualization
    # Use a more stable reference that doesn't depend on max value
    S_db = librosa.amplitude_to_db(S_magnitude, ref=1.0, top_db=80.0)
    
    return S_db, S_complex  # Return complex spec for audio

def calculate_original_baseline(original_audio_path, original_spec_path):
    """
    Calculate the baseline PESQ/STOI by comparing:
    - Original WAV file
    - vs Audio reconstructed from spectrogram (STFT→ISTFT)

    This measures the quality ceiling from the spectrogram representation itself.
    """
    try:
        # Load original WAV file
        audio_original_wav, sr_orig = librosa.load(original_audio_path, sr=None)

        # Load spectrogram and reconstruct audio
        spec_ri = np.load(original_spec_path)
        S_complex = spec_ri[..., 0] + 1j * spec_ri[..., 1]
        audio_from_spec = librosa.istft(S_complex, n_fft=N_FFT, hop_length=HOP_LENGTH, length=None)

        # Ensure both have same length
        min_len = min(len(audio_original_wav), len(audio_from_spec))
        audio_original_wav = audio_original_wav[:min_len]
        audio_from_spec = audio_from_spec[:min_len]

        # Normalize both
        max_val_orig = np.abs(audio_original_wav).max()
        if max_val_orig > 0:
            audio_original_wav = audio_original_wav / max_val_orig * 0.95

        max_val_spec = np.abs(audio_from_spec).max()
        if max_val_spec > 0:
            audio_from_spec = audio_from_spec / max_val_spec * 0.95

        # Calculate PESQ
        audio_orig_16k = librosa.resample(audio_original_wav, orig_sr=sr_orig, target_sr=16000)
        audio_spec_16k = librosa.resample(audio_from_spec, orig_sr=sr_orig, target_sr=16000)
        pesq_baseline = pesq(16000, audio_orig_16k, audio_spec_16k, 'wb')

        # Calculate STOI
        stoi_baseline = stoi(audio_original_wav, audio_from_spec, sr_orig, extended=False)

        return pesq_baseline, stoi_baseline

    except Exception as e:
        print(f"  Warning: Could not calculate baseline from WAV file: {e}")
        return None, None


def analyze_file(original_path, reconstructed_path, base_output_path, original_wav_dir=None):
    """
    Analyzes a single file: creates a plot and an audio file.
    Returns PESQ and STOI metrics for both reconstructed and original baseline.

    Args:
        original_path: Path to original spectrogram .npy
        reconstructed_path: Path to reconstructed spectrogram .npy
        base_output_path: Output directory
        original_wav_dir: Optional directory containing original .wav files for baseline calculation
    """
    file_id = os.path.basename(original_path).split('.')[0]
    print(f"--- Processing: {file_id} ---")
    """
    Analyzes a single file: creates a plot and an audio file.
    Returns PSNR and STOI metrics.
    """
    file_id = os.path.basename(original_path).split('.')[0]
    print(f"--- Processing: {file_id} ---")

    # --- 1. Load and prepare data ---
    spec_db_original, S_complex_original = load_and_convert_to_db(original_path)
    spec_db_reconstructed, S_complex_reconstructed = load_and_convert_to_db(reconstructed_path)

    if spec_db_original is None or spec_db_reconstructed is None:
        print(f"  Skipping file {file_id} due to loading error.")
        return None, None

    # --- 2. Calculate difference for analysis ---
    diff_db = spec_db_reconstructed - spec_db_original
    
    # --- 3. Generate and save plot ---
    plot_output_path = os.path.join(base_output_path, 'plots', f"{file_id}_comparison.png")
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    # Original spectrogram
    img1 = librosa.display.specshow(spec_db_original, ax=axes[0, 0], y_axis='log', x_axis='time',
                                     sr=ORIGINAL_SR, hop_length=HOP_LENGTH, cmap='viridis')
    axes[0, 0].set_title("Original Spectrogram", fontsize=14, fontweight='bold')
    plt.colorbar(img1, ax=axes[0, 0], format="%+2.0f dB")
    
    # Reconstructed spectrogram
    img2 = librosa.display.specshow(spec_db_reconstructed, ax=axes[0, 1], y_axis='log', x_axis='time',
                                     sr=ORIGINAL_SR, hop_length=HOP_LENGTH, cmap='viridis')
    axes[0, 1].set_title("Reconstructed (GaussianImage)", fontsize=14, fontweight='bold')
    plt.colorbar(img2, ax=axes[0, 1], format="%+2.0f dB")
    
    # Difference plot
    img3 = librosa.display.specshow(diff_db, ax=axes[1, 0], y_axis='log', x_axis='time',
                                     sr=ORIGINAL_SR, hop_length=HOP_LENGTH, cmap='RdBu_r', 
                                     vmin=-20, vmax=20)
    axes[1, 0].set_title("Difference (Reconstructed - Original)", fontsize=14, fontweight='bold')
    plt.colorbar(img3, ax=axes[1, 0], format="%+2.0f dB")
    
    # Magnitude comparison (linear scale for frequencies 0-8kHz)
    freq_bins = S_complex_original.shape[0]
    freq_hz = librosa.fft_frequencies(sr=ORIGINAL_SR, n_fft=N_FFT)
    max_freq_idx = np.argmax(freq_hz > 8000)
    
    time_idx = spec_db_original.shape[1] // 2  # Middle time frame
    axes[1, 1].plot(freq_hz[:max_freq_idx], spec_db_original[:max_freq_idx, time_idx], 
                    label='Original', alpha=0.7, linewidth=2)
    axes[1, 1].plot(freq_hz[:max_freq_idx], spec_db_reconstructed[:max_freq_idx, time_idx], 
                    label='Reconstructed', alpha=0.7, linewidth=2)
    axes[1, 1].set_xlabel('Frequency (Hz)', fontsize=12)
    axes[1, 1].set_ylabel('Magnitude (dB)', fontsize=12)
    axes[1, 1].set_title(f"Frequency Slice at t={time_idx}", fontsize=14, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(f"Spectrogram Analysis: {file_id}", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(plot_output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)  # Close the figure to save memory
    print(f"  ✅ Plot saved to: {plot_output_path}")

    # --- 4. Generate and save audio ---
    audio_output_path = os.path.join(base_output_path, 'audio', f"{file_id}_reconstructed.wav")

    # Perform iSTFT with proper parameters
    audio_waveform = librosa.istft(S_complex_reconstructed,
                                    n_fft=N_FFT,
                                    hop_length=HOP_LENGTH,
                                    length=None)  # Let librosa determine length

    # Normalize audio to prevent clipping
    max_val = np.abs(audio_waveform).max()
    if max_val > 0:
        audio_waveform = audio_waveform / max_val * 0.95

    # Save as .wav file
    sf.write(audio_output_path, audio_waveform, ORIGINAL_SR)
    print(f"  ✅ Audio saved to: {audio_output_path}")

    # Also save original for comparison and calculate metrics
    audio_waveform_orig = None
    if S_complex_original is not None:
        audio_original_path = os.path.join(base_output_path, 'audio', f"{file_id}_original.wav")
        audio_waveform_orig = librosa.istft(S_complex_original,
                                            n_fft=N_FFT,
                                            hop_length=HOP_LENGTH,
                                            length=None)
        max_val_orig = np.abs(audio_waveform_orig).max()
        if max_val_orig > 0:
            audio_waveform_orig = audio_waveform_orig / max_val_orig * 0.95
        sf.write(audio_original_path, audio_waveform_orig, ORIGINAL_SR)
        print(f"  ✅ Original audio saved to: {audio_original_path}")

    # --- 5. Calculate audio quality metrics (PESQ and STOI) ---
    pesq_value = None
    stoi_value = None
    pesq_original = None
    stoi_original = None

    if audio_waveform_orig is not None:
        # Ensure both waveforms have the same length
        min_len = min(len(audio_waveform), len(audio_waveform_orig))
        audio_waveform_trimmed = audio_waveform[:min_len]
        audio_waveform_orig_trimmed = audio_waveform_orig[:min_len]

        # Calculate PESQ and STOI for RECONSTRUCTED audio
        try:
            # Resample to 16kHz for PESQ (wideband mode)
            audio_orig_16k = librosa.resample(audio_waveform_orig_trimmed, orig_sr=ORIGINAL_SR, target_sr=16000)
            audio_recon_16k = librosa.resample(audio_waveform_trimmed, orig_sr=ORIGINAL_SR, target_sr=16000)

            # PESQ returns a score from -0.5 to 4.5 (higher is better)
            pesq_value = pesq(16000, audio_orig_16k, audio_recon_16k, 'wb')
        except Exception as e:
            print(f"  Warning: PESQ calculation failed: {e}")
            pesq_value = None

        # Calculate STOI (Short-Time Objective Intelligibility)
        try:
            stoi_value = stoi(audio_waveform_orig_trimmed, audio_waveform_trimmed, ORIGINAL_SR, extended=False)
        except Exception as e:
            print(f"  Warning: STOI calculation failed: {e}")
            stoi_value = None

        # Calculate PESQ and STOI for ORIGINAL audio baseline
        # If we have access to original WAV files, use those for accurate baseline
        if original_wav_dir is not None:
            # Try to find the original WAV file
            original_wav_path = os.path.join(original_wav_dir, f"{file_id}.wav")
            if os.path.exists(original_wav_path):
                print(f"  📁 Found original WAV file, calculating true baseline...")
                pesq_original, stoi_original = calculate_original_baseline(original_wav_path, original_path)
                if pesq_original is None:
                    # Fallback to theoretical max
                    pesq_original = 4.5
                    stoi_original = 1.0
                    print(f"  ⚠️  Baseline calculation failed, using theoretical max")
            else:
                print(f"  ⚠️  Original WAV not found, using theoretical max as baseline")
                pesq_original = 4.5
                stoi_original = 1.0
        else:
            # No WAV directory provided, use theoretical maximum
            print(f"  ℹ️  No WAV directory provided, using theoretical max as baseline")
            pesq_original = 4.5
            stoi_original = 1.0

        print(f"  📊 Reconstructed - PESQ: {pesq_value:.3f}, STOI: {stoi_value:.4f}" if pesq_value is not None else "  📊 Reconstructed - PESQ: N/A, STOI: N/A")
        print(f"  📊 Original Baseline - PESQ: {pesq_original:.3f}, STOI: {stoi_original:.4f}")

    return pesq_value, stoi_value, pesq_original, stoi_original


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze a full subset run (generate plots and audio).")
    
    # Path to the *original* spectrograms
    parser.add_argument(
        "-orig_dir", "--original_dir",
        type=str,
        default='./dataset/ljspeech_spectrograms/',
        help="Path to the directory with original .npy spectrograms."
    )

    parser.add_argument(
        "--original_wav_dir",
        type=str,
        default=None,
        help="(Optional) Path to directory with original .wav files for accurate baseline calculation."
    )
    
    # Path to the *results* of a train_subset run
    parser.add_argument(
        "-run_dir", "--run_dir", 
        type=str, 
        required=True,
        help="Path to the checkpoint directory of a train_subset run (e.g., ./checkpoints/ljspeech_subset_test/...)"
    )
    
    # Path to the main output folder
    parser.add_argument(
        "-o", "--output_dir", 
        type=str, 
        default="./outputs",
        help="Path to the main folder for saving plots and audio."
    )
    
    args = parser.parse_args()

    # --- Main script logic ---
    print(f"Starting run analysis...")
    print(f"Reconstruction source: {args.run_dir}")
    print(f"Original source: {args.original_dir}")
    print(f"Output directory: {args.output_dir}")

    # Ensure output directories exist
    os.makedirs(os.path.join(args.output_dir, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'audio'), exist_ok=True)
    
    # Find all reconstructed files in all subfolders of the run_dir
    # This looks for '.../LJ.../LJ..._fitting.npy'
    reconstructed_files = sorted(glob.glob(os.path.join(args.run_dir, '*', '*_fitting.npy')))
    
    if not reconstructed_files:
        print(f"\nError: No `..._fitting.npy` files found in directory: {args.run_dir}")
        print("Please ensure you ran `train_subset.py` with the `--save_imgs` flag and the path is correct.")
    else:
        print(f"\nFound {len(reconstructed_files)} reconstructed files. Starting processing...")

        # Collect metrics
        pesq_values = []
        stoi_values = []
        pesq_original_values = []
        stoi_original_values = []
        file_ids = []

        for i, recon_path in enumerate(reconstructed_files, 1):
            print(f"\n[{i}/{len(reconstructed_files)}]")
            file_id = os.path.basename(recon_path).replace('_fitting.npy', '')
            original_path = os.path.join(args.original_dir, f"{file_id}.npy")

            if not os.path.exists(original_path):
                print(f"  Warning: Original file not found for {file_id}. Skipping.")
                continue

            pesq_val, stoi_val, pesq_orig, stoi_orig = analyze_file(
                original_path, recon_path, args.output_dir, args.original_wav_dir
            )

            if pesq_val is not None and stoi_val is not None:
                file_ids.append(file_id)
                pesq_values.append(pesq_val)
                stoi_values.append(stoi_val)
                if pesq_orig is not None and stoi_orig is not None:
                    pesq_original_values.append(pesq_orig)
                    stoi_original_values.append(stoi_orig)

        # Calculate and save summary metrics
        if pesq_values and stoi_values:
            avg_pesq = np.mean(pesq_values)
            avg_stoi = np.mean(stoi_values)
            avg_pesq_original = np.mean(pesq_original_values) if pesq_original_values else 4.5
            avg_stoi_original = np.mean(stoi_original_values) if stoi_original_values else 1.0

            # Save metrics to CSV
            metrics_path = os.path.join(args.output_dir, 'metrics_summary.csv')
            with open(metrics_path, 'w') as f:
                f.write("file_id,pesq,stoi,pesq_original,stoi_original\n")
                for i, fid in enumerate(file_ids):
                    p_orig = pesq_original_values[i] if i < len(pesq_original_values) else avg_pesq_original
                    s_orig = stoi_original_values[i] if i < len(stoi_original_values) else avg_stoi_original
                    f.write(f"{fid},{pesq_values[i]:.4f},{stoi_values[i]:.4f},{p_orig:.4f},{s_orig:.4f}\n")
                f.write(f"\nAverage,{avg_pesq:.4f},{avg_stoi:.4f},{avg_pesq_original:.4f},{avg_stoi_original:.4f}\n")

            print("\n" + "="*60)
            print("--- Analysis Complete! ---")
            print(f"📊 Reconstructed - Average PESQ: {avg_pesq:.3f}, Average STOI: {avg_stoi:.4f}")
            print(f"📊 Original Baseline - Average PESQ: {avg_pesq_original:.3f}, Average STOI: {avg_stoi_original:.4f}")
            print(f"📊 Quality Gap - PESQ: {avg_pesq_original - avg_pesq:.3f}, STOI: {avg_stoi_original - avg_stoi:.4f}")
            print(f"Plot files are located in: {os.path.join(args.output_dir, 'plots')}")
            print(f"Audio files are located in: {os.path.join(args.output_dir, 'audio')}")
            print(f"Metrics saved to: {metrics_path}")
            print("="*60)
        else:
            print("\n" + "="*60)
            print("--- Analysis Complete (No metrics calculated) ---")
            print(f"Plot files are located in: {os.path.join(args.output_dir, 'plots')}")
            print(f"Audio files are located in: {os.path.join(args.output_dir, 'audio')}")
            print("="*60)
