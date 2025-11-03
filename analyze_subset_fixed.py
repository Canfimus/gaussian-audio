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

def analyze_file(original_path, reconstructed_path, base_output_path):
    """
    Analyzes a single file: creates a plot and an audio file.
    """
    file_id = os.path.basename(original_path).split('.')[0]
    print(f"--- Processing: {file_id} ---")

    # --- 1. Load and prepare data ---
    spec_db_original, S_complex_original = load_and_convert_to_db(original_path)
    spec_db_reconstructed, S_complex_reconstructed = load_and_convert_to_db(reconstructed_path)

    if spec_db_original is None or spec_db_reconstructed is None:
        print(f"  Skipping file {file_id} due to loading error.")
        return

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
    
    # Also save original for comparison
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze a full subset run (generate plots and audio).")
    
    # Path to the *original* spectrograms
    parser.add_argument(
        "-orig_dir", "--original_dir", 
        type=str, 
        default='./dataset/ljspeech_spectrograms/',
        help="Path to the directory with original .npy spectrograms."
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
        
        for i, recon_path in enumerate(reconstructed_files, 1):
            print(f"\n[{i}/{len(reconstructed_files)}]")
            file_id = os.path.basename(recon_path).replace('_fitting.npy', '')
            original_path = os.path.join(args.original_dir, f"{file_id}.npy")
            
            if not os.path.exists(original_path):
                print(f"  Warning: Original file not found for {file_id}. Skipping.")
                continue
            
            analyze_file(original_path, recon_path, args.output_dir)

        print("\n" + "="*60)
        print("--- Analysis Complete! ---")
        print(f"Plot files are located in: {os.path.join(args.output_dir, 'plots')}")
        print(f"Audio files are located in: {os.path.join(args.output_dir, 'audio')}")
        print("="*60)
