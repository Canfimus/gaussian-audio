import librosa
import numpy as np
import soundfile as sf
import os
import glob
import argparse
import matplotlib
matplotlib.use('Agg') # Use 'Agg' backend for saving plots on servers (no GUI)
import matplotlib.pyplot as plt

# --- 1. Spectrogram Parameters (Must match preprocess.py) ---
N_FFT = 1024
HOP_LENGTH = 256
ORIGINAL_SR = 22050

def load_and_convert_to_db(npy_path):
    """Loads a 2-ch (Real, Imag) .npy file and converts to log-magnitude (dB)."""
    try:
        spec_ri = np.load(npy_path)
    except Exception as e:
        print(f"  שגיאה בטעינת קובץ: {npy_path} | {e}")
        return None, None
        
    # Reconstruct the complex spectrogram
    S_complex = spec_ri[..., 0] + 1j * spec_ri[..., 1]
    
    # Convert to magnitude
    S_magnitude = np.abs(S_complex)
    
    # Convert to log-magnitude (decibels) for visualization
    S_db = librosa.amplitude_to_db(S_magnitude, ref=np.max)
    
    return S_db, S_complex # Return complex spec for audio

def analyze_file(original_path, reconstructed_path, base_output_path):
    """
    Analyzes a single file: creates a plot and an audio file.
    """
    file_id = os.path.basename(original_path).split('.')[0]
    print(f"--- מעבד את: {file_id} ---")

    # --- 1. Load and prepare data ---
    spec_db_original, _ = load_and_convert_to_db(original_path)
    spec_db_reconstructed, S_complex_reconstructed = load_and_convert_to_db(reconstructed_path)

    if spec_db_original is None or spec_db_reconstructed is None:
        print(f"  דילוג על הקובץ {file_id} עקב שגיאת טעינה.")
        return

    # --- 2. Generate and save plot ---
    plot_output_path = os.path.join(base_output_path, 'plots', f"{file_id}_comparison.png")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    
    librosa.display.specshow(spec_db_original, ax=ax1, y_axis='log', x_axis='time')
    ax1.set_title("Original Spectrogram")
    
    librosa.display.specshow(spec_db_reconstructed, ax=ax2, y_axis='log', x_axis='time')
    ax2.set_title("Reconstructed (GaussianImage)")
    
    fig.colorbar(librosa.display.specshow(spec_db_original, ax=ax1), ax=[ax1, ax2], format="%+2.0f dB")
    plt.suptitle(f"Comparison: {file_id}")
    plt.tight_layout()
    plt.savefig(plot_output_path)
    plt.close(fig) # Close the figure to save memory
    print(f"  ✅ תמונה נשמרה ב: {plot_output_path}")

    # --- 3. Generate and save audio ---
    audio_output_path = os.path.join(base_output_path, 'audio', f"{file_id}_reconstructed.wav")
    
    # Perform iSTFT
    audio_waveform = librosa.istft(S_complex_reconstructed, n_fft=N_FFT, hop_length=HOP_LENGTH)
    
    # Save as .wav file
    sf.write(audio_output_path, audio_waveform, ORIGINAL_SR)
    print(f"  ✅ אודיו נשמר ב: {audio_output_path}")


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
    print(f"מתחיל ניתוח ריצה...")
    print(f"מקור השחזורים: {args.run_dir}")
    print(f"מקור המקורים: {args.original_dir}")
    print(f"תיקיית פלט: {args.output_dir}")

    # Ensure output directories exist
    os.makedirs(os.path.join(args.output_dir, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'audio'), exist_ok=True)
    
    # Find all reconstructed files in all subfolders of the run_dir
    # This looks for '.../LJ.../LJ..._fitting.npy'
    reconstructed_files = sorted(glob.glob(os.path.join(args.run_dir, '*', '*_fitting.npy')))
    
    if not reconstructed_files:
        print(f"\nשגיאה: לא נמצאו קבצי `..._fitting.npy` בתיקייה: {args.run_dir}")
        print("אנא ודא שהרצת את `train_subset.py` עם הדגל `--save_imgs` ושהנתיב נכון.")
    else:
        print(f"\nנמצאו {len(reconstructed_files)} קבצים משוחזרים. מתחיל עיבוד...")
        
        for recon_path in reconstructed_files:
            file_id = os.path.basename(recon_path).replace('_fitting.npy', '')
            original_path = os.path.join(args.original_dir, f"{file_id}.npy")
            
            if not os.path.exists(original_path):
                print(f"  אזהרה: לא נמצא קובץ מקור תואם עבור {file_id}. מדלג.")
                continue
            
            analyze_file(original_path, recon_path, args.output_dir)

        print("\n--- ניתוח הושלם! ---")
        print(f"קבצי התמונה נמצאים ב: {os.path.join(args.output_dir, 'plots')}")
        print(f"קבצי האודיו נמצאים ב: {os.path.join(args.output_dir, 'audio')}")