import librosa
import numpy as np
import os
import glob
from tqdm import tqdm # For a nice progress bar

# --- 1. User Configuration ---
# (You only need to change these two paths)

# 📍 Path to the 'wavs' folder in your LJSpeech dataset
INPUT_WAV_DIR = './dataset/LJSpeech-1.1/wavs/'

# 📍 Path to the *new* empty folder where .npy files will be saved
OUTPUT_NPY_DIR = './dataset/ljspeech_spectrograms/'

# --- 2. Spectrogram Parameters ---
N_FFT = 1024        # FFT window size
HOP_LENGTH = 256    # Step size between frames

def preprocess_audio_files():
    """
    This script converts all .wav files from INPUT_WAV_DIR
    into 2-channel spectrograms (Real + Imaginary)
    and saves them as .npy files in OUTPUT_NPY_DIR.
    """
    
    print(f"Starting preprocessing...")
    print(f"Input WAV directory:  {INPUT_WAV_DIR}")
    print(f"Output NPY directory: {OUTPUT_NPY_DIR}")
    
    # 1. Ensure the output directory exists
    os.makedirs(OUTPUT_NPY_DIR, exist_ok=True)
    
    # 2. Find all .wav files in the input directory
    wav_files = glob.glob(os.path.join(INPUT_WAV_DIR, '*.wav'))
    
    if not wav_files:
        print(f"Warning: No .wav files found in {INPUT_WAV_DIR}")
        return

    print(f"Found {len(wav_files)} files to process.")

    # 3. Loop over all files with a progress bar
    for wav_path in tqdm(wav_files, desc="Processing audio files"):
        try:
            # 4. Load the audio file
            #    sr=None preserves the original sample rate (22050Hz for LJSpeech)
            audio, sr = librosa.load(wav_path, sr=None)
            
            # 5. Calculate the complex-valued STFT
            S_complex = librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH)
            
            # 6. Split the complex spectrogram into Real and Imaginary channels
            real_part = np.real(S_complex)
            imag_part = np.imag(S_complex)
            
            # 7. Stack them into a single NumPy array
            #    The resulting shape will be (H, W, 2)
            #    H = Frequency bins (1 + N_FFT / 2)
            #    W = Time frames
            #    2 = Channels (Real, Imag)
            #    This is the exact format our train_audio.py script expects!
            spectrogram_array = np.stack([real_part, imag_part], axis=-1)
            
            # 8. Create the output path
            #    e.g., '.../wavs/LJ001-0001.wav' -> '.../spectrograms/LJ001-0001.npy'
            file_id = os.path.basename(wav_path).split('.')[0] # 'LJ001-0001'
            output_path = os.path.join(OUTPUT_NPY_DIR, f"{file_id}.npy")
            
            # 9. Save the array as a .npy file
            np.save(output_path, spectrogram_array)
            
        except Exception as e:
            print(f"Error processing file {wav_path}: {e}")

    print("\n--- Preprocessing Complete! ---")
    print(f".npy files are saved in: {OUTPUT_NPY_DIR}")

# Run the main function when the script is executed
if __name__ == '__main__':
    preprocess_audio_files()