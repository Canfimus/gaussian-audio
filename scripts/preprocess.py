import librosa
import numpy as np
import os
import glob
import argparse
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

# --- 3. Representation Mode ---
REPRESENTATION_MODE = 'real_imag'  # Options: 'real_imag' or 'amp_phase'

def preprocess_audio_files(input_dir=INPUT_WAV_DIR, output_dir=OUTPUT_NPY_DIR, mode=REPRESENTATION_MODE):
    """
    This script converts all .wav files from input_dir
    into 2-channel spectrograms and saves them as .npy files.

    Args:
        input_dir: Path to WAV files
        output_dir: Path to save .npy files
        mode: Representation mode - 'real_imag' or 'amp_phase'
              - 'real_imag': Real + Imaginary components (default, current behavior)
              - 'amp_phase': Amplitude (magnitude) + Phase components
    """

    print(f"Starting preprocessing...")
    print(f"Input WAV directory:  {input_dir}")
    print(f"Output NPY directory: {output_dir}")
    print(f"Representation mode:  {mode}")
    
    # 1. Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # 2. Find all .wav files in the input directory
    wav_files = glob.glob(os.path.join(input_dir, '*.wav'))

    if not wav_files:
        print(f"Warning: No .wav files found in {input_dir}")
        return

    print(f"Found {len(wav_files)} files to process.")

    # 3. Loop over all files with a progress bar
    for wav_path in tqdm(wav_files, desc=f"Processing audio files ({mode})"):
        try:
            # 4. Load the audio file
            #    sr=None preserves the original sample rate (22050Hz for LJSpeech)
            audio, sr = librosa.load(wav_path, sr=None)

            # 5. Calculate the complex-valued STFT
            S_complex = librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH)

            # 6. Split the complex spectrogram based on mode
            if mode == 'real_imag':
                # Real + Imaginary representation
                channel1 = np.real(S_complex)
                channel2 = np.imag(S_complex)
                mode_str = "Real+Imaginary"
            elif mode == 'amp_phase':
                # Amplitude + Phase representation
                channel1 = np.abs(S_complex)  # Amplitude (magnitude)
                channel2 = np.angle(S_complex)  # Phase
                mode_str = "Amplitude+Phase"
            else:
                raise ValueError(f"Unknown representation mode: {mode}. Use 'real_imag' or 'amp_phase'")

            # 7. Stack them into a single NumPy array
            #    The resulting shape will be (H, W, 2)
            #    H = Frequency bins (1 + N_FFT / 2)
            #    W = Time frames
            #    2 = Channels
            spectrogram_array = np.stack([channel1, channel2], axis=-1)

            # 8. Create the output path
            #    e.g., '.../wavs/LJ001-0001.wav' -> '.../spectrograms/LJ001-0001.npy'
            file_id = os.path.basename(wav_path).split('.')[0] # 'LJ001-0001'
            output_path = os.path.join(output_dir, f"{file_id}.npy")

            # 9. Save the array as a .npy file
            np.save(output_path, spectrogram_array)

        except Exception as e:
            print(f"Error processing file {wav_path}: {e}")

    print("\n--- Preprocessing Complete! ---")
    print(f".npy files are saved in: {output_dir}")
    print(f"Mode used: {mode_str if 'mode_str' in locals() else mode}")

# Run the main function when the script is executed
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess audio files into spectrograms")
    parser.add_argument(
        '--mode',
        type=str,
        default='real_imag',
        choices=['real_imag', 'amp_phase'],
        help="Representation mode: 'real_imag' (Real+Imaginary) or 'amp_phase' (Amplitude+Phase)"
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        default=INPUT_WAV_DIR,
        help="Path to input WAV files directory"
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help="Path to output .npy files directory (auto-generated based on mode if not specified)"
    )

    args = parser.parse_args()

    # Auto-generate output directory based on mode if not specified
    if args.output_dir is None:
        if args.mode == 'amp_phase':
            args.output_dir = './dataset/ljspeech_spectrograms_amp_phase/'
        else:
            args.output_dir = OUTPUT_NPY_DIR

    preprocess_audio_files(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        mode=args.mode
    )