import librosa
import numpy as np
import soundfile as sf
import os
import argparse

# --- 1. Spectrogram Parameters ---
# ! IMPORTANT: These must be identical to the parameters
# ! used in the preprocess.py script
N_FFT = 1024
HOP_LENGTH = 256

# Original sample rate of LJSpeech
ORIGINAL_SR = 22050

def convert_spec_to_audio(input_file, output_file):
    """
    Loads a 2-channel spectrogram (Real, Imaginary) from a .npy file,
    performs an iSTFT to reconstruct the audio, and saves it as a .wav file.
    """
    
    print(f"טוען ספקטרוגרמה משוחזרת מ: {input_file}")
    
    # 1. Load the (H, W, 2) array
    spec_ri = np.load(input_file)
    
    # 2. Reconstruct the complex spectrogram
    #    spec_ri[..., 0] is the Real part
    #    spec_ri[..., 1] is the Imaginary part
    real_part = spec_ri[..., 0]
    imag_part = spec_ri[..., 1]
    S_complex = real_part + 1j * imag_part
    
    print("מאחד ערוצים ממשי ומדומה... בוצע.")

    # 3. Perform iSTFT (the inverse of STFT)
    #    Must use the same hop_length and n_fft as the original
    audio_waveform = librosa.istft(S_complex, n_fft=N_FFT, hop_length=HOP_LENGTH)
    
    print("iSTFT הושלם.")

    # 4. Save as a .wav file
    #    We use the original sample rate
    sf.write(output_file, audio_waveform, ORIGINAL_SR)
    
    print(f"\n✅ הצלחה! האודיו המשוחזר נשמר ב: {output_file}")
    print("עכשיו אתה יכול להוריד את הקובץ הזה ולהאזין לו.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert GaussianImage .npy output back to a .wav file.")
    
    # This script now requires input and output paths to be provided
    
    parser.add_argument(
        "-i", "--input", 
        type=str, 
        required=True, # Make this argument mandatory
        help="Path to the input .npy file (the reconstructed spectrogram)."
    )
    
    parser.add_argument(
        "-o", "--output", 
        type=str, 
        required=True, # Make this argument mandatory
        help="Path to save the output .wav file (e.g., ./my_reconstruction.wav)."
    )
    
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"שגיאה: קובץ הקלט לא נמצא בנתיב {args.input}")
        print("אנא ודא שהנתיב נכון ושביצעת אימון עם הדגל --save_imgs.")
    else:
        convert_spec_to_audio(args.input, args.output)