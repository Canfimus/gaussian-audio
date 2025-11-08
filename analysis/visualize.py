import numpy as np
import matplotlib
# Add this line right at the top, after imports
# This tells matplotlib not to try and open an interactive window
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import librosa
import argparse
import os

def load_and_convert_to_db(npy_path):
    """Loads a 2-ch (Real, Imag) .npy file and converts to log-magnitude (dB)."""
    # Load the (H, W, 2) array
    try:
        spec_ri = np.load(npy_path)
    except Exception as e:
        print(f"שגיאה בטעינת הקובץ: {npy_path}")
        print(f"פרטי השגיאה: {e}")
        return None
        
    # Reconstruct the complex spectrogram
    # spec_ri[..., 0] is Real, spec_ri[..., 1] is Imaginary
    S_complex = spec_ri[..., 0] + 1j * spec_ri[..., 1]
    
    # Convert to magnitude (absolute value of complex numbers)
    S_magnitude = np.abs(S_complex)
    
    # Convert to log-magnitude (decibels) for visualization
    S_db = librosa.amplitude_to_db(S_magnitude, ref=np.max)
    
    return S_db

def plot_spectrograms(original_path, reconstructed_path, output_filename):
    """Loads and plots two spectrograms side-by-side."""
    
    print(f"טוען מקור: {original_path}")
    spec_db_original = load_and_convert_to_db(original_path)
    if spec_db_original is None:
        return

    print(f"טוען שחזור: {reconstructed_path}")
    spec_db_reconstructed = load_and_convert_to_db(reconstructed_path)
    if spec_db_reconstructed is None:
        return
    
    # Create a plot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    
    # Plot Original
    img1 = librosa.display.specshow(spec_db_original, ax=ax1, y_axis='log', x_axis='time')
    ax1.set_title("Original Spectrogram")
    
    # Plot Reconstructed
    img2 = librosa.display.specshow(spec_db_reconstructed, ax=ax2, y_axis='log', x_axis='time')
    ax2.set_title("Reconstructed (GaussianImage)")
    
    # Add a colorbar
    fig.colorbar(img1, ax=[ax1, ax2], format="%+2.0f dB")
    
    plt.suptitle("Spectrogram Comparison")
    plt.tight_layout()
    
    # Save the figure to a file
    plt.savefig(output_filename)
    print(f"\n✅ הצלחה! התמונה נשמרה ב: {output_filename}")
    print("עכשיו אתה יכול לפתוח את קובץ התמונה ב-VS Code.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare original and reconstructed .npy spectrograms.")
    
    # Argument for the original ground-truth file
    parser.add_argument(
        "-orig", "--original", 
        type=str, 
        required=True,
        help="Path to the original ground-truth .npy file."
    )
    
    # Argument for the reconstructed file (from the model)
    parser.add_argument(
        "-recon", "--reconstructed", 
        type=str, 
        required=True,
        help="Path to the reconstructed .npy file (e.g., ..._fitting.npy)."
    )
    
    # Argument for the output plot file
    parser.add_argument(
        "-o", "--output", 
        type=str, 
        default="./comparison_plot.png",
        help="Path to save the output .png plot."
    )
    
    args = parser.parse_args()

    if not os.path.exists(args.original):
        print(f"שגיאה: קובץ המקור לא נמצא ב: {args.original}")
    elif not os.path.exists(args.reconstructed):
        print(f"שגיאה: הקובץ המשוחזר לא נמצא ב: {args.reconstructed}")
        print("אנא ודא שהנתיב נכון ושביצעת אימון עם הדגל --save_imgs.")
    else:
        plot_spectrograms(args.original, args.reconstructed, args.output)