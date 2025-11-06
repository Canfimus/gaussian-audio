"""
Baseline Codec Comparison Script
Generates audio using various codecs at different bitrates for comparison.
"""

import os
import subprocess
import glob
import argparse
from pathlib import Path
from tqdm import tqdm

def check_codec_installed(codec_name):
    """Check if a codec is installed."""
    try:
        subprocess.run([codec_name, '--version'], 
                      stdout=subprocess.DEVNULL, 
                      stderr=subprocess.DEVNULL)
        return True
    except FileNotFoundError:
        return False

def encode_opus(input_file, output_dir, bitrate):
    """Encode audio file using Opus codec."""
    base_name = Path(input_file).stem.replace('_original', '')
    opus_file = os.path.join(output_dir, f"{base_name}.opus")
    decoded_file = os.path.join(output_dir, f"{base_name}_reconstructed.wav")
    
    # Encode
    encode_cmd = ['opusenc', '--bitrate', str(bitrate), input_file, opus_file]
    subprocess.run(encode_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Decode
    decode_cmd = ['opusdec', opus_file, decoded_file]
    subprocess.run(decode_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Get file size for bpp calculation (optional)
    file_size = os.path.getsize(opus_file)
    
    # Clean up .opus file
    os.remove(opus_file)
    
    return decoded_file, file_size

def encode_mp3(input_file, output_dir, bitrate):
    """Encode audio file using MP3 (LAME)."""
    base_name = Path(input_file).stem.replace('_original', '')
    mp3_file = os.path.join(output_dir, f"{base_name}.mp3")
    decoded_file = os.path.join(output_dir, f"{base_name}_reconstructed.wav")
    
    # Encode
    encode_cmd = ['lame', '-b', str(bitrate), input_file, mp3_file]
    subprocess.run(encode_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Decode
    decode_cmd = ['lame', '--decode', mp3_file, decoded_file]
    subprocess.run(decode_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Get file size
    file_size = os.path.getsize(mp3_file)
    
    # Clean up .mp3 file
    os.remove(mp3_file)
    
    return decoded_file, file_size

def generate_baselines(input_dir, output_base_dir):
    """Generate baselines for all audio files."""
    
    print("="*60)
    print("Baseline Codec Generation")
    print("="*60)
    print()
    
    # Check which codecs are available
    codecs_available = {
        'opus': check_codec_installed('opusenc'),
        'mp3': check_codec_installed('lame')
    }
    
    print("Codec Availability:")
    for codec, available in codecs_available.items():
        status = "✅ Available" if available else "❌ Not installed"
        print(f"  {codec.upper()}: {status}")
    print()
    
    if not any(codecs_available.values()):
        print("❌ No codecs available. Please install:")
        print("  - Opus: sudo apt-get install opus-tools")
        print("  - MP3:  sudo apt-get install lame")
        return
    
    # Find all original audio files
    audio_files = sorted(glob.glob(os.path.join(input_dir, '*_original.wav')))
    
    if not audio_files:
        print(f"❌ No '*_original.wav' files found in {input_dir}")
        return
    
    print(f"Found {len(audio_files)} audio files to process.\n")
    
    # Define bitrates to test
    opus_bitrates = [16, 32, 64]  # kbps
    mp3_bitrates = [64, 128, 192]  # kbps
    
    # Generate baselines
    if codecs_available['opus']:
        print("Generating Opus baselines...")
        for bitrate in opus_bitrates:
            output_dir = os.path.join(output_base_dir, f'opus_{bitrate}k')
            os.makedirs(output_dir, exist_ok=True)
            
            for audio_file in tqdm(audio_files, desc=f"Opus @ {bitrate} kbps"):
                try:
                    encode_opus(audio_file, output_dir, bitrate)
                except Exception as e:
                    print(f"  Error processing {audio_file}: {e}")
        print()
    
    if codecs_available['mp3']:
        print("Generating MP3 baselines...")
        for bitrate in mp3_bitrates:
            output_dir = os.path.join(output_base_dir, f'mp3_{bitrate}k')
            os.makedirs(output_dir, exist_ok=True)
            
            for audio_file in tqdm(audio_files, desc=f"MP3 @ {bitrate} kbps"):
                try:
                    encode_mp3(audio_file, output_dir, bitrate)
                except Exception as e:
                    print(f"  Error processing {audio_file}: {e}")
        print()
    
    print("✅ Baseline generation complete!")
    print(f"Baselines saved in: {output_base_dir}")
    print()
    print("To evaluate, run:")
    if codecs_available['opus']:
        for bitrate in opus_bitrates:
            print(f"  python evaluate_audio_quality.py -orig {input_dir} -recon {output_base_dir}/opus_{bitrate}k -o opus_{bitrate}k_results.txt")
    if codecs_available['mp3']:
        for bitrate in mp3_bitrates:
            print(f"  python evaluate_audio_quality.py -orig {input_dir} -recon {output_base_dir}/mp3_{bitrate}k -o mp3_{bitrate}k_results.txt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate codec baselines for comparison")
    
    parser.add_argument(
        "-i", "--input_dir",
        type=str,
        default="./outputs/audio",
        help="Directory containing original audio files (*_original.wav)"
    )
    
    parser.add_argument(
        "-o", "--output_dir",
        type=str,
        default="./baselines",
        help="Base directory for baseline outputs"
    )
    
    args = parser.parse_args()
    
    generate_baselines(args.input_dir, args.output_dir)