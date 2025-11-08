#!/usr/bin/env python3
"""
Run focused experiment using Amplitude/Phase representation.
Tests the 2000-5000 gps range with Amplitude+Phase spectrograms.
"""

import subprocess
import os
import sys
import json
from pathlib import Path
import pandas as pd
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration ---
ORIGINAL_WAV_DIR = "./dataset/LJSpeech-1.1/wavs/"
AMP_PHASE_DATASET_DIR = "./dataset/ljspeech_spectrograms_amp_phase/"
DATA_NAME = "amp_phase_experiment"
ITERATIONS = 10000

# Focused gaussian rates in the sweet spot
GAUSSIAN_RATES = [2000, 2500, 3000, 3500, 4000, 4500, 5000]


def preprocess_amp_phase(input_dir, output_dir):
    """
    Run preprocessing to create Amplitude/Phase spectrograms.
    """
    print(f"\n{'='*80}")
    print("Step 1: Preprocessing Audio to Amplitude/Phase Spectrograms")
    print(f"{'='*80}\n")

    if os.path.exists(output_dir) and len(os.listdir(output_dir)) > 0:
        print(f"✅ Amplitude/Phase spectrograms already exist in {output_dir}")
        print(f"   Skipping preprocessing. Delete the directory to re-preprocess.")
        return True

    cmd = [
        "python", "preprocess.py",
        "--mode", "amp_phase",
        "--input_dir", input_dir,
        "--output_dir", output_dir
    ]

    print(f"Running: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        print(f"✅ Preprocessing completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Preprocessing failed: {e}")
        return False


def run_training(gps_rate, iterations, dataset_dir, data_name, use_quantization=True, save_imgs=True):
    """
    Run training for a specific gaussian per second rate.
    """
    quant_str = "with quantization" if use_quantization else "without quantization"
    print(f"\n{'='*80}")
    print(f"Training with {gps_rate} gaussians per second ({quant_str})")
    print(f"Using Amplitude/Phase representation")
    print(f"{'='*80}\n")

    cmd = [
        "python", "train_subset.py",
        "--dataset", dataset_dir,
        "--data_name", data_name,
        "--iterations", str(iterations),
        "--gaussians_per_second", str(gps_rate),
        "--lr", "1e-3",
        "--seed", "42",
    ]

    if use_quantization:
        cmd.append("--quantize")

    if save_imgs:
        cmd.append("--save_imgs")

    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        print(f"✅ Training completed successfully for {gps_rate} gps")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed for {gps_rate} gps: {e}")
        return False


def run_analysis(run_dir, output_dir, original_spec_dir, original_wav_dir):
    """
    Run analysis on a trained model to generate metrics.
    IMPORTANT: Uses --mode amp_phase for Amplitude/Phase representation.
    """
    print(f"\n{'='*80}")
    print(f"Analyzing: {run_dir}")
    print(f"Mode: Amplitude/Phase")
    print(f"{'='*80}\n")

    cmd = [
        "python", "analyze_subset_fixed.py",
        "--run_dir", run_dir,
        "--output_dir", output_dir,
        "--original_dir", original_spec_dir,
        "--mode", "amp_phase",  # CRITICAL: Must specify amp_phase mode
    ]

    # Add original WAV directory if it exists
    if original_wav_dir and os.path.exists(original_wav_dir):
        cmd.extend(["--original_wav_dir", original_wav_dir])

    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        print(f"✅ Analysis completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Analysis failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run Amplitude/Phase experiments")
    parser.add_argument("--skip_preprocessing", action="store_true", help="Skip preprocessing step")
    parser.add_argument("--skip_training", action="store_true", help="Skip training and only run analysis")
    parser.add_argument("--skip_analysis", action="store_true", help="Skip analysis and only run training")
    parser.add_argument("--iterations", type=int, default=ITERATIONS, help=f"Training iterations (default: {ITERATIONS})")
    parser.add_argument("--no_quantization", action="store_true", help="Disable quantization")
    args = parser.parse_args()

    use_quantization = not args.no_quantization

    experiments_dir = f"./experiments/{DATA_NAME}"
    os.makedirs(experiments_dir, exist_ok=True)

    # Save experiment configuration
    config = {
        'representation_mode': 'amplitude_phase',
        'dataset_dir': AMP_PHASE_DATASET_DIR,
        'data_name': DATA_NAME,
        'iterations': args.iterations,
        'gaussian_rates': GAUSSIAN_RATES,
        'quantization_enabled': use_quantization,
        'description': 'Amplitude/Phase representation experiment (2000-5000 gps)'
    }

    config_file = os.path.join(experiments_dir, "experiment_config.json")
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Experiment configuration saved to: {config_file}")

    # Step 1: Preprocess audio to Amplitude/Phase
    if not args.skip_preprocessing:
        success = preprocess_amp_phase(ORIGINAL_WAV_DIR, AMP_PHASE_DATASET_DIR)
        if not success:
            print("\n❌ Preprocessing failed. Please check the error messages above.")
            sys.exit(1)

    # Check if preprocessed data exists
    if not os.path.exists(AMP_PHASE_DATASET_DIR) or len(os.listdir(AMP_PHASE_DATASET_DIR)) == 0:
        print(f"\n❌ Error: Amplitude/Phase spectrograms not found in {AMP_PHASE_DATASET_DIR}")
        print("   Run without --skip_preprocessing to create them.")
        sys.exit(1)

    # Step 2: Training
    if not args.skip_training:
        print(f"\n{'='*80}")
        print(f"Step 2: Training with Amplitude/Phase Spectrograms")
        print(f"Range: {GAUSSIAN_RATES[0]}-{GAUSSIAN_RATES[-1]} gps ({len(GAUSSIAN_RATES)} points)")
        print(f"Quantization: {'ENABLED' if use_quantization else 'DISABLED'}")
        print(f"Iterations: {args.iterations}")
        print(f"{'='*80}\n")

        for gps_rate in GAUSSIAN_RATES:
            run_training(
                gps_rate=gps_rate,
                iterations=args.iterations,
                dataset_dir=AMP_PHASE_DATASET_DIR,
                data_name=DATA_NAME,
                use_quantization=use_quantization,
                save_imgs=True
            )

    # Step 3: Analysis
    if not args.skip_analysis:
        print(f"\n{'='*80}")
        print("Step 3: Analyzing Results (with Amplitude/Phase mode)")
        print(f"{'='*80}\n")

        for gps_rate in GAUSSIAN_RATES:
            run_dir = f"./checkpoints/{DATA_NAME}/GaussianImage_Cholesky_{args.iterations}_{gps_rate}gps"

            if os.path.exists(run_dir):
                output_dir = os.path.join(experiments_dir, f"analysis_{gps_rate}gps")
                run_analysis(
                    run_dir=run_dir,
                    output_dir=output_dir,
                    original_spec_dir=AMP_PHASE_DATASET_DIR,
                    original_wav_dir=ORIGINAL_WAV_DIR
                )
            else:
                print(f"⚠️  Checkpoint directory not found for {gps_rate} gps: {run_dir}")

    # Step 4: Collect results
    print(f"\n{'='*80}")
    print("Step 4: Collecting Results")
    print(f"{'='*80}\n")

    all_metrics = []
    for gps_rate in GAUSSIAN_RATES:
        output_dir = os.path.join(experiments_dir, f"analysis_{gps_rate}gps")
        metrics_file = os.path.join(output_dir, "metrics_summary.csv")

        if os.path.exists(metrics_file):
            try:
                df = pd.read_csv(metrics_file)
                avg_row = df[df['file_id'] == 'Average']

                if not avg_row.empty:
                    avg_pesq = avg_row['pesq'].values[0]
                    avg_stoi = avg_row['stoi'].values[0]

                    # Get baseline values
                    if 'pesq_original' in avg_row and 'stoi_original' in avg_row:
                        avg_pesq_original = avg_row['pesq_original'].values[0]
                        avg_stoi_original = avg_row['stoi_original'].values[0]
                    else:
                        avg_pesq_original = 4.5
                        avg_stoi_original = 1.0

                    all_metrics.append({
                        'gaussians_per_second': gps_rate,
                        'avg_pesq': avg_pesq,
                        'avg_stoi': avg_stoi,
                        'avg_pesq_original': avg_pesq_original,
                        'avg_stoi_original': avg_stoi_original,
                    })
                    print(f"  ✅ {gps_rate} gps: PESQ={avg_pesq:.3f}, STOI={avg_stoi:.4f}")
            except Exception as e:
                print(f"  ❌ Error reading metrics for {gps_rate} gps: {e}")

    if all_metrics:
        comparison_df = pd.DataFrame(all_metrics)
        comparison_df = comparison_df.sort_values('gaussians_per_second')

        comparison_file = os.path.join(experiments_dir, "amp_phase_comparison.csv")
        comparison_df.to_csv(comparison_file, index=False)

        print(f"\n{'='*80}")
        print("Amplitude/Phase Experiment Results:")
        print(f"{'='*80}")
        print(comparison_df.to_string(index=False))
        print(f"\n✅ Comparison saved to: {comparison_file}")
    else:
        print("\n⚠️  No metrics collected")

    print(f"\n{'='*80}")
    print("Amplitude/Phase Experiment Complete!")
    print(f"Results directory: {experiments_dir}")
    print(f"{'='*80}\n")

    print("Next steps:")
    print("1. Check the results CSV file for metrics")
    print("2. Compare with Real/Imaginary results (if you have them)")
    print("3. Look at individual plots in the analysis folders")


if __name__ == "__main__":
    main()
