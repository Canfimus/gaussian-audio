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

# Gaussian rates from very low to very high (including extremes)
GAUSSIAN_RATES = [1000, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 8000]


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
        "python", "scripts/preprocess.py",
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
        "python", "scripts/train_subset.py",
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


def calculate_compression_ratio(checkpoint_dir, original_wav_dir):
    """
    Calculate compression ratio by comparing compressed model size to ORIGINAL WAV files.

    This is the TRUE end-to-end compression:
    Original WAV → Gaussian Model (not Spectrogram → Gaussian Model)
    """
    import glob
    import librosa

    # Get model file size (the trained Gaussian parameters)
    model_files = glob.glob(os.path.join(checkpoint_dir, '*/gaussian_model.pth.tar'))
    if not model_files:
        return None, None

    # Calculate total size of all model files
    total_model_size = sum(os.path.getsize(f) for f in model_files)

    # Calculate total size of ORIGINAL WAV files for the same files
    trained_file_ids = [Path(mf).parent.name for mf in model_files]
    original_size = 0

    for file_id in trained_file_ids:
        wav_path = os.path.join(original_wav_dir, f"{file_id}.wav")
        if os.path.exists(wav_path):
            original_size += os.path.getsize(wav_path)
        else:
            print(f"  ⚠️  Warning: WAV file not found for {file_id}")

    if original_size == 0:
        return None, None

    # Calculate compression ratio (original / compressed)
    compression_ratio = original_size / total_model_size
    space_savings = (1 - (total_model_size / original_size)) * 100

    return compression_ratio, space_savings


def plot_amp_phase_results(comparison_df, experiments_dir):
    """
    Create visualization plots for Amplitude/Phase experiments.
    Shows PESQ, STOI, and compression ratio comparison.
    """
    if comparison_df is None or len(comparison_df) == 0:
        print("⚠️  No data to plot")
        return

    print(f"\n{'='*80}")
    print("Creating Amplitude/Phase visualization plots...")
    print(f"{'='*80}\n")

    # Create figure with 3 rows, 2 columns
    fig = plt.figure(figsize=(18, 16))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    gps_rates = comparison_df['gaussians_per_second'].values
    pesq_values = comparison_df['avg_pesq'].values
    stoi_values = comparison_df['avg_stoi'].values
    compression_ratios = comparison_df['compression_ratio'].values

    # Get actual measured baseline values
    if 'avg_pesq_original' in comparison_df and 'avg_stoi_original' in comparison_df:
        pesq_original_values = comparison_df['avg_pesq_original'].values
        stoi_original_values = comparison_df['avg_stoi_original'].values
        BASELINE_PESQ = np.mean(pesq_original_values)
        BASELINE_STOI = np.mean(stoi_original_values)
    else:
        BASELINE_PESQ = 4.5
        BASELINE_STOI = 1.0

    print(f"Using measured baseline: PESQ={BASELINE_PESQ:.3f}, STOI={BASELINE_STOI:.4f}")

    # Plot 1: PESQ vs Gaussian Rate
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(gps_rates, pesq_values, 'o-', linewidth=2.5, markersize=10, color='#2E86AB', label='Amp/Phase', zorder=3)
    ax1.axhline(y=BASELINE_PESQ, color='green', linestyle='--', linewidth=2, alpha=0.6, label=f'Baseline ({BASELINE_PESQ:.2f})', zorder=2)
    ax1.set_xlabel('Gaussians per Second', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Average PESQ', fontsize=13, fontweight='bold')
    ax1.set_title('PESQ vs Gaussian Rate (Amplitude/Phase)', fontsize=15, fontweight='bold')
    ax1.grid(True, alpha=0.3, zorder=1)
    ax1.legend(loc='lower right', fontsize=11)
    ax1.set_ylim([0, 5.0])

    for x, y in zip(gps_rates, pesq_values):
        ax1.annotate(f'{y:.2f}', (x, y), textcoords="offset points",
                    xytext=(0,10), ha='center', fontsize=9, fontweight='bold')

    # Plot 2: STOI vs Gaussian Rate
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(gps_rates, stoi_values, 'o-', linewidth=2.5, markersize=10, color='#A23B72', label='Amp/Phase', zorder=3)
    ax2.axhline(y=BASELINE_STOI, color='green', linestyle='--', linewidth=2, alpha=0.6, label=f'Baseline ({BASELINE_STOI:.3f})', zorder=2)
    ax2.set_xlabel('Gaussians per Second', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Average STOI', fontsize=13, fontweight='bold')
    ax2.set_title('STOI vs Gaussian Rate (Amplitude/Phase)', fontsize=15, fontweight='bold')
    ax2.grid(True, alpha=0.3, zorder=1)
    ax2.set_ylim([0, 1.05])
    ax2.legend(loc='lower right', fontsize=11)

    for x, y in zip(gps_rates, stoi_values):
        ax2.annotate(f'{y:.3f}', (x, y), textcoords="offset points",
                    xytext=(0,10), ha='center', fontsize=9, fontweight='bold')

    # Plot 3: Compression Ratio vs Gaussian Rate
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(gps_rates, compression_ratios, 'o-', linewidth=2.5, markersize=10, color='#F18F01', zorder=3)
    ax3.axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.4, label='No Compression (1x)', zorder=2)
    ax3.set_xlabel('Gaussians per Second', fontsize=13, fontweight='bold')
    ax3.set_ylabel('Compression Ratio (X:1)', fontsize=13, fontweight='bold')
    ax3.set_title('Compression Ratio vs Gaussian Rate', fontsize=15, fontweight='bold')
    ax3.grid(True, alpha=0.3, zorder=1)
    ax3.legend(loc='upper right', fontsize=11)

    for x, y in zip(gps_rates, compression_ratios):
        ax3.annotate(f'{y:.1f}x', (x, y), textcoords="offset points",
                    xytext=(0,10), ha='center', fontsize=9, fontweight='bold')

    # Plot 4: Quality-Compression Trade-off (PESQ vs Compression)
    ax4 = fig.add_subplot(gs[1, 1])
    scatter = ax4.scatter(compression_ratios, pesq_values, s=200, c=gps_rates,
                         cmap='viridis', edgecolors='black', linewidth=2, zorder=3)
    ax4.axhline(y=BASELINE_PESQ, color='green', linestyle='--', linewidth=2, alpha=0.4, label=f'Baseline PESQ', zorder=2)
    ax4.set_xlabel('Compression Ratio (X:1)', fontsize=13, fontweight='bold')
    ax4.set_ylabel('Average PESQ', fontsize=13, fontweight='bold')
    ax4.set_title('Quality vs Compression Trade-off', fontsize=15, fontweight='bold')
    ax4.grid(True, alpha=0.3, zorder=1)
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('Gaussians/sec', fontsize=11, fontweight='bold')
    ax4.legend(loc='lower right', fontsize=11)

    for x, y, gps in zip(compression_ratios, pesq_values, gps_rates):
        ax4.annotate(f'{int(gps)}', (x, y), textcoords="offset points",
                    xytext=(0,-15), ha='center', fontsize=8)

    # Plot 5: STOI vs Compression Trade-off
    ax5 = fig.add_subplot(gs[2, 0])
    scatter2 = ax5.scatter(compression_ratios, stoi_values, s=200, c=gps_rates,
                          cmap='plasma', edgecolors='black', linewidth=2, zorder=3)
    ax5.axhline(y=BASELINE_STOI, color='green', linestyle='--', linewidth=2, alpha=0.4, label=f'Baseline STOI', zorder=2)
    ax5.set_xlabel('Compression Ratio (X:1)', fontsize=13, fontweight='bold')
    ax5.set_ylabel('Average STOI', fontsize=13, fontweight='bold')
    ax5.set_title('Intelligibility vs Compression Trade-off', fontsize=15, fontweight='bold')
    ax5.grid(True, alpha=0.3, zorder=1)
    ax5.set_ylim([0, 1.05])
    cbar2 = plt.colorbar(scatter2, ax=ax5)
    cbar2.set_label('Gaussians/sec', fontsize=11, fontweight='bold')
    ax5.legend(loc='lower right', fontsize=11)

    for x, y, gps in zip(compression_ratios, stoi_values, gps_rates):
        ax5.annotate(f'{int(gps)}', (x, y), textcoords="offset points",
                    xytext=(0,-15), ha='center', fontsize=8)

    # Plot 6: Combined metrics bar chart
    ax6 = fig.add_subplot(gs[2, 1])
    x_pos = np.arange(len(gps_rates))
    width = 0.25

    # Normalize metrics to 0-1 for comparison
    pesq_norm = pesq_values / BASELINE_PESQ
    stoi_norm = stoi_values / BASELINE_STOI
    comp_norm = compression_ratios / compression_ratios.max()

    ax6.bar(x_pos - width, pesq_norm, width, label='PESQ (norm)', color='#2E86AB', alpha=0.8)
    ax6.bar(x_pos, stoi_norm, width, label='STOI (norm)', color='#A23B72', alpha=0.8)
    ax6.bar(x_pos + width, comp_norm, width, label='Compression (norm)', color='#F18F01', alpha=0.8)

    ax6.set_xlabel('Gaussians per Second', fontsize=13, fontweight='bold')
    ax6.set_ylabel('Normalized Score (0-1)', fontsize=13, fontweight='bold')
    ax6.set_title('All Metrics Comparison (Normalized)', fontsize=15, fontweight='bold')
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels([f'{int(g)}' for g in gps_rates])
    ax6.legend(loc='lower right', fontsize=11)
    ax6.grid(True, alpha=0.3, axis='y')
    ax6.set_ylim([0, 1.1])

    plt.suptitle(f'Amplitude/Phase Experiment Results\n(Baseline: PESQ={BASELINE_PESQ:.2f}, STOI={BASELINE_STOI:.3f})',
                 fontsize=18, fontweight='bold', y=0.998)

    # Save the plot
    plot_file = os.path.join(experiments_dir, "amp_phase_results.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"✅ Amplitude/Phase results plot saved to: {plot_file}")


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
        "python", "analysis/analyze_subset_fixed.py",
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

                    # Calculate compression ratio (comparing to ORIGINAL WAV files, not spectrograms)
                    checkpoint_dir = f"./checkpoints/{DATA_NAME}/GaussianImage_Cholesky_{args.iterations}_{gps_rate}gps"
                    comp_ratio, space_savings = calculate_compression_ratio(checkpoint_dir, ORIGINAL_WAV_DIR)

                    if comp_ratio is None:
                        comp_ratio = 0
                        space_savings = 0

                    all_metrics.append({
                        'gaussians_per_second': gps_rate,
                        'avg_pesq': avg_pesq,
                        'avg_stoi': avg_stoi,
                        'avg_pesq_original': avg_pesq_original,
                        'avg_stoi_original': avg_stoi_original,
                        'compression_ratio': comp_ratio,
                        'space_savings_percent': space_savings
                    })
                    print(f"  ✅ {gps_rate} gps: PESQ={avg_pesq:.3f}, STOI={avg_stoi:.4f}, Compression={comp_ratio:.1f}x ({space_savings:.1f}% savings)")
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

        # Create visualization plots
        plot_amp_phase_results(comparison_df, experiments_dir)
    else:
        print("\n⚠️  No metrics collected")

    print(f"\n{'='*80}")
    print("Amplitude/Phase Experiment Complete!")
    print(f"Results directory: {experiments_dir}")
    print(f"{'='*80}\n")

    print("Next steps:")
    print("1. Check the results CSV file for metrics")
    print("2. Check the visualization plot: amp_phase_results.png")
    print("3. Compare with Real/Imaginary results (if you have them)")
    print("4. Look at individual plots in the analysis folders")


if __name__ == "__main__":
    main()
