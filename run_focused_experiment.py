#!/usr/bin/env python3
"""
Focused experiment runner for the 2000-5000 gps range with quantization.
Tests more granular Gaussian rates in the "sweet spot" range and calculates compression ratios.
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
DATASET_DIR = "./dataset/ljspeech_spectrograms/"
DATA_NAME = "focused_quantized_experiments"
ITERATIONS = 10000  # Higher quality for focused experiments

# Focused gaussian rates in the sweet spot (more granular)
GAUSSIAN_RATES = [2000, 2500, 3000, 3500, 4000, 4500, 5000]


def calculate_compression_ratio(checkpoint_dir, original_spectrograms_dir):
    """
    Calculate compression ratio by comparing compressed model size to original data size.
    """
    import glob

    # Get model file size (the trained Gaussian parameters)
    model_files = glob.glob(os.path.join(checkpoint_dir, '*/gaussian_model.pth.tar'))
    if not model_files:
        return None, None

    # Calculate total size of all model files
    total_model_size = sum(os.path.getsize(f) for f in model_files)

    # Calculate total size of original spectrograms for the same files
    original_files = glob.glob(os.path.join(original_spectrograms_dir, '*.npy'))
    # Match only the files that were trained
    trained_file_ids = [Path(mf).parent.name for mf in model_files]
    original_size = 0
    for orig_file in original_files:
        file_id = Path(orig_file).stem
        if file_id in trained_file_ids:
            original_size += os.path.getsize(orig_file)

    if original_size == 0:
        return None, None

    # Calculate compression ratio
    compression_ratio = original_size / total_model_size
    space_savings = (1 - (total_model_size / original_size)) * 100

    return compression_ratio, space_savings


def run_training(gps_rate, iterations, dataset_dir, data_name, use_quantization=False, save_imgs=True):
    """
    Run training for a specific gaussian per second rate.
    """
    quant_str = "with quantization" if use_quantization else "without quantization"
    print(f"\n{'='*80}")
    print(f"Training with {gps_rate} gaussians per second ({quant_str})")
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


def run_analysis(run_dir, output_dir, original_dir):
    """
    Run analysis on a trained model to generate metrics.
    """
    print(f"\n{'='*80}")
    print(f"Analyzing: {run_dir}")
    print(f"{'='*80}\n")

    cmd = [
        "python", "analyze_subset_fixed.py",
        "--run_dir", run_dir,
        "--output_dir", output_dir,
        "--original_dir", original_dir,
    ]

    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        print(f"✅ Analysis completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Analysis failed: {e}")
        return False


def plot_focused_results(comparison_df, experiments_dir):
    """
    Create visualization plots focused on the 2000-5000 gps range.
    Shows PESQ, STOI, and compression ratio.
    """
    if comparison_df is None or len(comparison_df) == 0:
        print("⚠️  No data to plot")
        return

    print(f"\n{'='*80}")
    print("Creating focused visualization plots...")
    print(f"{'='*80}\n")

    # Create figure with 3 rows, 2 columns
    fig = plt.figure(figsize=(18, 16))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    gps_rates = comparison_df['gaussians_per_second'].values
    pesq_values = comparison_df['avg_pesq'].values
    stoi_values = comparison_df['avg_stoi'].values
    compression_ratios = comparison_df['compression_ratio'].values

    # Reference values
    PERFECT_STOI = 1.0
    MAX_PESQ = 4.5

    # Plot 1: PESQ vs Gaussian Rate (focused range)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(gps_rates, pesq_values, 'o-', linewidth=2.5, markersize=10, color='#2E86AB', label='Reconstructed', zorder=3)
    ax1.axhline(y=MAX_PESQ, color='green', linestyle='--', linewidth=2, alpha=0.6, label='Original (Max)', zorder=2)
    ax1.set_xlabel('Gaussians per Second', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Average PESQ', fontsize=13, fontweight='bold')
    ax1.set_title('Audio PESQ vs Gaussian Rate (Focused)', fontsize=15, fontweight='bold')
    ax1.grid(True, alpha=0.3, zorder=1)
    ax1.legend(loc='lower right', fontsize=11)
    ax1.set_ylim([0, 5.0])

    # Add value labels
    for x, y in zip(gps_rates, pesq_values):
        ax1.annotate(f'{y:.2f}', (x, y), textcoords="offset points",
                    xytext=(0,10), ha='center', fontsize=10, fontweight='bold')

    # Plot 2: STOI vs Gaussian Rate (focused range)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(gps_rates, stoi_values, 'o-', linewidth=2.5, markersize=10, color='#A23B72', label='Reconstructed', zorder=3)
    ax2.axhline(y=PERFECT_STOI, color='green', linestyle='--', linewidth=2, alpha=0.6, label='Original (Perfect)', zorder=2)
    ax2.set_xlabel('Gaussians per Second', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Average STOI', fontsize=13, fontweight='bold')
    ax2.set_title('Audio STOI vs Gaussian Rate (Focused)', fontsize=15, fontweight='bold')
    ax2.grid(True, alpha=0.3, zorder=1)
    ax2.set_ylim([0, 1.05])
    ax2.legend(loc='lower right', fontsize=11)

    # Add value labels
    for x, y in zip(gps_rates, stoi_values):
        ax2.annotate(f'{y:.3f}', (x, y), textcoords="offset points",
                    xytext=(0,10), ha='center', fontsize=10, fontweight='bold')

    # Plot 3: Compression Ratio vs Gaussian Rate
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(gps_rates, compression_ratios, 'o-', linewidth=2.5, markersize=10, color='#F18F01', zorder=3)
    ax3.axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.4, label='No Compression', zorder=2)
    ax3.set_xlabel('Gaussians per Second', fontsize=13, fontweight='bold')
    ax3.set_ylabel('Compression Ratio (X:1)', fontsize=13, fontweight='bold')
    ax3.set_title('Compression Ratio vs Gaussian Rate', fontsize=15, fontweight='bold')
    ax3.grid(True, alpha=0.3, zorder=1)
    ax3.legend(loc='upper right', fontsize=11)

    # Add value labels
    for x, y in zip(gps_rates, compression_ratios):
        ax3.annotate(f'{y:.1f}x', (x, y), textcoords="offset points",
                    xytext=(0,10), ha='center', fontsize=10, fontweight='bold')

    # Plot 4: Quality-Compression Trade-off (PESQ vs Compression)
    ax4 = fig.add_subplot(gs[1, 1])
    scatter = ax4.scatter(compression_ratios, pesq_values, s=200, c=gps_rates,
                         cmap='viridis', edgecolors='black', linewidth=2, zorder=3)
    ax4.axhline(y=MAX_PESQ, color='green', linestyle='--', linewidth=2, alpha=0.4, label='Max PESQ', zorder=2)
    ax4.set_xlabel('Compression Ratio (X:1)', fontsize=13, fontweight='bold')
    ax4.set_ylabel('Average PESQ', fontsize=13, fontweight='bold')
    ax4.set_title('Quality vs Compression Trade-off', fontsize=15, fontweight='bold')
    ax4.grid(True, alpha=0.3, zorder=1)
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('Gaussians/sec', fontsize=11, fontweight='bold')
    ax4.legend(loc='lower right', fontsize=11)

    # Annotate points with gps values
    for x, y, gps in zip(compression_ratios, pesq_values, gps_rates):
        ax4.annotate(f'{int(gps)}', (x, y), textcoords="offset points",
                    xytext=(0,-15), ha='center', fontsize=9)

    # Plot 5: STOI vs Compression Trade-off
    ax5 = fig.add_subplot(gs[2, 0])
    scatter2 = ax5.scatter(compression_ratios, stoi_values, s=200, c=gps_rates,
                          cmap='plasma', edgecolors='black', linewidth=2, zorder=3)
    ax5.axhline(y=PERFECT_STOI, color='green', linestyle='--', linewidth=2, alpha=0.4, label='Perfect STOI', zorder=2)
    ax5.set_xlabel('Compression Ratio (X:1)', fontsize=13, fontweight='bold')
    ax5.set_ylabel('Average STOI', fontsize=13, fontweight='bold')
    ax5.set_title('Intelligibility vs Compression Trade-off', fontsize=15, fontweight='bold')
    ax5.grid(True, alpha=0.3, zorder=1)
    ax5.set_ylim([0, 1.05])
    cbar2 = plt.colorbar(scatter2, ax=ax5)
    cbar2.set_label('Gaussians/sec', fontsize=11, fontweight='bold')
    ax5.legend(loc='lower right', fontsize=11)

    # Annotate points
    for x, y, gps in zip(compression_ratios, stoi_values, gps_rates):
        ax5.annotate(f'{int(gps)}', (x, y), textcoords="offset points",
                    xytext=(0,-15), ha='center', fontsize=9)

    # Plot 6: Combined metrics bar chart
    ax6 = fig.add_subplot(gs[2, 1])
    x_pos = np.arange(len(gps_rates))
    width = 0.25

    # Normalize metrics to 0-1 for comparison
    pesq_norm = pesq_values / MAX_PESQ
    stoi_norm = stoi_values / PERFECT_STOI
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

    plt.suptitle('Focused Experiment: 2000-5000 gps with Quantization',
                 fontsize=18, fontweight='bold', y=0.995)

    # Save the plot
    plot_file = os.path.join(experiments_dir, "focused_experiment_results.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"✅ Focused results plot saved to: {plot_file}")


def collect_all_metrics(experiments_dir, original_dir):
    """
    Collect metrics from all experiment runs including compression ratios.
    """
    print(f"\n{'='*80}")
    print("Collecting metrics from all experiments...")
    print(f"{'='*80}\n")

    all_metrics = []

    for gps_rate in GAUSSIAN_RATES:
        output_dir = os.path.join(experiments_dir, f"analysis_{gps_rate}gps")
        metrics_file = os.path.join(output_dir, "metrics_summary.csv")

        if os.path.exists(metrics_file):
            try:
                # Read the CSV file
                df = pd.read_csv(metrics_file)
                avg_row = df[df['file_id'] == 'Average']

                if not avg_row.empty:
                    avg_pesq = avg_row['pesq'].values[0]
                    avg_stoi = avg_row['stoi'].values[0]

                    # Calculate compression ratio
                    checkpoint_dir = f"./checkpoints/{DATA_NAME}/GaussianImage_Cholesky_{ITERATIONS}_{gps_rate}gps"
                    comp_ratio, space_savings = calculate_compression_ratio(checkpoint_dir, original_dir)

                    if comp_ratio is None:
                        comp_ratio = 0
                        space_savings = 0

                    all_metrics.append({
                        'gaussians_per_second': gps_rate,
                        'avg_pesq': avg_pesq,
                        'avg_stoi': avg_stoi,
                        'compression_ratio': comp_ratio,
                        'space_savings_percent': space_savings
                    })
                    print(f"  ✅ {gps_rate} gps: PESQ={avg_pesq:.3f}, STOI={avg_stoi:.4f}, Compression={comp_ratio:.1f}x ({space_savings:.1f}% savings)")
                else:
                    print(f"  ⚠️  No average metrics found for {gps_rate} gps")
            except Exception as e:
                print(f"  ❌ Error reading metrics for {gps_rate} gps: {e}")
        else:
            print(f"  ⚠️  Metrics file not found for {gps_rate} gps: {metrics_file}")

    if all_metrics:
        comparison_df = pd.DataFrame(all_metrics)
        comparison_df = comparison_df.sort_values('gaussians_per_second')

        comparison_file = os.path.join(experiments_dir, "focused_experiment_comparison.csv")
        comparison_df.to_csv(comparison_file, index=False)

        print(f"\n{'='*80}")
        print("Focused Experiment Comparison:")
        print(f"{'='*80}")
        print(comparison_df.to_string(index=False))
        print(f"\n✅ Comparison saved to: {comparison_file}")

        return comparison_df
    else:
        print("\n❌ No metrics collected from any experiment")
        return None


def main():
    parser = argparse.ArgumentParser(description="Run focused Gaussian rate experiments (2000-5000 gps)")
    parser.add_argument("--skip_training", action="store_true", help="Skip training and only run analysis")
    parser.add_argument("--skip_analysis", action="store_true", help="Skip analysis and only run training")
    parser.add_argument("--iterations", type=int, default=ITERATIONS, help=f"Training iterations (default: {ITERATIONS})")
    parser.add_argument("--no_quantization", action="store_true", help="Disable quantization (enabled by default)")
    args = parser.parse_args()

    use_quantization = not args.no_quantization

    experiments_dir = f"./experiments/{DATA_NAME}"
    os.makedirs(experiments_dir, exist_ok=True)

    # Save experiment configuration
    config = {
        'dataset_dir': DATASET_DIR,
        'data_name': DATA_NAME,
        'iterations': args.iterations,
        'gaussian_rates': GAUSSIAN_RATES,
        'quantization_enabled': use_quantization,
        'description': 'Focused experiment on 2000-5000 gps range with compression analysis'
    }

    config_file = os.path.join(experiments_dir, "experiment_config.json")
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Experiment configuration saved to: {config_file}")

    # Run experiments
    if not args.skip_training:
        print(f"\n{'='*80}")
        print(f"Starting Focused Experiments")
        print(f"Range: {GAUSSIAN_RATES[0]}-{GAUSSIAN_RATES[-1]} gps ({len(GAUSSIAN_RATES)} points)")
        print(f"Quantization: {'ENABLED' if use_quantization else 'DISABLED'}")
        print(f"Iterations: {args.iterations}")
        print(f"{'='*80}\n")

        for gps_rate in GAUSSIAN_RATES:
            run_training(
                gps_rate=gps_rate,
                iterations=args.iterations,
                dataset_dir=DATASET_DIR,
                data_name=DATA_NAME,
                use_quantization=use_quantization,
                save_imgs=True
            )

    if not args.skip_analysis:
        print(f"\n{'='*80}")
        print("Running Analysis on Trained Models")
        print(f"{'='*80}\n")

        for gps_rate in GAUSSIAN_RATES:
            run_dir = f"./checkpoints/{DATA_NAME}/GaussianImage_Cholesky_{args.iterations}_{gps_rate}gps"

            if os.path.exists(run_dir):
                output_dir = os.path.join(experiments_dir, f"analysis_{gps_rate}gps")
                run_analysis(run_dir=run_dir, output_dir=output_dir, original_dir=DATASET_DIR)
            else:
                print(f"⚠️  Checkpoint directory not found for {gps_rate} gps: {run_dir}")

    # Collect and compare all metrics
    comparison_df = collect_all_metrics(experiments_dir, DATASET_DIR)

    # Create visualization plots
    if comparison_df is not None:
        plot_focused_results(comparison_df, experiments_dir)

    print(f"\n{'='*80}")
    print("Focused experiments completed!")
    print(f"Results directory: {experiments_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
