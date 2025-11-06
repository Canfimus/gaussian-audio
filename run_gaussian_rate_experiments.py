#!/usr/bin/env python3
"""
Experiment runner to test different Gaussian per second rates.
This script will:
1. Train models with different gaussian rates (e.g., 1000, 2000, 5000, 10000 gaussians/sec)
2. Analyze each trained model
3. Collect and compare PSNR and STOI metrics
"""

import subprocess
import os
import sys
import json
from pathlib import Path
import pandas as pd
import argparse
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration ---
DATASET_DIR = "./dataset/ljspeech_spectrograms/"
DATA_NAME = "gaussian_rate_experiments"
ITERATIONS = 10000  # Training iterations per file
SUBSET_SIZE = 50  # Number of files to train on

# Different gaussian rates to test (gaussians per second)
GAUSSIAN_RATES = [500, 1000, 2000, 5000, 10000, 20000]


def run_training(gps_rate, iterations, dataset_dir, data_name, save_imgs=True):
    """
    Run training for a specific gaussian per second rate.
    """
    print(f"\n{'='*80}")
    print(f"Training with {gps_rate} gaussians per second")
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


def plot_experiment_results(comparison_df, experiments_dir):
    """
    Create visualization plots of experiment results.
    """
    if comparison_df is None or len(comparison_df) == 0:
        print("⚠️  No data to plot")
        return

    print(f"\n{'='*80}")
    print("Creating visualization plots...")
    print(f"{'='*80}\n")

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    gps_rates = comparison_df['gaussians_per_second'].values
    psnr_values = comparison_df['avg_psnr_db'].values
    stoi_values = comparison_df['avg_stoi'].values

    # Plot 1: PSNR vs Gaussian Rate
    axes[0, 0].plot(gps_rates, psnr_values, 'o-', linewidth=2, markersize=8, color='#2E86AB')
    axes[0, 0].set_xlabel('Gaussians per Second', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Average PSNR (dB)', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('Audio PSNR vs Gaussian Rate', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xscale('log')

    # Add value labels
    for x, y in zip(gps_rates, psnr_values):
        axes[0, 0].annotate(f'{y:.1f}', (x, y), textcoords="offset points",
                           xytext=(0,10), ha='center', fontsize=9)

    # Plot 2: STOI vs Gaussian Rate
    axes[0, 1].plot(gps_rates, stoi_values, 'o-', linewidth=2, markersize=8, color='#A23B72')
    axes[0, 1].set_xlabel('Gaussians per Second', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Average STOI', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('Audio STOI vs Gaussian Rate', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xscale('log')
    axes[0, 1].set_ylim([0, 1.0])

    # Add value labels
    for x, y in zip(gps_rates, stoi_values):
        axes[0, 1].annotate(f'{y:.3f}', (x, y), textcoords="offset points",
                           xytext=(0,10), ha='center', fontsize=9)

    # Plot 3: Bar chart comparison
    x_pos = np.arange(len(gps_rates))
    axes[1, 0].bar(x_pos, psnr_values, color='#2E86AB', alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Gaussians per Second', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Average PSNR (dB)', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('PSNR Comparison (Bar Chart)', fontsize=14, fontweight='bold')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels([f'{int(g)}' for g in gps_rates], rotation=45)
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, (x, y) in enumerate(zip(x_pos, psnr_values)):
        axes[1, 0].text(x, y + 0.5, f'{y:.1f}', ha='center', fontsize=9)

    # Plot 4: Dual axis plot (PSNR and STOI together)
    ax4 = axes[1, 1]
    ax4_twin = ax4.twinx()

    line1 = ax4.plot(gps_rates, psnr_values, 'o-', linewidth=2, markersize=8,
                     color='#2E86AB', label='PSNR')
    ax4.set_xlabel('Gaussians per Second', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Average PSNR (dB)', fontsize=12, fontweight='bold', color='#2E86AB')
    ax4.tick_params(axis='y', labelcolor='#2E86AB')
    ax4.set_xscale('log')
    ax4.grid(True, alpha=0.3)

    line2 = ax4_twin.plot(gps_rates, stoi_values, 's-', linewidth=2, markersize=8,
                         color='#A23B72', label='STOI')
    ax4_twin.set_ylabel('Average STOI', fontsize=12, fontweight='bold', color='#A23B72')
    ax4_twin.tick_params(axis='y', labelcolor='#A23B72')
    ax4_twin.set_ylim([0, 1.0])

    ax4.set_title('PSNR and STOI vs Gaussian Rate', fontsize=14, fontweight='bold')

    # Add legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, loc='best')

    plt.suptitle('Gaussian Rate Experiment Results', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    # Save the plot
    plot_file = os.path.join(experiments_dir, "experiment_results.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"✅ Results plot saved to: {plot_file}")


def collect_all_metrics(experiments_dir):
    """
    Collect metrics from all experiment runs and create a comparison table.
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

                # Get the average row (last row)
                avg_row = df[df['file_id'] == 'Average']

                if not avg_row.empty:
                    avg_psnr = avg_row['psnr_db'].values[0]
                    avg_stoi = avg_row['stoi'].values[0]

                    all_metrics.append({
                        'gaussians_per_second': gps_rate,
                        'avg_psnr_db': avg_psnr,
                        'avg_stoi': avg_stoi
                    })
                    print(f"  ✅ {gps_rate} gps: PSNR={avg_psnr:.2f} dB, STOI={avg_stoi:.4f}")
                else:
                    print(f"  ⚠️  No average metrics found for {gps_rate} gps")
            except Exception as e:
                print(f"  ❌ Error reading metrics for {gps_rate} gps: {e}")
        else:
            print(f"  ⚠️  Metrics file not found for {gps_rate} gps: {metrics_file}")

    if all_metrics:
        # Create comparison dataframe
        comparison_df = pd.DataFrame(all_metrics)
        comparison_df = comparison_df.sort_values('gaussians_per_second')

        # Save to CSV
        comparison_file = os.path.join(experiments_dir, "experiment_comparison.csv")
        comparison_df.to_csv(comparison_file, index=False)

        print(f"\n{'='*80}")
        print("Experiment Comparison:")
        print(f"{'='*80}")
        print(comparison_df.to_string(index=False))
        print(f"\n✅ Comparison saved to: {comparison_file}")

        return comparison_df
    else:
        print("\n❌ No metrics collected from any experiment")
        return None


def main():
    parser = argparse.ArgumentParser(description="Run Gaussian rate experiments")
    parser.add_argument(
        "--skip_training",
        action="store_true",
        help="Skip training and only run analysis/comparison"
    )
    parser.add_argument(
        "--skip_analysis",
        action="store_true",
        help="Skip analysis and only run training"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=ITERATIONS,
        help=f"Training iterations (default: {ITERATIONS})"
    )
    parser.add_argument(
        "--rates",
        type=str,
        default=None,
        help=f"Comma-separated list of gaussian rates to test (default: {','.join(map(str, GAUSSIAN_RATES))})"
    )
    args = parser.parse_args()

    # Parse custom rates if provided
    rates_to_test = GAUSSIAN_RATES
    if args.rates:
        rates_to_test = [int(r.strip()) for r in args.rates.split(',')]

    # Create experiments directory
    experiments_dir = f"./experiments/{DATA_NAME}"
    os.makedirs(experiments_dir, exist_ok=True)

    # Save experiment configuration
    config = {
        'dataset_dir': DATASET_DIR,
        'data_name': DATA_NAME,
        'iterations': args.iterations,
        'subset_size': SUBSET_SIZE,
        'gaussian_rates': rates_to_test,
    }

    config_file = os.path.join(experiments_dir, "experiment_config.json")
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Experiment configuration saved to: {config_file}")

    # Run experiments
    successful_runs = []

    if not args.skip_training:
        print(f"\n{'='*80}")
        print(f"Starting Experiments: Testing {len(rates_to_test)} different Gaussian rates")
        print(f"Subset size: {SUBSET_SIZE} files")
        print(f"Iterations per file: {args.iterations}")
        print(f"{'='*80}\n")

        for gps_rate in rates_to_test:
            success = run_training(
                gps_rate=gps_rate,
                iterations=args.iterations,
                dataset_dir=DATASET_DIR,
                data_name=DATA_NAME,
                save_imgs=True
            )

            if success:
                successful_runs.append(gps_rate)

        print(f"\n✅ Training completed for {len(successful_runs)}/{len(rates_to_test)} configurations")

    if not args.skip_analysis:
        print(f"\n{'='*80}")
        print("Running Analysis on Trained Models")
        print(f"{'='*80}\n")

        for gps_rate in rates_to_test:
            # Construct checkpoint directory path
            run_dir = f"./checkpoints/{DATA_NAME}/GaussianImage_Cholesky_{args.iterations}_{gps_rate}gps"

            if os.path.exists(run_dir):
                output_dir = os.path.join(experiments_dir, f"analysis_{gps_rate}gps")

                run_analysis(
                    run_dir=run_dir,
                    output_dir=output_dir,
                    original_dir=DATASET_DIR
                )
            else:
                print(f"⚠️  Checkpoint directory not found for {gps_rate} gps: {run_dir}")

    # Collect and compare all metrics
    comparison_df = collect_all_metrics(experiments_dir)

    # Create visualization plots
    if comparison_df is not None:
        plot_experiment_results(comparison_df, experiments_dir)

    print(f"\n{'='*80}")
    print("All experiments completed!")
    print(f"Results directory: {experiments_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
