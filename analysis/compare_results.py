"""
Results Comparison and Visualization
Compares GaussianImage results with baseline codecs
"""

import numpy as np
import matplotlib.pyplot as plt
import re
import argparse
from pathlib import Path

def parse_results_file(filepath):
    """Parse evaluation results from text file."""
    results = {}
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Extract average results
    avg_section = content.split("AVERAGE RESULTS")[-1]
    
    metrics = ['PESQ', 'STOI', 'VUV_F1', 'SNR_dB']
    for metric in metrics:
        pattern = rf"{metric}:\s*([\d.]+)"
        match = re.search(pattern, avg_section)
        if match:
            results[metric] = float(match.group(1))
        else:
            results[metric] = None
    
    return results

def create_comparison_plot(results_dict, output_path='comparison.png'):
    """Create a comparison plot of all methods."""
    
    methods = list(results_dict.keys())
    metrics = ['PESQ', 'STOI', 'VUV_F1']  # SNR_dB on separate plot
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Audio Quality Comparison', fontsize=16, fontweight='bold')
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(methods)))
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        # Get values for this metric
        values = []
        labels = []
        for method, results in results_dict.items():
            if results.get(metric) is not None:
                values.append(results[metric])
                labels.append(method)
        
        # Create bar plot
        bars = ax.bar(range(len(values)), values, color=colors[:len(values)])
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel(metric, fontweight='bold')
        ax.set_title(f'{metric} Comparison')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=9)
        
        # Add reference lines for "good" performance
        if metric == 'PESQ':
            ax.axhline(y=3.0, color='green', linestyle='--', alpha=0.5, label='Good (3.0)')
            ax.axhline(y=2.5, color='orange', linestyle='--', alpha=0.5, label='Acceptable (2.5)')
        elif metric == 'STOI':
            ax.axhline(y=0.85, color='green', linestyle='--', alpha=0.5, label='Good (0.85)')
            ax.axhline(y=0.70, color='orange', linestyle='--', alpha=0.5, label='Acceptable (0.70)')
        elif metric == 'VUV_F1':
            ax.axhline(y=0.90, color='green', linestyle='--', alpha=0.5, label='Good (0.90)')
            ax.axhline(y=0.80, color='orange', linestyle='--', alpha=0.5, label='Acceptable (0.80)')
        
        ax.legend(fontsize=8, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Comparison plot saved to: {output_path}")
    plt.close()

def create_summary_table(results_dict):
    """Create a text summary table."""
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Method':<25} {'PESQ':<10} {'STOI':<10} {'VUV_F1':<10} {'SNR_dB':<10}")
    print("-"*80)
    
    for method, results in results_dict.items():
        pesq = f"{results['PESQ']:.3f}" if results['PESQ'] else "N/A"
        stoi = f"{results['STOI']:.3f}" if results['STOI'] else "N/A"
        vuv = f"{results['VUV_F1']:.3f}" if results['VUV_F1'] else "N/A"
        snr = f"{results['SNR_dB']:.2f}" if results['SNR_dB'] else "N/A"
        print(f"{method:<25} {pesq:<10} {stoi:<10} {vuv:<10} {snr:<10}")
    
    print("="*80)
    print("\nInterpretation:")
    print("  PESQ:   > 3.0 = Good,  2.5-3.0 = Acceptable,  < 2.5 = Poor")
    print("  STOI:   > 0.85 = Good, 0.70-0.85 = Acceptable, < 0.70 = Poor")
    print("  VUV_F1: > 0.90 = Good, 0.80-0.90 = Acceptable, < 0.80 = Poor")
    print("  SNR_dB: > 20 = Good,   10-20 = Acceptable,     < 10 = Poor")
    print("="*80 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Compare evaluation results")
    
    parser.add_argument(
        "-r", "--results",
        nargs='+',
        required=True,
        help="Result files to compare (format: name:path)"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="comparison.png",
        help="Output plot filename"
    )
    
    args = parser.parse_args()
    
    # Parse results
    results_dict = {}
    for result_spec in args.results:
        if ':' in result_spec:
            name, path = result_spec.split(':', 1)
        else:
            name = Path(result_spec).stem
            path = result_spec
        
        if Path(path).exists():
            results_dict[name] = parse_results_file(path)
        else:
            print(f"⚠️  File not found: {path}")
    
    if not results_dict:
        print("❌ No valid result files found.")
        return
    
    # Create visualizations
    create_summary_table(results_dict)
    create_comparison_plot(results_dict, args.output)

if __name__ == "__main__":
    # Example usage is in the docstring below
    """
    Example usage:
    
    python compare_results.py \
        -r "GaussianImage:evaluation_results.txt" \
           "Opus-32k:opus_32k_results.txt" \
           "Opus-64k:opus_64k_results.txt" \
        -o comparison.png
    """
    main()