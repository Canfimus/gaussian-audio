# Gaussian Rate Experiments Guide

## Overview
This experimental setup allows you to test different rates of Gaussians per second and measure their impact on audio reconstruction quality using PSNR and STOI metrics.

## What's Changed

### 1. `train_subset.py`
- **New argument**: `--gaussians_per_second` (float) - Rate of Gaussians per second of audio
- If `--gaussians_per_second` is set, it overrides `--num_points`
- Automatically calculates the appropriate number of Gaussians based on audio duration
- Subset size changed from 10 to **50 files**

### 2. `analyze_subset_fixed.py`
- **New metrics**: Now calculates both PSNR and STOI for audio quality
- Saves metrics to `metrics_summary.csv` with per-file and average values
- Returns audio quality metrics for each reconstructed file

### 3. `run_gaussian_rate_experiments.py` (NEW)
- Automated experiment runner
- Tests multiple Gaussian rates in sequence
- Collects and compares all metrics
- Generates comparison tables

## Quick Start

### Run All Experiments (Recommended)
```bash
python run_gaussian_rate_experiments.py
```

This will:
1. Train models with 6 different Gaussian rates: 500, 1000, 2000, 5000, 10000, 20000 gps
2. Train on the first 50 spectrograms from your dataset
3. Run analysis and calculate PSNR/STOI for each
4. Generate a comparison table **AND** visualization plots

### Custom Gaussian Rates
```bash
python run_gaussian_rate_experiments.py --rates "1000,3000,5000,10000"
```

### Skip Training (Analysis Only)
If you've already trained models:
```bash
python run_gaussian_rate_experiments.py --skip_training
```

### Skip Analysis (Training Only)
If you just want to train:
```bash
python run_gaussian_rate_experiments.py --skip_analysis
```

## Manual Usage

### Training with Gaussians Per Second
```bash
python train_subset.py \
    --dataset ./dataset/ljspeech_spectrograms/ \
    --data_name my_experiment \
    --gaussians_per_second 5000 \
    --iterations 10000 \
    --save_imgs
```

### Training with Fixed Number of Points (old method)
```bash
python train_subset.py \
    --dataset ./dataset/ljspeech_spectrograms/ \
    --data_name my_experiment \
    --num_points 50000 \
    --iterations 10000 \
    --save_imgs
```

### Running Analysis
```bash
python analyze_subset_fixed.py \
    --run_dir ./checkpoints/my_experiment/GaussianImage_Cholesky_10000_5000gps \
    --output_dir ./outputs/my_analysis \
    --original_dir ./dataset/ljspeech_spectrograms/
```

## Output Structure

```
experiments/
└── gaussian_rate_experiments/
    ├── experiment_config.json          # Experiment configuration
    ├── experiment_comparison.csv        # Comparison of all runs
    ├── experiment_results.png           # 📊 Visualization plots (NEW!)
    ├── analysis_500gps/
    │   ├── plots/                       # Spectrogram comparison plots
    │   ├── audio/                       # Original and reconstructed audio
    │   └── metrics_summary.csv          # Per-file metrics
    ├── analysis_1000gps/
    │   └── ...
    └── ...
```

### Visualization Output

The experiment runner automatically generates `experiment_results.png` with 4 subplots:
1. **PSNR vs Gaussian Rate** - Line plot showing how PSNR changes with Gaussian rate
2. **STOI vs Gaussian Rate** - Line plot showing how STOI changes with Gaussian rate
3. **PSNR Bar Chart** - Easy-to-read bar comparison of all rates
4. **Dual Metric Plot** - Both PSNR and STOI on the same plot for direct comparison

## Metrics Explained

### PSNR (Peak Signal-to-Noise Ratio)
- Measured in decibels (dB)
- Higher is better
- Typical range: 20-50 dB
- >40 dB indicates very good reconstruction

### STOI (Short-Time Objective Intelligibility)
- Range: 0.0 to 1.0
- Higher is better
- Measures speech intelligibility
- >0.85 indicates good intelligibility

## Dependencies

Make sure you have these installed:
```bash
# Install all dependencies at once
pip install -r requirements_experiments.txt

# Or install manually
pip install pystoi librosa soundfile matplotlib numpy torch torchvision tqdm pytorch-msssim pandas
```

## Troubleshooting

### "No .npy files found"
Make sure you've run the preprocessing script:
```bash
python preprocess.py --input_dir <your_audio_dir> --output_dir ./dataset/ljspeech_spectrograms/
```

### "pystoi module not found"
Install the STOI library:
```bash
pip install pystoi
```

### Out of memory during training
- Reduce `--gaussians_per_second`
- Reduce `--iterations`
- Or reduce the subset size in `train_subset.py` (SUBSET_SIZE variable)

## Interpreting Results

The `experiment_comparison.csv` will show you:
- Which Gaussian rate gives the best PSNR
- Which Gaussian rate gives the best STOI
- The trade-off between model complexity and quality

Generally:
- **Lower rates** (500-1000 gps): Faster training, less memory, lower quality
- **Medium rates** (2000-5000 gps): Good balance
- **High rates** (10000+ gps): Better quality, slower training, more memory

## Next Steps

After running experiments, you can:
1. Listen to the reconstructed audio in `experiments/.../analysis_XXXgps/audio/`
2. View spectrogram comparisons in `experiments/.../analysis_XXXgps/plots/`
3. Analyze the metrics CSV files to find the optimal Gaussian rate
4. Fine-tune hyperparameters (learning rate, iterations) for the best-performing rate
