# Gaussian Audio Experiment Instructions

This guide explains how to run experiments with the updated features including UTMOS metrics and Amplitude/Phase representation.

## Table of Contents
1. [Quick Start](#quick-start)
2. [New Features](#new-features)
3. [Running Experiments](#running-experiments)
4. [Testing Amplitude/Phase Mode](#testing-amplitudephase-mode)
5. [Understanding the Metrics](#understanding-the-metrics)

## Quick Start

### 1. Install Dependencies

First, install the required dependencies:

```bash
pip install -r requirements_experiments.txt
```

**Note about UTMOS:** UTMOS is optional and not included in the basic requirements. The experiments will run fine without it, using PESQ and STOI metrics. If you want to add UTMOS later, see the [UTMOS Installation](#utmos-installation-optional) section below.

### 2. Run the Focused Experiment (Recommended)

The focused experiment tests the 2000-5000 gps range with quantization enabled:

```bash
python run_focused_experiment.py --iterations 10000
```

This will:
- Train models at 7 different Gaussian rates (2000, 2500, 3000, 3500, 4000, 4500, 5000 gps)
- Analyze results and calculate PESQ and STOI metrics (plus UTMOS if installed)
- Generate comprehensive comparison plots
- Calculate compression ratios

**Note:** If UTMOS is not installed, you'll see a warning but experiments will continue normally with PESQ and STOI.

### 3. Check Results

Results will be saved to `./experiments/focused_quantized_experiments/`:
- `focused_experiment_results.png` - Comprehensive 8-panel visualization
- `focused_experiment_comparison.csv` - Detailed metrics comparison
- Individual analysis folders for each Gaussian rate

## New Features

### UTMOS Metric (Optional)

**What is UTMOS?**
- Universal Text-free Model for Objective Speech Quality Assessment
- Neural network-based metric (more advanced than PESQ/STOI)
- Trained on large-scale MOS (Mean Opinion Score) data
- Range: typically 1.0-5.0 (higher is better)
- Better correlates with human perception than traditional metrics

**⚠️ UTMOS is OPTIONAL:**
- The experiments work perfectly fine with just PESQ and STOI
- UTMOS adds extra quality assessment but isn't required
- If UTMOS is not installed, the code automatically skips it
- See [UTMOS Installation](#utmos-installation-optional) section for manual installation

**When to use:**
- For speech/voice audio quality assessment
- When you need metrics that correlate well with human listening tests
- For comparing different compression/reconstruction methods

**Performance note:** UTMOS requires a pretrained neural network, so it's slower than PESQ/STOI.

### Amplitude/Phase Representation

**What is it?**
Instead of storing spectrograms as Real+Imaginary components, you can now use Amplitude+Phase:

- **Real/Imaginary (default):**
  - `channel1 = real part`, `channel2 = imaginary part`
  - Reconstruction: `S_complex = channel1 + 1j * channel2`

- **Amplitude/Phase:**
  - `channel1 = magnitude`, `channel2 = phase`
  - Reconstruction: `S_complex = channel1 * exp(1j * channel2)`

**Why use Amplitude/Phase?**
- Phase is often considered less perceptually important
- May allow better compression of phase channel
- Different characteristics for Gaussian splatting optimization
- Worth experimenting to see which works better for your use case

## Running Experiments

### Experiment 1: Standard Real/Imaginary Mode (Current)

This is what you've been using. No changes needed:

```bash
# Run focused experiment with Real/Imaginary (default)
python run_focused_experiment.py --iterations 10000

# Or run with original WAV files for accurate baseline
python run_focused_experiment.py --iterations 10000 \
    --original_wav_dir ./dataset/LJSpeech-1.1/wavs/
```

### Experiment 2: Test Different Metrics Only

If you just want to re-analyze existing results with UTMOS:

```bash
# Skip training, only run analysis (faster)
python run_focused_experiment.py --skip_training
```

### Experiment 3: Broader Range Experiment

Test a wider range of Gaussian rates (500 to 20000 gps):

```bash
python run_gaussian_rate_experiments.py --iterations 10000
```

## Testing Amplitude/Phase Mode

### Step 1: Preprocess Audio with Amplitude/Phase

Create a new dataset using Amplitude/Phase representation:

```bash
python preprocess.py \
    --mode amp_phase \
    --input_dir ./dataset/LJSpeech-1.1/wavs/ \
    --output_dir ./dataset/ljspeech_spectrograms_amp_phase/
```

This creates spectrograms in `./dataset/ljspeech_spectrograms_amp_phase/` using Amplitude+Phase encoding.

### Step 2: Train with Amplitude/Phase Dataset

```bash
python train_subset.py \
    --dataset ./dataset/ljspeech_spectrograms_amp_phase/ \
    --data_name amp_phase_test \
    --gaussians_per_second 3000 \
    --iterations 10000 \
    --save_imgs
```

### Step 3: Analyze with Amplitude/Phase Mode

**IMPORTANT:** You must specify `--mode amp_phase` when analyzing!

```bash
python analyze_subset_fixed.py \
    --run_dir ./checkpoints/amp_phase_test/GaussianImage_Cholesky_10000_3000gps \
    --output_dir ./outputs/amp_phase_analysis \
    --original_dir ./dataset/ljspeech_spectrograms_amp_phase/ \
    --original_wav_dir ./dataset/LJSpeech-1.1/wavs/ \
    --mode amp_phase
```

### Step 4: Compare Real/Imag vs Amp/Phase

Run both experiments and compare the metrics:

1. Real/Imag experiment (already done)
2. Amp/Phase experiment (steps above)
3. Compare the CSV files:
   - `./outputs/real_imag_analysis/metrics_summary.csv`
   - `./outputs/amp_phase_analysis/metrics_summary.csv`

**What to look for:**
- Which mode gives better PESQ/STOI/UTMOS scores?
- Which mode compresses better with quantization?
- Are phase errors more tolerable than real/imaginary errors?

## Understanding the Metrics

### Metrics Summary

| Metric | Range | Higher is Better? | What it Measures |
|--------|-------|-------------------|------------------|
| **PESQ** | -0.5 to 4.5 | ✅ Yes | Perceptual speech quality (ITU standard) |
| **STOI** | 0.0 to 1.0 | ✅ Yes | Speech intelligibility |
| **UTMOS** | 1.0 to 5.0 | ✅ Yes | Mean Opinion Score prediction (human-like) |
| **Compression Ratio** | >1.0 | ✅ Yes | How much smaller the model is vs original |

### Baseline Values

The "Original Baseline" shows the quality ceiling from the spectrogram representation itself:

- **Theoretical Max:** PESQ=4.5, STOI=1.0 (perfect reconstruction)
- **Measured Baseline:** Typically PESQ≈3.8-4.2, STOI≈0.92-0.97
- This measures quality loss from STFT→ISTFT process alone
- Your Gaussian reconstruction should approach this baseline

### Interpreting Results

**Good PESQ scores:**
- 4.0-4.5: Excellent (near transparent)
- 3.5-4.0: Good (minor artifacts)
- 3.0-3.5: Fair (noticeable but acceptable)
- <3.0: Poor (significant degradation)

**Good STOI scores:**
- 0.95-1.0: Excellent intelligibility
- 0.90-0.95: Good intelligibility
- 0.85-0.90: Fair intelligibility
- <0.85: Poor intelligibility

**Good UTMOS scores:**
- 4.0-5.0: Excellent
- 3.0-4.0: Good
- 2.0-3.0: Fair
- <2.0: Poor

### Trade-off Analysis

The focused experiment generates plots showing:
1. **Quality vs Gaussian Rate:** More Gaussians = better quality
2. **Quality vs Compression:** Better compression = lower quality (usually)
3. **Sweet Spot:** Find the rate that balances quality and compression

**Example interpretation:**
- 2000 gps: High compression (8x), but PESQ=3.2 (fair quality)
- 5000 gps: Lower compression (3x), but PESQ=3.9 (excellent quality)
- **Optimal:** Maybe 3500 gps with PESQ=3.7 and compression=5x

## Advanced Usage

### Custom Experiment Range

Edit the Gaussian rates in `run_focused_experiment.py`:

```python
# Line 25
GAUSSIAN_RATES = [2000, 2500, 3000, 3500, 4000, 4500, 5000]

# Change to your desired rates, e.g.:
GAUSSIAN_RATES = [1000, 2000, 3000, 4000, 5000, 6000, 8000, 10000]
```

### Running Without UTMOS (Default/Recommended)

**By default, experiments run without UTMOS** since it's not in the requirements:

1. Just install the standard requirements: `pip install -r requirements_experiments.txt`
2. Run experiments normally
3. The code will automatically skip UTMOS and show a one-time warning
4. Analysis uses PESQ and STOI (industry-standard metrics)
5. Results are still fully valid and comprehensive

### Subset Size

For faster debugging, edit `train_subset.py`:

```python
# Line 174
SUBSET_SIZE = 3  # Change this (currently 3 for debugging)

# For full experiment:
SUBSET_SIZE = 50
```

## UTMOS Installation (Optional)

**IMPORTANT: You don't need UTMOS to run experiments!** The system works perfectly with PESQ and STOI.

If you want to add UTMOS later, here are some options:

### Option 1: Skip UTMOS (Recommended for Now)

Just run the experiments without UTMOS:
- Install dependencies: `pip install -r requirements_experiments.txt`
- Run experiments normally
- You'll see a warning that UTMOS is not available (this is fine)
- Results will show PESQ and STOI only

### Option 2: Try Alternative UTMOS Installation (Advanced)

There are several UTMOS implementations. You can try:

```bash
# Option A: Try installing from a specific source
pip install git+https://github.com/tarepan/SpeechMOS.git

# Option B: Try a different package name
pip install speechmos

# Option C: Install from a research repository
# (Check GitHub for "UTMOS" implementations)
```

**Note:** These may or may not work depending on your system. If installation fails, don't worry - just use PESQ and STOI.

### What You'll See Without UTMOS

When running experiments without UTMOS:
- ✅ PESQ and STOI metrics work perfectly
- ✅ All plots are generated (without UTMOS panels)
- ✅ CSV output includes empty UTMOS columns
- ⚠️ You'll see: "Warning: speechmetrics not available. UTMOS metric will not be calculated."
- ✅ Everything else works normally

## Troubleshooting

### UTMOS Issues

**Problem:** Cannot install UTMOS / speechmetrics

**Solution:**
- **This is fine!** Just run without UTMOS
- PESQ and STOI are industry-standard metrics
- The experiments are designed to work without UTMOS

**Problem:** UTMOS calculation is very slow (if you installed it)

**Solution:**
- This is normal - UTMOS uses a neural network
- Consider using `--skip_training` to only analyze new models
- Or use smaller SUBSET_SIZE for faster testing

### Mode Mismatch Errors

**Problem:** `ValueError: Unknown representation mode`

**Solution:**
- Make sure to use `--mode amp_phase` in `analyze_subset_fixed.py` if you used `amp_phase` in preprocessing
- The mode must match between preprocessing and analysis
- Default is `real_imag` if not specified

### Missing Original WAV Files

**Problem:** Baseline shows theoretical max (PESQ=4.5, STOI=1.0)

**Solution:**
- This means original WAV files weren't found
- Provide `--original_wav_dir` path when running analysis or experiments
- Example: `--original_wav_dir ./dataset/LJSpeech-1.1/wavs/`

## Next Steps

1. **Install dependencies (UTMOS is optional and not required):**
   ```bash
   pip install -r requirements_experiments.txt
   ```

2. **Start with the focused experiment:**
   ```bash
   python run_focused_experiment.py --iterations 10000
   ```

   You'll see a warning about UTMOS not being available - **this is normal and fine!**

3. **Check the results** in `./experiments/focused_quantized_experiments/`
   - Plots will show PESQ and STOI metrics
   - CSV files will have empty UTMOS columns

4. **Try Amplitude/Phase mode** if you want to experiment with different representations

5. **Analyze the trade-offs** between quality and compression to find your optimal configuration

## Questions?

Check the existing documentation:
- `EXPERIMENT_GUIDE.md` - Original experiment guide
- `FOCUSED_EXPERIMENT_GUIDE.md` - Focused experiment details
- `README.md` - Project overview

For issues, check that:
- All dependencies are installed (`pip install -r requirements_experiments.txt`)
- Paths are correct (use absolute paths if relative paths fail)
- Mode matches between preprocessing and analysis
