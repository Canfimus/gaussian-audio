# Amplitude/Phase Experiment Guide

This guide shows you how to run experiments using Amplitude+Phase representation instead of Real+Imaginary.

## Quick Start

### Option 1: Automated Script (Easiest)

I've created a script that does everything for you:

```bash
# Run the complete Amplitude/Phase experiment
python run_amp_phase_experiment.py --iterations 10000
```

This will:
1. ✅ Preprocess audio to Amplitude/Phase spectrograms (automatically)
2. ✅ Train models at all GPS rates (2000, 2500, 3000, 3500, 4000, 4500, 5000)
3. ✅ Analyze results with correct mode (amp_phase)
4. ✅ Generate comparison CSV

**Current Configuration:**
- **SUBSET_SIZE = 3** (only 3 spectrograms for fast testing)
- **7 Gaussian rates** (2000-5000 gps range)
- **Quantization enabled** by default

### Option 2: Step-by-Step Manual

If you prefer to run each step manually:

#### Step 1: Preprocess Audio

```bash
python preprocess.py \
    --mode amp_phase \
    --input_dir ./dataset/LJSpeech-1.1/wavs/ \
    --output_dir ./dataset/ljspeech_spectrograms_amp_phase/
```

This creates Amplitude+Phase spectrograms in `./dataset/ljspeech_spectrograms_amp_phase/`

#### Step 2: Train Models

Train for each GPS rate you want to test:

```bash
# Example: 3000 gps
python train_subset.py \
    --dataset ./dataset/ljspeech_spectrograms_amp_phase/ \
    --data_name amp_phase_experiment \
    --gaussians_per_second 3000 \
    --iterations 10000 \
    --quantize \
    --save_imgs
```

Repeat for each rate: 2000, 2500, 3000, 3500, 4000, 4500, 5000

#### Step 3: Analyze Results

**IMPORTANT:** Must specify `--mode amp_phase`!

```bash
python analyze_subset_fixed.py \
    --run_dir ./checkpoints/amp_phase_experiment/GaussianImage_Cholesky_10000_3000gps \
    --output_dir ./outputs/amp_phase_3000gps \
    --original_dir ./dataset/ljspeech_spectrograms_amp_phase/ \
    --original_wav_dir ./dataset/LJSpeech-1.1/wavs/ \
    --mode amp_phase
```

Repeat for each trained model.

## Understanding SUBSET_SIZE

The code is currently set to use **only 3 spectrograms** for fast debugging.

**Current setting in `train_subset.py` line 19:**
```python
SUBSET_SIZE = 3  # Using 3 for debugging
```

This means:
- ✅ Training is FAST (only 3 images)
- ✅ Good for testing if everything works
- ⚠️ Results are NOT representative (too small sample)
- 📊 For real experiments, change to 50 or more

**To change it:**
1. Open `train_subset.py`
2. Find line 19: `SUBSET_SIZE = 3`
3. Change to: `SUBSET_SIZE = 50` (or whatever you want)
4. Save and run experiments again

## Script Options

The automated script has several options:

```bash
# Skip preprocessing (if already done)
python run_amp_phase_experiment.py --skip_preprocessing

# Skip training (if already trained, just re-analyze)
python run_amp_phase_experiment.py --skip_training

# Run without quantization
python run_amp_phase_experiment.py --no_quantization

# Change iterations
python run_amp_phase_experiment.py --iterations 20000
```

## Comparing Amp/Phase vs Real/Imag

To compare the two representations:

1. **Run Real/Imaginary experiment** (your original mode):
   ```bash
   python run_focused_experiment.py --iterations 10000
   ```
   Results in: `./experiments/focused_quantized_experiments/`

2. **Run Amplitude/Phase experiment**:
   ```bash
   python run_amp_phase_experiment.py --iterations 10000
   ```
   Results in: `./experiments/amp_phase_experiment/`

3. **Compare the CSV files:**
   - Real/Imag: `./experiments/focused_quantized_experiments/focused_experiment_comparison.csv`
   - Amp/Phase: `./experiments/amp_phase_experiment/amp_phase_comparison.csv`

4. **Look for:**
   - Which mode gives better PESQ scores?
   - Which mode gives better STOI scores?
   - Are there any patterns? (e.g., one mode better at low/high GPS rates)

## Expected Output

When you run the experiment, you'll see:

```
================================================================================
Step 1: Preprocessing Audio to Amplitude/Phase Spectrograms
================================================================================
Processing audio files (amp_phase): 100%|████████████| 13100/13100

================================================================================
Step 2: Training with Amplitude/Phase Spectrograms
================================================================================
Training with 2000 gaussians per second (with quantization)
...

================================================================================
Step 3: Analyzing Results (with Amplitude/Phase mode)
================================================================================
Analyzing: ./checkpoints/amp_phase_experiment/...
Mode: Amplitude/Phase
...

================================================================================
Step 4: Collecting Results
================================================================================
  ✅ 2000 gps: PESQ=3.234, STOI=0.8934
  ✅ 2500 gps: PESQ=3.456, STOI=0.9123
  ...

================================================================================
Amplitude/Phase Experiment Results:
================================================================================
   gaussians_per_second  avg_pesq  avg_stoi  ...
...
```

## Files Structure

After running, you'll have:

```
./dataset/ljspeech_spectrograms_amp_phase/    # Preprocessed Amp/Phase data
./checkpoints/amp_phase_experiment/           # Trained models
./experiments/amp_phase_experiment/           # Analysis results
    ├── amp_phase_comparison.csv              # Main results
    ├── experiment_config.json                # Configuration
    ├── analysis_2000gps/                     # Per-rate analysis
    │   ├── metrics_summary.csv
    │   ├── plots/
    │   └── audio/
    ├── analysis_2500gps/
    ...
```

## Troubleshooting

### Problem: "Mode mismatch" error

**Solution:** Make sure you always use `--mode amp_phase` when analyzing Amplitude/Phase data.

### Problem: "No .npy files found"

**Solution:**
- Check if preprocessing completed successfully
- Look in `./dataset/ljspeech_spectrograms_amp_phase/` for .npy files
- Re-run preprocessing without `--skip_preprocessing`

### Problem: Training is slow

**Solution:**
- This is normal - you're training 7 models (one for each GPS rate)
- With SUBSET_SIZE=3 and iterations=10000, each model takes ~5-10 minutes
- Total time: ~35-70 minutes for all 7 rates
- To go faster: reduce iterations (e.g., `--iterations 5000`)

### Problem: Want to test just one GPS rate first

**Solution:**
Edit `run_amp_phase_experiment.py` line 20:

```python
# Change from:
GAUSSIAN_RATES = [2000, 2500, 3000, 3500, 4000, 4500, 5000]

# To (just one rate):
GAUSSIAN_RATES = [3000]
```

Or run manually (Step-by-Step Manual option above).

## Next Steps

1. **Start with the automated script:**
   ```bash
   python run_amp_phase_experiment.py --iterations 10000
   ```

2. **Wait for it to complete** (~35-70 minutes with SUBSET_SIZE=3)

3. **Check results:**
   ```bash
   cat ./experiments/amp_phase_experiment/amp_phase_comparison.csv
   ```

4. **Compare with Real/Imaginary mode** (if you've run it)

5. **If satisfied, increase SUBSET_SIZE to 50** and run again for real results

## Known Issues

### PESQ Values > 4.5

You may see warnings like: `⚠️ PESQ = 4.64 (>4.5) - clamping to 4.5`

**Why this happens:**
- The Python `pesq` library has a known scaling issue
- It can return values > 4.5 even though the ITU-T standard defines 4.5 as the theoretical maximum
- Even identical signals can produce PESQ ~4.64 instead of 4.5

**Solution:**
- The analysis code automatically clamps PESQ values to 4.5
- This ensures results stay within the valid ITU-T range (-0.5 to 4.5)
- Relative comparisons between experiments remain valid
- Run `python tools/diagnostic_pesq.py` to verify this behavior

**Impact:**
- Does not affect experiment comparisons
- All PESQ values are clamped consistently
- Results can be safely interpreted using standard PESQ ranges

## Questions?

- SUBSET_SIZE controls how many spectrograms to use (currently 10)
- The script automatically handles everything for Amplitude/Phase mode
- Results will tell you if Amp/Phase is better or worse than Real/Imag
- All the same metrics (PESQ, STOI) are calculated
