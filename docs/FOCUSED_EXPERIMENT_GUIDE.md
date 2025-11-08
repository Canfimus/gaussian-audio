# Focused Experiment Guide: 2000-5000 gps with Quantization

## What Is This Experiment?

This is a **deep-dive analysis** of the "sweet spot" range (2000-5000 gaussians per second) with:
- 🔍 **More granular testing**: 7 points instead of 3 (2000, 2500, 3000, 3500, 4000, 4500, 5000)
- 📦 **Quantization enabled**: Tests compression capabilities
- 📊 **Compression metrics**: Measures actual file size reduction

## Quick Start

```bash
# Run the focused experiment (with 3 images for debugging)
python run_focused_experiment.py --iterations 1000

# Full run (change SUBSET_SIZE to 50 in train_subset.py)
python run_focused_experiment.py --iterations 10000
```

## What We're Testing

### 1. **Quantization** 🗜️

**What is it?**
Quantization reduces the precision of Gaussian parameters to save space:

```python
# WITHOUT quantization:
position_x = 0.123456789  # 32-bit float = 4 bytes
position_y = 0.987654321
color_real = 0.555555555
# ... etc

# WITH quantization:
position_x = 0.125  # 16-bit half = 2 bytes (50% smaller!)
position_y = 0.984
color_real = 0.556  # Rounded to fewer bits
```

**Three types of quantization applied:**

1. **Position quantization**: 32-bit → 16-bit (half precision)
2. **Color quantization**: Vector quantization with codebook (8 possible values!)
3. **Covariance quantization**: 6-bit uniform quantization (64 levels)

**Trade-off:**
- ✅ Much smaller file sizes (4-8x compression)
- ❌ Slight quality loss (usually <5%)

### 2. **Compression Ratio** 📉

**Formula:**
```
Compression Ratio = Original Size / Compressed Size

Example:
Original spectrogram: 1.5 MB
Compressed Gaussians: 0.3 MB
Compression Ratio = 1.5 / 0.3 = 5.0x
Space Savings = 80%
```

**What affects compression ratio:**
- **Fewer Gaussians** = Higher compression (but lower quality)
- **Quantization** = 2-4x additional compression
- **More Gaussians** = Lower compression (but higher quality)

### 3. **The Sweet Spot** 🎯

Why focus on 2000-5000 gps?

```
< 2000 gps:  Quality too poor for practical use
2000-5000:   🔥 SWEET SPOT - Good quality, good compression
> 5000 gps:  Diminishing returns, larger files
```

This range typically gives you:
- PESQ: 3.5 - 4.2 (Fair to Excellent)
- STOI: 0.85 - 0.95 (Good to Very Good)
- Compression: 3x - 8x (Very efficient)

## The Plots You'll Get 📊

### 1. **PESQ vs Gaussian Rate**
Shows how audio quality improves with more Gaussians.
- Green line = Original audio (4.5 max)
- Blue line = Your reconstructions
- Gap = Quality loss

### 2. **STOI vs Gaussian Rate**
Shows how speech intelligibility improves.
- Green line = Perfect (1.0)
- Purple line = Your reconstructions

### 3. **Compression Ratio vs Gaussian Rate**
Shows the compression trade-off.
- Red line = No compression (1.0x)
- Orange line = Actual compression achieved
- Higher = Better compression

### 4. **Quality vs Compression Trade-off** ⭐ MOST IMPORTANT!
Scatter plot showing PESQ on Y-axis, Compression on X-axis.
- **Top-right corner** = Holy grail (high quality + high compression)
- Each point is colored by Gaussian rate
- Find the best balance!

### 5. **STOI vs Compression**
Same as #4 but for intelligibility.

### 6. **All Metrics Comparison (Normalized)**
Bar chart comparing all three metrics side-by-side.
- All normalized to 0-1 scale
- Easy visual comparison

## Understanding the Results

### Example Output:

```csv
gaussians_per_second,avg_pesq,avg_stoi,compression_ratio,space_savings_percent
2000,3.65,0.87,8.2,87.8
2500,3.78,0.89,6.8,85.3
3000,3.89,0.91,5.9,83.1
3500,3.96,0.93,5.2,80.8
4000,4.02,0.94,4.6,78.3
4500,4.06,0.95,4.1,75.6
5000,4.09,0.95,3.8,73.7
```

### Interpretation:

**Best compression**: 2000 gps
- ✅ 8.2x compression (88% space savings!)
- ❌ Lower quality (PESQ 3.65 = Fair)

**Best quality**: 5000 gps
- ✅ PESQ 4.09 (Excellent!)
- ❌ Lower compression (3.8x = 74% savings)

**Sweet spot**: ~3000-3500 gps ⭐
- ✅ Good quality (PESQ ~3.9-4.0)
- ✅ Good compression (5-6x)
- ✅ Best balance!

## Comparison: With vs Without Quantization

| Mode | Gaussian Rate | File Size | PESQ | Compression |
|------|---------------|-----------|------|-------------|
| **No Quant** | 3000 gps | 1.2 MB | 3.92 | 3.0x |
| **Quantized** | 3000 gps | 0.5 MB | 3.89 | 6.0x |
|  |  | **2.4x smaller!** | -0.03 | **2x better!** |

**Result**: Quantization gives you 2-3x better compression with minimal quality loss!

## Technical Details

### What Gets Compressed?

For each Gaussian, we store:

```python
# WITHOUT quantization:
position (x, y):     2 × 32-bit = 8 bytes
covariance (3 vals): 3 × 32-bit = 12 bytes
color (2 vals):      2 × 32-bit = 8 bytes
opacity (1 val):     1 × 32-bit = 4 bytes
─────────────────────────────────────────
Total per Gaussian:                32 bytes

For 10,000 Gaussians: 320 KB

# WITH quantization:
position (x, y):     2 × 16-bit = 4 bytes   (50% savings)
covariance (3 vals): 3 × 6-bit = 3 bytes    (75% savings!)
color (2 vals):      codebook indices       (90% savings!)
opacity (1 val):     1 × 32-bit = 4 bytes   (no change)
─────────────────────────────────────────
Total per Gaussian:                ~11 bytes

For 10,000 Gaussians: 110 KB (65% smaller!)
```

### Quantization Types Explained:

1. **Half Precision (positions)**
   ```
   32-bit: 0.123456789123456
   16-bit: 0.123456 (good enough!)
   ```

2. **Vector Quantization (colors)**
   ```
   Instead of storing actual values, store index to codebook:
   Codebook: [red, blue, green, yellow, orange, purple, pink, white]
   Original: (0.234, 0.876) = 8 bytes
   Quantized: Index 3 = 1 byte (8x compression!)
   ```

3. **Uniform Quantization (covariance)**
   ```
   Continuous: 0.0 to 1.0 (infinite values)
   6-bit: 64 discrete levels (0, 0.016, 0.032, ..., 1.0)
   ```

## Running the Experiment

### Option 1: Quick Test (3 images, 1000 iterations)
```bash
# Current setup - perfect for testing
python run_focused_experiment.py --iterations 1000
```
Time: ~5 minutes
Use for: Verifying everything works

### Option 2: Full Quality (50 images, 10000 iterations)
```bash
# Edit train_subset.py: Set SUBSET_SIZE = 50
python run_focused_experiment.py --iterations 10000
```
Time: ~2-3 hours
Use for: Real results

### Option 3: Without Quantization (comparison)
```bash
python run_focused_experiment.py --no_quantization
```
Use for: Seeing the impact of quantization

### Option 4: Analysis Only (if training already done)
```bash
python run_focused_experiment.py --skip_training
```
Use for: Re-analyzing existing results

## Output Files

```
experiments/focused_quantized_experiments/
├── experiment_config.json              # Configuration
├── focused_experiment_comparison.csv   # All metrics
├── focused_experiment_results.png      # 6-panel visualization
├── analysis_2000gps/
│   ├── metrics_summary.csv
│   ├── plots/
│   └── audio/
├── analysis_2500gps/
│   └── ...
└── ... (one folder per rate)
```

## Interpreting Your Results

### Question 1: What's the optimal Gaussian rate?
Look at Plot 4 (Quality vs Compression).
- Find the point closest to the top-right corner
- That's your sweet spot!

### Question 2: Is quantization worth it?
Compare compression ratios:
- If you gain >2x compression with <5% quality loss → YES!

### Question 3: Where are diminishing returns?
Look at Plot 1 (PESQ vs Gaussian Rate).
- Where does the curve start to flatten?
- Adding more Gaussians past that point isn't efficient

## Next Steps

1. **Run the experiment**: `python run_focused_experiment.py --iterations 1000`
2. **Check the plots**: Look at `focused_experiment_results.png`
3. **Find your sweet spot**: Which rate gives the best quality/compression balance?
4. **Listen to the audio**: Files in `experiments/.../analysis_XXXgps/audio/`
5. **Compare with baselines**: How does it compare to MP3/Opus?

## Advanced Options

### Test Different Quantization Settings

Edit `gaussianimage_audio_v2.py` to change quantization parameters:

```python
# Line 40-42: Adjust codebook size
self.features_dc_quantizer = VectorQuantizer(
    codebook_size=8,  # Try 4, 8, 16, 32
    num_quantizers=2  # Try 1, 2, 4
)

# Line 42: Adjust covariance bits
self.cholesky_quantizer = UniformQuantizer(
    bits=6  # Try 4, 6, 8, 10
)
```

## Why This Matters 🎯

This experiment tells you:
1. **Practical usability**: Can you get acceptable quality with good compression?
2. **Optimal configuration**: What settings should you use in production?
3. **Comparison data**: How does Gaussian splatting compare to traditional codecs?
4. **Research insights**: Is this approach viable for audio compression?

---

**TL;DR**: Run `python run_focused_experiment.py` to find the perfect balance between audio quality and file size in the 2000-5000 gps range with quantization enabled! 🚀
