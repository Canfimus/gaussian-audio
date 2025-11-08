# Project Structure

This document describes the organization of the Gaussian Audio repository.

## Directory Structure

```
gaussian-audio/
├── README.md                      # Project overview
├── LICENSE                        # License file
├── requirements.txt               # Core dependencies
├── requirements_experiments.txt   # Experiment dependencies
│
├── docs/                          # 📚 All documentation
│   ├── PROJECT_STRUCTURE.md       # This file
│   ├── AMP_PHASE_EXPERIMENT_GUIDE.md
│   ├── EXPERIMENT_GUIDE.md
│   ├── EXPERIMENT_INSTRUCTIONS.md
│   └── FOCUSED_EXPERIMENT_GUIDE.md
│
├── src/                           # 🔧 Core source code
│   ├── __init__.py
│   ├── utils.py                   # Utility functions
│   ├── optimizer.py               # Custom optimizers
│   ├── quantize.py                # Quantization utilities
│   └── models/                    # Model implementations
│       ├── __init__.py
│       ├── gaussianimage_audio.py
│       ├── gaussianimage_audio_v2.py
│       ├── gaussianimage_cholesky.py
│       ├── gaussianimage_rs.py
│       └── gaussiansplatting_3d.py
│
├── scripts/                       # 🚀 Training and preprocessing scripts
│   ├── preprocess.py              # Convert audio to spectrograms
│   ├── train.py                   # Main training script
│   ├── train_subset.py            # Train on subset (recommended)
│   ├── train_onespectrogram.py    # Train single file
│   ├── prepare_spectrogram_data.py
│   └── legacy/                    # Old/deprecated scripts
│       ├── train_backup.py
│       └── train_quantize.py
│
├── analysis/                      # 📊 Analysis and evaluation scripts
│   ├── analyze_subset_fixed.py    # Main analysis script (recommended)
│   ├── evaluate_audio_quality.py
│   ├── evaluate_compression.py
│   ├── compare_results.py
│   ├── generate_baselines.py
│   ├── verify_metrics.py
│   ├── visualize.py
│   └── legacy/
│       └── analyze_subset.py
│
├── experiments/                   # 🧪 Automated experiment runners
│   ├── run_focused_experiment.py       # 2000-5000 gps (recommended)
│   ├── run_gaussian_rate_experiments.py # Broader range
│   └── run_amp_phase_experiment.py      # Amplitude/Phase mode
│
├── tools/                         # 🔨 Utility tools
│   ├── spectrogram_to_audio.py
│   ├── debug_find.py
│   └── test_quantize.py
│
├── tests/                         # ✅ Unit tests
│   └── ... (various test files)
│
├── gsplat/                        # External dependency (submodule)
│
└── [Not in git - created at runtime]
    ├── dataset/                   # Training data
    ├── checkpoints/               # Saved models
    ├── outputs/                   # Analysis outputs
    └── experiments/               # Experiment results
```

## Quick Start Paths

### Running Experiments

**Most common workflow:**
```bash
# From repository root
python experiments/run_focused_experiment.py --iterations 10000
```

**Amplitude/Phase mode:**
```bash
python experiments/run_amp_phase_experiment.py --iterations 10000
```

### Preprocessing Data

```bash
# Real/Imaginary mode (default)
python scripts/preprocess.py

# Amplitude/Phase mode
python scripts/preprocess.py --mode amp_phase
```

### Training

```bash
# Train on subset
python scripts/train_subset.py \
    --dataset ./dataset/ljspeech_spectrograms/ \
    --gaussians_per_second 3000 \
    --iterations 10000

# Train single file (for testing)
python scripts/train_onespectrogram.py
```

### Analysis

```bash
python analysis/analyze_subset_fixed.py \
    --run_dir ./checkpoints/... \
    --output_dir ./outputs/... \
    --original_dir ./dataset/ljspeech_spectrograms/
```

## Key Files

### User-Facing Scripts

Most users will primarily use these:

1. **experiments/run_focused_experiment.py** - Main experiment runner
2. **experiments/run_amp_phase_experiment.py** - Amplitude/Phase experiments
3. **scripts/preprocess.py** - Data preprocessing
4. **scripts/train_subset.py** - Training script
5. **analysis/analyze_subset_fixed.py** - Results analysis

### Documentation

- **docs/EXPERIMENT_INSTRUCTIONS.md** - Comprehensive guide (start here!)
- **docs/AMP_PHASE_EXPERIMENT_GUIDE.md** - Amplitude/Phase mode guide
- **README.md** - Project overview

### Configuration

- **scripts/train_subset.py** - Line 22: `SUBSET_SIZE = 3` (change for more images)
- **requirements_experiments.txt** - Dependencies for experiments

## Import Structure

Since the codebase has been reorganized, scripts use:

```python
# Scripts in scripts/, experiments/, analysis/, tools/ directories
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.utils import *
from src.models.gaussianimage_cholesky import GaussianImage_Cholesky
```

## Migration Notes

If you have old commands, update them:

| Old Command | New Command |
|-------------|-------------|
| `python train_subset.py` | `python scripts/train_subset.py` |
| `python analyze_subset_fixed.py` | `python analysis/analyze_subset_fixed.py` |
| `python preprocess.py` | `python scripts/preprocess.py` |
| `python run_focused_experiment.py` | `python experiments/run_focused_experiment.py` |

## Development

When adding new code:

- **Model implementations** → `src/models/`
- **Training scripts** → `scripts/`
- **Analysis scripts** → `analysis/`
- **Experiment runners** → `experiments/`
- **Utility tools** → `tools/`
- **Documentation** → `docs/`
- **Tests** → `tests/`

## Legacy Code

Old/deprecated code is in:
- `scripts/legacy/`
- `analysis/legacy/`

These are kept for reference but not recommended for new work.
