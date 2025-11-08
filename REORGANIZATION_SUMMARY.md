# Repository Reorganization Summary

## ✅ What Was Done

1. **Organized all files into logical directories:**
   - 📚 `docs/` - All documentation and guides
   - 🔧 `src/` - Core source code and models
   - 🚀 `scripts/` - Training and preprocessing scripts
   - 📊 `analysis/` - Analysis and evaluation tools
   - 🧪 `experiments/` - Automated experiment runners
   - 🔨 `tools/` - Utility tools and debugging scripts

2. **Translated all Hebrew text to English:**
   - `scripts/train_subset.py` - 13 translations
   - Now fully accessible to international collaborators

3. **Updated all file paths:**
   - All experiment runners now use correct paths
   - Import statements updated to work with new structure

## 🚀 How to Use (Quick Reference)

### Running Experiments (Main Use Case)

**For Real/Imaginary mode (current setup):**
```bash
python experiments/run_focused_experiment.py --iterations 10000
```

**For Amplitude/Phase mode:**
```bash
python experiments/run_amp_phase_experiment.py --iterations 10000
```

### Other Common Commands

**Preprocessing:**
```bash
# Real/Imaginary (default)
python scripts/preprocess.py

# Amplitude/Phase
python scripts/preprocess.py --mode amp_phase
```

**Training (manual):**
```bash
python scripts/train_subset.py \
    --dataset ./dataset/ljspeech_spectrograms/ \
    --gaussians_per_second 3000 \
    --iterations 10000
```

**Analysis (manual):**
```bash
python analysis/analyze_subset_fixed.py \
    --run_dir ./checkpoints/... \
    --output_dir ./outputs/...
```

## 📁 Key File Locations

### Most Important (What You'll Use)

| File | Purpose |
|------|---------|
| `experiments/run_focused_experiment.py` | Main experiment runner (2000-5000 gps) |
| `experiments/run_amp_phase_experiment.py` | Amplitude/Phase experiments |
| `scripts/train_subset.py` | Training script (SUBSET_SIZE=3) |
| `scripts/preprocess.py` | Convert audio to spectrograms |
| `analysis/analyze_subset_fixed.py` | Analyze results |

### Documentation

| File | Purpose |
|------|---------|
| `docs/EXPERIMENT_INSTRUCTIONS.md` | Complete guide (START HERE!) |
| `docs/AMP_PHASE_EXPERIMENT_GUIDE.md` | Amplitude/Phase mode guide |
| `docs/PROJECT_STRUCTURE.md` | Full directory structure |
| `README.md` | Project overview |

## 🔄 Command Changes

If you have old commands or scripts, update them:

| Old Command | New Command |
|-------------|-------------|
| `python train_subset.py` | `python scripts/train_subset.py` |
| `python analyze_subset_fixed.py` | `python analysis/analyze_subset_fixed.py` |
| `python preprocess.py` | `python scripts/preprocess.py` |
| `python run_focused_experiment.py` | `python experiments/run_focused_experiment.py` |
| `python run_amp_phase_experiment.py` | `python experiments/run_amp_phase_experiment.py` |

## ⚙️ Current Configuration

- **SUBSET_SIZE = 3** (in `scripts/train_subset.py` line 22)
- **All Hebrew text translated to English**
- **All paths updated and working**

## 🎯 Next Steps

1. **Pull the changes in VS Code**
2. **Run your first experiment:**
   ```bash
   python experiments/run_amp_phase_experiment.py --iterations 10000
   ```
3. **Check the results in** `./experiments/amp_phase_experiment/`

## 📖 Documentation

- **Read first:** `docs/EXPERIMENT_INSTRUCTIONS.md`
- **For Amp/Phase:** `docs/AMP_PHASE_EXPERIMENT_GUIDE.md`
- **Structure details:** `docs/PROJECT_STRUCTURE.md`

## ✨ Benefits

- ✅ Much cleaner and more organized
- ✅ All text in English (no more Hebrew)
- ✅ Easy to find related files
- ✅ Documentation in one place
- ✅ Git history preserved (used `git mv`)
- ✅ All experiment runners work from repo root

## 🐛 Troubleshooting

**If you get import errors:**
- Make sure you're running from the repository root
- The scripts automatically add paths to find `src/`

**If paths don't work:**
- Use the new paths (see table above)
- Run commands from repository root directory

**Still having issues?**
- Check `docs/PROJECT_STRUCTURE.md` for complete structure
- See `docs/EXPERIMENT_INSTRUCTIONS.md` for detailed guide

---

Everything is ready to use! 🎉
