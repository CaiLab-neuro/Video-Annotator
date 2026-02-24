# Installation Guide for Tarsier2 Video Annotation Tool

## Quick Start

```bash
# Clone the repository (if not already done)
git clone <repository-url>
cd video_annotator

# Run the automated installer
python install.py

# Activate environment and start using
conda activate tarsier
bash scripts/run_preset.sh
```

## Installation Options

### Basic Usage

```bash
# Interactive installation (recommended for first-time users)
python install.py

# Specify custom environment name
python install.py --env-name my_tarsier_env

# Non-interactive mode (for automation/scripts)
python install.py --force --non-interactive
```

### Advanced Options

```bash
# Verify existing installation without installing
python install.py --verify-only --env-name tarsier

# Use existing conda environment (skip creation)
python install.py --skip-env-creation --env-name tarsier

# Only install project dependencies (skip Tarsier)
python install.py --skip-tarsier --env-name tarsier

# Force reinstall (remove existing tarsier/ directory)
python install.py --force

# Skip system checks
python install.py --skip-checks

# Disable colored output
python install.py --no-color
```

## What the Script Does

### 1. System Checks
- ✅ Verifies git is installed (required)
- ✅ Checks for ffmpeg (warns if missing)
- ✅ Detects CUDA/GPU availability (optional)
- ✅ Verifies sufficient disk space (≥10 GB recommended)

### 2. Environment Setup
- Detects if conda is available
- Auto-detects active conda environment
- Creates new environment or uses existing one
- Verifies Python 3.9 is available

### 3. Tarsier Installation
- Checks if `tarsier/` directory exists and is empty
- Clones Tarsier repository from GitHub
- Checks out `tarsier2` branch
- Runs `setup.sh` to install dependencies:
  - PyTorch 2.1.0 with CUDA 12.1
  - flash-attention 2.5.7 (pre-built wheel for cu122/torch2.1/Python 3.9)
  - transformers, decord, and other dependencies

> **Note:** The flash-attention wheel in `setup.sh` is tied to torch 2.1+cu122. If your
> system has a newer CUDA version, `setup.sh` may fail or install an incompatible PyTorch.
> You may need to manually upgrade PyTorch and reinstall flash-attn afterward.
> See **"flash-attention / PyTorch CUDA mismatch"** in the Troubleshooting section.

### 4. Project Dependencies
- Installs `pympi-ling` (ELAN file processing)
- Installs pandas, matplotlib, scikit-learn, numpy
- Upgrades pip to latest version

### 5. Verification
- Tests all critical imports
- Checks CUDA availability through PyTorch
- Displays verification report with status

### 6. Usage Instructions
- Shows environment activation command
- Lists model information
- Provides example annotation command
- Offers troubleshooting tips

## Smart Features

### Active Environment Detection
If you already have a conda environment active, the script will detect it and ask if you want to use it:

```bash
conda activate my_existing_env
python install.py
# Prompts: "Currently active: my_existing_env, requested: tarsier. Use my_existing_env? [Y/n]"
```

### Empty Directory Handling
The script intelligently handles the `tarsier/` directory:
- **Empty directory**: Proceeds with installation without prompting
- **Non-empty directory**: Prompts user to confirm deletion (unless `--force` is used)

This avoids confusing prompts when the empty `tarsier/` folder exists from git clone.

### Model Auto-Download
Models are NOT pre-downloaded during installation. Instead, they auto-download on first use:
- **Model**: `omni-research/Tarsier2-7b-0115` (recommended)
- **Alternative**: `omni-research/Tarsier2-Recap-7b`
- **Cache location**: `~/.cache/huggingface/hub/`

## Troubleshooting

### Git Not Found
```bash
# Install git first
sudo apt-get install git  # Ubuntu/Debian
sudo yum install git      # CentOS/RHEL
brew install git          # macOS
```

### FFmpeg Not Found
```bash
# Install ffmpeg
sudo apt-get install ffmpeg  # Ubuntu/Debian
sudo yum install ffmpeg      # CentOS/RHEL
brew install ffmpeg          # macOS
```

### CUDA Not Available
The script will warn if CUDA is not detected but will continue anyway. You can still use CPU mode, though it will be slower.

To enable GPU:
1. Install NVIDIA drivers
2. Install CUDA toolkit (12.1 or compatible)
3. Verify with: `nvidia-smi`

### Import Failures
If verification shows import failures:

```bash
# Reinstall Tarsier
python install.py --force

# Manually check imports
conda activate tarsier
python -c "from tarsier.tasks.utils import load_model_and_processor"
python -c "import torch; print(torch.cuda.is_available())"
```

### Setup Script Fails
If Tarsier's `setup.sh` fails:

```bash
# Try manual installation
cd tarsier/
bash setup.sh
# Check error messages and install missing dependencies manually
```

### flash-attention / PyTorch CUDA mismatch

`setup.sh` pins torch 2.1.0+cu121 and a matching flash-attention wheel. If your GPU
requires a newer CUDA version, you'll need to upgrade both manually after running the
installer.

**Step 1: Install the correct PyTorch for your CUDA version.**
Use the official PyTorch selector to get the exact `pip install` command:
https://pytorch.org/get-started/locally/

Example for CUDA 12.8:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

**Step 2: Reinstall flash-attn to match the new torch:**
```bash
pip install flash-attn --no-build-isolation
```
The `--no-build-isolation` flag tells pip to compile flash-attn against the torch that is
already installed, rather than in an isolated environment where torch may not be visible.

**Known API issue:** Newer flash-attn (≥2.6) changed `bert_padding.unpad_input` to return
5 values; older `transformers` expects 4, causing a `ValueError` when `attention_mask` is
passed. This is already worked around in `run_prompt_presets.py` by passing
`attention_mask=None`.

### Environment Issues
```bash
# List conda environments
conda env list

# Remove and recreate
conda env remove -n tarsier
python install.py

# Use existing environment
python install.py --skip-env-creation --env-name your_env
```

## Manual Installation (Alternative)

If the automated script doesn't work, follow manual steps:

```bash
# 1. Create conda environment
conda create -n tarsier python=3.9 -y
conda activate tarsier

# 2. Clone Tarsier
git clone --branch tarsier2 https://github.com/bytedance/tarsier.git tarsier/
cd tarsier
bash setup.sh
cd ..

# 3. Install project dependencies
pip install pympi-ling pandas matplotlib scikit-learn

# 4. Verify installation
python -c "from tarsier.tasks.utils import load_model_and_processor"
```

## Post-Installation

After successful installation:

1. **Activate environment:**
   ```bash
   conda activate tarsier
   ```

2. **Configure your video path** in `scripts/run_preset.sh`

3. **Run annotation:**
   ```bash
   bash scripts/run_preset.sh
   ```

4. **Model downloads automatically** on first run (may take several minutes)

## Getting Help

- **Script help**: `python install.py --help`
- **Tarsier docs**: https://github.com/bytedance/tarsier
- **Project README**: `README.md`
- **Developer guide**: `CLAUDE.md`
- **Report issues**: Create an issue in the project repository

## Requirements Summary

### System Requirements
- **OS**: Linux (tested), macOS (should work), Windows (via WSL)
- **Python**: 3.9 (required)
- **Disk**: ~10 GB free space
- **RAM**: 8 GB minimum, 16 GB+ recommended
- **GPU**: Optional but recommended (NVIDIA with CUDA support)

### Software Dependencies
- git (required)
- ffmpeg (required for video processing)
- conda or Python 3.9 (required)
- CUDA toolkit (optional, for GPU acceleration)

### Python Packages (installed automatically)
- PyTorch 2.1.0+ (may need manual upgrade to match your CUDA version)
- transformers 4.47.0+
- flash-attention (version must match installed torch; may need `pip install flash-attn --no-build-isolation` after upgrading torch)
- pympi-ling, pandas, matplotlib, scikit-learn
- decord, gradio, openai, Pillow, scipy, safetensors, tiktoken

## Version Information

- **Tarsier Version**: Tarsier2 (branch: tarsier2)
- **Python Version**: 3.9
- **PyTorch Version**: 2.1.0
- **CUDA Version**: 12.1 (recommended)
- **Model**: omni-research/Tarsier2-7b-0115

Last updated: February 2026
