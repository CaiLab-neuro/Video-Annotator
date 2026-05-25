# Annotation Tool for Videos Content (Video Content Annotator)

`Video Content Annotator`, part of the `GazeBehavior Annotation Toolkit (GBAT)`, provides a general-purpose Vision-Language Model (VLM)-based annotator for labeling video content using a question-answering framework. For human behavioral research, it supports the annotation of human behaviors from egocentric and third-person video recordings.

## Main Scripts

- `annotate_video.py`: Main video annotation entry point; clips a video, runs preset prompts, and writes segment-level labels to CSV.
- `run_prompt_presets.py`: Loads the Tarsier model and applies prompt presets to a single clip.
- `eaf_to_csv.py`: Converts ELAN `.eaf` human annotations into CSV format for comparison.
- `visualizer.py`: Compares model and human annotation CSVs and saves metrics and plots.
- `scripts/run_preset.sh`: Example shell wrapper for annotating one configured video.
- `scripts/run_all_exo_videos.sh`: Batch wrapper for annotating side/room exocentric videos in a directory.
- `scripts/visualize.sh`: Example shell wrapper for generating comparison visualizations.

## Installation
**Requirements:** Python 3.9, git, ffmpeg, ~10 GB disk space

### Option1: Setup Script (`install.py`)

```bash
python install.py
conda activate tarsier
```
Options: `--env-name <name>`, `--force`, `--verify-only`

### Option2: Manual Installation of Dependencies

```bash
# Create environment
conda create -n tarsier python=3.9 -y && conda activate tarsier

# Install Tarsier
git clone --branch tarsier2 https://github.com/bytedance/tarsier.git tarsier/
cd tarsier && bash setup.sh && cd ..

# Install project dependencies
pip install pympi-ling pandas matplotlib scikit-learn
```
**Note:** Model (`omni-research/Tarsier2-7b-0115`) auto-downloads on first use to `~/.cache/huggingface/hub/`

## Annotating Videos with Preset Promputs (run_preset.sh)

**Purpose**: Analyze videos by extracting short clips and automatically annotate each clips with an event label based on prompted questions (e.g. For human behaviors: child and parent hand actions).

### Usage

**Linux/Mac:**

1. Configure the `scripts/run_preset.sh` script with your parameters:
   ```bash
   vim scripts/run_preset.sh
   ```
   
2. Key variables to set:
   - `VIDEO`: Path to your input video file
   - `OUT_CSV`: Path where the output annotations CSV will be saved
   - `PRESETS`: Path to the prompts configuration (e.g., `prompts/presets_short.json`)
   - `CONFIG`: Path to Tarsier config (e.g., `tarsier/configs/tarser2_default_config.yaml`)
   - `TARSIER_MODEL`: Model identifier (e.g., `omni-research/Tarsier2-7b-0115`)

3. Run the annotation script:
   ```bash
   conda activate tarsier
   bash scripts/run_preset.sh
   ```

**Windows (Command Prompt or PowerShell):**

```cmd
conda activate tarsier
python -m annotate_video ^
  --video data/videos/YOUR_VIDEO.mp4 ^
  --model omni-research/Tarsier2-7b-0115 ^
  --config tarsier/configs/tarser2_default_config.yaml ^
  --prompts prompts/presets_short.json ^
  --out_csv data/results/OUTPUT.csv ^
  --clip_sec 3.0 ^
  --stride_sec 3.0 ^
  --start_sec 60 ^
  --limit_sec 180
```

**All platforms (direct Python):**

```bash
conda activate tarsier
python -m annotate_video \
  --video data/videos/YOUR_VIDEO.mp4 \
  --model omni-research/Tarsier2-7b-0115 \
  --config tarsier/configs/tarser2_default_config.yaml \
  --prompts prompts/presets_short.json \
  --out_csv data/results/OUTPUT.csv \
  --clip_sec 3.0 \
  --stride_sec 3.0 \
  --start_sec 60 \
  --limit_sec 180
```

**Output:** A CSV file with one row per video segment, containing the following behavioral annotations:
- `t_start`, `t_end`: Temporal boundaries of each clip
- `child_hand_action`: Action type (e.g., grabbing toy, manipulating toy, pointing, etc.)
- `child_proximity_behavior`: Spatial relationship to adult (close, mid-distance, far; facing toward/away)
- `current_toy`: Which toy the child is holding
- `adult_hand_action`: Parent's hand activity
- `child_body_orientation_action`: Body orientation and movement
- `interaction_flow`: Quality of interaction exchange
- `pose`: Overall body pose

#### Tuning Analysis Parameters

The script accepts several parameters to control the analysis:

| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `CLIP_DURATION` | Length of each video clip in seconds (should match model's training) | 3.0 | `1.5` (shorter) or `5.0` (longer) |
| `STRIDE` | Time interval (in seconds) between consecutive clip starts (controls temporal resolution) | 3.0 | `1.0` (dense, more clips) or `5.0` (sparse, fewer clips) |
| `START_SEC` | Begin processing from this timestamp in seconds | 60 | `0` (start of video) or `120` (skip first 2 minutes) |
| `LIMIT_SEC` | Maximum duration to process (in seconds) | 180 | `600` (10 minutes) or `1200` (20 minutes) |

**Parameter Tuning Tips:**

- **For dense temporal resolution** (frame-by-frame analysis): Reduce `STRIDE` (e.g., `0.5` or `1.0`) and use matching `CLIP_DURATION`
- **For quick overview**: Increase `STRIDE` (e.g., `5.0` or `10.0`) to reduce computational time
- **For specific time windows**: Use `START_SEC` and `LIMIT_SEC` to isolate particular segments (e.g., analyze only toy play segments 1:00–5:00)
- **Computational cost**: Total clips ≈ `(LIMIT_SEC - CLIP_DURATION) / STRIDE`, so larger strides mean fewer clips to process

**Test Run: analyze 3-minute window with 1-second intervals:**
```bash
START_SEC=60
LIMIT_SEC=180
CLIP_DURATION=3.0
STRIDE=1.0
```
This would generate ~180 annotated clips covering the 1:00–4:00 range of the video.

## Contact 
Iba Baig (baig.i@northeastern.edu)
Dr. Mingbo Cai (mingbocai@gmail.com)


## Citation

If you use the tool, please cite:

```bibtex
@misc{baig2026gazebehaviorannotationtoolkitgbat,
      title={GazeBehavior Annotation Toolkit (GBAT): AI-powered toolkit for automatic annotation of egocentric eye-tracking and video data of child-caregiver interaction}, 
      author={Iba Baig and Kevin Li and Yanbin Xu and Seiji Cattelain and Marie Hallo and Hayato Ono and Sho Tsuji and Ming Bo Cai},
      year={2026},
      eprint={2605.22962},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2605.22962}, 
}
```
