<div align="center">

<h2>Annotator Tool of Behavioral Video Data</h2>

Iba Baig (Northeastern University), Mingbo Cai, Ph.D. (University of Miami).

</div>

# Overview

### Abstract

How do children and parents switch attention and choose actions during their dynamic interactions? To study this requires naturalist video recording of their activity. Current research in developmental psychology relies heavily on frame-to-frame manual coding of video recordings, which is time-intensive and subjective. We investigate whether multimodal AI can automatically classify behavioral states during such interaction given language prompts by building a video analysis pipeline employing Tarsier, a large video-language model developed by ByteDance. Our pipeline extracts sequential overlapping brief (1-3 second) video clips from 20-minute videos of child-parent playing activity in the lab, and submits them to the video-language model with standardized prompts for classifying child’s attention direction, hand actions, toy manipulation styles, and parental gesturing behaviors. This process generates a time series of these behavioral features reflecting the moment-to-moment changes of the action and attention of children and parents. Preliminary analysis of the pipeline’s output in comparison to manual annotation confirms its validity. Further, by allowing the model to analyze an entire 20-minute video, we find that the model can reveal quantifiable visual cues such as child’s pose, child’s longest toy play time, distraction, and parental gestures with a single prompt and can also infer the child’s preference among the toys played during the recording session. This approach demonstrates that AI can be utilized for objective measures of social engagement previously inefficient through manual observation and speeds up the analysis of large-scale video datasets of children’s behavior. The methodology has implications for early developmental screening, personalized intervention design, and advancing our understanding of how visual interaction dynamics influence social development. Future work will integrate audio analysis and eye-tracking data.

### Simple Model Structure
Tarsier takes a simple sturcture that use a MLP projection layer to connect visual encoder (CLIP ViT) and text decoder (LLM). Frames are encoded independently and concatenated to input into LLM.

# Usage
This section provides guidance on how to run, evaluate and deploy this model.

## Setup
Following all are running under the virtual environment of python 3.9. First, clone this git repo, then continue to the following instructions to add the Tarsier model. Install main dependencies

### Model Prepare
Download the model checkpoints from Hugging Face: [Tarsier2-Recap-7b](https://huggingface.co/omni-research/Tarsier2-Recap-7b). Change model_path to "omni-research/Tarsier2-7b-0115".
Complete instructions to integrate Tarsier2 model in folder annotations_module: [Tarsier2](https://github.com/bytedance/tarsier).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r tarsier/requirements.txt
```

### Get Started

#### Annotating Videos with Presets

This pipeline automatically analyzes behavioral videos by extracting short clips and classifying behavioral features across multiple dimensions (e.g., child hand action, toy engagement, spatial proximity, parental gestures).

**Quick Start:**

1. Configure the `scripts/run_preset.sh` script with your parameters:
   ```bash
   vim scripts/run_preset.sh
   ```

2. Key variables to set:
   - `PYTHON_BIN`: Path to your Python executable
   - `VIDEO`: Path to your input video file
   - `OUT_CSV`: Path where the output annotations CSV will be saved
   - `PRESETS`: Path to the prompts configuration (e.g., `prompts/presets_short.json`)
   - `CONFIG`: Path to Tarsier config (e.g., `tarsier/configs/tarser2_default_config.yaml`)
   - `TARSIER_MODEL`: Model identifier (e.g., `omni-research/Tarsier2-7b-0115`)

3. Run the annotation script:
   ```bash
   bash scripts/run_preset.sh
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
