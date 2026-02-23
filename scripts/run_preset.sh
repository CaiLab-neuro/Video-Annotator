#!/bin/bash
# NOTE: Activate your conda environment before running this script:
#   conda activate tarsier

VIDEO="/data/Cai_gaze/Tsuji_lab_collaboration/results/aligned_video_YB_finalized_version/2_side_11_35584_merged.mp4"
OUT_CSV="/data/Cai_gaze/Tsuji_lab_collaboration/results/video_annotation/2/2_side.csv"

PRESETS="prompts/presets_shorter.json"
CONFIG="tarsier/configs/tarser2_default_config.yaml" #tarser is correct spelling, typo on developer's end
TARSIER_MODEL="omni-research/Tarsier2-7b-0115"
CLIP_DURATION="1.0"
STRIDE="0.5"

START_SEC=0
LIMIT_SEC=1185.5 # 3 minute window
# Toy 1: 1:00 to 5:01 = 60 to 301 seconds
# Toy 2: 5:41 to 8:41 = 341 to 521 seconds

# --- Speed / quality trade-off ---
# N_FRAMES: frames sampled per clip before model processing.
#   Config default is 16; tarsier doubles each frame internally (use_multi_images_for_video),
#   so 8 here → 16 images fed to the vision encoder.
#   Fewer frames = faster but less temporal context.
N_FRAMES=8

# MAX_PIXELS: max resolution per frame (width * height).
#   Config default is 460800 (~678x678). Lower values reduce vision encoder memory and time.
#   200704 = 448x448  (roughly half the area, ~2x faster vision encoding)
#   65536  = 256x256  (fast testing)
MAX_PIXELS=240000

# --- Debugging ---
# Set RAW_DIR to a path to save per-clip JSONL files with the model's raw text output.
# Useful for inspecting what the model actually said before label normalization.
# Leave empty to discard intermediate files (default).
RAW_DIR=""
# RAW_DIR="${OUT_CSV%.csv}_raw"  # uncomment to auto-name next to the output CSV

export CUDA_VISIBLE_DEVICES=0

python -m annotate_video \
  --video "$VIDEO" \
  --model "$TARSIER_MODEL" \
  --config "$CONFIG" \
  --prompts "$PRESETS" \
  --out_csv "$OUT_CSV" \
  --clip_sec "$CLIP_DURATION" \
  --stride_sec "$STRIDE" \
  --start_sec "$START_SEC" \
  --limit_sec "$LIMIT_SEC" \
  --n_frames "$N_FRAMES" \
  --max_pixels "$MAX_PIXELS" \
  ${RAW_DIR:+--raw_dir "$RAW_DIR"}