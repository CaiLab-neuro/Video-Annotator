#!/bin/bash
# Annotate all exocentric videos (side and room views) in VIDEO_DIR.
# Child and parent (egocentric) videos are excluded — they need separate presets.
#
# Activate your conda environment before running:
#   conda activate tarsier
#
# Output CSVs are named after the first two fields of the video filename,
# e.g. 10_side_3_36826_merged.mp4 -> 10_side.csv

VIDEO_DIR="/data/Cai_gaze/Tsuji_lab_collaboration/results/aligned_video_YB_finalized_version"
OUT_DIR="/data/Cai_gaze/Tsuji_lab_collaboration/results/video_annotation/all"

PRESETS="prompts/presets_shorter.json"
CONFIG="tarsier/configs/tarser2_default_config.yaml"  # "tarser" is a typo from the original developers
TARSIER_MODEL="omni-research/Tarsier2-7b-0115"

CLIP_DURATION="1.0"
STRIDE="0.25"
START_SEC=0
N_FRAMES=15
MAX_PIXELS=307200

# Set to a path to save per-clip JSONL files with raw model text output.
# Leave empty to discard (default).
RAW_DIR=""

export CUDA_VISIBLE_DEVICES=0

mkdir -p "$OUT_DIR"

for video in "$VIDEO_DIR"/*.mp4; do
    # Extract the view type (second underscore-separated field).
    # e.g. "10_side_3_36826_merged.mp4" -> view_type="side", key="10_side"
    basename=$(basename "$video" .mp4)
    view_type=$(echo "$basename" | cut -d'_' -f2)
    key=$(echo "$basename" | cut -d'_' -f1,2)

    # Only process exocentric views (side and room).
    if [[ "$view_type" != "side" && "$view_type" != "room" ]]; then
        echo "[skip] $key (not an exo view)"
        continue
    fi

    out_csv="$OUT_DIR/${key}.csv"

    # Get video duration in seconds via ffprobe.
    duration=$(ffprobe -v quiet -show_entries format=duration -of csv=p=0 "$video")
    if [ -z "$duration" ]; then
        echo "[error] Could not read duration for $video, skipping."
        continue
    fi

    # Compute LIMIT_SEC as the end of the last complete clip that fits within the video:
    #   last_start = floor((duration - clip_sec) / stride) * stride
    #   LIMIT_SEC  = last_start + clip_sec
    # This guarantees no clip extends beyond the video end.
    LIMIT_SEC=$(python3 -c "
import math
d, c, s = float('$duration'), float('$CLIP_DURATION'), float('$STRIDE')
print(round(math.floor((d - c) / s) * s + c, 6))
")

    echo "[run] $key  duration=${duration}s  LIMIT_SEC=${LIMIT_SEC}s"
    echo "      -> $out_csv"

    python -m annotate_video \
      --video "$video" \
      --model "$TARSIER_MODEL" \
      --config "$CONFIG" \
      --prompts "$PRESETS" \
      --out_csv "$out_csv" \
      --clip_sec "$CLIP_DURATION" \
      --stride_sec "$STRIDE" \
      --start_sec "$START_SEC" \
      --limit_sec "$LIMIT_SEC" \
      --n_frames "$N_FRAMES" \
      --max_pixels "$MAX_PIXELS" \
      ${RAW_DIR:+--raw_dir "$RAW_DIR"}

    echo "[done] $key"
    echo ""
done

echo "All exo videos processed."
