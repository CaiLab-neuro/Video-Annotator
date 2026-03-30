#!/bin/bash
# Annotate all exocentric videos (side and room views) in VIDEO_DIR.
# Child and parent (egocentric) videos are excluded — they need separate presets.
#
# Activate your conda environment before running:
#   conda activate tarsier
#
# Output CSVs are named after the first two fields of the video filename,
# e.g. 10_side_3_36826_merged.mp4 -> 10_side.csv
#
# Usage:
#   bash run_all_exo_videos.sh              # process all subjects
#   bash run_all_exo_videos.sh 10 27 28     # process only subjects 10, 27, 28
#   bash run_all_exo_videos.sh --resume     # skip already-complete outputs; prompt on incomplete/wrong-format
#   bash run_all_exo_videos.sh --resume 10 27 28

# Parse optional arguments.
RESUME=0
SUBJECT_IDS=()
for _arg in "$@"; do
    if [[ "$_arg" == "--resume" ]]; then
        RESUME=1
    else
        SUBJECT_IDS+=("$_arg")
    fi
done

VIDEO_DIR="/data/Cai_gaze/Tsuji_lab_collaboration/results/aligned_video_YB_finalized_version"
OUT_DIR="/data/Cai_gaze/Tsuji_lab_collaboration/results/video_annotation/ICDL_3s_token16"

PRESETS="prompts/presets_short_delta.json"
CONFIG="tarsier/configs/tarser2_default_config.yaml"  # "tarser" is a typo from the original developers
TARSIER_MODEL="omni-research/Tarsier2-7b-0115"

CLIP_DURATION="3.0"
STRIDE="1.0"
START_SEC=0
N_FRAMES=30
MAX_PIXELS=307200

# Set to a path to save per-clip JSONL files with raw model text output.
# Leave empty to discard (default).
RAW_DIR=""

export CUDA_VISIBLE_DEVICES=3

mkdir -p "$OUT_DIR"

# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------

OVERWRITE_ALL=0

# check_csv_status <csv_path> <presets_json> <expected_limit_sec>
# Prints one of: complete | incomplete | wrong_format | error
check_csv_status() {
    python3 - "$1" "$2" "$3" <<'PYEOF'
import csv, json, sys
csv_path, presets_path, limit_sec = sys.argv[1], sys.argv[2], float(sys.argv[3])
try:
    with open(presets_path) as f:
        data = json.load(f)
    tasks = [p["task"] for p in data["presets"]]
    expected_cols = ["video_path", "t_start", "t_end"] + tasks
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        if list(reader.fieldnames) != expected_cols:
            print("wrong_format")
            sys.exit(0)
        last_row = None
        for row in reader:
            last_row = row
    if last_row is None:
        print("incomplete")
        sys.exit(0)
    if abs(float(last_row["t_end"]) - limit_sec) < 0.01:
        print("complete")
    else:
        print("incomplete")
except Exception:
    print("error")
PYEOF
}

mapfile -t _videos < <(printf '%s\n' "$VIDEO_DIR"/*.mp4 | sort -V)
for video in "${_videos[@]}"; do
    # Extract the view type (second underscore-separated field).
    # e.g. "10_side_3_36826_merged.mp4" -> view_type="side", key="10_side"
    basename=$(basename "$video" .mp4)
    view_type=$(echo "$basename" | cut -d'_' -f2)
    key=$(echo "$basename" | cut -d'_' -f1,2)

    # Only process exocentric views (side and room).
    if [[ "$view_type" != "side" && "$view_type" != "room" ]]; then # 
        echo "[skip] $key (not an exo view)"
        continue
    fi

    # If subject IDs were specified, skip videos not in the list.
    if [ ${#SUBJECT_IDS[@]} -gt 0 ]; then
        subject_id=$(echo "$basename" | cut -d'_' -f1)
        match=0
        for id in "${SUBJECT_IDS[@]}"; do
            if [[ "$subject_id" == "$id" ]]; then
                match=1
                break
            fi
        done
        if [ $match -eq 0 ]; then
            echo "[skip] $key (subject $subject_id not in filter)"
            continue
        fi
    fi

    out_csv="$OUT_DIR/${key}.csv"

    # --resume: if the output CSV already exists, check whether it is complete.
    if [[ $RESUME -eq 1 && -f "$out_csv" ]]; then
        _dur=$(ffprobe -v quiet -show_entries format=duration -of csv=p=0 "$video")
        if [ -z "$_dur" ]; then
            echo "[error] Could not read duration for $video, skipping."
            continue
        fi
        _lim=$(python3 -c "
import math
d, c, s = float('$_dur'), float('$CLIP_DURATION'), float('$STRIDE')
print(round(math.floor((d - c) / s) * s + c, 6))
")
        _status=$(check_csv_status "$out_csv" "$PRESETS" "$_lim")
        if [[ "$_status" == "complete" ]]; then
            echo "[skip] $key (output already complete)"
            continue
        else
            if [[ $OVERWRITE_ALL -eq 0 ]]; then
                echo "[resume] $key: CSV exists but status='$_status' (last t_end may differ from expected $_lim)."
                echo "         Overwrite?  y=yes  a=yes to all future  n=skip (default: n)"
                read -r -p "         Choice [y/a/n]: " _choice
                case "$_choice" in
                    y|Y) ;;
                    a|A) OVERWRITE_ALL=1 ;;
                    *)
                        echo "[skip] $key"
                        continue
                        ;;
                esac
            fi
            echo "[overwrite] $key (status was '$_status')"
        fi
    fi

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
      ${RAW_DIR:+--raw_dir "$RAW_DIR"} \
      --independent_questions

    echo "[done] $key"
    echo ""
done

if [ ${#SUBJECT_IDS[@]} -gt 0 ]; then
    echo "Done. Processed subjects: ${SUBJECT_IDS[*]}"
else
    echo "All exo videos processed."
fi
