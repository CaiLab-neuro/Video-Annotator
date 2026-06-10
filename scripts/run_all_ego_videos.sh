#!/bin/bash
# Annotate all egocentric videos (child or adult/parent perspective) in VIDEO_DIR.
# Side and room views are excluded — use run_all_exo_videos.sh for those.
#
# NOTE: In the video filenames, the naming is swapped relative to the study roles:
#   view_type "parent" in filename  → child's egocentric camera  (use --perspective child)
#   view_type "child"  in filename  → adult's egocentric camera  (use --perspective adult|parent)
#
# Activate your conda environment before running:
#   conda activate tarsier
#
# Output CSVs use the canonical perspective name in the filename:
#   child  perspective → {subject}_child.csv
#   adult/parent perspective → {subject}_parent.csv
#
# Usage:
#   bash run_all_ego_videos.sh --perspective child              # process all child-ego videos
#   bash run_all_ego_videos.sh --perspective adult              # process all adult-ego videos
#   bash run_all_ego_videos.sh --perspective parent             # same as adult
#   bash run_all_ego_videos.sh --perspective child 10 27 28    # filter to subjects 10, 27, 28
#   bash run_all_ego_videos.sh --perspective child --resume     # skip already-complete outputs; prompt on incomplete/wrong-format

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
PERSPECTIVE=""
SUBJECT_IDS=()
RESUME=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --perspective)
            PERSPECTIVE="$2"
            shift 2
            ;;
        --resume)
            RESUME=1
            shift
            ;;
        *)
            SUBJECT_IDS+=("$1")
            shift
            ;;
    esac
done

if [[ "$PERSPECTIVE" != "child" && "$PERSPECTIVE" != "adult" && "$PERSPECTIVE" != "parent" ]]; then
    echo "Error: --perspective must be 'child', 'adult', or 'parent' (adult and parent are aliases)."
    echo "Usage: bash run_all_ego_videos.sh --perspective child|adult|parent [subject_id ...]"
    exit 1
fi

# Normalize aliases: "adult" and "parent" are the same perspective.
if [[ "$PERSPECTIVE" == "adult" ]]; then
    PERSPECTIVE="parent"
fi

# Map user-facing perspective to the view_type string used in video filenames.
# Filename convention is swapped: "parent" = child's camera, "child" = adult's camera.
if [[ "$PERSPECTIVE" == "child" ]]; then
    VIEW_TYPE_IN_FILENAME="parent"
    OUT_KEY_PREFIX="child"    # output key = {subject}_child
    PRESETS="prompts/presets_ego_child.json"
else
    # perspective == "parent" (adult)
    VIEW_TYPE_IN_FILENAME="child"
    OUT_KEY_PREFIX="parent"   # output key = {subject}_parent
    PRESETS="prompts/presets_ego_adult.json"
fi

# ---------------------------------------------------------------------------
# Paths and parameters
# ---------------------------------------------------------------------------
VIDEO_DIR="/data/Cai_gaze/Tsuji_lab_collaboration/results/aligned_video_YB_finalized_version"
OUT_DIR="/data/Cai_gaze/Tsuji_lab_collaboration/results/video_annotation/ICDL_3s_ego_${PERSPECTIVE}"  # "parent" for adult/parent, "child" for child

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

echo "Perspective : $PERSPECTIVE (view_type='$VIEW_TYPE_IN_FILENAME' in filenames)"
echo "Presets     : $PRESETS"
echo "Output dir  : $OUT_DIR"
echo ""

# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------

# Track whether the user chose "yes to all" for overwrite prompts.
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

# ---------------------------------------------------------------------------
# Process videos
# ---------------------------------------------------------------------------
mapfile -t _videos < <(printf '%s\n' "$VIDEO_DIR"/*.mp4 | sort -V)
for video in "${_videos[@]}"; do
    basename=$(basename "$video" .mp4)
    view_type=$(echo "$basename" | cut -d'_' -f2)
    subject_id=$(echo "$basename" | cut -d'_' -f1)
    key="${subject_id}_${OUT_KEY_PREFIX}"   # always use canonical name (e.g. 10_parent)

    # Only process the requested egocentric view.
    if [[ "$view_type" != "$VIEW_TYPE_IN_FILENAME" ]]; then
        echo "[skip] $key (view '$view_type' != '$VIEW_TYPE_IN_FILENAME')"
        continue
    fi

    # If subject IDs were specified, skip videos not in the list.
    if [ ${#SUBJECT_IDS[@]} -gt 0 ]; then
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
        # Need LIMIT_SEC to verify completeness, so compute it now.
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
            # Incomplete or wrong format — ask unless user already said yes-to-all.
            if [[ $OVERWRITE_ALL -eq 0 ]]; then
                echo "[resume] $key: CSV exists but status='$_status' (last t_end may differ from expected $_lim)."
                echo "         Overwrite?  y=yes  a=yes to all future  n=skip (default: n)"
                read -r -p "         Choice [y/a/n]: " _choice
                case "$_choice" in
                    y|Y) ;;           # proceed once
                    a|A) OVERWRITE_ALL=1 ;;  # proceed and remember
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

    # Compute LIMIT_SEC as the end of the last complete clip that fits within the video.
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
    echo "Done. Processed $PERSPECTIVE-ego videos for subjects: ${SUBJECT_IDS[*]}"
else
    echo "All $PERSPECTIVE-ego videos processed."
fi
