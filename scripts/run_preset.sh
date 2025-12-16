PYTHON_BIN="/home/IBaig/.conda/envs/video/bin/python"

VIDEO="data/videos/6_side.mp4"
OUT_CSV="data/results_6/6_side_toy1.csv"

PRESETS="prompts/presets_short.json"
CONFIG="tarsier/configs/tarser2_default_config.yaml" #tarser is correct spelling, typo on developer's end
TARSIER_MODEL="omni-research/Tarsier2-7b-0115"
CLIP_DURATION="3.0"
STRIDE="3.0"

START_SEC=60  
LIMIT_SEC=180 # 3 minute window
# Toy 1: 1:00 to 5:01 = 60 to 301 seconds
# Toy 2: 5:41 to 8:41 = 341 to 521 seconds 

"$PYTHON_BIN" -m annotate_video \
  --video "$VIDEO" \
  --model "$TARSIER_MODEL" \
  --config "$CONFIG" \
  --prompts "$PRESETS" \
  --out_csv "$OUT_CSV" \
  --clip_sec "$CLIP_DURATION" \
  --stride_sec "$STRIDE" \
  --start_sec "$START_SEC" \
  --limit_sec "$LIMIT_SEC"