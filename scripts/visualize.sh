#!/bin/bash
# NOTE: Activate your conda environment before running this script:
#   conda activate tarsier

# Visualize results from CSV files (model and human annotations)

python -m visualizer \
  --csv_model data/results_27/27_side_toy2.csv \
  --csv_human data/results_27/27_side_human_toy2.csv \
  --out_dir data/results_27/visualizations_27


# Single camera (as before)
# python visualizer.py --input data/side_clips.csv --output results/side

# Comparison side vs room camera
# python visualizer.py \
#   --input data/side_clips.csv \
#   --side_csv data/side_clips.csv \
#   --room_csv data/room_clips.csv \
#   --output results/side_room
