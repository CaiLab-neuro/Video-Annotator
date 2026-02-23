
import argparse
import csv
import json
import math
import os
import shutil
import signal
import subprocess
import tempfile
import sys
from tqdm import tqdm
import time

# Segments a long video into clips, applies structured prompt-based inference
# with a video–language model, and writes per-clip behavioral annotations to a CSV.

_current_proc = None  # tracks the active subprocess so signal handlers can kill it

def _kill_current_proc():
    global _current_proc
    if _current_proc is not None and _current_proc.poll() is None:
        try:
            os.killpg(os.getpgid(_current_proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass

def _signal_handler(sig, frame):
    _kill_current_proc()
    sys.exit(1)

signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)

def sh(cmd):
    global _current_proc
    _current_proc = subprocess.Popen(
        cmd, shell=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        preexec_fn=os.setsid,  # new process group so killpg reaches all grandchildren
    )
    out, _ = _current_proc.communicate()
    code = _current_proc.returncode
    _current_proc = None
    return code, out

def ffprobe_duration(path):
    code, out = sh(f'ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "{path}"')
    if code != 0:
        raise RuntimeError(f"ffprobe failed on {path}:\n{out}")
    try:
        return float(out.strip())
    except:
        raise RuntimeError(f"Could not parse duration from ffprobe output:\n{out}")

def cut_clip(src, dst, start, dur):
    code, out = sh(
        f'ffmpeg -y -ss {start:.3f} -i "{src}" -t {dur:.3f} '
        f'-vf "scale=640:360" -c:v libx264 -preset ultrafast -an "{dst}"'
    )
    if code != 0 or not os.path.exists(dst) or os.path.getsize(dst) == 0:
        raise RuntimeError(f"ffmpeg failed to cut clip:\n{out}")

def run_presets(python_bin, model, config, prompts, clip_path, out_jsonl,
                n_frames=None, max_pixels=None):
    cmd = (
        f'"{python_bin}" -m run_prompt_presets '
        f'--model_name_or_path "{model}" '
        f'--config "{config}" '
        f'--input_path "{clip_path}" '
        f'--prompts "{prompts}" '
        f'--output "{out_jsonl}"'
    )
    if n_frames is not None:
        cmd += f' --n_frames {n_frames}'
    if max_pixels is not None:
        cmd += f' --max_pixels {max_pixels}'
    code, out = sh(cmd)
    if code != 0:
        raise RuntimeError(f"inference_presets failed:\n{out}")
    return out

def jsonl_labels_only(jsonl_path):
    """
    Read the output JSONL from inference_presets and return:
    label_dict: mapping task -> normalized label
    """
    label_dict = {}
    try:
        with open(jsonl_path, "r") as f:
            for line in f:
                obj = json.loads(line)
                task = obj["task"]
                label_dict[task] = obj.get("label", "unknown")
    except FileNotFoundError:
        print(f"Output file not found: {jsonl_path}")
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON in {jsonl_path}: {e}")
    except Exception as e:
        print(f"Error reading {jsonl_path}: {e}")
    return label_dict

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Path to a behavioral video")
    ap.add_argument("--model", required=True, help="Model path or hub id (e.g., omni-research/Tarsier2-7b-0115)")
    ap.add_argument("--config", default="configs/tarser2_default_config.yaml")
    ap.add_argument("--prompts", required=True, help="Preset bank JSON (your prompts file)")
    ap.add_argument("--python", default=sys.executable,
                    help="Python interpreter to run inference_presets (defaults to the current interpreter).")
    ap.add_argument("--out_csv", default="data/clips.csv")
    ap.add_argument("--clip_sec", type=float, default=0.8)
    ap.add_argument("--stride_sec", type=float, default=5.0)
    ap.add_argument("--start_sec", type=float, default=0.0, help="Start processing from this timestamp (in seconds)")
    ap.add_argument("--limit_sec", type=float, default=None, help="Optional: only process first N seconds")
    ap.add_argument("--n_frames", type=int, default=None,
                    help="Frames sampled per clip (overrides config). "
                         "Fewer frames = faster. Default from config (16, doubled to 32 by tarsier).")
    ap.add_argument("--max_pixels", type=int, default=None,
                    help="Max pixels per frame (overrides config). "
                         "Lower = faster. E.g. 200704 (448x448) vs default 460800 (678x678).")
    ap.add_argument("--raw_dir", default=None,
                    help="Optional directory to save per-clip JSONL files with raw model outputs. "
                         "Useful for debugging label normalization. If omitted, intermediate files are deleted.")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)

    # Optional persistent directory for per-clip JSONL files (raw model output).
    # Each file contains the model's raw text and normalized label for every task.
    if args.raw_dir:
        os.makedirs(args.raw_dir, exist_ok=True)
        print(f"[info] Raw per-clip outputs will be saved to: {args.raw_dir}/")

    full_duration = ffprobe_duration(args.video)
    effective_duration = full_duration

    if args.limit_sec:
        effective_duration = min(full_duration, args.start_sec + args.limit_sec)

    print(f"Full video duration: {full_duration:.2f}s")
    print(f"Processing duration: {effective_duration:.2f}s; sampling every {args.stride_sec}s with clip {args.clip_sec}s")

    fieldnames = [
        "video_path", "t_start", "t_end",
        "toy_in_environment",
        "parent_holding_toy",
        "child_holding_toy",
        "child_hand_action",
        "child_proximity_behavior",
        "adult_hand_action",
        "child_pose",
        "adult_pose",
    ]

    tmpdir = tempfile.mkdtemp(prefix="tarsier_clips_")
    try:
        end_time = effective_duration
        n = max(0, int(math.floor((end_time - args.start_sec - args.clip_sec) / args.stride_sec)) + 1)
        print(f"[info] Will process {n} segments starting from {args.start_sec}s")

        # Open CSV immediately so partial results survive if the run is interrupted.
        with open(args.out_csv, "w", newline="") as csv_f:
            w = csv.DictWriter(csv_f, fieldnames=fieldnames)
            w.writeheader()

            pbar = tqdm(range(n), desc="Annotating", unit="clip")
            for i in pbar:
                t0 = args.start_sec + (i * args.stride_sec)
                t1 = t0 + args.clip_sec
                clip_path = os.path.join(tmpdir, f"clip_{i:05d}.mp4")

                pbar.set_postfix({"segment": f"{t0:.1f}-{t1:.1f}s"})

                # JSONL goes to raw_dir (persistent) if requested, otherwise temp dir (deleted after).
                if args.raw_dir:
                    out_jsonl = os.path.join(args.raw_dir, f"clip_{i:05d}_{t0:.2f}-{t1:.2f}.jsonl")
                else:
                    out_jsonl = os.path.join(tmpdir, f"clip_{i:05d}.jsonl")

                try:
                    cut_clip(args.video, clip_path, t0, args.clip_sec)
                    run_presets(args.python, args.model, args.config, args.prompts, clip_path, out_jsonl,
                                n_frames=args.n_frames, max_pixels=args.max_pixels)
                    pred = jsonl_labels_only(out_jsonl)
                except Exception as e:
                    tqdm.write(f"[warn] failed at segment {i} ({t0:.2f}-{t1:.2f}s): {e}")
                    pred = {}

                row_data = {
                    "video_path": args.video,
                    "t_start": round(t0, 3),
                    "t_end": round(t1, 3),
                    "toy_in_environment": pred.get("toy_in_environment", "unknown"),
                    "parent_holding_toy": pred.get("parent_holding_toy", "unknown"),
                    "child_holding_toy": pred.get("child_holding_toy", "unknown"),
                    "child_hand_action": pred.get("child_hand_action", "unknown"),
                    "child_proximity_behavior": pred.get("child_proximity_behavior", "unknown"),
                    "adult_hand_action": pred.get("adult_hand_action", "unknown"),
                    "child_pose": pred.get("child_pose", "unknown"),
                    "adult_pose": pred.get("adult_pose", "unknown"),
                }

                w.writerow(row_data)
                csv_f.flush()  # persist each row immediately

                tqdm.write(f"  [{i+1}/{n}] {t0:.1f}-{t1:.1f}s | "
                           f"toy={row_data['toy_in_environment']}, "
                           f"child_hand={row_data['child_hand_action']}, "
                           f"adult_hand={row_data['adult_hand_action']}, "
                           f"prox={row_data['child_proximity_behavior']}, "
                           f"child_pose={row_data['child_pose']}")

        print(f"[ok] wrote {n} rows → {args.out_csv}")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

if __name__ == "__main__":
    main()