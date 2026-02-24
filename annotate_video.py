
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
from pathlib import Path
from tqdm import tqdm
import time

# Segments a long video into clips, applies structured prompt-based inference
# with a video–language model, and writes per-clip behavioral annotations to a CSV.

# Ensure project directory is on sys.path so run_prompt_presets is importable
# regardless of the working directory.
_project_dir = Path(__file__).parent.resolve()
if str(_project_dir) not in sys.path:
    sys.path.insert(0, str(_project_dir))

import yaml
import torch
import run_prompt_presets as rpp

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Path to a behavioral video")
    ap.add_argument("--model", required=True, help="Model path or hub id (e.g., omni-research/Tarsier2-7b-0115)")
    ap.add_argument("--config", default="configs/tarser2_default_config.yaml")
    ap.add_argument("--prompts", required=True, help="Preset bank JSON (your prompts file)")
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
    ap.add_argument("--no_kv_cache", action="store_true",
                    help="Disable KV-cache reuse in run_prompt_presets; re-encode video for every question.")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)

    # Optional persistent directory for per-clip JSONL files (raw model output).
    if args.raw_dir:
        os.makedirs(args.raw_dir, exist_ok=True)
        print(f"[info] Raw per-clip outputs will be saved to: {args.raw_dir}/")

    # ------------------------------------------------------------------
    # Load model ONCE — stays in VRAM for all clips
    # ------------------------------------------------------------------
    with open(args.config, "r") as f:
        data_config = yaml.safe_load(f)
    if args.n_frames is not None:
        data_config["n_frames"] = int(args.n_frames)
    if args.max_pixels is not None:
        data_config["max_pixels"] = int(args.max_pixels)

    model, processor = rpp.load_model_and_processor(args.model, data_config)

    with open(args.prompts, "r") as f:
        bank = json.load(f)
    system_prompt = bank.get("global_instructions", {}).get("system_prompt", "") or ""
    defaults = bank.get("global_instructions", {}).get("defaults", {}) or {}
    presets = bank.get("presets", []) or []
    if not presets:
        raise ValueError(f"No 'presets' found in {args.prompts}")

    gen_kwargs = {
        "max_new_tokens": int(defaults.get("max_new_tokens", 12)),
        "temperature":    float(defaults.get("temperature", 0.0)),
        "top_p":          float(defaults.get("top_p", 1.0)),
    }
    n_frames = data_config.get("n_frames", 8)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    chat = rpp.Chat(model, processor, device, debug=False)

    # ------------------------------------------------------------------
    # Build dynamic fieldnames from presets (no hardcoded task list)
    # ------------------------------------------------------------------
    task_names = [p["task"] for p in presets]
    fieldnames = ["video_path", "t_start", "t_end"] + task_names

    full_duration = ffprobe_duration(args.video)
    effective_duration = full_duration

    if args.limit_sec:
        effective_duration = min(full_duration, args.start_sec + args.limit_sec)

    print(f"Full video duration: {full_duration:.2f}s")
    print(f"Processing duration: {effective_duration:.2f}s; sampling every {args.stride_sec}s with clip {args.clip_sec}s")

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

                try:
                    cut_clip(args.video, clip_path, t0, args.clip_sec)
                    pred, raw_rows = rpp.run_clip_inference(
                        chat=chat, clip_path=clip_path, presets=presets,
                        system_prompt=system_prompt, gen_kwargs=gen_kwargs,
                        n_frames=n_frames, no_kv_cache=args.no_kv_cache,
                    )
                    if args.raw_dir:
                        out_jsonl = os.path.join(args.raw_dir, f"clip_{i:05d}_{t0:.2f}-{t1:.2f}.jsonl")
                        with open(out_jsonl, "w") as jf:
                            for row in raw_rows:
                                jf.write(json.dumps(row) + "\n")
                except Exception as e:
                    tqdm.write(f"[warn] failed at segment {i} ({t0:.2f}-{t1:.2f}s): {e}")
                    pred = {}

                row_data = {
                    "video_path": args.video,
                    "t_start": round(t0, 3),
                    "t_end": round(t1, 3),
                    **{task: pred.get(task, "unknown") for task in task_names},
                }

                w.writerow(row_data)
                csv_f.flush()  # persist each row immediately

                tqdm.write(f"  [{i+1}/{n}] {t0:.1f}-{t1:.1f}s | " +
                           ", ".join(f"{k}={v}" for k, v in pred.items()))

        print(f"[ok] wrote {n} rows → {args.out_csv}")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

if __name__ == "__main__":
    main()
