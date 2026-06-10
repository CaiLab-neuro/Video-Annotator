#!/usr/bin/env python3
"""
summarize_unknowns.py — Aggregate unknown-label entries into merged time ranges.

Reads one or more *_unknowns.jsonl files produced by annotate_video.py,
groups entries that share the same (task, raw response), merges overlapping
or adjacent time windows within each group, and prints a sorted summary.

Usage
-----
    python summarize_unknowns.py data/results_27/27_side_unknowns.jsonl
    python summarize_unknowns.py run1_unknowns.jsonl run2_unknowns.jsonl
    python summarize_unknowns.py data/results_27/27_side_unknowns.jsonl --out report.txt
    python summarize_unknowns.py data/results_27/27_side_unknowns.jsonl --by_task
"""

import argparse
import json
import sys
from collections import defaultdict


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def load_entries(paths):
    entries = []
    for path in paths:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    e = json.loads(line)
                    e["_source"] = path
                    entries.append(e)
    return entries


def merge_windows(windows):
    """Merge overlapping or adjacent [t_start, t_end] intervals.

    'Adjacent' means t_start of the next clip == t_end of the current one,
    which happens when stride_sec == clip_sec (no gap between clips).
    We also treat a gap of at most GAP_TOL seconds as adjacent.
    """
    GAP_TOL = 0.01   # seconds; handles floating-point rounding
    sorted_w = sorted(windows, key=lambda x: x[0])
    merged = [list(sorted_w[0])]
    for start, end in sorted_w[1:]:
        if start <= merged[-1][1] + GAP_TOL:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]


def build_groups(entries):
    """Return list of group dicts, one per (source, task, raw) triple."""
    import os
    buckets = defaultdict(list)
    choices_map = {}
    for e in entries:
        source = e.get("_source", "")
        key = (source, e["task"], e.get("raw", "").strip())
        buckets[key].append((float(e["t_start"]), float(e["t_end"])))
        if key not in choices_map:
            choices_map[key] = e.get("choices", [])

    groups = []
    for (source, task, raw), windows in buckets.items():
        merged = merge_windows(windows)
        groups.append({
            "source":      source,
            "source_name": os.path.basename(source),
            "task":        task,
            "raw":         raw,
            "choices":     choices_map[(source, task, raw)],
            "count":       len(windows),
            "windows":     merged,
            "first_start": merged[0][0],
        })

    groups.sort(key=lambda g: (g["source"], g["task"], g["first_start"]))
    return groups


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def fmt_time(sec):
    """Format seconds as M:SS.s"""
    m = int(sec) // 60
    s = sec - m * 60
    return f"{m}:{s:04.1f}"


def fmt_range(start, end):
    return f"{fmt_time(start)}–{fmt_time(end)}  ({start:.1f}s–{end:.1f}s)"


def format_report(groups, by_task=False, multi_source=False):
    lines = []

    if by_task:
        # Group by task first, then source within each task
        tasks = sorted({g["task"] for g in groups})
        for task in tasks:
            task_groups = [g for g in groups if g["task"] == task]
            n_clips   = sum(g["count"] for g in task_groups)
            n_windows = sum(len(g["windows"]) for g in task_groups)
            lines.append(f"\n{'━'*70}")
            lines.append(f"  TASK: {task}   "
                         f"({n_clips} unknown clip(s) → {n_windows} merged window(s))")
            lines.append(f"{'━'*70}")
            for g in sorted(task_groups, key=lambda x: (x["source"], x["first_start"])):
                raw_display = g["raw"] if g["raw"] else "(empty response)"
                lines.append(f"\n  Raw response: \"{raw_display}\"")
                lines.append(f"  Valid choices: {', '.join(g['choices'])}")
                lines.append(f"  Occurrences: {g['count']} clip(s)  "
                             f"→  {len(g['windows'])} merged window(s)")
                for s, e in g["windows"]:
                    prefix = f"[{g['source_name']}] " if multi_source else ""
                    lines.append(f"    • {prefix}{fmt_range(s, e)}")
    else:
        # Chronological per source, then interleaved
        all_windows = []
        for g in groups:
            for s, e in g["windows"]:
                all_windows.append((g["source"], s, e, g))
        all_windows.sort(key=lambda x: (x[0], x[1]))

        lines.append(f"\n{'━'*70}")
        lines.append(f"  {len(groups)} unknown pattern(s) across "
                     f"{sum(len(g['windows']) for g in groups)} merged window(s)")
        lines.append(f"{'━'*70}")

        cur_source = None
        for source, s, e, g in all_windows:
            if multi_source and source != cur_source:
                lines.append(f"\n  ── {g['source_name']} ──")
                cur_source = source
            raw_display = g["raw"] if g["raw"] else "(empty response)"
            lines.append(f"\n  {fmt_range(s, e)}")
            lines.append(f"  task   : {g['task']}")
            lines.append(f"  raw    : \"{raw_display}\"")
            lines.append(f"  choices: {', '.join(g['choices'])}")

    lines.append("")  # trailing newline
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Summarise unknown-label entries into merged time ranges.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("unknowns", nargs="+",
                    help="One or more *_unknowns.jsonl files")
    ap.add_argument("--out", default=None,
                    help="Write report to this file in addition to stdout")
    ap.add_argument("--by_task", action="store_true",
                    help="Group output by task rather than chronologically")
    args = ap.parse_args()

    entries = load_entries(args.unknowns)
    if not entries:
        print("No unknown entries found.", file=sys.stderr)
        sys.exit(0)

    multi_source = len(args.unknowns) > 1
    groups  = build_groups(entries)
    report  = format_report(groups, by_task=args.by_task, multi_source=multi_source)

    print(report)

    if args.out:
        with open(args.out, "w") as f:
            f.write(report)
        print(f"Report written to: {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
