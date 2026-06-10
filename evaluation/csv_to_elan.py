"""
csv_to_elan.py — Convert wide-format annotation CSVs to ELAN-importable TSV.

BACKGROUND
----------
annotate_video.py uses a sliding window with overlap, so consecutive rows share
time. For example with clip_sec=1.0 and stride=0.25:

    row 0:  t_start=0.00,  t_end=1.00
    row 1:  t_start=0.25,  t_end=1.25
    row 2:  t_start=0.50,  t_end=1.50
    ...

Naively converting each row to a tier annotation would produce heavily overlapping
ELAN annotations, which is not useful.

ASSIGNMENT RULE
---------------
Each clip is authoritative for the "new" time it adds past the previous clip:

    bin_start[0] = t_start[0]          (first clip covers its full window)
    bin_start[i] = t_end[i-1]          (subsequent clips: start where the
    bin_end[i]   = t_end[i]             previous clip ended)

With the example above:
    row 0 → bin [0.00, 1.00)
    row 1 → bin [1.00, 1.25)
    row 2 → bin [1.25, 1.50)
    ...

This is independent of clip_sec/stride — the script infers the bins entirely
from the t_start/t_end values in the CSV.

After bin assignment, consecutive bins with the same annotation for a tier are
merged into a single longer annotation.

OUTPUT FORMAT
-------------
Tab-delimited, one row per merged annotation segment:

    tier                begin_s     end_s   annotation
    child_hand_action   0.000       1.250   pointing
    current_toy         0.000       2.500   giraffe
    ...

Times are decimal seconds (e.g. 0.280), which ELAN auto-detects.

ELAN IMPORT STEPS
-----------------
File > Import > CSV/Tab-delimited text file, then map:
    tier       -> Tier
    begin_s    -> Begin Time
    end_s      -> End Time
    annotation -> Annotation

Usage
-----
# Single file (outputs 10_side_elan.tsv alongside the CSV):
python csv_to_elan.py annotation.csv

# Multiple files merged into one output:
python csv_to_elan.py 10_side.csv 11_side.csv -o merged_elan.tsv

# Omit rows where the model couldn't normalise the label:
python csv_to_elan.py annotation.csv --skip-unknown
"""

import argparse
import csv
import sys
from pathlib import Path

# Columns that are metadata, not behavioral tiers.
_META_COLS = {"video_path", "t_start", "t_end"}


def convert(input_csvs, output_tsv, skip_unknown=False):
    # ------------------------------------------------------------------
    # 1. Load all rows across input files
    # ------------------------------------------------------------------
    all_rows = []   # list of (t_start, t_end, dict_row)
    tier_cols = None

    for csv_path in input_csvs:
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                print(f"[warn] {csv_path}: empty or missing header, skipping.")
                continue
            cols = [c for c in reader.fieldnames if c not in _META_COLS]
            if not cols:
                print(f"[warn] {csv_path}: no task columns found, skipping.")
                continue
            if tier_cols is None:
                tier_cols = cols
            for row in reader:
                try:
                    t_start = float(row["t_start"])
                    t_end   = float(row["t_end"])
                except (KeyError, ValueError) as e:
                    print(f"[warn] {csv_path}: could not parse times ({e}), skipping row.")
                    continue
                all_rows.append((t_start, t_end, row))

    if not all_rows:
        sys.exit("[error] No data rows found in input files.")

    # Sort by clip start time so bin boundaries are well-defined.
    all_rows.sort(key=lambda x: x[0])

    # ------------------------------------------------------------------
    # 2. Assign non-overlapping bins
    #    bin_start[0] = t_start[0]  (first clip covers its full window)
    #    bin_start[i] = t_end[i-1]  (each subsequent clip starts where the
    #    bin_end[i]   = t_end[i]     previous one ended)
    # ------------------------------------------------------------------
    bins = []
    for i, (t_start, t_end, row) in enumerate(all_rows):
        bin_start = t_start if i == 0 else all_rows[i - 1][1]
        bins.append((bin_start, t_end, row))

    # ------------------------------------------------------------------
    # 3. Per tier: collect bins, then merge consecutive same-label runs
    # ------------------------------------------------------------------
    rows_written = 0
    with open(output_tsv, "w", newline="") as out_f:
        out_f.write("tier\tbegin_s\tend_s\tannotation\n")

        for tier in tier_cols:
            # Collect this tier's (bin_start, bin_end, annotation) sequence.
            segments = []
            for bin_start, bin_end, row in bins:
                annotation = row.get(tier, "").strip()
                if skip_unknown and annotation == "unknown":
                    continue
                segments.append((bin_start, bin_end, annotation))

            # Merge consecutive segments that share the same annotation
            # (only merge when they are temporally adjacent, i.e. no gap).
            merged = []
            for bin_start, bin_end, annotation in segments:
                if (merged
                        and merged[-1][2] == annotation
                        and abs(merged[-1][1] - bin_start) < 1e-6):
                    # Extend the last segment's end time.
                    merged[-1] = (merged[-1][0], bin_end, annotation)
                else:
                    merged.append((bin_start, bin_end, annotation))

            for begin_s, end_s, annotation in merged:
                out_f.write(f"{tier}\t{begin_s:.3f}\t{end_s:.3f}\t{annotation}\n")
                rows_written += 1

    print(f"[ok] wrote {rows_written} annotation segments -> {output_tsv}")


def main():
    ap = argparse.ArgumentParser(
        description="Convert wide-format annotation CSVs to ELAN-importable TSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("inputs", nargs="+", metavar="CSV", help="One or more input CSV files.")
    ap.add_argument(
        "-o", "--output", default=None,
        help="Output TSV path. Defaults to <first-input-stem>_elan.tsv.",
    )
    ap.add_argument(
        "--skip-unknown", action="store_true",
        help="Omit segments where the annotation is 'unknown' (model normalisation failed).",
    )
    args = ap.parse_args()

    input_paths = [Path(p) for p in args.inputs]
    for p in input_paths:
        if not p.exists():
            sys.exit(f"[error] File not found: {p}")

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_paths[0].with_name(input_paths[0].stem + "_elan.tsv")

    convert(input_paths, output_path, skip_unknown=args.skip_unknown)


if __name__ == "__main__":
    main()
