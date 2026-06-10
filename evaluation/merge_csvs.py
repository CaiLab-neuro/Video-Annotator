#!/usr/bin/env python3
"""
merge_csvs.py — Merge two annotation CSVs produced by different prompt runs.

When both CSVs contain the same task column the user chooses which takes
priority (either via --priority for a global default, or interactively
per column).  Rows are aligned by t_start.  Columns present in only one
CSV are included as-is.

Usage
-----
    python merge_csvs.py \\
        --csv1  data/results/output_full.csv \\
        --csv2  data/results/output_delta.csv \\
        --out   data/results/output_merged.csv \\
        [--priority csv1|csv2]

Typical workflow
----------------
1. Run full annotation with the original prompts  →  output_full.csv
2. Add new choices via resolve_unknowns.py        →  output_delta_prompts.json
3. Re-run annotation using the delta prompts      →  output_delta.csv
4. Merge, giving priority to the delta for updated tasks:

    python merge_csvs.py \\
        --csv1 output_full.csv --csv2 output_delta.csv \\
        --out  output_merged.csv --priority csv2
"""

import argparse
import csv
import os

TOL = 0.002  # seconds tolerance for t_start alignment
STRUCTURAL = {"video_path", "t_start", "t_end"}


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_csv(path):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        rows = list(reader)
    return fieldnames, rows


def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def build_index(rows):
    """Return {round(t_start, 3): row} for fast lookup."""
    idx = {}
    for r in rows:
        try:
            idx[round(float(r["t_start"]), 3)] = r
        except (ValueError, TypeError, KeyError):
            pass
    return idx


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Merge two annotation CSVs, choosing priority per task column.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--csv1", required=True, help="First annotation CSV")
    ap.add_argument("--csv2", required=True, help="Second annotation CSV")
    ap.add_argument("--out",  required=True, help="Output merged CSV path")
    ap.add_argument("--priority", choices=["csv1", "csv2"], default=None,
                    help="Global priority for overlapping task columns. "
                         "If omitted, asks interactively per column.")
    args = ap.parse_args()

    fn1, rows1 = load_csv(args.csv1)
    fn2, rows2 = load_csv(args.csv2)

    cols1 = [c for c in fn1 if c not in STRUCTURAL]
    cols2 = [c for c in fn2 if c not in STRUCTURAL]
    set1, set2 = set(cols1), set(cols2)

    overlap = [c for c in fn1 if c in set2 and c not in STRUCTURAL]
    only1   = [c for c in cols1 if c not in set2]
    only2   = [c for c in cols2 if c not in set1]

    print(f"\n  CSV 1: {os.path.basename(args.csv1)}  "
          f"({len(rows1)} rows, {len(cols1)} task column(s))")
    print(f"  CSV 2: {os.path.basename(args.csv2)}  "
          f"({len(rows2)} rows, {len(cols2)} task column(s))")
    print(f"\n  Overlapping  ({len(overlap)}): {', '.join(overlap) or 'none'}")
    print(f"  Only in CSV1 ({len(only1)}):  {', '.join(only1)  or 'none'}")
    print(f"  Only in CSV2 ({len(only2)}):  {', '.join(only2)  or 'none'}")

    # -----------------------------------------------------------------------
    # Determine priority for each overlapping column
    # -----------------------------------------------------------------------
    priority = {}   # col -> "csv1" | "csv2"

    if args.priority:
        for col in overlap:
            priority[col] = args.priority
        if overlap:
            print(f"\n  Priority for all overlapping columns: {args.priority}")
    elif overlap:
        print(f"\n  For each overlapping column choose which CSV takes priority.")
        print(f"  Enter '1' for CSV 1, '2' for CSV 2,")
        print(f"  or 'a1' / 'a2' to apply that choice to all remaining columns.\n")

        apply_all = None
        for col in overlap:
            if apply_all:
                priority[col] = apply_all
                print(f"  [{col}]  → {apply_all} (applied to all)")
                continue
            while True:
                try:
                    ans = input(f"  [{col}] priority? [1/2/a1/a2]: ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    print("\nInterrupted.")
                    return
                if ans in ("1", "csv1"):
                    priority[col] = "csv1"
                    break
                elif ans in ("2", "csv2"):
                    priority[col] = "csv2"
                    break
                elif ans == "a1":
                    apply_all = "csv1"
                    priority[col] = "csv1"
                    print("  Applying 'csv1' to all remaining columns.")
                    break
                elif ans == "a2":
                    apply_all = "csv2"
                    priority[col] = "csv2"
                    print("  Applying 'csv2' to all remaining columns.")
                    break
                else:
                    print("  Please enter 1, 2, a1, or a2.")

    # -----------------------------------------------------------------------
    # Build row index and merge
    # -----------------------------------------------------------------------
    idx1 = build_index(rows1)
    idx2 = build_index(rows2)
    all_keys = sorted(set(idx1) | set(idx2))

    # Output fieldnames: structural columns from CSV1, then all task columns
    # in CSV1 order, then any extra columns from CSV2
    out_fields = [c for c in fn1]
    for c in fn2:
        if c not in set(out_fields):
            out_fields.append(c)

    merged_rows = []
    for key in all_keys:
        r1 = idx1.get(key, {})
        r2 = idx2.get(key, {})
        row = {}
        for col in out_fields:
            if col in STRUCTURAL:
                row[col] = r1.get(col) or r2.get(col, "")
            elif col in priority:
                src = r1 if priority[col] == "csv1" else r2
                row[col] = src.get(col, "")
            elif col in set1:
                row[col] = r1.get(col, "")
            else:
                row[col] = r2.get(col, "")
        merged_rows.append(row)

    write_csv(args.out, out_fields, merged_rows)
    print(f"\n  Merged CSV written to: {args.out}  ({len(merged_rows)} rows)\n")


if __name__ == "__main__":
    main()
