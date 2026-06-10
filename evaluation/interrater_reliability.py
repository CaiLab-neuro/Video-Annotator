#!/usr/bin/env python3
"""
interrater_reliability.py

Compute inter-rater reliability between two human annotation CSVs produced by
eaf_to_csv_ICDL_miami.py and/or eaf_to_csv_ICDL_ENS.py.

The two CSVs must overlap in their t_start/t_end grid (i.e. the same video
was annotated by two raters using possibly different annotation systems).

Four behavioral dimensions are compared:
  1. child_hand_action
  2. adult_hand_action
  3. child_pose
  4. adult_pose

Both CSVs are mapped to a shared canonical label space (ENS-style) before
comparison.  Only labels present in the SHARED_LABELS set for each dimension
are counted; rows where either rater outputs "unknown" or a label outside the
shared vocabulary are excluded from the metrics (but reported as "excluded").

Outputs
-------
  - Console: per-dimension agreement, Cohen's kappa, and exclusion stats
  - <out_dir>/interrater_metrics.csv
  - <out_dir>/interrater_accuracy.png        (bar chart of agreement / kappa)
  - <out_dir>/interrater_confusion_<dim>.png (confusion matrix per dimension)
  - <out_dir>/interrater_timecourse_<dim>.png (Gantt-chart, rater1 vs rater2)

Usage
-----
    python interrater_reliability.py \\
        --csv_rater1 data/results_6/6_side_human.csv \\
        --csv_rater2 data/results_4/4_side_human.csv \\
        --out_dir    data/interrater/
"""

import argparse
import os
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score, ConfusionMatrixDisplay, confusion_matrix


# ---------------------------------------------------------------------------
# Shared canonical label space  (ENS-style)
# Only rows where *both* raters are in the shared set are included.
# ---------------------------------------------------------------------------

SHARED_LABELS = {
    "child_hand_action": {
        "manipulating toy",
        "holding toy still",
        "touching adult",
        "on the ground/touching some furniture/resting",
        # shared grabbing/giving if both systems annotate them:
        "grabbing toy",
        "giving away the toy",
        "gesturing",
        "pointing",
    },
    "adult_hand_action": {
        "manipulating toy",
        "holding toy still",
        "pointing",
        "touching child",
        "touching box/toy bag/eye-tracker components",
        "gesturing",
        "waving",
        "handing toy to child",
        "taking toy from child",
        "moving toy",
        "holding paper",
        "on the ground/touching some furniture/resting",
    },
    "child_pose": {
        "sitting (kneeling)",
        "standing still",
        "walking",
        "crawling",
        "turning around",
        "invisible",
    },
    "adult_pose": {
        "sitting (kneeling)",
        "standing still",
        "walking",
        "crawling",
        "turning around",
        "invisible",
    },
}

# ---------------------------------------------------------------------------
# Normalization: map all label variants to ENS-canonical form
# Applied to both raters so legacy Miami-style outputs are handled even if
# the Miami script hasn't been re-run yet.
# ---------------------------------------------------------------------------

CANONICALIZE = {
    "child_hand_action": {
        "on some furniture":    "on the ground/touching some furniture/resting",
        "on furniture":         "on the ground/touching some furniture/resting",
        "on the ground":        "on the ground/touching some furniture/resting",
        "resting":              "on the ground/touching some furniture/resting",
        "resting on body":      "on the ground/touching some furniture/resting",
        "none":                 "on the ground/touching some furniture/resting",
        "holding toy":          "holding toy still",
        "opening a box or bag": "touching box/toy bag/eye-tracker components",
        "closing a box or bag": "touching box/toy bag/eye-tracker components",
        "touching glasses":     "touching box/toy bag/eye-tracker components",
    },
    "adult_hand_action": {
        "holding toy":          "holding toy still",
        "on some furniture":    "on the ground/touching some furniture/resting",
        "on furniture":         "on the ground/touching some furniture/resting",
        "on the ground":        "on the ground/touching some furniture/resting",
        "resting":              "on the ground/touching some furniture/resting",
        "resting on body":      "on the ground/touching some furniture/resting",
        "none":                 "on the ground/touching some furniture/resting",
        "opening a box or bag": "touching box/toy bag/eye-tracker components",
        "closing a box or bag": "touching box/toy bag/eye-tracker components",
        "touching glasses":     "touching box/toy bag/eye-tracker components",
        "showing":              "holding toy still",
        "giving away the toy":  "handing toy to child",
    },
    "child_pose": {
        "sitting still":        "sitting (kneeling)",
        "sitting":              "sitting (kneeling)",
        "not visible":          "invisible",
        "lying on ground":      "crawling",     # Miami maps this to crawling
    },
    "adult_pose": {
        "sitting":              "sitting (kneeling)",
        "sitting still":        "sitting (kneeling)",
        "not visible":          "invisible",
    },
}

DIMENSIONS = [
    "child_hand_action",
    "adult_hand_action",
    "child_pose",
    "adult_pose",
]

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.edgecolor":   "black",
    "text.color":       "black",
    "font.size":        13,
    "axes.titlesize":   14,
    "axes.labelsize":   13,
    "xtick.labelsize":  11,
    "ytick.labelsize":  11,
    "legend.fontsize":  11,
})

COLORS = {
    "rater1": "#85d2d0",   # teal
    "rater2": "#887bb0",   # purple
    "agree":  "#59a14f",   # green
    "kappa":  "#ffbd59",   # orange
}

LABEL_COLORS = {
    "manipulating toy":      "#59a14f",
    "holding toy still":     "#76b7b2",
    "grabbing toy":          "#f28e2b",
    "giving away the toy":   "#e15759",
    "handing toy to child":  "#f28e2b",
    "taking toy from child": "#e15759",
    "touching adult":        "#b07aa1",
    "touching child":        "#b07aa1",
    "pointing":              "#4e79a7",
    "gesturing":             "#edc948",
    "waving":                "#ff9da7",
    "moving toy":            "#3a9142",
    "holding paper":         "#a0cbe8",
    "on the ground/touching some furniture/resting": "#bab0ac",
    "touching box/toy bag/eye-tracker components":   "#edc948",
    "none":                  "#eeeeee",
    "sitting (kneeling)":    "#4e79a7",
    "standing still":        "#f28e2b",
    "walking":               "#59a14f",
    "crawling":              "#e15759",
    "turning around":        "#b07aa1",
    "bending over":          "#9c755f",
    "crouching":             "#8b6914",
    "lying on the floor":    "#d4a017",
    "invisible":             "#cccccc",
    "unknown":               "#eeeeee",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def canonicalize_series(series: pd.Series, dim: str) -> pd.Series:
    """Normalize a label series to ENS-canonical form for the given dimension."""
    mapping = CANONICALIZE.get(dim, {})
    s = series.astype(str).str.strip().str.lower()
    s = s.replace("nan", "unknown").fillna("unknown")
    if mapping:
        s = s.replace(mapping)
    return s


def resample_to_1s_grid(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample a variable-stride/clip-length CSV to a 1-second grid.
    For each integer second t, picks the label from the row whose t_start
    is the largest value <= t (i.e. the most recently started clip that
    covers t).  Returns a new DataFrame with t_start=t, t_end=t+1.
    """
    non_label_cols = {"video_path", "t_start", "t_end"}
    label_cols = [c for c in df.columns if c not in non_label_cols]

    t_min = int(np.floor(df["t_start"].min()))
    t_max = int(np.ceil(df["t_end"].max()))

    rows = []
    for t in range(t_min, t_max):
        # Clips that contain second t: t_start <= t < t_end
        mask = (df["t_start"] <= t) & (df["t_end"] > t)
        candidates = df[mask]
        if candidates.empty:
            continue
        # Among candidates, prefer the one whose t_start is closest to t
        row = candidates.iloc[(t - candidates["t_start"]).abs().argsort().iloc[0]]
        new_row = {"t_start": float(t), "t_end": float(t + 1)}
        for col in label_cols:
            new_row[col] = row[col]
        rows.append(new_row)

    return pd.DataFrame(rows)


def load_and_merge(csv1: str, csv2: str, rater1_name: str, rater2_name: str) -> pd.DataFrame:
    """
    Load both CSVs and inner-join on t_start + t_end.
    Columns from csv2 get the suffix '_r2'; csv1 columns keep their original names.
    If the exact join yields 0 rows (e.g. different clip lengths/strides),
    both CSVs are resampled to a common 1-second grid before merging.
    """
    df1 = pd.read_csv(csv1)
    df2 = pd.read_csv(csv2)

    for col in ("t_start", "t_end"):
        if col not in df1.columns or col not in df2.columns:
            raise ValueError(f"Both CSVs must contain '{col}'.")

    n1, n2 = len(df1), len(df2)

    df = pd.merge(df1, df2, on=["t_start", "t_end"], suffixes=("", "_r2"),
                  how="inner", validate="one_to_one")
    nm = len(df)
    print(f"[merge] {rater1_name}: {n1} rows  |  {rater2_name}: {n2} rows  |  "
          f"matched: {nm} rows")

    if nm == 0:
        print("[merge] Exact join failed (different clip grids). "
              "Resampling both CSVs to a 1-second grid …")
        df1r = resample_to_1s_grid(df1)
        df2r = resample_to_1s_grid(df2)
        df = pd.merge(df1r, df2r, on=["t_start", "t_end"], suffixes=("", "_r2"),
                      how="inner")
        nm = len(df)
        print(f"[merge] After resampling: {len(df1r)} rows  |  {len(df2r)} rows  |  "
              f"matched: {nm} rows")

    if nm == 0:
        raise ValueError("No overlapping t_start/t_end rows found. "
                         "Do both CSVs cover the same video segment?")
    return df


def compute_agreement(s1: pd.Series, s2: pd.Series, shared: set):
    """
    Given two canonicalized label series, compute:
      - percent agreement (strict: both labels known and in shared vocab)
      - Cohen's kappa (same subset)
      - n_included: rows used for strict metric
      - n_excluded: rows excluded (unknown or OOV)

    Returns (agreement, kappa, n_included, n_excluded, excluded_breakdown)
    where excluded_breakdown is a dict counting reasons for exclusion.
    """
    n_total = len(s1)
    mask_unknown = (s1 == "unknown") | (s2 == "unknown")
    mask_oov = ~s1.isin(shared) | ~s2.isin(shared)
    mask_exclude = mask_unknown | mask_oov

    excluded_breakdown = {
        "either_unknown": int(mask_unknown.sum()),
        "oov_rater1": int((~mask_unknown & ~s1.isin(shared)).sum()),
        "oov_rater2": int((~mask_unknown & s1.isin(shared) & ~s2.isin(shared)).sum()),
    }

    mask_include = ~mask_exclude
    n_included = int(mask_include.sum())
    n_excluded = n_total - n_included

    if n_included < 2:
        return np.nan, np.nan, n_included, n_excluded, excluded_breakdown

    y1 = s1[mask_include]
    y2 = s2[mask_include]

    agreement = float((y1 == y2).mean())
    try:
        kappa = cohen_kappa_score(y1, y2)
    except Exception:
        kappa = np.nan

    return agreement, kappa, n_included, n_excluded, excluded_breakdown


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_metrics_bar(metrics_rows: list, out_path: str):
    """Bar chart: percent agreement and Cohen's kappa per dimension."""
    dims   = [r["dimension"] for r in metrics_rows]
    agrees = [r["agreement"] for r in metrics_rows]
    kappas = [r["kappa"]     for r in metrics_rows]

    x = np.arange(len(dims))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars_a = ax.bar(x - width / 2, agrees, width, color=COLORS["agree"],  label="% Agreement")
    bars_k = ax.bar(x + width / 2, kappas, width, color=COLORS["kappa"],  label="Cohen's κ")

    # Annotate values
    for bar in list(bars_a) + list(bars_k):
        h = bar.get_height()
        if not np.isnan(h):
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels([d.replace("_", "\n") for d in dims], rotation=0)
    ax.set_ylim(0.0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Inter-Rater Reliability per Behavioral Dimension")
    ax.axhline(0.6, ls="--", lw=0.8, color="gray", label="κ = 0.6 threshold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_confusion(s1: pd.Series, s2: pd.Series, shared: set,
                   dim: str, rater1_name: str, rater2_name: str, out_path: str):
    """Confusion matrix (row = rater1, col = rater2) restricted to shared labels."""
    mask = s1.isin(shared) & s2.isin(shared)
    y1 = s1[mask]
    y2 = s2[mask]
    if len(y1) < 2:
        return

    labels = sorted(shared & set(y1) & set(y2))
    cm = confusion_matrix(y1, y2, labels=labels)

    fig, ax = plt.subplots(figsize=(max(6, len(labels)), max(5, len(labels))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, colorbar=False, xticks_rotation=45)
    ax.set_xlabel(rater2_name)
    ax.set_ylabel(rater1_name)
    ax.set_title(f"Confusion Matrix: {dim}\n(n={len(y1)} shared segments)")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def spans_from_series(t_starts, t_ends, labels):
    """Merge consecutive segments with the same label into contiguous spans."""
    spans = []
    t_starts = list(t_starts)
    t_ends   = list(t_ends)
    labels   = list(labels)
    if not labels:
        return spans

    cur_label = labels[0]
    cur_t0    = t_starts[0]
    cur_t1    = t_ends[0]

    for t0, t1, lbl in zip(t_starts[1:], t_ends[1:], labels[1:]):
        if lbl == cur_label:
            cur_t1 = t1
        else:
            spans.append((cur_t0, cur_t1, cur_label))
            cur_label, cur_t0, cur_t1 = lbl, t0, t1

    spans.append((cur_t0, cur_t1, cur_label))
    return spans


def plot_timecourse(df: pd.DataFrame, s1: pd.Series, s2: pd.Series,
                    dim: str, rater1_name: str, rater2_name: str, out_path: str):
    """Gantt-chart timecourse for one dimension, rater1 on top, rater2 on bottom."""
    t_start = df["t_start"]
    t_end   = df["t_end"]

    all_labels = sorted(set(s1) | set(s2))
    tab20 = plt.cm.tab20
    extra = {lbl: tab20(i % 20) for i, lbl in enumerate(
        lbl for lbl in all_labels if lbl not in LABEL_COLORS)}

    def get_color(lbl):
        return LABEL_COLORS.get(lbl, extra.get(lbl, "#999999"))

    fig, ax = plt.subplots(figsize=(14, 3))
    bar_h = 0.6
    y_pos = {rater1_name: 1.0, rater2_name: 0.0}

    for row_name, series in [(rater1_name, s1), (rater2_name, s2)]:
        y = y_pos[row_name]
        for (t0, t1, lbl) in spans_from_series(t_start, t_end, series):
            ax.broken_barh([(t0, t1 - t0)], (y - bar_h / 2, bar_h),
                           facecolors=get_color(lbl), edgecolor="none")

    ax.set_yticks([0.0, 1.0])
    ax.set_yticklabels([rater2_name, rater1_name])
    ax.set_xlabel("Time (s)")
    ax.set_title(f"{dim.replace('_', ' ').title()} – Rater Comparison")
    ax.set_xlim(t_start.min() - 0.5, t_end.max() + 0.5)
    ax.set_ylim(-0.6, 1.6)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=get_color(lbl),
                       edgecolor="black", linewidth=0.5, label=lbl)
        for lbl in all_labels
    ]
    ax.legend(handles=legend_handles, title="Label",
              loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Compute inter-rater reliability between two human annotation CSVs."
    )
    ap.add_argument("--csv_rater1", required=True,
                    help="CSV produced by eaf_to_csv_ICDL_miami.py or _ENS.py (rater 1).")
    ap.add_argument("--csv_rater2", required=True,
                    help="CSV for rater 2 (same video, different annotator/system).")
    ap.add_argument("--rater1_name", default="Rater1",
                    help="Display name for rater 1 (default: Rater1).")
    ap.add_argument("--rater2_name", default="Rater2",
                    help="Display name for rater 2 (default: Rater2).")
    ap.add_argument("--out_dir", default="../data/interrater",
                    help="Directory to save outputs (default: ../data/interrater).")
    ap.add_argument("--min_shared_n", type=int, default=5,
                    help="Skip a dimension if fewer than this many shared rows (default: 5).")
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    df = load_and_merge(args.csv_rater1, args.csv_rater2,
                        args.rater1_name, args.rater2_name)

    metrics_rows = []

    for dim in DIMENSIONS:
        col_r1 = dim
        col_r2 = dim + "_r2"

        if col_r1 not in df.columns:
            print(f"[skip] '{dim}': column not found in rater-1 CSV.")
            continue
        if col_r2 not in df.columns:
            print(f"[skip] '{dim}': column not found in rater-2 CSV.")
            continue

        s1 = canonicalize_series(df[col_r1], dim)
        s2 = canonicalize_series(df[col_r2], dim)

        shared = SHARED_LABELS[dim]

        agree, kappa, n_inc, n_exc, exc_bd = compute_agreement(s1, s2, shared)

        if n_inc < args.min_shared_n:
            print(f"[skip] '{dim}': only {n_inc} shared-vocabulary rows (< {args.min_shared_n}).")
            continue

        kappa_str = f"{kappa:.3f}" if not np.isnan(kappa) else "n/a"
        print(
            f"[{dim}]  agreement={agree:.3f}  kappa={kappa_str}  "
            f"n={n_inc}  excluded={n_exc} "
            f"(unknown={exc_bd['either_unknown']}, "
            f"oov_r1={exc_bd['oov_rater1']}, oov_r2={exc_bd['oov_rater2']})"
        )

        metrics_rows.append({
            "dimension":   dim,
            "agreement":   round(agree, 4),
            "kappa":       round(kappa, 4) if not np.isnan(kappa) else np.nan,
            "n_included":  n_inc,
            "n_excluded":  n_exc,
            "n_unknown":   exc_bd["either_unknown"],
            "n_oov_r1":    exc_bd["oov_rater1"],
            "n_oov_r2":    exc_bd["oov_rater2"],
        })

        # Confusion matrix
        plot_confusion(
            s1, s2, shared, dim,
            args.rater1_name, args.rater2_name,
            out_path=os.path.join(args.out_dir, f"interrater_confusion_{dim}.png"),
        )

        # Timecourse
        plot_timecourse(
            df, s1, s2, dim,
            args.rater1_name, args.rater2_name,
            out_path=os.path.join(args.out_dir, f"interrater_timecourse_{dim}.png"),
        )

    if not metrics_rows:
        print("[warn] No dimensions had sufficient shared annotations.")
        return

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_csv = os.path.join(args.out_dir, "interrater_metrics.csv")
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"\n[saved] metrics → {metrics_csv}")

    plot_metrics_bar(
        metrics_rows,
        out_path=os.path.join(args.out_dir, "interrater_accuracy.png"),
    )
    print(f"[saved] figures → {args.out_dir}/")


if __name__ == "__main__":
    main()
