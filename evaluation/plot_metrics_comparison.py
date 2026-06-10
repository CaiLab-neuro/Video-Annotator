"""
Plot model-vs-human, inter-rater, and inter-camera consistency metrics.

Subjects with model-vs-human data: 4, 6, 27, 28  (side + top cameras)
Subjects with inter-rater data:    6, 27, 28
Subjects with inter-camera data:   4, 6, 27, 28  (side vs top model predictions)

Output figures
--------------
combined_agreement.png  – model-vs-human (pooled) vs inter-rater vs inter-camera, agreement
combined_kappa.png      – same comparison for Cohen's κ
mvh_agreement.png       – model-vs-human side vs top camera, agreement + chance lines
mvh_kappa.png           – model-vs-human side vs top camera, kappa
"""

import os
import re
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)  # repo root (one level up from evaluation/)

# Raw model output CSVs for inter-camera consistency
RAW_DIR = "/data/Cai_gaze/Tsuji_lab_collaboration/results/video_annotation/ICDL_3s"


def _discover_mvh_files(camera):
    """
    Scan data/results_<N>/ for visualizations_<N>_<camera>_3s/model_vs_human_metrics.csv.
    Returns dict {subject_int: relative_path}.
    """
    result = {}
    data_dir = os.path.join(ROOT, "data")
    for name in sorted(os.listdir(data_dir)):
        m = re.fullmatch(r"results_(\d+)", name)
        if not m:
            continue
        subj = int(m.group(1))
        rel = f"data/{name}/visualizations_{subj}_{camera}_3s/model_vs_human_metrics.csv"
        if os.path.exists(os.path.join(ROOT, rel)):
            result[subj] = rel
    return result


def _discover_interrater_files():
    """
    Scan data/interrater/results_<N>/ for interrater_metrics.csv.
    Returns dict {subject_int: relative_path}.
    """
    result = {}
    itr_dir = os.path.join(ROOT, "data", "interrater")
    if not os.path.isdir(itr_dir):
        return result
    for name in sorted(os.listdir(itr_dir)):
        m = re.fullmatch(r"results_(\d+)", name)
        if not m:
            continue
        subj = int(m.group(1))
        rel = f"data/interrater/{name}/interrater_metrics.csv"
        if os.path.exists(os.path.join(ROOT, rel)):
            result[subj] = rel
    return result


def _discover_raw_subjects():
    """Return sorted list of subjects with both side and room CSVs in RAW_DIR."""
    if not os.path.isdir(RAW_DIR):
        return []
    subjects = set()
    for fname in os.listdir(RAW_DIR):
        m = re.fullmatch(r"(\d+)_(side|room)\.csv", fname)
        if m:
            subjects.add(int(m.group(1)))
    # Keep only subjects that have both cameras
    return sorted(s for s in subjects
                  if os.path.exists(os.path.join(RAW_DIR, f"{s}_side.csv"))
                  and os.path.exists(os.path.join(RAW_DIR, f"{s}_room.csv")))


MVH_SIDE_FILES = _discover_mvh_files("side")
MVH_ROOM_FILES = _discover_mvh_files("room")
INTERRATER_FILES = _discover_interrater_files()
RAW_SUBJECTS = _discover_raw_subjects()

print(f"Discovered MVH side subjects:   {sorted(MVH_SIDE_FILES)}")
print(f"Discovered MVH room subjects:   {sorted(MVH_ROOM_FILES)}")
print(f"Discovered inter-rater subjects:{sorted(INTERRATER_FILES)}")
print(f"Discovered inter-camera subjects:{RAW_SUBJECTS}")

# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------
TASKS = [
    "Child Hand Action",
    "Parent Hand Action",
    "Child Pose",
    "Parent Pose",
]

SHORT_TASKS = ["Child\nHand Action", "Parent\nHand Action", "Child\nPose", "Parent\nPose"]

INTERRATER_TASK_MAP = {
    "child_hand_action": "Child Hand Action",
    "adult_hand_action": "Parent Hand Action",
    "child_pose":        "Child Pose",
    "adult_pose":        "Parent Pose",
}

# Column names in the raw model CSV → canonical task name
RAW_COL_MAP = {
    "child_hand_action": "Child Hand Action",
    "adult_hand_action": "Parent Hand Action",
    "child_pose":        "Child Pose",
    "adult_pose":        "Parent Pose",
}

# Chance level: 1/k  (k = meaningful choices, excl. "none"/"not visible")
CHANCE = {
    "Child Hand Action":  1 / 10,
    "Parent Hand Action": 1 / 10,
    "Child Pose":         1 / 6,
    "Parent Pose":        1 / 6,
}

OUT_DIR = os.path.join(ROOT, "data", "comparison_plots")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_mvh(file_dict):
    """Load model-vs-human metrics CSVs → dict task -> {agreement, kappa} lists."""
    records = {t: {"agreement": [], "kappa": []} for t in TASKS}
    for subj, rel_path in file_dict.items():
        path = os.path.join(ROOT, rel_path)
        if not os.path.exists(path):
            print(f"  [warn] missing: {path}")
            continue
        df = pd.read_csv(path).dropna(subset=["behavior"])
        for _, row in df.iterrows():
            task = row["behavior"]
            if task not in records:
                continue
            records[task]["agreement"].append(float(row["agreement_all"]))
            if pd.notna(row["kappa"]):
                records[task]["kappa"].append(float(row["kappa"]))
    return records


def load_interrater():
    """Load inter-rater metrics CSVs → dict task -> {agreement, kappa} lists."""
    records = {t: {"agreement": [], "kappa": []} for t in TASKS}
    for subj, rel_path in INTERRATER_FILES.items():
        path = os.path.join(ROOT, rel_path)
        df = pd.read_csv(path).dropna(subset=["dimension"])
        for _, row in df.iterrows():
            task = INTERRATER_TASK_MAP.get(row["dimension"])
            if task is None:
                continue
            records[task]["agreement"].append(float(row["agreement"]))
            if pd.notna(row["kappa"]):
                records[task]["kappa"].append(float(row["kappa"]))
    return records


def load_intercamera():
    """
    Compute inter-camera consistency by comparing model predictions from side
    vs top (room) camera for the same timestamps (inner join on t_start).
    Returns dict task -> {agreement, kappa} lists (one entry per subject).
    """
    records = {t: {"agreement": [], "kappa": []} for t in TASKS}

    for subj in RAW_SUBJECTS:
        side_path = os.path.join(RAW_DIR, f"{subj}_side.csv")
        room_path = os.path.join(RAW_DIR, f"{subj}_room.csv")
        if not os.path.exists(side_path) or not os.path.exists(room_path):
            print(f"  [warn] missing raw CSVs for subject {subj}")
            continue

        side_df = pd.read_csv(side_path)
        room_df = pd.read_csv(room_path)

        # Align on t_start; round to avoid float precision mismatches
        side_df["t_start"] = side_df["t_start"].round(3)
        room_df["t_start"] = room_df["t_start"].round(3)

        merged = pd.merge(side_df, room_df, on="t_start", suffixes=("_side", "_room"))
        if merged.empty:
            print(f"  [warn] no overlapping timestamps for subject {subj}")
            continue

        for col, task in RAW_COL_MAP.items():
            col_side = f"{col}_side"
            col_room = f"{col}_room"
            if col_side not in merged or col_room not in merged:
                # Try columns without suffix if only one CSV had this column
                if col in side_df.columns and col in room_df.columns:
                    col_side, col_room = col, col  # shouldn't happen after merge
                else:
                    continue

            paired = merged[[col_side, col_room]].dropna()
            if paired.empty:
                continue

            y_side = paired[col_side].astype(str)
            y_room = paired[col_room].astype(str)

            agreement = (y_side == y_room).mean()
            records[task]["agreement"].append(float(agreement))

            # Kappa requires at least 2 unique labels
            if y_side.nunique() > 1 or y_room.nunique() > 1:
                try:
                    kappa = cohen_kappa_score(y_side, y_room)
                    records[task]["kappa"].append(float(kappa))
                except Exception:
                    pass

    return records


def pool_mvh(side, room):
    """Pool side+room model-vs-human values per task."""
    pooled = {t: {"agreement": [], "kappa": []} for t in TASKS}
    for t in TASKS:
        pooled[t]["agreement"] = side[t]["agreement"] + room[t]["agreement"]
        pooled[t]["kappa"]     = side[t]["kappa"]     + room[t]["kappa"]
    return pooled


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def summary(values):
    """Return (mean, SE)."""
    arr = np.array([v for v in values if not np.isnan(v)], dtype=float)
    if len(arr) == 0:
        return np.nan, 0.0
    mean = arr.mean()
    se   = arr.std(ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else 0.0
    return mean, se


def build_arrays(records, metric):
    means, ses = [], []
    for t in TASKS:
        m, s = summary(records[t][metric])
        means.append(m)
        ses.append(s)
    return np.array(means), np.array(ses)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

FS_TITLE  = 18
FS_LABEL  = 16
FS_TICK   = 15
FS_ANNOT  = 13
FS_LEGEND = 14


def make_fig(title, fs_title=FS_TITLE):
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(title, fontsize=fs_title) # , fontweight="bold"
    return fig, ax


def style_ax(ax, ylabel, ylim=(0.0, 1.05), fs_label=FS_LABEL, fs_tick=FS_TICK):
    ax.set_ylabel(ylabel, fontsize=fs_label)
    ax.set_ylim(*ylim)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticks(np.arange(len(TASKS)))
    ax.set_xticklabels(SHORT_TASKS, fontsize=fs_tick)
    ax.tick_params(axis="y", labelsize=fs_tick)


def add_bar(ax, x_offset, width, means, ses, color, label, fs_annot=FS_ANNOT):
    bars = ax.bar(np.arange(len(TASKS)) + x_offset, means, width,
                  color=color, label=label,
                  yerr=ses, capsize=4, error_kw={"elinewidth": 1.3},
                  zorder=3)
    for bar, m, s in zip(bars, means, ses):
        if not np.isnan(m):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    m + s + 0.02, f"{m:.2f}",
                    ha="center", va="bottom", fontsize=fs_annot, zorder=4)
    return bars


def add_chance_lines(ax, n_groups, width_per_bar):
    """Dashed chance lines spanning all bars in each task group; single legend entry."""
    half = (n_groups * width_per_bar) / 2 + 0.02
    for xi, task in enumerate(TASKS):
        ax.hlines(CHANCE[task], xi - half, xi + half,
                  colors="crimson", linestyles="dashed", linewidth=1.5, zorder=5)
    ax.plot([], [], color="crimson", linestyle="dashed", linewidth=1.5,
            label="chance level (1/k)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    mvh_side = load_mvh(MVH_SIDE_FILES)
    mvh_room = load_mvh(MVH_ROOM_FILES)
    mvh_pool = pool_mvh(mvh_side, mvh_room)
    itr      = load_interrater()
    icc      = load_intercamera()

    n_side = len(MVH_SIDE_FILES)    # 4
    n_room = len(MVH_ROOM_FILES)    # 4
    n_itr  = len(INTERRATER_FILES)  # 3
    n_icc  = len(RAW_SUBJECTS)      # 4

    pool_agr_m, pool_agr_s = build_arrays(mvh_pool, "agreement")
    pool_kap_m, pool_kap_s = build_arrays(mvh_pool, "kappa")
    itr_agr_m,  itr_agr_s  = build_arrays(itr,      "agreement")
    itr_kap_m,  itr_kap_s  = build_arrays(itr,      "kappa")
    icc_agr_m,  icc_agr_s  = build_arrays(icc,      "agreement")
    icc_kap_m,  icc_kap_s  = build_arrays(icc,      "kappa")

    side_agr_m, side_agr_s = build_arrays(mvh_side, "agreement")
    side_kap_m, side_kap_s = build_arrays(mvh_side, "kappa")
    room_agr_m, room_agr_s = build_arrays(mvh_room, "agreement")
    room_kap_m, room_kap_s = build_arrays(mvh_room, "kappa")

    C_POOL = "#4C72B0"   # blue  – model-vs-human pooled
    C_SIDE = "#4C72B0"   # blue  – side camera
    C_ROOM = "#64B5CD"   # light blue – top camera
    C_ITR  = "#DD8452"   # orange – inter-rater
    C_ICC  = "#55A868"   # green  – inter-camera

    W3 = 0.25   # bar width for 3-group plots
    W2 = 0.35   # bar width for 2-group plots

    # -----------------------------------------------------------------------
    # Fig 1: combined_agreement  (3 groups)
    # -----------------------------------------------------------------------
    fig, ax = make_fig(
        f"Agreement: Model vs Human  |  Inter-Rater  |  Inter-Camera\n"
        f"(mean ± SE;  MVH n={n_side+n_room} recordings,  "
        f"Inter-rater n={n_itr},  Inter-camera n={n_icc} subjects)"
    )
    add_bar(ax, -W3, W3, pool_agr_m, pool_agr_s, C_POOL,
            f"Model vs Human (n={n_side+n_room})")
    add_bar(ax,   0, W3, itr_agr_m,  itr_agr_s,  C_ITR,
            f"Inter-Rater (n={n_itr})")
    add_bar(ax,  W3, W3, icc_agr_m,  icc_agr_s,  C_ICC,
            f"Inter-Camera (n={n_icc})")
    style_ax(ax, "Accuracy")
    ax.legend(fontsize=FS_LEGEND, loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "combined_agreement.png"), dpi=150, bbox_inches="tight")
    print("Saved: combined_agreement.png")
    plt.close()

    # -----------------------------------------------------------------------
    # Fig 2: combined_kappa  (3 groups)
    # -----------------------------------------------------------------------
    fig, ax = make_fig(
        f"Cohen's κ: Model vs Human  |  Inter-Rater  |  Inter-Camera\n"
        f"(mean ± SE;  MVH n={n_side+n_room} recordings,  "
        f"Inter-rater n={n_itr},  Inter-camera n={n_icc} subjects)"
    )
    add_bar(ax, -W3, W3, pool_kap_m, pool_kap_s, C_POOL,
            f"Model vs Human (n={n_side+n_room})")
    add_bar(ax,   0, W3, itr_kap_m,  itr_kap_s,  C_ITR,
            f"Inter-Rater (n={n_itr})")
    add_bar(ax,  W3, W3, icc_kap_m,  icc_kap_s,  C_ICC,
            f"Inter-Camera (n={n_icc})")
    style_ax(ax, "Cohen's κ", ylim=(-0.35, 1.0))
    ax.legend(fontsize=FS_LEGEND, loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "combined_kappa.png"), dpi=150, bbox_inches="tight")
    print("Saved: combined_kappa.png")
    plt.close()

    # -----------------------------------------------------------------------
    # Fig 3: mvh_agreement — side vs top, with chance lines
    # -----------------------------------------------------------------------
    MVH_FS_TITLE  = 24
    MVH_FS_LABEL  = 22
    MVH_FS_TICK   = 20
    MVH_FS_ANNOT  = 22
    MVH_FS_LEGEND = 20

    fig, ax = make_fig(
        f"Model-human consistency (mean ± SE)\n",
        fs_title=MVH_FS_TITLE,
    )
    add_bar(ax, -W2/2, W2, side_agr_m, side_agr_s, C_SIDE,
            f"Side camera (n={n_side})", fs_annot=MVH_FS_ANNOT)
    add_bar(ax,  W2/2, W2, room_agr_m, room_agr_s, C_ROOM,
            f"Top camera (n={n_room})", fs_annot=MVH_FS_ANNOT)
    add_chance_lines(ax, n_groups=2, width_per_bar=W2)
    style_ax(ax, "Accuracy", fs_label=MVH_FS_LABEL, fs_tick=MVH_FS_TICK)
    ax.legend(fontsize=MVH_FS_LEGEND, loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "mvh_agreement.png"), dpi=150, bbox_inches="tight")
    print("Saved: mvh_agreement.png")
    plt.close()

    # -----------------------------------------------------------------------
    # Fig 4: mvh_kappa — side vs top
    # -----------------------------------------------------------------------
    fig, ax = make_fig(
        f"Model vs Human Cohen's κ  (Side vs Top camera)\n"
        f"(mean ± SE across {n_side} subjects)"
    )
    add_bar(ax, -W2/2, W2, side_kap_m, side_kap_s, C_SIDE,
            f"Side camera (n={n_side})")
    add_bar(ax,  W2/2, W2, room_kap_m, room_kap_s, C_ROOM,
            f"Top camera (n={n_room})")
    style_ax(ax, "Cohen's κ", ylim=(-0.35, 1.0))
    ax.legend(fontsize=FS_LEGEND, loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "mvh_kappa.png"), dpi=150, bbox_inches="tight")
    print("Saved: mvh_kappa.png")
    plt.close()


if __name__ == "__main__":
    main()
