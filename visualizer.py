#!/usr/bin/env python3
"""
Unified visualizer for parent–child interaction:

Inputs:
    --csv_model: CSV with model predictions
    --csv_human: CSV with human annotations
    Both must share t_start and t_end to be merged.

Outputs (all saved individually in out_dir):
    - model_vs_human_metrics.csv  (per-behavior accuracy, agreement, kappa)
    - accuracy_vs_agreement.png   (strict accuracy vs agreement including unknowns)
    - toy_possession_breakdown_model.png
    - hand_action_mix_child_parent.png
    - toy_holder_over_time.png
    - toy_contact_vs_child_activity.png
    - transitions_child_hand_action_bar.png
    - transitions_parent_hand_action_bar.png
    - transitions_child_pose_bar.png
    - transitions_parent_pose_bar.png
"""

import argparse
import os
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score


def pairwise(iterable):
    """Return successive pairs (s0,s1), (s1,s2), ..."""
    it = iter(iterable)
    prev = next(it, None)
    for item in it:
        yield prev, item
        prev = item


# ----------------------
# Global styling / colors
# ----------------------

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "text.color": "black",
    "savefig.facecolor": "white",
    "savefig.edgecolor": "white",
})

# Palette
COLORS = {
    "teal": "#85d2d0",
    "purple": "#887bb0",
    "orange": "#ffbd59",
    "pale_yellow": "#fff4bd",
}

# ----------------------
# Column mapping (model vs human)
# After merging, human columns are suffixed with `_human`.
# ----------------------

COLUMN_MAP = {
    "time_start": "t_start",
    "time_end": "t_end",

    # model columns
    "toy_in_environment_model": "toy_in_environment",
    "parent_holding_toy_model": "parent_holding_toy",
    "child_holding_toy_model": "child_holding_toy",
    "child_hand_action_model": "child_hand_action",
    "child_proximity_behavior_model": "child_proximity_behavior",
    "current_toy_model": "current_toy",
    "adult_hand_action_model": "adult_hand_action",
    "child_pose_model": "child_pose",
    "adult_pose_model": "adult_pose",

    # human columns (post-merge, suffixed with _human)
    "toy_in_environment_human": "toy_in_environment_human",
    "parent_holding_toy_human": "parent_holding_toy_human",
    "child_holding_toy_human": "child_holding_toy_human",
    "child_hand_action_human": "child_hand_action_human",
    "child_proximity_behavior_human": "child_proximity_behavior_human",
    "current_toy_human": "current_toy_human",
    "adult_hand_action_human": "adult_hand_action_human",
    "child_pose_human": "child_pose_human",
    "adult_pose_human": "adult_pose_human",
}

# ----------------------
# Helpers
# ----------------------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_and_merge(model_csv: str, human_csv: str) -> pd.DataFrame:
    """
    Load model + human CSVs and merge on t_start, t_end.
    Human columns get `_human` suffix on overlap.
    """
    df_model = pd.read_csv(model_csv)
    df_human = pd.read_csv(human_csv)

    required_keys = [COLUMN_MAP["time_start"], COLUMN_MAP["time_end"]]
    for k in required_keys:
        if k not in df_model.columns or k not in df_human.columns:
            raise ValueError(f"Both CSVs must contain '{k}' for merging.")

    df = pd.merge(
        df_model,
        df_human,
        on=required_keys,
        suffixes=("", "_human"),
        validate="one_to_one",
    )

    return df


def compute_accuracy_and_agreement(df: pd.DataFrame, model_col: str, human_col: str):
    """
    Compute:
      - strict accuracy: agreement excluding unknowns / NaNs
      - agreement_all: raw agreement including unknowns
      - kappa: Cohen's kappa on strict subset
    """
    s_model = df[model_col].astype(str)
    s_human = df[human_col].astype(str)

    # Strict subset: exclude NaNs and 'unknown'
    mask_strict = (
        (s_model != "nan") & (s_human != "nan") &
        (s_model != "unknown") & (s_human != "unknown")
    )
    if mask_strict.sum() > 0:
        y_model = s_model[mask_strict]
        y_human = s_human[mask_strict]
        acc_strict = (y_model == y_human).mean()
        try:
            kappa = cohen_kappa_score(y_human, y_model)
        except Exception:
            kappa = np.nan
    else:
        acc_strict = np.nan
        kappa = np.nan

    # Agreement including unknowns (raw proportion of equal labels)
    mask_all = (s_model != "nan") & (s_human != "nan")
    if mask_all.sum() > 0:
        y_model_all = s_model[mask_all]
        y_human_all = s_human[mask_all]
        agreement_all = (y_model_all == y_human_all).mean()
    else:
        agreement_all = np.nan

    return acc_strict, agreement_all, kappa


def derive_toy_holder_category(parent_hold: pd.Series, child_hold: pd.Series) -> pd.Series:
    """
    Derive 'who is holding a toy' category from parent/child yes/no/unknown.
    Categories: 'none', 'child only', 'parent only', 'both', 'unknown'.
    """
    p = parent_hold.astype(str).fillna("unknown")
    c = child_hold.astype(str).fillna("unknown")

    categories = []
    for ph, ch in zip(p, c):
        ph_yes = (ph.lower() == "yes")
        ch_yes = (ch.lower() == "yes")

        if ph_yes and ch_yes:
            categories.append("both")
        elif ph_yes and not ch_yes:
            categories.append("parent only")
        elif ch_yes and not ph_yes:
            categories.append("child only")
        elif (ph.lower() in ["no", "none"]) and (ch.lower() in ["no", "none"]):
            categories.append("none")
        else:
            categories.append("unknown")
    return pd.Series(categories, index=p.index)


def compute_transition_counts(series: pd.Series) -> Counter:
    """
    Compute frequency of transitions in a label series.
    Returns Counter with keys like "A → B".
    """
    s = series.astype(str).fillna("unknown")
    s = s.replace("nan", "unknown")

    counts = Counter()
    for a, b in pairwise(s):
        if a is None or b is None:
            continue
        key = f"{a} → {b}"
        counts[key] += 1
    return counts


# ----------------------
# Plotting
# ----------------------

def plot_toy_possession_pie_model(model_cat: pd.Series, out_path: str):
    """
    Pie chart of toy possession breakdown for the model only.
    """
    counts = Counter(model_cat)
    order = ["none", "child only", "parent only", "both", "unknown"]
    labels = [l for l in order if l in counts]

    values = [counts.get(l, 0) for l in labels]
    colors = [
        COLORS["pale_yellow"],
        COLORS["teal"],
        COLORS["purple"],
        COLORS["orange"],
        "#cccccc",
    ][:len(labels)]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(values, labels=labels, colors=colors, autopct="%1.1f%%")
    ax.set_title("Toy Possession Breakdown (Model)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_child_vs_parent_hand_mix(df: pd.DataFrame, out_path: str):
    """
    Two side-by-side plots:

      Left: Child hand actions (Model vs Human)
      Right: Parent hand actions (Model vs Human)

    Each plot has x-axis = action categories, with two bars per category (Model, Human)
    and its own legend.
    """
    ch_model = df[COLUMN_MAP["child_hand_action_model"]].astype(str)
    ch_human = df[COLUMN_MAP["child_hand_action_human"]].astype(str)
    ad_model = df[COLUMN_MAP["adult_hand_action_model"]].astype(str)
    ad_human = df[COLUMN_MAP["adult_hand_action_human"]].astype(str)

    child_actions = sorted(set(ch_model) | set(ch_human))
    parent_actions = sorted(set(ad_model) | set(ad_human))

    ch_model_counts = Counter(ch_model)
    ch_human_counts = Counter(ch_human)
    ad_model_counts = Counter(ad_model)
    ad_human_counts = Counter(ad_human)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    # --- Child subplot ---
    x_child = np.arange(len(child_actions))
    width = 0.35

    ax_child = axes[0]
    child_model_vals = [ch_model_counts.get(a, 0) for a in child_actions]
    child_human_vals = [ch_human_counts.get(a, 0) for a in child_actions]

    ax_child.bar(x_child - width / 2, child_model_vals, width, color=COLORS["teal"], label="Model")
    ax_child.bar(x_child + width / 2, child_human_vals, width, color=COLORS["purple"], label="Human")
    ax_child.set_xticks(x_child)
    ax_child.set_xticklabels(child_actions, rotation=45, ha="right")
    ax_child.set_ylabel("Count")
    ax_child.set_xlabel("Child Hand Action")
    ax_child.set_title("Child Hand Action Mix (Model vs Human)")
    ax_child.legend()

    # --- Parent subplot ---
    x_parent = np.arange(len(parent_actions))
    parent_model_vals = [ad_model_counts.get(a, 0) for a in parent_actions]
    parent_human_vals = [ad_human_counts.get(a, 0) for a in parent_actions]

    ax_parent = axes[1]
    ax_parent.bar(x_parent - width / 2, parent_model_vals, width, color=COLORS["teal"], label="Model")
    ax_parent.bar(x_parent + width / 2, parent_human_vals, width, color=COLORS["purple"], label="Human")
    ax_parent.set_xticks(x_parent)
    ax_parent.set_xticklabels(parent_actions, rotation=45, ha="right")
    ax_parent.set_ylabel("Count")
    ax_parent.set_xlabel("Parent Hand Action")
    ax_parent.set_title("Parent Hand Action Mix (Model vs Human)")
    ax_parent.legend()

    fig.suptitle("Child and Parent Hand Action Mix (Model vs Human)")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path)
    plt.close(fig)


def plot_toy_holder_timeline(df: pd.DataFrame, model_cat: pd.Series,
                             human_cat: pd.Series, out_path: str):
    """
    Combined model + human toy holder timeline on ONE compact vertical axis:
        y=1 -> Model
        y=0 -> Human
    Legend placed outside.
    """
    t_start = df[COLUMN_MAP["time_start"]]
    t_end = df[COLUMN_MAP["time_end"]]
    t_mid = (t_start + t_end) / 2.0

    categories = ["none", "child only", "parent only", "both", "unknown"]
    color_map = {
        "none": COLORS["pale_yellow"],
        "child only": COLORS["teal"],
        "parent only": COLORS["purple"],
        "both": COLORS["orange"],
        "unknown": "#cccccc",
    }

    fig, ax = plt.subplots(figsize=(12, 3.5))

    # y positions: 1 = Model, 0 = Human
    y_model = 1.0
    y_human = 0.0

    # --- Model points at y = 1 ---
    for cat in categories:
        mask = (model_cat == cat)
        ax.scatter(
            t_mid[mask],
            np.ones(mask.sum()) * y_model,
            s=16,
            color=color_map[cat],
            edgecolor="black",
            linewidth=0.3,
        )

    # --- Human points at y = 0 ---
    for cat in categories:
        mask = (human_cat == cat)
        ax.scatter(
            t_mid[mask],
            np.ones(mask.sum()) * y_human,
            s=16,
            color=color_map[cat],
            edgecolor="black",
            linewidth=0.3,
            marker="s",
        )

    ax.set_yticks([y_human, y_model])
    ax.set_yticklabels(["Human", "Model"])
    ax.set_xlabel("Time (s)")
    ax.set_title("Who Is Holding a Toy Over Time (Model vs Human)")
    ax.set_xlim(t_mid.min() - 1, t_mid.max() + 1)

    # Legend outside
    legend_handles = [
        plt.Line2D(
            [0], [0],
            marker='o',
            linestyle='',
            color='w',
            markerfacecolor=color_map[c],
            markeredgecolor='black',
            markersize=8,
            label=c,
        )
        for c in categories
    ]

    ax.legend(
        handles=legend_handles,
        title="Toy Holder",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
    )

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_toy_contact_vs_child_activity(df: pd.DataFrame, out_path: str):
    """
    Bar chart: toy contact (child_holding_toy yes/no) vs child activity (pose),
    model vs human side-by-side per pose.
    """
    ch_pose_m = df[COLUMN_MAP["child_pose_model"]].astype(str)
    ch_pose_h = df[COLUMN_MAP["child_pose_human"]].astype(str)
    ch_hold_m = df[COLUMN_MAP["child_holding_toy_model"]].astype(str)
    ch_hold_h = df[COLUMN_MAP["child_holding_toy_human"]].astype(str)

    def bucket_toy(x: str) -> str:
        return "holding toy" if x.lower() == "yes" else "not holding toy"

    toy_m = ch_hold_m.map(bucket_toy)
    toy_h = ch_hold_h.map(bucket_toy)

    combo_m = Counter(zip(toy_m, ch_pose_m))
    combo_h = Counter(zip(toy_h, ch_pose_h))

    toy_states = sorted(set(toy_m) | set(toy_h))
    poses = sorted(set(ch_pose_m) | set(ch_pose_h))

    x = np.arange(len(poses))
    width = 0.35

    fig, axes = plt.subplots(1, len(toy_states), figsize=(max(8, 4 * len(toy_states)), 4), sharey=True)

    if len(toy_states) == 1:
        axes = [axes]

    for ax, toy_state in zip(axes, toy_states):
        model_vals = [combo_m.get((toy_state, p), 0) for p in poses]
        human_vals = [combo_h.get((toy_state, p), 0) for p in poses]

        ax.bar(x - width/2, model_vals, width, color=COLORS["teal"], label="Model")
        ax.bar(x + width/2, human_vals, width, color=COLORS["purple"], label="Human")

        ax.set_xticks(x)
        ax.set_xticklabels(poses, rotation=45, ha="right")
        ax.set_ylabel("Count")
        ax.set_xlabel("Child Pose")
        ax.set_title(f"Toy Contact & Child Activity ({toy_state})")
        ax.legend()

    fig.suptitle("Toy Contact & Child Activity (Model vs Human)")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path)
    plt.close(fig)


def plot_accuracy_vs_agreement(metrics_df: pd.DataFrame, out_path: str):
    """
    Per-behavior bar chart:
      - x-axis: behavior names
      - y-axis: proportion
      - two bars per behavior:
          * strict accuracy (excluding unknowns)
          * agreement including unknowns
    """
    behaviors = metrics_df["behavior"].tolist()
    x = np.arange(len(behaviors))
    width = 0.35

    acc_strict = metrics_df["accuracy_strict"].values
    agree_all = metrics_df["agreement_all"].values

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, acc_strict, width, color=COLORS["teal"], label="Strict accuracy (no unknowns)")
    ax.bar(x + width/2, agree_all, width, color=COLORS["purple"], label="Agreement (including unknowns)")

    ax.set_xticks(x)
    ax.set_xticklabels(behaviors, rotation=45, ha="right")
    ax.set_ylabel("Proportion")
    ax.set_xlabel("Behavior")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Per-Behavior Accuracy vs Agreement (Model vs Human)")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_transition_bars(counts_model: Counter, counts_human: Counter,
                         title: str, out_path: str, top_k: int = 5):
    """
    Bar graph of transition frequencies, ordered from highest to lowest
    by total (model + human) count. Model and Human are two bars per transition.
    Only the top_k most frequent transitions are shown (default: 5).
    """
    all_keys = set(counts_model.keys()) | set(counts_human.keys())
    # sort by total count desc
    sorted_keys = sorted(
        all_keys,
        key=lambda k: counts_model.get(k, 0) + counts_human.get(k, 0),
        reverse=True,
    )
    if top_k is not None:
        sorted_keys = sorted_keys[:top_k]

    x = np.arange(len(sorted_keys))
    model_vals = [counts_model.get(k, 0) for k in sorted_keys]
    human_vals = [counts_human.get(k, 0) for k in sorted_keys]

    width = 0.4
    fig, ax = plt.subplots(figsize=(max(8, 1.6 * len(sorted_keys)), 4.5))
    ax.bar(x - width/2, model_vals, width, color=COLORS["teal"], label="Model")
    ax.bar(x + width/2, human_vals, width, color=COLORS["purple"], label="Human")

    ax.set_xticks(x)
    ax.set_xticklabels(sorted_keys, rotation=45, ha="right")
    ax.set_ylabel("Transition Count")
    ax.set_xlabel("Transition (prev → next)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ----------------------
# Main
# ----------------------

def main():
    ap = argparse.ArgumentParser(description="Unified visualizer for model vs human annotations.")
    ap.add_argument("--csv_model", required=True, help="CSV with model predictions.")
    ap.add_argument("--csv_human", required=True, help="CSV with human annotations.")
    ap.add_argument("--out_dir", default="figures", help="Directory to save figures and metrics.")
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    df = load_and_merge(args.csv_model, args.csv_human)

    # --- Per-behavior accuracy & agreement (for all preset behaviors) ---
    behaviors = [
        ("Toy In Environment",
         COLUMN_MAP["toy_in_environment_model"],
         COLUMN_MAP["toy_in_environment_human"]),
        ("Parent Holding Toy",
         COLUMN_MAP["parent_holding_toy_model"],
         COLUMN_MAP["parent_holding_toy_human"]),
        ("Child Holding Toy",
         COLUMN_MAP["child_holding_toy_model"],
         COLUMN_MAP["child_holding_toy_human"]),
        ("Child Hand Action",
         COLUMN_MAP["child_hand_action_model"],
         COLUMN_MAP["child_hand_action_human"]),
        ("Child Proximity Behavior",
         COLUMN_MAP["child_proximity_behavior_model"],
         COLUMN_MAP["child_proximity_behavior_human"]),
        ("Current Toy",
         COLUMN_MAP["current_toy_model"],
         COLUMN_MAP["current_toy_human"]),
        ("Parent Hand Action",
         COLUMN_MAP["adult_hand_action_model"],
         COLUMN_MAP["adult_hand_action_human"]),
        ("Child Pose",
         COLUMN_MAP["child_pose_model"],
         COLUMN_MAP["child_pose_human"]),
        ("Parent Pose",
         COLUMN_MAP["adult_pose_model"],
         COLUMN_MAP["adult_pose_human"]),
    ]

    metrics_rows = []
    for name, m_col, h_col in behaviors:
        acc_strict, agree_all, kappa = compute_accuracy_and_agreement(df, m_col, h_col)
        metrics_rows.append({
            "behavior": name,
            "accuracy_strict": acc_strict,
            "agreement_all": agree_all,
            "kappa": kappa,
        })
        print(f"[metrics] {name}: "
              f"strict_acc={acc_strict:.3f}  "
              f"agree_all={agree_all:.3f}  "
              f"kappa={kappa:.3f}")

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(os.path.join(args.out_dir, "model_vs_human_metrics.csv"), index=False)

    # Plot per-behavior accuracy vs agreement
    plot_accuracy_vs_agreement(
        metrics_df,
        out_path=os.path.join(args.out_dir, "accuracy_vs_agreement.png"),
    )

    # --- Toy possession breakdown (model only) ---
    toy_cat_model = derive_toy_holder_category(
        df[COLUMN_MAP["parent_holding_toy_model"]],
        df[COLUMN_MAP["child_holding_toy_model"]],
    )
    toy_cat_human = derive_toy_holder_category(  # still used for toy_holder_over_time
        df[COLUMN_MAP["parent_holding_toy_human"]],
        df[COLUMN_MAP["child_holding_toy_human"]],
    )
    plot_toy_possession_pie_model(
        toy_cat_model,
        out_path=os.path.join(args.out_dir, "toy_possession_breakdown_model.png"),
    )

    # --- Child vs Parent hand action mix (side-by-side plots) ---
    plot_child_vs_parent_hand_mix(
        df,
        out_path=os.path.join(args.out_dir, "hand_action_mix_child_parent.png"),
    )

    # --- Who is holding a toy over time (combined model + human) ---
    plot_toy_holder_timeline(
        df,
        toy_cat_model,
        toy_cat_human,
        out_path=os.path.join(args.out_dir, "toy_holder_over_time.png"),
    )

    # --- Toy contact & child activity ---
    plot_toy_contact_vs_child_activity(
        df,
        out_path=os.path.join(args.out_dir, "toy_contact_vs_child_activity.png"),
    )

    # --- Transition frequency bars (child/parent hand actions & poses) ---
    # Child hand actions
    ch_counts_m = compute_transition_counts(df[COLUMN_MAP["child_hand_action_model"]])
    ch_counts_h = compute_transition_counts(df[COLUMN_MAP["child_hand_action_human"]])
    plot_transition_bars(
        ch_counts_m,
        ch_counts_h,
        title="Child Hand Action Transitions (Model vs Human)",
        out_path=os.path.join(args.out_dir, "transitions_child_hand_action_bar.png"),
    )

    # Parent hand actions
    ad_counts_m = compute_transition_counts(df[COLUMN_MAP["adult_hand_action_model"]])
    ad_counts_h = compute_transition_counts(df[COLUMN_MAP["adult_hand_action_human"]])
    plot_transition_bars(
        ad_counts_m,
        ad_counts_h,
        title="Parent Hand Action Transitions (Model vs Human)",
        out_path=os.path.join(args.out_dir, "transitions_parent_hand_action_bar.png"),
    )

    # Child pose transitions
    cp_counts_m = compute_transition_counts(df[COLUMN_MAP["child_pose_model"]])
    cp_counts_h = compute_transition_counts(df[COLUMN_MAP["child_pose_human"]])
    plot_transition_bars(
        cp_counts_m,
        cp_counts_h,
        title="Child Pose Transitions (Model vs Human)",
        out_path=os.path.join(args.out_dir, "transitions_child_pose_bar.png"),
    )

    # Parent pose transitions
    ap_counts_m = compute_transition_counts(df[COLUMN_MAP["adult_pose_model"]])
    ap_counts_h = compute_transition_counts(df[COLUMN_MAP["adult_pose_human"]])
    plot_transition_bars(
        ap_counts_m,
        ap_counts_h,
        title="Parent Pose Transitions (Model vs Human)",
        out_path=os.path.join(args.out_dir, "transitions_parent_pose_bar.png"),
    )

    print(f"[info] Saved figures and metrics to {args.out_dir}")


if __name__ == "__main__":
    main()
