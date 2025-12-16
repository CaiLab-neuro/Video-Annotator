#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gesture-only visual behavior analysis (no audio, no eye-gaze inference).

Designed for the SHORT preset CSV schema produced by quick_clip_csv.py:

Columns expected:
- video_path
- t_start, t_end
- toy_in_environment
- parent_holding_toy
- child_holding_toy
- child_hand_action
- child_proximity_behavior
- current_toy
- adult_hand_action
- child_pose
- adult_pose

Optional:
- compare two camera views (side vs room) using --side_csv and --room_csv
- compare model vs human labels using --human_csv (with --input as model CSV).
"""

import argparse
import os
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# ---------------------- Helpers ---------------------- #

def safe_unique(series):
    if series is None:
        return []
    vals = series.dropna().unique().tolist()
    return [v for v in vals if str(v).lower() != "unknown"]


def map_categories(series):
    """Return a mapping {label: int} excluding 'unknown' and NaNs, sorted by label."""
    cats = sorted([str(x) for x in safe_unique(series)])
    return {c: i for i, c in enumerate(cats)}


def scatter_categorical(ax, df, time_col, label_col, ymap,
                        color, size=60, alpha=0.75, y_offset=0.0):
    if label_col not in df.columns or not ymap:
        return
    for _, row in df.iterrows():
        label = str(row.get(label_col, "unknown"))
        if label and label.lower() != "unknown" and label in ymap:
            ax.scatter(
                row[time_col],
                ymap[label] + y_offset,
                s=size,
                alpha=alpha,
                color=color
            )


def count_per_minute(df, time_col, label_col, positive_set):
    """
    Simple per-minute count of frames where label_col is in positive_set.
    Useful for comparing activity over time.
    """
    if label_col not in df.columns or df.empty:
        return pd.DataFrame(columns=["minute", "count"])

    tmp = df.copy()
    tmp["minute"] = np.floor(tmp[time_col]).astype(int)
    tmp["flag"] = tmp[label_col].isin(positive_set).astype(int)
    out = tmp.groupby("minute")["flag"].sum().reset_index(name="count")
    return out


def ensure_cols(df, cols):
    for c in cols:
        if c not in df.columns:
            df[c] = "unknown"
    return df


# ---------------------- Timeline Plots ---------------------- #

def create_timeline_plots(df, output_prefix):
    """
    Timeline visualizations with strictly visual/gestural cues
    for the SHORT-preset schema.

    Kept timelines:
    - Current toy in child's hands over time (with legend)
    - Who is holding a toy over time (lines connecting points)
    - Child pose over time (categorical y, time in minutes on x)
    """
    print("Creating timeline plots...")
    df = df.copy()
    df["t_min"] = df["t_start"] / 60.0

    expected = [
        "toy_in_environment",
        "parent_holding_toy",
        "child_holding_toy",
        "child_hand_action",
        "child_proximity_behavior",
        "current_toy",
        "adult_hand_action",
        "child_pose",
        "adult_pose",
    ]
    df = ensure_cols(df, expected)

    fig, axes = plt.subplots(3, 1, figsize=(18, 14), sharex=True)
    fig.suptitle(
        "Visual Behavior Analysis (Short Clips) — Toy, Pose, and Possession Timelines",
        fontsize=18,
        fontweight="bold",
        y=0.97
    )

    toy_colors = {
        "giraffe": "#FF6B6B",
        "elephant": "#4ECDC4",
        "yellow toy": "#FFE66D",
        "green toy": "#6A0572",
        "none": "#999999",
    }

    # -------- (1) Current toy identity over time -------- #
    ax = axes[0]
    seen_toys = set()
    for _, row in df.iterrows():
        toy = str(row["current_toy"])
        if toy.lower() != "unknown":
            seen_toys.add(toy)
            # y = 0.5 when some toy, 0.2 when none
            y = 0.5 if toy != "none" else 0.2
            ax.scatter(
                row["t_min"],
                y,
                color=toy_colors.get(toy, "#CCCCCC"),
                alpha=0.9,
                s=110,
                marker="s"
            )

    # Legend for toy identities
    handles = []
    labels = []
    for toy in sorted(seen_toys):
        handles.append(
            plt.Line2D(
                [0], [0],
                marker="s",
                linestyle="",
                color=toy_colors.get(toy, "#CCCCCC"),
                label=toy,
                markersize=10
            )
        )
        labels.append(toy)
    if handles:
        ax.legend(handles=handles, loc="upper right", fontsize=9, title="Current Toy")

    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.5])
    ax.set_yticklabels(["No Toy", "Toy in Child's Hands"])
    ax.set_ylabel("Toy Identity", fontweight="bold")
    ax.set_title("Current Toy in Child's Hands Over Time", fontweight="bold")
    ax.grid(True, alpha=0.2)

    # -------- (2) Toy possession over time (lines) -------- #
    ax = axes[1]

    # Create line series where y is constant level when holder == yes, else NaN
    t = df["t_min"].values

    child_yes = df["child_holding_toy"].astype(str).str.lower() == "yes"
    parent_yes = df["parent_holding_toy"].astype(str).str.lower() == "yes"
    any_yes = df["toy_in_environment"].astype(str).str.lower() == "yes"

    y_child = np.where(child_yes, 0.0, np.nan)
    y_parent = np.where(parent_yes, 1.0, np.nan)
    y_any = np.where(any_yes, 2.0, np.nan)

    ax.plot(t, y_any, marker="o", linewidth=2, alpha=0.9,
            label="Toy held (anyone)")
    ax.plot(t, y_parent, marker="s", linewidth=2, alpha=0.9,
            label="Parent holding toy")
    ax.plot(t, y_child, marker="^", linewidth=2, alpha=0.9,
            label="Child holding toy")

    ax.set_yticks([0.0, 1.0, 2.0])
    ax.set_yticklabels(["Child", "Parent", "Any holder"])
    ax.set_ylabel("Toy Possession", fontweight="bold")
    ax.set_title("Who Is Holding a Toy Over Time? (Lines show continuity)", fontweight="bold")
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=9)

    # -------- (3) Child pose over time (categorical) -------- #
    ax = axes[2]
    ymap_pose = map_categories(df["child_pose"])
    if ymap_pose:
        for _, row in df.iterrows():
            pose = str(row["child_pose"])
            if pose.lower() != "unknown" and pose in ymap_pose:
                ax.scatter(
                    row["t_min"],
                    ymap_pose[pose],
                    s=60,
                    alpha=0.85
                )
        ax.set_yticks(list(ymap_pose.values()))
        ax.set_yticklabels(list(ymap_pose.keys()), fontsize=9)
    ax.set_ylabel("Child Pose", fontweight="bold")
    ax.set_title("Child Pose Over Time", fontweight="bold")
    ax.grid(True, alpha=0.2)

    # Common x-label
    axes[2].set_xlabel("Time (minutes)", fontweight="bold")

    plt.tight_layout()
    plt.savefig(f"{output_prefix}_timeline.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved timeline plot: {output_prefix}_timeline.png")


# ---------------------- Statistics & Transitions ---------------------- #

def calculate_behavioral_statistics(df):
    """
    Compute simple proportions, transitions, and engagement-like metrics
    for the new schema. No chi-square associations anymore.
    """
    print("Calculating behavioral statistics...")
    stats = {}

    df = df.copy()
    df = df.replace("unknown", np.nan)

    needed = [
        "toy_in_environment",
        "parent_holding_toy",
        "child_holding_toy",
        "child_hand_action",
        "child_proximity_behavior",
        "current_toy",
        "adult_hand_action",
        "child_pose",
        "adult_pose",
    ]
    for c in needed:
        if c not in df.columns:
            df[c] = np.nan

    # 1) Proportions for all columns
    stats["proportions"] = {}
    for col in needed:
        counts = df[col].dropna().value_counts(normalize=True)
        stats["proportions"][col] = counts.to_dict()

    # 2) Transitions (common changes over time) for key behaviors
    stats["transitions"] = {}
    key_behaviors = [
        "child_hand_action",
        "adult_hand_action",
        "child_pose",
    ]

    for behavior in key_behaviors:
        series = df[behavior].dropna().astype(str).tolist()
        transitions = []
        for i in range(1, len(series)):
            if series[i] != series[i - 1]:
                transitions.append((series[i - 1], series[i]))
        if transitions:
            tdf = pd.DataFrame(transitions, columns=["from", "to"])
            common = (
                tdf.groupby(["from", "to"])
                .size()
                .sort_values(ascending=False)
                .head(8)
            )
            stats["transitions"][behavior] = {
                f"{k[0]} → {k[1]}": int(v) for k, v in common.items()
            }

    # 3) Simple engagement-like metrics based on toy + hand activity (no gaze)
    n = len(df)
    child_toy_rate = (df["child_holding_toy"] == "yes").sum() / n if n else 0.0
    parent_toy_rate = (df["parent_holding_toy"] == "yes").sum() / n if n else 0.0
    toy_any_rate = (df["toy_in_environment"] == "yes").sum() / n if n else 0.0

    active_child_set = {
        "pointing",
        "grabbing toy",
        "giving away the toy",
        "holding toy still",
        "manipulating toy",
        "gesturing",
        "touching adult",
    }
    active_child_rate = df["child_hand_action"].isin(active_child_set).sum() / n if n else 0.0

    stats["engagement_metrics"] = {
        "toy_any_rate": float(toy_any_rate),
        "child_toy_rate": float(child_toy_rate),
        "parent_toy_rate": float(parent_toy_rate),
        "active_child_hand_rate": float(active_child_rate),
        "total_segments": int(n),
    }

    # 4) Alignment of holding_toy vs "toy"-related hand-actions
    stats["toy_alignment"] = {}

    # Child
    child_hold = df["child_holding_toy"].astype(str).str.lower() == "yes"
    child_hand_has_toy = df["child_hand_action"].astype(str).str.contains("toy", case=False, na=False)
    child_any_engaged = child_hold | child_hand_has_toy

    stats["toy_alignment"]["child"] = {
        "n_holding_yes": int(child_hold.sum()),
        "n_hand_mentions_toy": int(child_hand_has_toy.sum()),
        "n_both": int((child_hold & child_hand_has_toy).sum()),
        "n_either": int(child_any_engaged.sum()),
    }

    # Parent
    parent_hold = df["parent_holding_toy"].astype(str).str.lower() == "yes"
    parent_hand_has_toy = df["adult_hand_action"].astype(str).str.contains("toy", case=False, na=False)
    parent_any_engaged = parent_hold | parent_hand_has_toy

    stats["toy_alignment"]["parent"] = {
        "n_holding_yes": int(parent_hold.sum()),
        "n_hand_mentions_toy": int(parent_hand_has_toy.sum()),
        "n_both": int((parent_hold & parent_hand_has_toy).sum()),
        "n_either": int(parent_any_engaged.sum()),
    }

    return stats


def create_statistical_summary(df, output_prefix):
    """
    Statistical summary figure:
    - Bar graph of toy contact & child activity.
    - Common child hand-action transitions.
    - Common parent hand-action transitions.
    - Common child pose transitions.
    """
    print("Creating statistical summary...")
    stats = calculate_behavioral_statistics(df)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "Behavioral Analysis: Statistical Summary (Gesture/Object Only)",
        fontsize=18,
        fontweight="bold"
    )

    # 1) Key proportions (toy + activity)
    ax = axes[0, 0]
    m = stats.get("engagement_metrics", {})
    key_vals = {
        "Toy present (any holder)": m.get("toy_any_rate", 0.0),
        "Child holding toy": m.get("child_toy_rate", 0.0),
        "Parent holding toy": m.get("parent_toy_rate", 0.0),
        "Active child hand": m.get("active_child_hand_rate", 0.0),
    }
    labels = list(key_vals.keys())
    vals = list(key_vals.values())
    colors = ["#FF9F1C", "#2A9D8F", "#E76F51", "#2E86AB"]
    bars = ax.bar(labels, vals, alpha=0.85, color=colors)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Proportion of Segments", fontweight="bold")
    ax.set_title("Toy Contact & Child Activity", fontweight="bold")
    ax.tick_params(axis="x", rotation=25)
    for b, v in zip(bars, vals):
        ax.text(
            b.get_x() + b.get_width() / 2.0,
            v + 0.02,
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontweight="bold"
        )

    # 2) Common child hand-action transitions
    ax = axes[0, 1]
    trans_child = stats.get("transitions", {}).get("child_hand_action", {})
    if trans_child:
        labels = list(trans_child.keys())
        vals = list(trans_child.values())
        ax.barh(labels, vals, color="#7A5195", alpha=0.85)
        ax.set_xlabel("Count", fontweight="bold")
        ax.set_title("Common Child Hand-Action Transitions", fontweight="bold")
    else:
        ax.text(0.5, 0.5, "No child hand-action transitions found", ha="center", va="center")

    # 3) Common parent hand-action transitions
    ax = axes[1, 0]
    trans_parent = stats.get("transitions", {}).get("adult_hand_action", {})
    if trans_parent:
        labels = list(trans_parent.keys())
        vals = list(trans_parent.values())
        ax.barh(labels, vals, color="#F18F01", alpha=0.85)
        ax.set_xlabel("Count", fontweight="bold")
        ax.set_title("Common Parent Hand-Action Transitions", fontweight="bold")
    else:
        ax.text(0.5, 0.5, "No parent hand-action transitions found", ha="center", va="center")

    # 4) Child pose transitions
    ax = axes[1, 1]
    trans_pose = stats.get("transitions", {}).get("child_pose", {})
    if trans_pose:
        labels = list(trans_pose.keys())
        vals = list(trans_pose.values())
        ax.barh(labels, vals, color="#4ECDC4", alpha=0.85)
        ax.set_xlabel("Count", fontweight="bold")
        ax.set_title("Common Child Pose Transitions", fontweight="bold")
    else:
        ax.text(0.5, 0.5, "No child pose transitions found", ha="center", va="center")

    plt.tight_layout()
    plt.savefig(f"{output_prefix}_stats.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved statistical summary: {output_prefix}_stats.png")

    # Also persist JSON
    stats_file = f"{output_prefix}_stats.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved detailed statistics: {stats_file}")

    return stats


# ---------------------- Camera Comparison (Side vs Room) ---------------------- #

def create_camera_comparison(df_side, df_room, output_prefix):
    """
    Compare side vs room camera CSVs:
    - Align by t_start
    - Compute agreement rates per behavior
    - Visualize agreement/disagreement over time for key behaviors
    """
    print("Creating camera comparison (side vs room)...")

    # Ensure expected columns
    expected = [
        "toy_in_environment",
        "parent_holding_toy",
        "child_holding_toy",
        "child_hand_action",
        "child_proximity_behavior",
        "current_toy",
        "adult_hand_action",
        "child_pose",
        "adult_pose",
    ]
    df_side = ensure_cols(df_side.copy(), expected)
    df_room = ensure_cols(df_room.copy(), expected)

    # Merge by t_start (assumes the same stride/grid across cameras)
    merged = pd.merge(
        df_side,
        df_room,
        on="t_start",
        how="inner",
        suffixes=("_side", "_room")
    )

    if merged.empty:
        print("No overlapping timepoints between side and room views; skipping camera comparison.")
        return {}

    merged["t_min"] = merged["t_start"] / 60.0

    compare_cols = [
        "child_hand_action",
        "child_pose",
        "adult_hand_action",
        "adult_pose",
        "child_holding_toy",
        "parent_holding_toy",
        "current_toy",
    ]

    comparison_stats = {"per_behavior": {}}

    # Agreement stats per behavior
    for col in compare_cols:
        side_col = f"{col}_side"
        room_col = f"{col}_room"
        both_known = (
            (merged[side_col].str.lower() != "unknown") &
            (merged[room_col].str.lower() != "unknown")
        )
        agree = both_known & (merged[side_col] == merged[room_col])
        n_both = both_known.sum()
        n_agree = agree.sum()
        agreement_rate = float(n_agree / n_both) if n_both else float("nan")

        comparison_stats["per_behavior"][col] = {
            "n_overlap": int(n_both),
            "n_agree": int(n_agree),
            "agreement_rate": agreement_rate,
        }

    # ---------- Visualization 1: Agreement rates per behavior ---------- #
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    fig.suptitle(
        "Side vs Room Camera Comparison (Short-Clip Labels)",
        fontsize=18,
        fontweight="bold"
    )

    ax = axes[0]
    behaviors = []
    rates = []
    for col, info in comparison_stats["per_behavior"].items():
        if np.isfinite(info["agreement_rate"]):
            behaviors.append(col)
            rates.append(info["agreement_rate"])
    if behaviors:
        ax.bar(behaviors, rates, color="#4E79A7", alpha=0.85)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Agreement Rate (Side vs Room)", fontweight="bold")
        ax.set_title("Per-Behavior Agreement Across Cameras", fontweight="bold")
        ax.tick_params(axis="x", rotation=25)
        for x, r in zip(behaviors, rates):
            ax.text(
                x, r + 0.02,
                f"{r:.2f}",
                ha="center",
                va="bottom",
                fontweight="bold"
            )
    else:
        ax.text(0.5, 0.5, "No overlapping labeled segments", ha="center", va="center")

    # ---------- Visualization 2: Time-series agreement status ---------- #
    ax = axes[1]
    key_for_timeline = ["child_hand_action", "child_pose", "adult_hand_action"]

    # Status encoding: 2 = agree, 1 = disagree, 0 = unknown
    status_colors = {
        2: "#2ECC71",   # agree
        1: "#E67E22",   # disagree
        0: "#95A5A6",   # unknown
    }

    for idx, col in enumerate(key_for_timeline):
        side_col = f"{col}_side"
        room_col = f"{col}_room"
        statuses = []
        times = []
        y_level = idx  # separate row per behavior
        for _, row in merged.iterrows():
            s = str(row[side_col]).lower()
            r = str(row[room_col]).lower()
            if s == "unknown" or r == "unknown":
                status = 0
            elif s == r:
                status = 2
            else:
                status = 1
            statuses.append(status)
            times.append(row["t_min"])

        # plot as colored dots
        for t, st in zip(times, statuses):
            ax.scatter(
                t,
                y_level + (st - 1) * 0.15,  # small vertical shift by status
                color=status_colors.get(st, "#95A5A6"),
                s=25,
                alpha=0.8
            )

        ax.text(
            merged["t_min"].min() - 0.1,
            y_level,
            col,
            va="center",
            ha="right",
            fontsize=9,
            fontweight="bold"
        )

    ax.set_yticks([])
    ax.set_xlabel("Time (minutes)", fontweight="bold")
    ax.set_title(
        "Time-Series Agreement Status (2 = agree, 1 = disagree, 0 = unknown)",
        fontweight="bold"
    )
    ax.grid(True, axis="x", alpha=0.2)

    # Legend for statuses
    legend_handles = [
        plt.Line2D([0], [0], marker='o', linestyle='',
                   color=status_colors[2], label="Agree", markersize=8),
        plt.Line2D([0], [0], marker='o', linestyle='',
                   color=status_colors[1], label="Disagree", markersize=8),
        plt.Line2D([0], [0], marker='o', linestyle='',
                   color=status_colors[0], label="Unknown", markersize=8),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig(f"{output_prefix}_camera_compare.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved camera comparison plot: {output_prefix}_camera_compare.png")

    # Save comparison stats as JSON
    stats_file = f"{output_prefix}_camera_compare.json"
    with open(stats_file, "w") as f:
        json.dump(comparison_stats, f, indent=2)
    print(f"Saved camera comparison statistics: {stats_file}")

    return comparison_stats


# ---------------------- Model vs Human Comparison ---------------------- #

def create_model_vs_human_comparison(df_model, df_human, output_prefix):
    """
    Compare model CSV (df_model) vs human CSV (df_human):

    - Align by t_start (and t_end) assuming they come from the same template.
    - For each behavior column:
        * strict_accuracy: accuracy only on rows where human != 'unknown'
        * match_incl_unknown: fraction of rows where model == human
          (including 'unknown' == 'unknown').
    - Visualization:
        * Bar chart of strict accuracy per behavior.
        * Time-series agreement status for key behaviors.
    """
    print("Creating model vs human comparison...")

    expected = [
        "toy_in_environment",
        "parent_holding_toy",
        "child_holding_toy",
        "child_hand_action",
        "child_proximity_behavior",
        "current_toy",
        "adult_hand_action",
        "child_pose",
        "adult_pose",
    ]

    df_model = ensure_cols(df_model.copy(), expected)
    df_human = ensure_cols(df_human.copy(), expected)

    merged = pd.merge(
        df_model,
        df_human,
        on=["t_start", "t_end"],
        how="inner",
        suffixes=("_model", "_human")
    )

    if merged.empty:
        print("No overlapping timepoints between model and human CSVs; skipping comparison.")
        return {}

    merged["t_min"] = merged["t_start"] / 60.0

    comparison_stats = {"per_behavior": {}}

    for col in expected:
        mc = f"{col}_model"
        hc = f"{col}_human"

        # Normalize labels to lowercase strings, treating NaN as 'unknown'
        gt = merged[hc].fillna("unknown").astype(str).str.lower()
        pred = merged[mc].fillna("unknown").astype(str).str.lower()

        n_total = len(merged)

        # strict accuracy: only where human != 'unknown'
        mask_valid = gt != "unknown"
        n_valid = int(mask_valid.sum())
        if n_valid > 0:
            strict_acc = float((gt[mask_valid] == pred[mask_valid]).mean())
            n_correct_valid = int((gt[mask_valid] == pred[mask_valid]).sum())
        else:
            strict_acc = float("nan")
            n_correct_valid = 0

        # overall agreement (including unknown vs unknown)
        n_agree_all = int((gt == pred).sum())
        overall_match = float((gt == pred).mean()) if n_total > 0 else float("nan")

        comparison_stats["per_behavior"][col] = {
            "n_total": int(n_total),
            "n_valid_gt_not_unknown": n_valid,
            "n_correct_on_valid": n_correct_valid,
            "strict_accuracy": strict_acc,
            "n_agree_all": n_agree_all,
            "match_incl_unknown": overall_match,
        }

    # ---------- Visualization: Accuracy + Time-series agreement ---------- #
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    fig.suptitle(
        "Model vs Human Comparison (Short-Clip Labels)",
        fontsize=18,
        fontweight="bold"
    )

    # (1) Strict accuracy per behavior
    ax = axes[0]
    behaviors = []
    strict_accs = []
    overall_matches = []
    for col, info in comparison_stats["per_behavior"].items():
        if np.isfinite(info["strict_accuracy"]):
            behaviors.append(col)
            strict_accs.append(info["strict_accuracy"])
            overall_matches.append(info["match_incl_unknown"])

    if behaviors:
        x = np.arange(len(behaviors))
        width = 0.35
        ax.bar(x - width/2, strict_accs, width, label="Strict accuracy (GT != 'unknown')")
        ax.bar(x + width/2, overall_matches, width, label="Agreement incl. 'unknown'")
        ax.set_xticks(x)
        ax.set_xticklabels(behaviors, rotation=25)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Proportion", fontweight="bold")
        ax.set_title("Per-Behavior Accuracy vs Agreement", fontweight="bold")
        ax.legend()
        for i, (sa, om) in enumerate(zip(strict_accs, overall_matches)):
            ax.text(i - width/2, sa + 0.02, f"{sa:.2f}", ha="center", va="bottom", fontsize=8)
            ax.text(i + width/2, om + 0.02, f"{om:.2f}", ha="center", va="bottom", fontsize=8)
    else:
        ax.text(0.5, 0.5, "No valid behaviors for accuracy computation", ha="center", va="center")

    # (2) Time-series agreement status for key behaviors
    ax = axes[1]
    key_for_timeline = [
        "child_hand_action",
        "child_proximity_behavior",
        "child_pose",
        "adult_hand_action",
    ]

    # Status encoding:
    #   2 = GT known and model == human
    #   1 = GT known and model != human
    #   0 = GT unknown
    status_colors = {
        2: "#2ECC71",   # correct on known label
        1: "#E67E22",   # mismatch on known label
        0: "#95A5A6",   # GT unknown
    }

    for idx, col in enumerate(key_for_timeline):
        mc = f"{col}_model"
        hc = f"{col}_human"
        statuses = []
        times = []
        y_level = idx  # separate row per behavior
        for _, row in merged.iterrows():
            gt = str(row[hc]).lower() if pd.notna(row[hc]) else "unknown"
            pred = str(row[mc]).lower() if pd.notna(row[mc]) else "unknown"

            if gt == "unknown":
                status = 0
            elif gt == pred:
                status = 2
            else:
                status = 1

            statuses.append(status)
            times.append(row["t_min"])

        for t, st in zip(times, statuses):
            ax.scatter(
                t,
                y_level + (st - 1) * 0.15,
                color=status_colors.get(st, "#95A5A6"),
                s=25,
                alpha=0.8
            )

        ax.text(
            merged["t_min"].min() - 0.1,
            y_level,
            col,
            va="center",
            ha="right",
            fontsize=9,
            fontweight="bold"
        )

    ax.set_yticks([])
    ax.set_xlabel("Time (minutes)", fontweight="bold")
    ax.set_title(
        "Model vs Human Agreement Over Time (2=correct, 1=wrong, 0=GT unknown)",
        fontweight="bold"
    )
    ax.grid(True, axis="x", alpha=0.2)

    legend_handles = [
        plt.Line2D([0], [0], marker='o', linestyle='',
                   color=status_colors[2], label="Correct (GT known)", markersize=8),
        plt.Line2D([0], [0], marker='o', linestyle='',
                   color=status_colors[1], label="Incorrect (GT known)", markersize=8),
        plt.Line2D([0], [0], marker='o', linestyle='',
                   color=status_colors[0], label="GT unknown", markersize=8),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig(f"{output_prefix}_model_vs_human.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved model vs human comparison plot: {output_prefix}_model_vs_human.png")

    # Save stats JSON
    stats_file = f"{output_prefix}_model_vs_human.json"
    with open(stats_file, "w") as f:
        json.dump(comparison_stats, f, indent=2)
    print(f"Saved model vs human comparison statistics: {stats_file}")

    return comparison_stats


# ---------------------- Insights ---------------------- #

def create_behavioral_insights(df, output_prefix):
    """
    Gesture-only behavioral insights:
    - Toy possession breakdown
    - Child vs parent hand-action mix
    - Child pose distribution
    - Child proximity distribution
    - Current toy distribution
    - Time series of toy engagement (any vs child vs parent) where
      engagement = holding_toy=='yes' OR hand action contains "toy".
    """
    print("Creating behavioral insights...")
    df = df.copy()
    df = ensure_cols(df, [
        "toy_in_environment",
        "parent_holding_toy",
        "child_holding_toy",
        "adult_hand_action",
        "child_hand_action",
        "child_pose",
        "adult_pose",
        "child_proximity_behavior",
        "current_toy",
    ])

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle("Behavioral Insights (Short Clips, Gesture/Object Only)",
                 fontsize=18, fontweight="bold")

    # 1) Toy possession breakdown (pie)
    ax = axes[0, 0]
    toy_states = pd.DataFrame({
        "toy_in_environment": df["toy_in_environment"],
        "parent_holding_toy": df["parent_holding_toy"],
        "child_holding_toy": df["child_holding_toy"],
    }).replace("unknown", np.nan).dropna(how="all")
    if not toy_states.empty:
        def cat(row):
            if row["child_holding_toy"] == "yes":
                if row["parent_holding_toy"] == "yes":
                    return "Both"
                else:
                    return "Child only"
            elif row["parent_holding_toy"] == "yes":
                return "Parent only"
            elif row["toy_in_environment"] == "yes":
                return "Other holder"
            else:
                return "No toy held"

        toy_poss_cat = toy_states.apply(cat, axis=1)
        counts = toy_poss_cat.value_counts(normalize=True)
        ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=90)
        ax.set_title("Toy Possession Breakdown", fontweight="bold")
    else:
        ax.text(0.5, 0.5, "No toy possession data", ha="center", va="center")

    # 2) Child vs parent hand-action mix
    ax = axes[0, 1]
    child = df["child_hand_action"].replace("unknown", np.nan).dropna()
    adult = df["adult_hand_action"].replace("unknown", np.nan).dropna()
    if not child.empty or not adult.empty:
        # Show top 6 labels for each as side-by-side bars
        child_counts = child.value_counts().head(6)
        adult_counts = adult.value_counts().head(6)

        labels = sorted(set(child_counts.index).union(adult_counts.index))
        child_vals = [child_counts.get(l, 0) for l in labels]
        adult_vals = [adult_counts.get(l, 0) for l in labels]

        x = np.arange(len(labels))
        width = 0.4
        ax.bar(x - width/2, child_vals, width, label="Child")
        ax.bar(x + width/2, adult_vals, width, label="Parent")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylabel("Count", fontweight="bold")
        ax.set_title("Child vs Parent Hand-Action Mix", fontweight="bold")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No hand-action data", ha="center", va="center")

    # 3) Child pose distribution
    ax = axes[0, 2]
    pose = df["child_pose"].replace("unknown", np.nan).dropna()
    if not pose.empty:
        counts = pose.value_counts()
        ax.bar(counts.index, counts.values)
        ax.set_ylabel("Count", fontweight="bold")
        ax.set_title("Child Pose Distribution", fontweight="bold")
        ax.tick_params(axis="x", rotation=25)
    else:
        ax.text(0.5, 0.5, "No child pose data", ha="center", va="center")

    # 4) Child proximity distribution
    ax = axes[1, 0]
    prox = df["child_proximity_behavior"].replace("unknown", np.nan).dropna()
    if not prox.empty:
        counts = prox.value_counts()
        ax.bar(counts.index, counts.values)
        ax.set_ylabel("Count", fontweight="bold")
        ax.set_title("Child Proximity Categories", fontweight="bold")
        ax.tick_params(axis="x", rotation=25)
    else:
        ax.text(0.5, 0.5, "No proximity data", ha="center", va="center")

    # 5) Toy distribution
    ax = axes[1, 1]
    toy = df["current_toy"].replace("unknown", np.nan).dropna()
    if not toy.empty:
        counts = toy.value_counts()
        ax.bar(counts.index, counts.values)
        ax.set_ylabel("Count", fontweight="bold")
        ax.set_title("Current Toy Labels (Child’s Hands)", fontweight="bold")
        ax.tick_params(axis="x", rotation=15)
    else:
        ax.text(0.5, 0.5, "No toy label data", ha="center", va="center")

    # 6) Time series: toy engagement (any vs child vs parent)
    ax = axes[1, 2]
    df["t_min"] = df["t_start"] / 60.0
    df["minute"] = np.floor(df["t_min"]).astype(int)

    # engagement = holding_toy == "yes" OR hand_action contains "toy"
    child_hold = df["child_holding_toy"].astype(str).str.lower() == "yes"
    child_hand_toy = df["child_hand_action"].astype(str).str.contains("toy", case=False, na=False)
    child_engaged = (child_hold | child_hand_toy).astype(int)

    parent_hold = df["parent_holding_toy"].astype(str).str.lower() == "yes"
    parent_hand_toy = df["adult_hand_action"].astype(str).str.contains("toy", case=False, na=False)
    parent_engaged = (parent_hold | parent_hand_toy).astype(int)

    any_engaged = ((child_engaged == 1) | (parent_engaged == 1)).astype(int)

    ts = pd.DataFrame({
        "minute": df["minute"],
        "child": child_engaged,
        "parent": parent_engaged,
        "any": any_engaged,
    })
    ts = ts.groupby("minute").sum().reset_index()

    if not ts.empty:
        ax.plot(ts["minute"], ts["any"], marker="o", linewidth=2, label="Any engaged")
        ax.plot(ts["minute"], ts["child"], marker="^", linewidth=2, label="Child engaged")
        ax.plot(ts["minute"], ts["parent"], marker="s", linewidth=2, label="Parent engaged")
        ax.set_xlabel("Time (minutes)", fontweight="bold")
        ax.set_ylabel("Engagement Events / Minute", fontweight="bold")
        ax.set_title("Toy Engagement Over Time (Any vs Child vs Parent)", fontweight="bold")
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, "No engagement data", ha="center", va="center")

    plt.tight_layout()
    plt.savefig(f"{output_prefix}_insights.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved behavioral insights: {output_prefix}_insights.png")


def main():
    parser = argparse.ArgumentParser(
        description="Gesture-only visual behavior analysis for SHORT preset CSV, with optional side vs room comparison and model vs human comparison."
    )
    parser.add_argument(
        "--input", "-i",
        default="data/clips.csv"
    )
    parser.add_argument(
        "--output", "-o",
        default="results/visualization"
    )
    parser.add_argument(
        "--side_csv",
        default=None,
        help="Optional: CSV for side-view camera for comparison."
    )
    parser.add_argument(
        "--room_csv",
        default=None,
        help="Optional: CSV for room-view camera for comparison."
    )
    parser.add_argument(
        "--human_csv",
        default=None,
        help="Optional: human annotation CSV (ground truth) to compare against model CSV (given by --input)."
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)

    print(f"Loading single-camera (model) data from {args.input}...")
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} time segments")

    if not df.empty:
        print("\nData Overview:")
        try:
            print(f"Time range: {df['t_start'].min():.1f}s to {df['t_end'].max():.1f}s")
            step = df["t_start"].diff().dropna()
            if not step.empty:
                print(f"Time resolution (median): {step.median():.2f}s between samples")
        except Exception:
            pass

    create_timeline_plots(df, args.output)
    stats = create_statistical_summary(df, args.output)
    create_behavioral_insights(df, args.output)

    print("\n" + "=" * 60)
    print("KEY BEHAVIORAL FINDINGS (Gesture/Object Only, Short Clips)")
    print("=" * 60)
    if "engagement_metrics" in stats:
        m = stats["engagement_metrics"]
        print(f"• Toy present (any holder): {m['toy_any_rate']:.1%}")
        print(f"• Child holding toy:        {m['child_toy_rate']:.1%}")
        print(f"• Parent holding toy:       {m['parent_toy_rate']:.1%}")
        print(f"• Active child hand:        {m['active_child_hand_rate']:.1%}")
        print(f"• Total segments:           {m['total_segments']}")

    ta = stats.get("toy_alignment", {})
    if ta:
        print("\nTOY ENGAGEMENT ALIGNMENT (holding_toy vs 'toy' in hand-actions)")
        child = ta.get("child", {})
        parent = ta.get("parent", {})
        print(f"Child:  holding_yes={child.get('n_holding_yes', 0)}, "
              f"hand_mentions_toy={child.get('n_hand_mentions_toy', 0)}, "
              f"both={child.get('n_both', 0)}, either={child.get('n_either', 0)}")
        print(f"Parent: holding_yes={parent.get('n_holding_yes', 0)}, "
              f"hand_mentions_toy={parent.get('n_hand_mentions_toy', 0)}, "
              f"both={parent.get('n_both', 0)}, either={parent.get('n_either', 0)}")

    # Optional: camera comparison
    if args.side_csv and args.room_csv:
        print("\n" + "=" * 60)
        print("SIDE vs ROOM CAMERA COMPARISON")
        print("=" * 60)
        try:
            df_side = pd.read_csv(args.side_csv)
            df_room = pd.read_csv(args.room_csv)
            cam_stats = create_camera_comparison(df_side, df_room, args.output)
            for col, info in cam_stats.get("per_behavior", {}).items():
                ar = info["agreement_rate"]
                if np.isfinite(ar):
                    print(f"- {col}: agreement={ar:.1%} (n_overlap={info['n_overlap']})")
        except Exception as e:
            print(f"[warn] Failed to create camera comparison: {e}")

    # Optional: model vs human comparison
    if args.human_csv:
        print("\n" + "=" * 60)
        print("MODEL vs HUMAN COMPARISON")
        print("=" * 60)
        try:
            df_human = pd.read_csv(args.human_csv)
            mh_stats = create_model_vs_human_comparison(df, df_human, args.output)
            for col, info in mh_stats.get("per_behavior", {}).items():
                sa = info["strict_accuracy"]
                om = info["match_incl_unknown"]
                if np.isfinite(sa) or np.isfinite(om):
                    print(
                        f"- {col}: strict_acc={sa:.1%} "
                        f"(GT!=unknown, n={info['n_valid_gt_not_unknown']}), "
                        f"agreement_incl_unknown={om:.1%} (n_total={info['n_total']})"
                    )
        except Exception as e:
            print(f"Failed to create model vs human comparison: {e}")

    print(f"\nAll visualizations saved with prefix: {args.output}_*")


if __name__ == "__main__":
    main()