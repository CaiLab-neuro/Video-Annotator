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
import json
import os
import textwrap
from collections import Counter
from typing import Optional

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
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 12,
    "legend.title_fontsize": 13,
})

# Palette
COLORS = {
    "teal": "#85d2d0",
    "purple": "#887bb0",
    "orange": "#ffbd59",
    "pale_yellow": "#fff4bd",
}

# Fixed colors for known behavioral labels (used by Gantt-style timecourse plots).
# Labels not listed here fall back to plt.cm.tab20.
#
# Color design: semantic proximity → perceptual proximity.
#
# Hand actions are grouped into semantic families:
#   - Toy transfer (orange spectrum, darker=receiving, lighter=giving)
#   - Toy holding/manipulation (amber-yellow, static→active)
#   - Communicative gestures (blue, directed→broad)
#   - Social contact (purple)
#   - Non-toy object handling (teal)
#   - Resting/passive (warm grey-brown)
#
# Poses follow a cool→warm gradient: passive/low → active/upright/moving
#
# Proximity: hue family = distance (blue=close, orange=mid, green=far);
#            lightness = orientation (dark=toward adult, light=away)
#
# Toy names: approximate physical colors of the actual toys.

LABEL_COLORS = {
    # ---- Hand actions: toy transfer (orange-red spectrum) ----
    # Spread across dark-red → orange → light-amber to maximise within-family contrast.
    "grabbing toy":             "#e05818",   # warm orange — acquiring toy
    "taking toy from child":    "#a02000",   # dark red — clearly darker/redder than grabbing
    "handing toy to child":     "#f5a840",   # light amber — giving to child
    "giving away the toy":      "#f5a840",   # same — giving/handing overlap
    "placing toy on something": "#604808",   # very dark brown — distinct from amber-gold below

    # ---- Hand actions: toy holding / manipulation (gold→green, wider spread) ----
    "holding toy":              "#d4a800",   # bright gold — static possession
    "holding toy still":        "#d4a800",   # same canonical
    "moving toy":               "#9cbc00",   # lime-yellow — shifted clearly from gold
    "manipulating toy":         "#58a010",   # green — distinct from lime-yellow

    # ---- Hand actions: communicative gestures (blue, wider spread) ----
    "pointing":                 "#0a3888",   # dark navy — directed, specific target
    "reaching out":             "#2a70c8",   # medium blue — extends toward target
    "gesturing":                "#5898d8",   # lighter blue — general expressive gesture
    "waving":                   "#88b8f0",   # very light blue — open social, non-directed

    # ---- Hand actions: social / body contact (purple) ----
    "touching adult":           "#8848b0",   # medium purple — touching a person
    "touching child":           "#8848b0",   # same hue — touching a person

    # ---- Hand actions: non-toy object handling (teal, wide lightness spread) ----
    "holding paper":                             "#70d8e0",  # light cyan — static flat object
    "touching box/toy bag/eye-tracker components": "#087068", # very dark teal — prop object
    "opening a box or bag":                      "#30b080",  # medium teal-green — active prop use
    "closing a box or bag":                      "#30b080",  # same — opposite but same object/type
    "touching glasses":                          "#28c0c8",  # bright cyan — prop accessory

    # ---- Hand actions: resting / passive (desaturated warm grey-brown) ----
    # Shifted to desaturated warm greys so the family is clearly distinct from
    # the saturated gold of "holding toy still" despite sharing similar hue angle.
    "on the ground/touching some furniture/resting": "#9a8070",  # medium warm grey-brown
    "on the ground":            "#9a8070",   # same compound group
    "on some furniture":        "#b8a890",   # lighter warm grey
    "on furniture":             "#b8a890",   # same
    "resting":                  "#d0c8b0",   # light beige
    "resting on body":          "#d0c8b0",   # same

    # ---- Hand actions: null / invisible ----
    "none":                     "#e0e0e0",   # near-white grey — no action

    # ---- Poses: cool→warm gradient, passive→active ----
    # Wider lightness/hue steps so adjacent poses remain distinguishable.
    "lying on the floor":       "#4030a0",   # violet-blue — most passive, horizontal
    "sitting (kneeling)":       "#1858a8",   # deep blue — low, stationary
    "sitting still":            "#1858a8",   # alias
    "sitting":                  "#5080c8",   # medium-light blue — clearly lighter than above
    "crouching":                "#1088b8",   # teal-blue — distinct from pure blue above
    "bending over":             "#10a878",   # teal-green — distinct from teal-blue
    "crawling":                 "#30b060",   # green — locomotion close to ground
    "turning around":           "#70b030",   # yellow-green — rotation, moderate effort
    "standing still":           "#98a820",   # olive-yellow — upright, stationary
    "walking":                  "#d07020",   # warm orange — active locomotion

    # ---- Proximity: hue=distance, lightness=orientation ----
    # Blue family = close; orange family = mid; green family = far
    # Darker/saturated = facing toward adult; lighter = facing away
    "close and facing toward adult":           "#1a6ea8",   # deep blue
    "close but facing away from adult":        "#80b8d8",   # light blue
    "mid distance and facing toward adult":    "#c86018",   # deep orange
    "mid distance and facing away from adult": "#f0b070",   # light orange
    "far and facing toward adult":             "#288040",   # deep green
    "far but facing away from adult":          "#78c888",   # light green

    # ---- Toy holder ----
    "child only":  "#45a8a8",   # teal — child
    "parent only": "#7868a8",   # purple — parent
    "both":        "#e8a030",   # orange — joint possession

    # ---- Yes / No ----
    "yes": "#4a9050",   # green
    "no":  "#c05050",   # red

    # ---- Unknown / not visible / missing ----
    # Light neutral grey — unobtrusive, clearly distinct from warm khaki resting.
    "unknown":     "#cccccc",   # light grey — data absent / uncodable
    "invisible":   "#cccccc",   # same — person not in frame
    "not visible": "#cccccc",   # same
}

# ----------------------
# Label grouping for model-vs-human comparison
# ----------------------

# Remap MODEL labels before comparison (known semantic equivalences).
# The delta preset uses fine-grained sub-labels; map them to the ENS compound
# labels that appear verbatim in human CSVs generated by eaf_to_csv_ICDL_ENS.py.
MODEL_COMPARISON_GROUPS = {
    "child_hand_action": {
        # delta sub-labels → ENS compound EAF label
        "on the ground": "on the ground/touching some furniture/resting",
        "on furniture": "on the ground/touching some furniture/resting",
        "on some furniture": "on the ground/touching some furniture/resting",
        "resting": "on the ground/touching some furniture/resting",
        "resting on body": "on the ground/touching some furniture/resting",
        "none": "on the ground/touching some furniture/resting",
        "opening a box or bag": "touching box/toy bag/eye-tracker components",
        "closing a box or bag": "touching box/toy bag/eye-tracker components",
        "touching glasses": "touching box/toy bag/eye-tracker components",
        # unaliased model may output "holding toy" instead of "holding toy still"
        "holding toy": "holding toy still",
    },
    "adult_hand_action": {
        # delta sub-labels → ENS compound EAF label
        "on the ground": "on the ground/touching some furniture/resting",
        "on furniture": "on the ground/touching some furniture/resting",
        "on some furniture": "on the ground/touching some furniture/resting",
        "resting": "on the ground/touching some furniture/resting",
        "resting on body": "on the ground/touching some furniture/resting",
        "none": "on the ground/touching some furniture/resting",
        "opening a box or bag": "touching box/toy bag/eye-tracker components",
        "closing a box or bag": "touching box/toy bag/eye-tracker components",
        "touching glasses": "touching box/toy bag/eye-tracker components",
        # unaliased model may output "holding toy" instead of "holding toy still"
        "holding toy": "holding toy still",
        # model uses adult-centric label; human scheme uses child-centric "grabbing toy"
        "taking toy from child": "grabbing toy",
    },
    "child_pose": {
        # delta canonical → ENS canonical (both are "sitting (kneeling)"; this
        # covers any older model CSVs that output the pre-delta "sitting still")
        "sitting still": "sitting (kneeling)",
        "not visible": "invisible",
    },
    "adult_pose": {
        "sitting": "sitting (kneeling)",
        "sitting still": "sitting (kneeling)",
        "not visible": "invisible",
    },
}

# Remap HUMAN labels before comparison (fix known normalisation gaps in
# existing human CSVs generated before certain aliases were added).
HUMAN_COMPARISON_GROUPS = {
    "adult_hand_action": {
        # Pre-delta human CSVs had "holding toy" as the canonical adult label;
        # delta (and updated ENS script) use "holding toy still".
        "holding toy": "holding toy still",
        # Pre-update CSVs may have the old "resting" collapse instead of compound.
        "resting": "on the ground/touching some furniture/resting",
    },
    "child_hand_action": {
        # Pre-update CSVs may have the old "resting" / "on some furniture" collapse.
        "resting": "on the ground/touching some furniture/resting",
        "on some furniture": "on the ground/touching some furniture/resting",
    },
    "child_proximity_behavior": {
        # ENS: older CSVs omit the trailing " adult" on proximity labels.
        "mid distance and facing toward": "mid distance and facing toward adult",
        "close and facing toward": "close and facing toward adult",
        "far and facing toward": "far and facing toward adult",
        "close and facing away": "close but facing away from adult",
        "mid distance and facing away": "mid distance and facing away from adult",
        "far and facing away": "far but facing away from adult",
    },
    "child_pose": {
        # Pre-delta human CSVs used "sitting still" as the canonical child pose.
        "sitting still": "sitting (kneeling)",
        "not visible": "invisible",
    },
    "adult_pose": {
        # Pre-delta human CSVs used "sitting" as the canonical adult pose.
        "sitting": "sitting (kneeling)",
        "not visible": "invisible",
    },
}


def apply_comparison_groups(series: pd.Series, col_name: str) -> pd.Series:
    """Remap model labels per MODEL_COMPARISON_GROUPS[col_name]."""
    mapping = MODEL_COMPARISON_GROUPS.get(col_name, {})
    if not mapping:
        return series
    return series.replace(mapping)


def apply_human_comparison_groups(series: pd.Series, col_name: str) -> pd.Series:
    """Remap human labels per HUMAN_COMPARISON_GROUPS[col_name]."""
    mapping = HUMAN_COMPARISON_GROUPS.get(col_name, {})
    if not mapping:
        return series
    return series.replace(mapping)


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


def compute_accuracy_and_agreement(df: pd.DataFrame, model_col: str, human_col: str,
                                   human_vocab: Optional[dict] = None):
    """
    Compute:
      - strict accuracy: agreement excluding unknowns / NaNs and OOV model predictions
      - agreement_all: raw agreement including unknowns (but still OOV-filtered)
      - kappa: Cohen's kappa on strict subset

    Model labels are remapped via MODEL_COMPARISON_GROUPS[model_col] and human
    labels via HUMAN_COMPARISON_GROUPS[human_col] before comparison.

    OOV filtering: rows where the (remapped) model label is not in the human
    annotation vocabulary are excluded.  The vocabulary is the union of the
    labels in human_vocab[col] (from the sidecar _vocab.json, which reflects
    the full annotation scheme) and the labels actually observed in the human
    column.  This ensures that, e.g., a Miami subject whose human CSV only
    shows "holding toy still" and "manipulating toy" does not penalise the
    model for predicting categories (like "pointing") that were outside the
    human annotator's scheme entirely.
    """
    s_model = apply_comparison_groups(df[model_col].astype(str), model_col)
    bare_col = human_col.removesuffix("_human")
    s_human = apply_human_comparison_groups(df[human_col].astype(str), bare_col)

    # Build effective human vocabulary:
    #   vocab JSON  (full intended scheme)
    #   ∪ observed human labels (catches any pass-through EAF values)
    # The vocab JSON uses bare column names (e.g. "child_hand_action") while
    # the merged DataFrame suffixes the human column with "_human".
    observed_human = set(s_human[(s_human != "nan") & (s_human != "unknown")].unique())
    if human_vocab and bare_col in human_vocab:
        # apply HUMAN_COMPARISON_GROUPS so vocab entries and observed labels
        # are in the same normalised space before taking the union
        remap = HUMAN_COMPARISON_GROUPS.get(bare_col, {})
        remapped_vocab = {remap.get(v, v) for v in human_vocab[bare_col]}
        eff_human_vocab = remapped_vocab | observed_human
    else:
        eff_human_vocab = observed_human

    # OOV mask: model predicted a label outside the human annotation scheme
    oov_mask = ~s_model.isin(eff_human_vocab)
    n_oov = int(((s_model != "nan") & (s_model != "unknown") &
                 (s_human != "nan") & oov_mask).sum())
    if n_oov > 0:
        oov_labels = sorted(
            s_model[(s_model != "nan") & (s_model != "unknown") & oov_mask].unique()
        )
        print(f"  [oov] {model_col}: excluding {n_oov} rows with model labels "
              f"outside human vocab: {oov_labels}")

    # Strict subset: exclude NaNs, 'unknown', 'other', and OOV model predictions
    mask_strict = (
        (s_model != "nan") & (s_human != "nan") &
        (s_model != "unknown") & (s_human != "unknown") &
        (s_model != "other") &
        ~oov_mask
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

    # Agreement including unknowns, still OOV-filtered; exclude 'other'
    mask_all = (s_model != "nan") & (s_human != "nan") & (s_model != "other") & ~oov_mask
    if mask_all.sum() > 0:
        agreement_all = (s_model[mask_all] == s_human[mask_all]).mean()
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


def spans_from_series(t_starts, t_ends, labels):
    """
    Merge consecutive timesteps with the same label into contiguous spans.

    Returns a list of (t0, t1, label) tuples suitable for broken_barh().
    Consecutive rows are merged regardless of small time gaps.
    """
    spans = []
    t_starts = list(t_starts)
    t_ends = list(t_ends)
    labels = list(labels)

    if not labels:
        return spans

    cur_label = labels[0]
    cur_t0 = t_starts[0]
    cur_t1 = t_ends[0]

    for t0, t1, lbl in zip(t_starts[1:], t_ends[1:], labels[1:]):
        if lbl == cur_label:
            cur_t1 = t1
        else:
            spans.append((cur_t0, cur_t1, cur_label))
            cur_label = lbl
            cur_t0 = t0
            cur_t1 = t1

    spans.append((cur_t0, cur_t1, cur_label))
    return spans


def plot_behavior_timecourse(df: pd.DataFrame, model_col: str, human_col: str,
                             title: str, out_path: str):
    """
    Gantt-chart timecourse for any categorical behavior.
    Two horizontal rows: Model (top, y=1) and Human (bottom, y=0).
    Each contiguous run of the same label is rendered as a filled bar via
    broken_barh(). Colors come from LABEL_COLORS; unknown labels fall back
    to plt.cm.tab20.
    """
    t_start = df[COLUMN_MAP["time_start"]]
    t_end = df[COLUMN_MAP["time_end"]]

    model_series = df[model_col].astype(str).fillna("unknown").replace("nan", "unknown")
    model_series = apply_comparison_groups(model_series, model_col)
    human_series = df[human_col].astype(str).fillna("unknown").replace("nan", "unknown")
    bare_human_col = human_col.removesuffix("_human")
    human_series = apply_human_comparison_groups(human_series, bare_human_col)

    # Suppress OOV model labels (not in human vocab) so the timecourse only
    # shows categories that appear in the EAF annotation scheme.
    human_label_set = set(human_series) | {"unknown"}
    model_series = model_series.map(lambda x: x if x in human_label_set else "unknown")

    all_labels = sorted(set(model_series) | set(human_series))

    tab20 = plt.cm.tab20
    extra_labels = [l for l in all_labels if l not in LABEL_COLORS]
    extra_colors = {lbl: tab20(i % 20) for i, lbl in enumerate(extra_labels)}

    def get_color(lbl):
        return LABEL_COLORS.get(lbl, extra_colors.get(lbl, "#999999"))

    fig, ax = plt.subplots(figsize=(14, 4))
    bar_height = 0.6
    y_positions = {"Model": 1.0, "Human": 0.0}

    for row_label, series in [("Model", model_series), ("Human", human_series)]:
        y = y_positions[row_label]
        spans = spans_from_series(t_start, t_end, series)
        for (t0, t1, lbl) in spans:
            ax.broken_barh(
                [(t0, t1 - t0)],
                (y - bar_height / 2, bar_height),
                facecolors=get_color(lbl),
                edgecolor="none",
            )

    ax.set_yticks([y_positions["Human"], y_positions["Model"]])
    ax.set_yticklabels(["Human", "Model"], fontsize=24)
    ax.set_xlabel("Time (s)", fontsize=26, labelpad=12)
    ax.set_title(title, fontsize=26)
    ax.tick_params(axis="x", labelsize=18)
    ax.set_xlim(t_start.min() - 0.5, t_end.max() + 0.5)
    ax.set_ylim(-0.6, 1.6)

    _LEGEND_RENAME = {"unknown": "other", "invisible": "other", "not visible": "other"}
    # Deduplicate: if multiple "other" aliases appear, show only one legend entry.
    seen_display = set()
    legend_handles = []
    for lbl in all_labels:
        display = _LEGEND_RENAME.get(lbl, lbl)
        if display in seen_display:
            continue
        seen_display.add(display)
        # Wrap long labels so the legend column stays narrow
        wrapped = textwrap.fill(display, width=16)
        legend_handles.append(
            plt.Rectangle((0, 0), 1, 1, facecolor=get_color(lbl),
                           edgecolor="black", linewidth=0.5, label=wrapped)
        )
    # tight_layout first so the legend (placed outside axes) doesn't compress the bars
    fig.tight_layout()
    ax.legend(
        handles=legend_handles,
        title="Label",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=20,
        ncol=2,
        title_fontsize=22,
    )
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_toy_holder_timeline(df: pd.DataFrame, model_cat: pd.Series,
                             human_cat: pd.Series, out_path: str):
    """
    Gantt-chart style toy-holder timeline.
    Two horizontal rows: Model (y=1) and Human (y=0).
    Each contiguous run of the same category is a filled horizontal bar.
    """
    t_start = df[COLUMN_MAP["time_start"]]
    t_end = df[COLUMN_MAP["time_end"]]

    categories = ["none", "child only", "parent only", "both", "unknown"]
    color_map = {
        "none": COLORS["pale_yellow"],
        "child only": COLORS["teal"],
        "parent only": COLORS["purple"],
        "both": COLORS["orange"],
        "unknown": "#cccccc",
    }

    fig, ax = plt.subplots(figsize=(14, 3))
    bar_height = 0.6
    y_positions = {"Model": 1.0, "Human": 0.0}

    for row_label, cat_series in [("Model", model_cat), ("Human", human_cat)]:
        y = y_positions[row_label]
        spans = spans_from_series(t_start, t_end, cat_series)
        for (t0, t1, lbl) in spans:
            ax.broken_barh(
                [(t0, t1 - t0)],
                (y - bar_height / 2, bar_height),
                facecolors=color_map.get(lbl, "#999999"),
                edgecolor="none",
            )

    ax.set_yticks([y_positions["Human"], y_positions["Model"]])
    ax.set_yticklabels(["Human", "Model"])
    ax.set_xlabel("Time (s)")
    ax.set_title("Who Is Holding a Toy Over Time (Model vs Human)")
    ax.set_xlim(t_start.min() - 0.5, t_end.max() + 0.5)
    ax.set_ylim(-0.6, 1.6)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=color_map[c],
                       edgecolor="black", linewidth=0.5, label=c)
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
      - one bar per behavior: strict accuracy (excluding unknowns)
    """
    behaviors = metrics_df["behavior"].tolist()
    x = np.arange(len(behaviors))
    width = 0.5

    acc_strict = metrics_df["accuracy_strict"].values

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x, acc_strict, width, color=COLORS["teal"], label="Strict accuracy (no unknowns)")

    ax.set_xticks(x)
    ax.set_xticklabels(behaviors, rotation=45, ha="right")
    ax.set_ylabel("Proportion")
    ax.set_xlabel("Behavior")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Per-Behavior Accuracy (Model vs Human)")
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
    ap.add_argument(
        "--human_vocab_json",
        default=None,
        help=(
            "JSON file listing possible labels per column for the human annotation scheme "
            "(produced by eaf_to_csv_ICDL*.py alongside each human CSV). "
            "If omitted, the visualizer looks for a sibling *_vocab.json next to --csv_human; "
            "if that is also absent, OOV filtering falls back to observed human labels only."
        ),
    )
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    df = load_and_merge(args.csv_model, args.csv_human)

    # Load human vocab (for OOV filtering in accuracy computation)
    human_vocab = None
    vocab_path = args.human_vocab_json
    if vocab_path is None:
        # Auto-detect sibling *_vocab.json
        candidate = args.csv_human.replace(".csv", "_vocab.json")
        if os.path.isfile(candidate):
            vocab_path = candidate
    if vocab_path is not None:
        with open(vocab_path) as f:
            human_vocab = json.load(f)
        print(f"[info] Loaded human vocab from {vocab_path}")
    else:
        print("[info] No human vocab JSON found; OOV filtering uses observed labels only.")

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
        # Skip behaviors where the human column is absent from the merged DataFrame
        # (e.g. the human CSV was produced with a reduced preset that omits some tasks).
        if h_col not in df.columns:
            print(f"[info] Skipping '{name}': column '{h_col}' not in human CSV.")
            continue
        # Skip behaviors where human annotations are essentially absent (e.g.
        # current_toy for ENS subjects where the tier was left empty).
        h_series = df[h_col].astype(str)
        n_valid_human = ((h_series != "unknown") & (h_series != "nan")).sum()
        if n_valid_human < 10:
            print(f"[info] Skipping '{name}': only {n_valid_human} non-unknown human values.")
            continue

        acc_strict, agree_all, kappa = compute_accuracy_and_agreement(
            df, m_col, h_col, human_vocab=human_vocab)
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

    # Plot per-behavior accuracy vs agreement (skip if nothing to plot)
    if metrics_df.empty:
        print("[info] No behaviors with sufficient human annotations; skipping accuracy plot.")
    else:
        plot_accuracy_vs_agreement(
            metrics_df,
            out_path=os.path.join(args.out_dir, "accuracy_vs_agreement.png"),
        )

    # --- Toy possession breakdown (model only) ---
    toy_cat_model = derive_toy_holder_category(
        df[COLUMN_MAP["parent_holding_toy_model"]],
        df[COLUMN_MAP["child_holding_toy_model"]],
    )
    plot_toy_possession_pie_model(
        toy_cat_model,
        out_path=os.path.join(args.out_dir, "toy_possession_breakdown_model.png"),
    )

    # --- Child vs Parent hand action mix (side-by-side plots) ---
    if (COLUMN_MAP["child_hand_action_human"] in df.columns and
            COLUMN_MAP["adult_hand_action_human"] in df.columns):
        plot_child_vs_parent_hand_mix(
            df,
            out_path=os.path.join(args.out_dir, "hand_action_mix_child_parent.png"),
        )
    else:
        print("[info] Skipping hand action mix plot: human columns absent.")

    # --- Who is holding a toy over time (combined model + human) ---
    if (COLUMN_MAP["parent_holding_toy_human"] in df.columns and
            COLUMN_MAP["child_holding_toy_human"] in df.columns):
        toy_cat_human = derive_toy_holder_category(
            df[COLUMN_MAP["parent_holding_toy_human"]],
            df[COLUMN_MAP["child_holding_toy_human"]],
        )
        plot_toy_holder_timeline(
            df,
            toy_cat_model,
            toy_cat_human,
            out_path=os.path.join(args.out_dir, "toy_holder_over_time.png"),
        )
    else:
        print("[info] Skipping toy holder timeline: human holding columns absent.")

    # --- Toy contact & child activity ---
    if (COLUMN_MAP["child_pose_human"] in df.columns and
            COLUMN_MAP["child_holding_toy_human"] in df.columns):
        plot_toy_contact_vs_child_activity(
            df,
            out_path=os.path.join(args.out_dir, "toy_contact_vs_child_activity.png"),
        )
    else:
        print("[info] Skipping toy contact vs child activity: human columns absent.")

    # --- Behavior timecourse (Gantt-style) ---
    if COLUMN_MAP["child_hand_action_human"] in df.columns:
        plot_behavior_timecourse(
            df,
            model_col=COLUMN_MAP["child_hand_action_model"],
            human_col=COLUMN_MAP["child_hand_action_human"],
            title="Child Hand Action Over Time (Model vs Human)",
            out_path=os.path.join(args.out_dir, "timecourse_child_hand_action.png"),
        )
    if COLUMN_MAP["adult_hand_action_human"] in df.columns:
        plot_behavior_timecourse(
            df,
            model_col=COLUMN_MAP["adult_hand_action_model"],
            human_col=COLUMN_MAP["adult_hand_action_human"],
            title="Adult Hand Action Over Time (Model vs Human)",
            out_path=os.path.join(args.out_dir, "timecourse_adult_hand_action.png"),
        )
    if COLUMN_MAP["child_pose_human"] in df.columns:
        plot_behavior_timecourse(
            df,
            model_col=COLUMN_MAP["child_pose_model"],
            human_col=COLUMN_MAP["child_pose_human"],
            title="Child Pose Over Time (Model vs Human)",
            out_path=os.path.join(args.out_dir, "timecourse_child_pose.png"),
        )

    # --- Transition frequency bars (child/parent hand actions & poses) ---
    # Child hand actions
    if COLUMN_MAP["child_hand_action_human"] in df.columns:
        ch_counts_m = compute_transition_counts(df[COLUMN_MAP["child_hand_action_model"]])
        ch_counts_h = compute_transition_counts(df[COLUMN_MAP["child_hand_action_human"]])
        plot_transition_bars(
            ch_counts_m,
            ch_counts_h,
            title="Child Hand Action Transitions (Model vs Human)",
            out_path=os.path.join(args.out_dir, "transitions_child_hand_action_bar.png"),
        )

    # Parent hand actions
    if COLUMN_MAP["adult_hand_action_human"] in df.columns:
        ad_counts_m = compute_transition_counts(df[COLUMN_MAP["adult_hand_action_model"]])
        ad_counts_h = compute_transition_counts(df[COLUMN_MAP["adult_hand_action_human"]])
        plot_transition_bars(
            ad_counts_m,
            ad_counts_h,
            title="Parent Hand Action Transitions (Model vs Human)",
            out_path=os.path.join(args.out_dir, "transitions_parent_hand_action_bar.png"),
        )

    # Child pose transitions
    if COLUMN_MAP["child_pose_human"] in df.columns:
        cp_counts_m = compute_transition_counts(df[COLUMN_MAP["child_pose_model"]])
        cp_counts_h = compute_transition_counts(df[COLUMN_MAP["child_pose_human"]])
        plot_transition_bars(
            cp_counts_m,
            cp_counts_h,
            title="Child Pose Transitions (Model vs Human)",
            out_path=os.path.join(args.out_dir, "transitions_child_pose_bar.png"),
        )

    # Parent pose transitions
    if COLUMN_MAP["adult_pose_human"] in df.columns:
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
