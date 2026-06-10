"""
eaf_to_csv_ICDL_ENS.py

Converts ELAN .eaf annotations (ENS cohort) to a human-annotation CSV that
matches the model output schema.

ENS cohort annotation style differs from Miami: hand actions are coded as
single multiclass tiers (C_hand_action, P_hand_action) with direct label
values, rather than binary yes/no tiers per action.

Usage:
    python eaf_to_csv_ICDL_ENS.py \
        --eaf data/annotations/4_vid.eaf \
        --template_csv /path/to/model_output.csv \
        --out_csv data/results_4/4_side_human.csv

No EAF time offset is applied by default (ENS videos are not affected by the
YB finalized-video timing shift that the Miami cohort has).
"""

import argparse
import csv
import json
import os
import pandas as pd
import pympi  # pip install pympi-ling

DEFAULT_EAF = "../data/annotations/4_vid.eaf"
DEFAULT_TEMPLATE_CSV = "/data/Cai_gaze/Tsuji_lab_collaboration/results/video_annotation/unaliased/4_side_resolved.csv"
DEFAULT_OUT_CSV = "../data/results_4/4_side_human.csv"

# Tiers in ELAN that have multiclass labels matching the prompt options.
# For ENS, C_hand_action and P_hand_action are also multiclass tiers.
MULTICLASS_TIERS = {
    "C_proximity_behavior": "child_proximity_behavior",
    "C_pose": "child_pose",
    "P_pose": "adult_pose",
    "current_toy": "current_toy",
    # ENS-specific: hand actions as direct multiclass labels
    "C_hand_action": "child_hand_action",
    "P_hand_action": "adult_hand_action",
}

# Which child hand-actions imply child_holding_toy = yes
CHILD_HOLDING_TOY_ACTIONS = {
    "grabbing toy",
    "giving away the toy",
    "holding toy still",
    "manipulating toy",
}

# Which parent hand-actions imply parent_holding_toy = yes
PARENT_HOLDING_TOY_ACTIONS = {
    "handing toy to child",
    "taking toy from child",
    "holding toy",
    "holding toy still",
    "grabbing toy",       # appears in EAF P_hand_action
    "manipulating toy",
    "moving toy",
    "giving away the toy",
}

# -------------------------------------------------------------------
# Canonical prompt choices + alias maps (so EAF text matches model)
# -------------------------------------------------------------------

CHILD_HAND_CHOICES = [
    "pointing",
    "grabbing toy",
    "giving away the toy",
    "holding toy still",
    "manipulating toy",
    "gesturing",
    "touching adult",
    "holding paper",                               # appears in EAF C_hand_action
    "on the ground/touching some furniture/resting",   # ENS compound label kept verbatim
    "touching box/toy bag/eye-tracker components",     # ENS compound label kept verbatim
    "none",
]

CHILD_HAND_ALIASES = {
    "grasping": "holding toy still",
    "playing": "manipulating toy",
    "moving toy": "manipulating toy",
    "waving": "gesturing",
    "showing": "gesturing",
    "reaching": "gesturing",
    # delta-style sub-labels that map to the ENS compound (if a newer preset is used)
    "on the ground": "on the ground/touching some furniture/resting",
    "on furniture": "on the ground/touching some furniture/resting",
    "on some furniture": "on the ground/touching some furniture/resting",
    "resting": "on the ground/touching some furniture/resting",
    "resting on body": "on the ground/touching some furniture/resting",
    "opening a box or bag": "touching box/toy bag/eye-tracker components",
    "closing a box or bag": "touching box/toy bag/eye-tracker components",
    "touching glasses": "touching box/toy bag/eye-tracker components",
}

CHILD_PROX_CHOICES = [
    "close and facing toward adult",
    "close but facing away from adult",
    "mid distance and facing toward adult",
    "mid distance and facing away from adult",
    "far and facing toward adult",
    "far but facing away from adult",
    "not visible",
]

CHILD_PROX_ALIASES = {
    # prompt-style / descriptive
    "leaning toward parent": "close and facing toward adult",
    "close and oriented toward adult": "close and facing toward adult",

    # human CSV style with "+" (facing toward adult)
    "close + facing adult": "close and facing toward adult",
    "close + facing toward adult": "close and facing toward adult",
    "close + facing toward": "close and facing toward adult",
    "mid distance + facing adult": "mid distance and facing toward adult",
    "mid distance + facing toward adult": "mid distance and facing toward adult",
    "mid distance + facing toward": "mid distance and facing toward adult",
    "far + facing adult": "far and facing toward adult",
    "far + facing toward adult": "far and facing toward adult",
    "far + facing toward": "far and facing toward adult",

    # human CSV style with "+" (facing away from adult)
    "close + facing away": "close but facing away from adult",
    "mid distance + facing away": "mid distance and facing away from adult",
    "far + facing away": "far but facing away from adult",

    # small textual variations (toward)
    "mid distance and facing adult": "mid distance and facing toward adult",
    "far and facing adult": "far and facing toward adult",
    "close and facing adult": "close and facing toward adult",

    # small textual variations (away)
    "close and facing away": "close but facing away from adult",
    "mid distance and facing away": "mid distance and facing away from adult",
    "far and facing away": "far but facing away from adult",
}

CURRENT_TOY_CHOICES = [
    "giraffe",
    "elephant",
    "yellow toy",
    "green toy",
    "none",
]

CURRENT_TOY_ALIASES = {
    "red toy": "yellow toy",
    "not interacting": "none",
    "grey toy": "none",
    "unknown": "none",
    "no toy present": "none",
    "green toy with a yellow flower-like shape": "green toy",
}

ADULT_HAND_CHOICES = [
    "pointing",
    "gesturing",                                   # own choice in delta (not → pointing)
    "grabbing toy",                                # appears in EAF P_hand_action + delta choice
    "holding toy still",                           # delta canonical (replaces "holding toy")
    "handing toy to child",
    "taking toy from child",
    "holding paper",
    "moving toy",
    "manipulating toy",                            # appears in EAF P_hand_action + delta choice
    "touching child",
    "waving",
    "on the ground/touching some furniture/resting",   # ENS compound label kept verbatim
    "touching box/toy bag/eye-tracker components",     # ENS compound label kept verbatim
    "none",
]

ADULT_HAND_ALIASES = {
    "showing": "holding toy still",               # was "holding toy"; delta uses "holding toy still"
    "holding toy": "holding toy still",            # old canonical → new canonical
    "holding book": "holding paper",
    "holding object": "holding paper",
    "illustrating toy": "moving toy",
    "patting": "touching child",
    "guiding": "touching child",
    "holding child": "touching child",
    "directing": "pointing",
    # Note: "gesturing" is NOT mapped to "pointing" anymore — it is its own delta choice.
    "giving away the toy": "handing toy to child",
    "operating toy": "manipulating toy",
    # delta-style sub-labels that map to ENS compound (if a newer preset is used)
    "on the ground": "on the ground/touching some furniture/resting",
    "on furniture": "on the ground/touching some furniture/resting",
    "on some furniture": "on the ground/touching some furniture/resting",
    "resting": "on the ground/touching some furniture/resting",
    "resting on body": "on the ground/touching some furniture/resting",
    "opening a box or bag": "touching box/toy bag/eye-tracker components",
    "closing a box or bag": "touching box/toy bag/eye-tracker components",
    "touching glasses": "touching box/toy bag/eye-tracker components",
}

CHILD_POSE_CHOICES = [
    "sitting (kneeling)",   # delta canonical; matches EAF label exactly
    "standing still",
    "walking",
    "crawling",
    "turning around",
    "bending over",         # delta choice (not seen in current EAF, but possible)
    "crouching",            # delta choice
    "lying on the floor",   # delta choice (replaces "lying on ground")
    "invisible",            # delta canonical (replaces "not visible"; not seen in EAF in practice)
]

CHILD_POSE_ALIASES = {
    "kneeling": "sitting (kneeling)",
    "sitting": "sitting (kneeling)",
    "sitting still": "sitting (kneeling)",
    "lying on ground": "lying on the floor",
    "lying down": "lying on the floor",
    "squatting": "crouching",
    "not visible": "invisible",
}

ADULT_POSE_CHOICES = [
    "sitting (kneeling)",   # delta canonical; matches EAF label exactly
    "standing still",
    "walking",
    "crawling",
    "turning around",
    "bending over",         # delta choice
    "crouching",            # delta choice
    "invisible",            # delta canonical (replaces "not visible")
]

ADULT_POSE_ALIASES = {
    "kneeling": "sitting (kneeling)",
    "sitting": "sitting (kneeling)",
    "sitting still": "sitting (kneeling)",
    "squatting": "crouching",
    "not visible": "invisible",
}


def normalize_label(col, value):
    """
    Normalize EAF / human annotation labels so they match
    the model's prompt choices (for accuracy comparison).
    """
    if value is None:
        return "unknown"

    s = str(value).strip().lower()

    if s == "" or s == "nan":
        return "unknown"
    if s == "unknown":
        return "unknown"

    s = s.replace("\t", " ")
    s = s.replace("towards", "toward")
    s = " ".join(s.split())

    # ---------------- Binary flags (yes/no) ---------------- #
    if col in ("toy_in_environment", "parent_holding_toy", "child_holding_toy"):
        if s.startswith("y"):
            return "yes"
        if s.startswith("n"):
            return "no"
        if s == "none":
            return "no"
        return s

    # ---------------- Child hand action ---------------- #
    if col == "child_hand_action":
        if s in CHILD_HAND_ALIASES:
            s = CHILD_HAND_ALIASES[s]
        if s in CHILD_HAND_CHOICES:
            return s
        return s

    # ---------------- Child proximity / orientation ---------------- #
    if col == "child_proximity_behavior":
        if s in CHILD_PROX_ALIASES:
            s = CHILD_PROX_ALIASES[s]

        if "+" in s:
            s = s.replace("+", "and")
            s = " ".join(s.split())

        if s in CHILD_PROX_ALIASES:
            s = CHILD_PROX_ALIASES[s]

        if s in CHILD_PROX_CHOICES:
            return s

        if "not visible" in s:
            return "not visible"

        if s in ("none", "no label", "na"):
            return "unknown"

        return s

    # ---------------- Current toy ---------------- #
    if col == "current_toy":
        if s in CURRENT_TOY_ALIASES:
            s = CURRENT_TOY_ALIASES[s]

        if "giraffe" in s:
            s = "giraffe"
        elif "elephant" in s:
            s = "elephant"
        elif "yellow" in s and "toy" in s:
            s = "yellow toy"
        elif "green" in s and "toy" in s:
            s = "green toy"
        elif "no toy present" in s or s == "no toy":
            s = "none"

        if s in CURRENT_TOY_CHOICES:
            return s

        if "unknown" in s:
            return "unknown"

        return s

    # ---------------- Adult hand action ---------------- #
    if col == "adult_hand_action":
        if s in ADULT_HAND_ALIASES:
            s = ADULT_HAND_ALIASES[s]
        if s in ADULT_HAND_CHOICES:
            return s
        return s

    # ---------------- Child pose ---------------- #
    if col == "child_pose":
        if s in CHILD_POSE_ALIASES:
            s = CHILD_POSE_ALIASES[s]

        if "sitting" in s and "kneel" in s:
            s = "sitting (kneeling)"
        if s == "sitting":
            s = "sitting (kneeling)"

        if s in CHILD_POSE_CHOICES:
            return s

        if "not visible" in s or s == "invisible":
            return "invisible"

        return s

    # ---------------- Adult pose ---------------- #
    if col == "adult_pose":
        if s in ADULT_POSE_ALIASES:
            s = ADULT_POSE_ALIASES[s]

        if "sitting" in s and "kneel" in s:
            s = "sitting (kneeling)"

        if s in ADULT_POSE_CHOICES:
            return s

        if "not visible" in s or s == "invisible":
            return "invisible"

        return s

    return s


def extract_multiclass_label(eaf, tier_name, start, end, min_overlap_frac=0.5):
    """
    Return the label of the annotation with the greatest overlap with [start, end],
    provided that overlap covers at least min_overlap_frac of the window duration.
    Returns 'unknown' if the tier is missing or no annotation meets the threshold.
    """
    if tier_name not in eaf.tiers:
        return "unknown"
    clip_dur = end - start
    if clip_dur <= 0:
        return "unknown"

    best_val = "unknown"
    best_overlap = 0
    for t0, t1, val in eaf.get_annotation_data_for_tier(tier_name):
        overlap = max(0, min(t1, end) - max(t0, start))
        if overlap > best_overlap:
            best_overlap = overlap
            best_val = val if val else "unknown"

    if best_overlap / clip_dur < min_overlap_frac:
        return "unknown"
    return best_val


def get_eaf_annotation_range(eaf):
    """Return (min_sec, max_sec) across behavioral tiers only (skips 'default' tier)."""
    times = []
    for tier in eaf.tiers:
        if tier == "default":
            continue
        for t0, t1, _ in eaf.get_annotation_data_for_tier(tier):
            times.extend([t0, t1])
    if not times:
        return 0.0, float("inf")
    return min(times) / 1000.0, max(times) / 1000.0


def process_time_segment(eaf, start_sec, end_sec, min_overlap_frac=0.5):
    """
    Compute the human label for a single [start_sec, end_sec] interval
    using the ELAN tiers, then normalize to prompt-style labels.
    """
    start_ms = int(start_sec * 1000)
    end_ms = int(end_sec * 1000)

    result = {
        "toy_in_environment": "no",
        "parent_holding_toy": "no",
        "child_holding_toy": "no",
        "child_hand_action": "unknown",
        "child_proximity_behavior": "unknown",
        "current_toy": "unknown",
        "adult_hand_action": "unknown",
        "child_pose": "unknown",
        "adult_pose": "unknown",
    }

    # 1) Read all multiclass tiers (including hand action tiers for ENS)
    for tier_name, outcol in MULTICLASS_TIERS.items():
        raw_val = extract_multiclass_label(
            eaf, tier_name, start_ms, end_ms,
            min_overlap_frac=min_overlap_frac,
        )
        result[outcol] = raw_val

    # 2) Derive holding/toy_in_environment flags from raw (pre-normalization) labels
    #    Raw EAF values like "manipulating toy" directly match CHILD_HOLDING_TOY_ACTIONS
    child_hand_raw = result["child_hand_action"].lower() if result["child_hand_action"] else ""
    parent_hand_raw = result["adult_hand_action"].lower() if result["adult_hand_action"] else ""

    if child_hand_raw in CHILD_HOLDING_TOY_ACTIONS:
        result["child_holding_toy"] = "yes"

    if parent_hand_raw in PARENT_HOLDING_TOY_ACTIONS:
        result["parent_holding_toy"] = "yes"

    if result["child_holding_toy"] == "yes" or result["parent_holding_toy"] == "yes":
        result["toy_in_environment"] = "yes"

    # 3) Normalize everything to canonical prompt-style labels
    for key in list(result.keys()):
        result[key] = normalize_label(key, result[key])

    return result


FIELDNAMES = [
    "video_path",
    "t_start",
    "t_end",
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


def main():
    parser = argparse.ArgumentParser(
        description="Convert ENS-cohort ELAN annotations (multiclass tier style) to model-schema CSV."
    )
    parser.add_argument(
        "--eaf",
        default=DEFAULT_EAF,
        help=f"Path to ELAN .eaf file (default: {DEFAULT_EAF})",
    )
    parser.add_argument(
        "--template_csv",
        default=DEFAULT_TEMPLATE_CSV,
        help=f"Template CSV (model output) to align times from (default: {DEFAULT_TEMPLATE_CSV})",
    )
    parser.add_argument(
        "--out_csv",
        default=DEFAULT_OUT_CSV,
        help=f"Output CSV with human annotations in model schema (default: {DEFAULT_OUT_CSV})",
    )
    parser.add_argument(
        "--min_overlap_frac",
        type=float,
        default=0.5,
        help=(
            "Minimum fraction of the clip window that an EAF annotation must "
            "cover to be accepted. Range: 0.0–1.0 (default: 0.5)."
        ),
    )
    args = parser.parse_args()

    eaf = pympi.Elan.Eaf(args.eaf)
    ann_start, ann_end = get_eaf_annotation_range(eaf)
    print(f"[info] EAF annotation range: {ann_start:.1f}s – {ann_end:.1f}s")

    template = pd.read_csv(args.template_csv)

    rows = []
    for _, row in template.iterrows():
        t0 = float(row["t_start"])
        t1 = float(row["t_end"])

        # Only process rows that fall within the EAF-annotated range
        if t0 < ann_start or t0 > ann_end:
            continue

        human = process_time_segment(eaf, t0, t1, min_overlap_frac=args.min_overlap_frac)
        human["video_path"] = row["video_path"]
        human["t_start"] = t0
        human["t_end"] = t1
        rows.append(human)

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(rows)

    print(f"[info] Wrote {len(rows)} segments to {args.out_csv}")

    # Emit a vocab JSON so the visualizer knows the full set of possible labels.
    # For ENS, hand actions are multiclass tiers; the vocab is the union of
    # CHOICES and all alias target values (the normalized outputs).
    child_vocab = sorted(set(CHILD_HAND_CHOICES) | set(CHILD_HAND_ALIASES.values()))
    adult_vocab = sorted(set(ADULT_HAND_CHOICES) | set(ADULT_HAND_ALIASES.values()))
    vocab = {
        "toy_in_environment": ["yes", "no"],
        "parent_holding_toy": ["yes", "no"],
        "child_holding_toy": ["yes", "no"],
        "child_hand_action": child_vocab,
        "adult_hand_action": adult_vocab,
        "child_proximity_behavior": CHILD_PROX_CHOICES,
        "current_toy": CURRENT_TOY_CHOICES,
        "child_pose": CHILD_POSE_CHOICES,
        "adult_pose": ADULT_POSE_CHOICES,
    }
    vocab_path = args.out_csv.replace(".csv", "_vocab.json")
    with open(vocab_path, "w") as f:
        json.dump(vocab, f, indent=2)
    print(f"[info] Wrote human vocabulary to {vocab_path}")


if __name__ == "__main__":
    main()
