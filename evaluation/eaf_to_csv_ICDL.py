import argparse
import csv
import json
import os
import pandas as pd
import pympi  # pip install pympi-ling

DEFAULT_EAF = "../data/27annotationscomplete.eaf"
DEFAULT_TEMPLATE_CSV = "../data/results_27/27_side_toy2.csv"
DEFAULT_OUT_CSV = "../data/results_27/27_side_human_toy2.csv"

# Toy 1: START_SEC = 60.0, END_SEC = 351.0
# Toy 2: 5:41 to 8:41 = 341 to 521 seconds 
START_SEC = 569.0
END_SEC = 797.0

# Tiers in ELAN that have multiclass labels matching the prompt options
MULTICLASS_TIERS = {
    # NOTE: capital C, matches TIER_ID="C_proximity_behavior" in the .eaf
    "C_proximity_behavior": "child_proximity_behavior",
    "C_pose": "child_pose",
    "P_pose": "adult_pose",
    "current_toy": "current_toy",
}

# Child hand-action tiers: each tier is a yes/no flag, value 'yes' means that label is active.
# Listed in priority order (first match wins).
CHILD_HAND_TIERS = [
    ("C_manipulating_toy", "manipulating toy"),
    ("C_holding_toy", "holding toy still"),
    ("C_touching_adult", "touching adult"),
    ("C_on_furniture", "on some furniture"),
    ("C_resting", "resting"),
    # C_touching_curtain has no model equivalent; omitted intentionally -> returns "unknown"
]

# Parent hand-action tiers: same logic as child hand tiers.
PARENT_HAND_TIERS = [
    ("P_holding_toy", "holding toy"),
    ("P_manipulating_toy", "manipulating toy"),
    ("P_opening_box_or_bag", "opening a box or bag"),
    ("P_pointing_toy", "pointing"),
    ("P_touching_child", "touching child"),
    # P_closing_box_or_bag exists but has no annotations in this file, so we skip it for now
]

# Which child hand-actions imply child_holding_toy = yes
CHILD_HOLDING_TOY_ACTIONS = {
    "grabbing toy",        # if you add these tiers later
    "giving away the toy",
    "holding toy still",
    "manipulating toy",
}

# Which parent hand-actions imply parent_holding_toy = yes
PARENT_HOLDING_TOY_ACTIONS = {
    "handing toy to child",  # if you add this tier later
    "taking toy from child",
    "holding toy",
    "manipulating toy",      # parent manipulating toy is still toy contact
    "moving toy",            # if you add a P_moving_toy tier later
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
    "on the ground",
    "on some furniture",
    "resting",
    "none",
]

CHILD_HAND_ALIASES = {
    "grasping": "holding toy still",
    "playing": "manipulating toy",
    "moving toy": "manipulating toy",
    "waving": "gesturing",
    "showing": "gesturing",
    "reaching": "gesturing",
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
    "mid distance + facing adult": "mid distance and facing toward adult",
    "mid distance + facing toward adult": "mid distance and facing toward adult",
    "far + facing adult": "far and facing toward adult",
    "far + facing toward adult": "far and facing toward adult",

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
    "handing toy to child",
    "taking toy from child",
    "holding toy",
    "holding paper",
    "moving toy",
    "touching child",
    "resting",
    "opening a box or bag",
    "waving",
    "none",
]

ADULT_HAND_ALIASES = {
    "showing": "holding toy",
    "holding book": "holding paper",
    "holding object": "holding paper",
    "illustrating toy": "moving toy",
    "patting": "touching child",
    "guiding": "touching child",
    "directing": "pointing",
    "gesturing": "pointing",
}

CHILD_POSE_CHOICES = [
    "sitting still",
    "standing still",
    "walking",
    "crawling",
    "turning around",
    "not visible",
]

CHILD_POSE_ALIASES = {
    "kneeling": "sitting still",
    "sitting (kneeling)": "sitting still",
    "sitting": "sitting still",
    "lying on ground": "crawling",   # no exact model equivalent; crawling is closest floor posture
}

ADULT_POSE_CHOICES = [
    "sitting",
    "standing still",
    "walking",
    "crawling",
    "turning around",
    "not visible",
]

ADULT_POSE_ALIASES = {
    "kneeling": "sitting",
    "sitting (kneeling)": "sitting",
}


def normalize_label(col, value):
    """
    Normalize EAF / human annotation labels so they match
    the model's prompt choices (for accuracy comparison).
    """

    if value is None:
        return "unknown"

    s = str(value).strip().lower()

    # treat blank / nan as unknown (but NOT 'none', which is a real label)
    if s == "" or s == "nan":
        return "unknown"
    if s == "unknown":
        return "unknown"

    # generic cleanup
    s = s.replace("\t", " ")
    s = s.replace("towards", "toward")   # unify toward/towards
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

        # normalize '+' to 'and'
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

        # heuristic collapsing to canonical names
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
            s = "sitting still"
        if s == "sitting":
            s = "sitting still"

        if s in CHILD_POSE_CHOICES:
            return s

        if "not visible" in s:
            return "not visible"

        return s

    # ---------------- Adult pose ---------------- #
    if col == "adult_pose":
        if s in ADULT_POSE_ALIASES:
            s = ADULT_POSE_ALIASES[s]

        if "sitting" in s and "kneel" in s:
            s = "sitting"

        if s in ADULT_POSE_CHOICES:
            return s

        if "not visible" in s:
            return "not visible"

        return s

    # ---------------- Default fallback ---------------- #
    return s


def extract_first_matching_action(eaf, tiers_actions, start, end,
                                   min_overlap_frac=0.5):
    """
    For a given time window [start, end] in ms, scan all binary (yes/no) tiers and
    return the label of the tier whose 'yes' annotation has the greatest overlap with
    the window, provided that overlap covers at least min_overlap_frac of the window
    duration. Ties among tiers are broken by priority order (earlier in tiers_actions
    wins). Returns 'unknown' if no annotation meets the threshold.
    """
    clip_dur = end - start
    if clip_dur <= 0:
        return "unknown"

    best_label = "unknown"
    best_overlap = 0

    for tier_name, label in tiers_actions:
        if tier_name not in eaf.tiers:
            continue
        for t0, t1, val in eaf.get_annotation_data_for_tier(tier_name):
            if not (val and str(val).lower() == "yes"):
                continue
            overlap = max(0, min(t1, end) - max(t0, start))
            # Strictly greater: tier priority (order in list) breaks ties
            if overlap > best_overlap:
                best_overlap = overlap
                best_label = label

    if best_overlap / clip_dur < min_overlap_frac:
        return "unknown"
    return best_label


def extract_multiclass_label(eaf, tier_name, start, end, min_overlap_frac=0.5):
    """
    For tiers that directly store the category label (e.g., 'close + facing adult').
    Return the label of the annotation with the greatest overlap with [start, end],
    provided that overlap covers at least min_overlap_frac of the window duration.
    Returns 'unknown' if no annotation meets the threshold.
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


def process_time_segment(eaf, start_sec, end_sec, min_overlap_frac=0.5):
    """
    Compute the human label for a single [start_sec, end_sec] interval
    using the ELAN tiers, then normalize to prompt-style labels.
    min_overlap_frac: minimum fraction of the clip that an annotation must
    cover to be accepted (passed through to extract functions).
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

    # 1) Hand actions from yes/no tiers
    child_hand_label = extract_first_matching_action(
        eaf, CHILD_HAND_TIERS, start_ms, end_ms,
        min_overlap_frac=min_overlap_frac,
    )
    result["child_hand_action"] = child_hand_label

    parent_hand_label = extract_first_matching_action(
        eaf, PARENT_HAND_TIERS, start_ms, end_ms,
        min_overlap_frac=min_overlap_frac,
    )
    result["adult_hand_action"] = parent_hand_label

    # 2) Multiclass tiers: directly read the label
    for tier_name, outcol in MULTICLASS_TIERS.items():
        raw_val = extract_multiclass_label(
            eaf, tier_name, start_ms, end_ms,
            min_overlap_frac=min_overlap_frac,
        )
        result[outcol] = raw_val

    # 3) Derive holding/toy_in_environment flags
    if child_hand_label in CHILD_HOLDING_TOY_ACTIONS:
        result["child_holding_toy"] = "yes"

    if parent_hand_label in PARENT_HOLDING_TOY_ACTIONS:
        result["parent_holding_toy"] = "yes"

    if result["child_holding_toy"] == "yes" or result["parent_holding_toy"] == "yes":
        result["toy_in_environment"] = "yes"

    # 4) Normalize everything to canonical prompt-style labels
    for key in list(result.keys()):
        result[key] = normalize_label(key, result[key])

    return result


def main():
    parser = argparse.ArgumentParser()
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
        "--eaf_time_offset",
        type=float,
        default=None,
        help=(
            "Seconds to add to model time when looking up EAF annotations. "
            "Compensates for the YB finalized video starting a few frames later "
            "than the original aligned-audio video used for EAF coding. "
            "Use the values in data/annotations/video_timing_offsets.json. "
            "If not supplied, falls back to --eaf_time_offset_file + --subject."
        ),
    )
    parser.add_argument(
        "--eaf_time_offset_file",
        default="../data/annotations/video_timing_offsets.json",
        help="JSON file with per-subject offsets (default: ../data/annotations/video_timing_offsets.json)",
    )
    parser.add_argument(
        "--subject",
        type=str,
        default=None,
        help="Subject ID to look up in --eaf_time_offset_file (e.g. '27')",
    )
    parser.add_argument(
        "--min_overlap_frac",
        type=float,
        default=0.5,
        help=(
            "Minimum fraction of the clip window that an EAF annotation must "
            "cover to be accepted. If the best-matching annotation covers less "
            "than this fraction, the segment is labelled 'unknown'. "
            "Range: 0.0–1.0 (default: 0.5 = at least 50%% of the clip)."
        ),
    )
    args = parser.parse_args()

    # Resolve EAF time offset
    eaf_time_offset = 0.0
    if args.eaf_time_offset is not None:
        eaf_time_offset = args.eaf_time_offset
    elif args.subject is not None:
        with open(args.eaf_time_offset_file) as f:
            offset_data = json.load(f)
        if args.subject not in offset_data["subjects"]:
            raise ValueError(
                f"Subject '{args.subject}' not found in {args.eaf_time_offset_file}. "
                f"Available: {list(offset_data['subjects'].keys())}"
            )
        eaf_time_offset = offset_data["subjects"][args.subject]["eaf_time_offset_sec"]
        print(f"[info] Using EAF time offset {eaf_time_offset:.4f}s for subject {args.subject}")
    else:
        print("[info] No EAF time offset specified; using 0.0s (no correction).")

    # Load ELAN and template
    eaf = pympi.Elan.Eaf(args.eaf)
    template = pd.read_csv(args.template_csv)

    rows = []
    for _, row in template.iterrows():
        t0 = float(row["t_start"])
        t1 = float(row["t_end"])

        # Only keep the segment within [START_SEC, END_SEC] (model/template time)
        if t0 < START_SEC or t0 > END_SEC:
            continue

        # Look up EAF at the shifted time to account for different video starts
        human = process_time_segment(
            eaf, t0 + eaf_time_offset, t1 + eaf_time_offset,
            min_overlap_frac=args.min_overlap_frac,
        )
        human["video_path"] = row["video_path"]
        human["t_start"] = t0
        human["t_end"] = t1
        rows.append(human)

    fieldnames = [
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

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} segments to {args.out_csv}")


if __name__ == "__main__":
    main()