"""
eaf_to_csv_ICDL_miami.py

Converts ELAN .eaf annotations (Miami cohort, binary yes/no tier style) to a
human-annotation CSV that matches the model output schema.

Runs over all three Miami subjects (27, 28, 6) in a single call, saving
per-subject output CSVs. For subject 27, both toy1 and toy2 model CSVs are
combined into one output because the EAF annotation is fully continuous.

EAF annotation ranges (from the .eaf files):
  Subject 27: 116.5s – 966.1s  (continuous; old toy1/toy2 split was artificial)
    Legacy segment comments:
      Toy 1: START_SEC = 60.0,  END_SEC = 351.0
      Toy 2: START_SEC = 569.0, END_SEC = 797.0  (5:41 to 13:17)
  Subject 28:  83.0s – 927.4s
  Subject  6: 130.1s – 1046.2s

Time offset (Miami cohort only):
  EAF annotations were coded on an older video version; the YB finalized video
  used for model inference starts a few frames later. Formula:
    eaf_lookup_time = model_time + eaf_time_offset_sec
  Per-subject offsets are loaded from data/annotations/video_timing_offsets.json.
"""

import argparse
import csv
import json
import os
import pandas as pd
import pympi  # pip install pympi-ling

# -----------------------------------------------------------------------
# Per-subject configuration
# Each entry lists one or more template_csvs (rows from the model output to
# match against the EAF).  All rows are combined into a single out_csv.
# -----------------------------------------------------------------------
SUBJECTS = [
    {
        "subject": "27",
        "eaf": "../data/annotations/27annotationscomplete.eaf",
    },
    {
        "subject": "28",
        "eaf": "../data/annotations/28annotationscomplete.eaf",
    },
    {
        "subject": "6",
        "eaf": "../data/annotations/6annotationscomplete.eaf",
    },
]

TEMPLATE_BASE = "/data/Cai_gaze/Tsuji_lab_collaboration/results/video_annotation"


def build_subject_paths(subject_cfg, resolution):
    """Return template_csvs and out_csv for a subject given a resolution tag (e.g. '3s')."""
    s = subject_cfg["subject"]
    template_csv = f"{TEMPLATE_BASE}/ICDL_{resolution}/{s}_side.csv"
    out_csv = f"../data/results_{s}/{s}_side_human_miami_{resolution}.csv"
    return [template_csv], out_csv

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
    ("C_on_furniture", "on the ground/touching some furniture/resting"),
    ("C_resting", "on the ground/touching some furniture/resting"),
    # C_touching_curtain has no model equivalent; omitted intentionally -> returns "unknown"
]

# Parent hand-action tiers: same logic as child hand tiers.
PARENT_HAND_TIERS = [
    ("P_holding_toy", "holding toy still"),
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
    "holding toy still",
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
    "on the ground/touching some furniture/resting",
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
    "holding toy still",
    "holding paper",
    "moving toy",
    "touching child",
    "resting",
    "opening a box or bag",
    "waving",
    "none",
]

ADULT_HAND_ALIASES = {
    "showing": "holding toy still",
    "holding toy": "holding toy still",
    "holding book": "holding paper",
    "holding object": "holding paper",
    "illustrating toy": "moving toy",
    "patting": "touching child",
    "guiding": "touching child",
    "directing": "pointing",
    "gesturing": "pointing",
}

CHILD_POSE_CHOICES = [
    "sitting (kneeling)",
    "standing still",
    "walking",
    "crawling",
    "turning around",
    "invisible",
]

CHILD_POSE_ALIASES = {
    "kneeling": "sitting (kneeling)",
    "sitting still": "sitting (kneeling)",
    "sitting": "sitting (kneeling)",
    "not visible": "invisible",
    "lying on ground": "crawling",   # no exact model equivalent; crawling is closest floor posture
}

ADULT_POSE_CHOICES = [
    "sitting (kneeling)",
    "standing still",
    "walking",
    "crawling",
    "turning around",
    "invisible",
]

ADULT_POSE_ALIASES = {
    "kneeling": "sitting (kneeling)",
    "sitting": "sitting (kneeling)",
    "sitting still": "sitting (kneeling)",
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
        # legacy alias: "holding toy" → "holding toy still"
        if s == "holding toy":
            s = "holding toy still"
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


def get_eaf_annotation_range(eaf):
    """Return (min_sec, max_sec) across all annotated tiers."""
    times = []
    for tier in eaf.tiers:
        for t0, t1, _ in eaf.get_annotation_data_for_tier(tier):
            times.extend([t0, t1])
    if not times:
        return 0.0, float("inf")
    return min(times) / 1000.0, max(times) / 1000.0


def process_time_segment(eaf, start_sec, end_sec, min_overlap_frac=0.5):
    """
    Compute the human label for a single [start_sec, end_sec] interval
    using the ELAN tiers, then normalize to prompt-style labels.
    The times passed here are already in EAF time (offset already applied).
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


def process_subject(subject_cfg, offset_data, args, template_csvs, out_csv):
    subject = subject_cfg["subject"]
    eaf_path = subject_cfg["eaf"]

    # Resolve EAF time offset for this subject.
    # Formula (from video_timing_offsets.json):
    #   eaf_lookup_time = model_time + eaf_time_offset_sec
    # The YB finalized video starts a few frames later than the EAF-coded video,
    # so we add a small positive offset to convert model timestamps to EAF time.
    eaf_time_offset = 0.0
    if args.eaf_time_offset is not None:
        eaf_time_offset = args.eaf_time_offset
        print(f"[subject {subject}] Using override EAF time offset: {eaf_time_offset:.4f}s")
    elif offset_data and subject in offset_data.get("subjects", {}):
        eaf_time_offset = offset_data["subjects"][subject]["eaf_time_offset_sec"]
        print(f"[subject {subject}] EAF time offset from file: {eaf_time_offset:.4f}s")
    else:
        print(f"[subject {subject}] No EAF time offset found; using 0.0s")

    eaf = pympi.Elan.Eaf(eaf_path)
    ann_start, ann_end = get_eaf_annotation_range(eaf)
    print(f"[subject {subject}] EAF annotation range: {ann_start:.1f}s – {ann_end:.1f}s")

    rows = []
    for template_path in template_csvs:
        template = pd.read_csv(template_path)
        n_before = len(rows)
        for _, row in template.iterrows():
            t0 = float(row["t_start"])
            t1 = float(row["t_end"])

            # Convert model time to EAF time for the range check
            t0_eaf = t0 + eaf_time_offset
            t1_eaf = t1 + eaf_time_offset

            # Only process rows that fall within the EAF-annotated range
            if t0_eaf < ann_start or t0_eaf > ann_end:
                continue

            human = process_time_segment(
                eaf, t0_eaf, t1_eaf,
                min_overlap_frac=args.min_overlap_frac,
            )
            # Store original model timestamps (not EAF-shifted) in the output
            human["video_path"] = row["video_path"]
            human["t_start"] = t0
            human["t_end"] = t1
            rows.append(human)

        print(f"[subject {subject}] {template_path}: {len(rows) - n_before} segments matched")

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(rows)

    print(f"[subject {subject}] Wrote {len(rows)} total segments to {out_csv}")

    vocab = {
        "toy_in_environment": ["yes", "no"],
        "parent_holding_toy": ["yes", "no"],
        "child_holding_toy": ["yes", "no"],
        "child_hand_action": [label for _, label in CHILD_HAND_TIERS],
        "adult_hand_action": [label for _, label in PARENT_HAND_TIERS],
        "child_proximity_behavior": CHILD_PROX_CHOICES,
        "current_toy": CURRENT_TOY_CHOICES,
        "child_pose": CHILD_POSE_CHOICES,
        "adult_pose": ADULT_POSE_CHOICES,
    }
    vocab_path = out_csv.replace(".csv", "_vocab.json")
    with open(vocab_path, "w") as f:
        json.dump(vocab, f, indent=2)
    print(f"[subject {subject}] Wrote human vocabulary to {vocab_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Miami-cohort ELAN annotations to model-schema CSV for all subjects."
    )
    parser.add_argument(
        "--subjects",
        nargs="*",
        default=None,
        help="Subject IDs to process (default: all). E.g. --subjects 27 6",
    )
    parser.add_argument(
        "--eaf_time_offset",
        type=float,
        default=None,
        help=(
            "Override EAF time offset (seconds) applied to all subjects. "
            "If not set, per-subject values are read from --eaf_time_offset_file. "
            "Formula: eaf_lookup_time = model_time + offset."
        ),
    )
    parser.add_argument(
        "--eaf_time_offset_file",
        default="../data/annotations/video_timing_offsets.json",
        help="JSON file with per-subject offsets (default: ../data/annotations/video_timing_offsets.json)",
    )
    parser.add_argument(
        "--resolution",
        default="3s",
        help=(
            "Time resolution tag used to locate the template CSV directory and name "
            "the output file. E.g. '3s' → template dir ICDL_3s/, output *_human_miami_3s.csv. "
            "(default: 3s)"
        ),
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

    # Load offset file once
    offset_data = None
    if os.path.exists(args.eaf_time_offset_file):
        with open(args.eaf_time_offset_file) as f:
            offset_data = json.load(f)
    else:
        print(f"[warn] Offset file not found: {args.eaf_time_offset_file}; using 0.0s for all subjects")

    # Filter subjects if requested
    subjects_to_run = SUBJECTS
    if args.subjects:
        subjects_to_run = [s for s in SUBJECTS if s["subject"] in args.subjects]
        if not subjects_to_run:
            print(f"No matching subjects found for: {args.subjects}")
            return

    for subject_cfg in subjects_to_run:
        template_csvs, out_csv = build_subject_paths(subject_cfg, args.resolution)
        process_subject(subject_cfg, offset_data, args, template_csvs, out_csv)


if __name__ == "__main__":
    main()
