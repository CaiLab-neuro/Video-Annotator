#!/usr/bin/env python3
"""
resolve_unknowns.py — Interactively review unknown annotation labels,
propose new aliases, and patch one or more annotation CSVs.

Single-run usage
----------------
    python resolve_unknowns.py \\
        --unknowns  data/results/output_unknowns.jsonl \\
        --csv       data/results/output.csv \\
        [--prompts  prompts/presets_short.json]

Multi-run usage (unknowns pooled, aliases defined once for all files)
----------------------------------------------------------------------
    python resolve_unknowns.py \\
        --unknowns  run1_unknowns.jsonl run2_unknowns.jsonl \\
        --csv       run1.csv            run2.csv \\
        [--prompts  prompts/presets_short.json]

--unknowns and --csv must be the same length and paired by position.

Output location — choose one:
  --out_dir PATH       All outputs go into PATH/ with automatic names.
  --out_csv FILE ...   Explicit resolved-CSV path(s) (one per --csv input).
                       Other outputs land in the same directory.
  (neither)            Outputs land alongside each input CSV.

Auto-named outputs (one pair per input, never overwrites originals)
-------------------------------------------------------------------
    <out_csv_stem>_unknowns_remaining.jsonl   entries still unresolved
    <out_dir>/<prompts_stem>_updated.json     prompts with new aliases merged in
                                              (only written when --prompts is given)
    <out_dir>/<prompts_stem>_delta.json       prompts with only tasks that gained new
                                              choices — use for re-running annotation
                                              (only written when new choices are defined)
"""

import argparse
import csv
import json
import os
from collections import defaultdict


# ---------------------------------------------------------------------------
# Normalisation (mirrors run_prompt_presets.py — keep in sync)
# ---------------------------------------------------------------------------

def normalize_to_choices(raw_text, choices, aliases=None):
    """Map free-form model output to one closed-set choice, or 'unknown'."""
    if not raw_text:
        return "unknown"
    s = raw_text.strip().lower()

    # Step 1 — Exact match
    if s in choices:
        return s

    # Step 2 — Alias-based mapping (user-defined; first match wins)
    if aliases:
        for k, v in aliases.items():
            k_norm = k.strip().lower()
            v_norm = v.strip().lower()
            if not v_norm or v_norm not in choices:
                continue
            if s == k_norm or k_norm in s:
                return v_norm

    # Step 3 — Word-based match (only if unambiguous)
    words = s.split()
    word_matches = [c for c in choices if c in words]
    if len(word_matches) == 1:
        return word_matches[0]

    # Step 4 — Prefix match (only if unambiguous)
    prefix_matches = [c for c in choices if s and (c.startswith(s) or s.startswith(c))]
    if len(prefix_matches) == 1:
        return prefix_matches[0]

    # Step 5 — Yes/no special handling
    if set(choices) == {"yes", "no"}:
        if any(w in s for w in ["yes", "yeah", "ya", "correct", "true", "is"]):
            return "yes"
        if any(w in s for w in ["no", "nope", "not", "false", "isn"]):
            return "no"

    # Step 6 — Substring match (only if unambiguous)
    substr_matches = [c for c in choices if c in s]
    if len(substr_matches) == 1:
        return substr_matches[0]

    return "unknown"


def hint_matches(raw_text, choices):
    """Return (word_matches, prefix_matches, substr_matches) for display hints."""
    s = raw_text.strip().lower()
    words = s.split()
    word_m   = [c for c in choices if c in words]
    prefix_m = [c for c in choices if s and (c.startswith(s) or s.startswith(c))]
    substr_m = [c for c in choices if c in s]
    return word_m, prefix_m, substr_m


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_unknowns(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


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


def load_existing_aliases(prompts_path):
    """Return {task: {alias_key: choice}} from a prompts JSON file."""
    if not prompts_path or not os.path.exists(prompts_path):
        return {}
    with open(prompts_path) as f:
        bank = json.load(f)
    return {
        p["task"]: dict(p.get("aliases") or {})
        for p in bank.get("presets", [])
        if p.get("task")
    }


def load_full_presets(prompts_path):
    """Return {task: preset_dict} for all presets in a prompts JSON file."""
    if not prompts_path or not os.path.exists(prompts_path):
        return {}
    with open(prompts_path) as f:
        bank = json.load(f)
    return {p["task"]: p for p in bank.get("presets", []) if p.get("task")}


def save_updated_prompts(prompts_path, new_aliases, out_path):
    with open(prompts_path) as f:
        bank = json.load(f)
    for preset in bank.get("presets", []):
        task = preset.get("task")
        if task in new_aliases and new_aliases[task]:
            existing = preset.get("aliases") or {}
            existing.update(new_aliases[task])
            preset["aliases"] = existing
    with open(out_path, "w") as f:
        json.dump(bank, f, indent=2)
    print(f"  Updated prompts written to: {out_path}")


def save_delta_prompts(prompts_path, full_presets_by_task, new_choices_by_task,
                       new_aliases_by_task, out_path):
    """Write a delta prompts JSON containing only tasks that gained new choices."""
    import datetime

    with open(prompts_path) as f:
        bank = json.load(f)

    delta_presets = []
    for task, new_clist in sorted(new_choices_by_task.items()):
        if task not in full_presets_by_task:
            continue
        preset = dict(full_presets_by_task[task])          # shallow copy
        preset["choices"] = list(preset.get("choices", [])) + new_clist
        if task in new_aliases_by_task and new_aliases_by_task[task]:
            existing = dict(preset.get("aliases") or {})
            existing.update(new_aliases_by_task[task])
            preset["aliases"] = existing
        delta_presets.append(preset)

    # Build updated meta that reflects this file's provenance
    orig_meta = bank.get("meta") or {}
    orig_name = orig_meta.get("name", os.path.splitext(os.path.basename(prompts_path))[0])
    tasks_updated = sorted(new_choices_by_task.keys())
    delta_meta = {
        **orig_meta,
        "name":        orig_name + "_delta",
        "created_utc": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "derived_from": orig_name,
        "notes": (
            f"Delta of '{orig_name}'. Contains only tasks with new choices added via "
            f"resolve_unknowns.py: {', '.join(tasks_updated)}."
        ),
    }

    delta_bank = {
        **{k: v for k, v in bank.items() if k not in ("meta", "presets")},
        "meta":    delta_meta,
        "presets": delta_presets,
    }
    with open(out_path, "w") as f:
        json.dump(delta_bank, f, indent=2)
    print(f"  Delta prompts written to:   {out_path}")


# ---------------------------------------------------------------------------
# Phase 1 — Interactive alias collection (pooled across all source files)
# ---------------------------------------------------------------------------

def phase1_collect_aliases(all_unknowns, existing_aliases_by_task):
    """
    Group all unknown entries by (task, raw_text) regardless of source file,
    rank by combined frequency, and prompt for alias assignments.

    Returns:
        new_aliases_by_task  — {task: {raw_text: choice}}
        new_choices_by_task  — {task: [newly added choice, ...]}
    """
    counts = defaultdict(int)
    choices_by_task = {}
    for u in all_unknowns:
        key = (u["task"], u["raw"])
        counts[key] += 1
        choices_by_task[u["task"]] = list(u["choices"])   # mutable copy per task

    ranked = sorted(counts.items(), key=lambda x: -x[1])
    total  = len(ranked)

    print(f"\n{'═'*62}")
    print(f"  PHASE 1 OF 3 — Review Unknown Labels")
    print(f"  {total} unique (task, raw_text) pair(s) across {len(all_unknowns)} "
          f"entr{'y' if len(all_unknowns)==1 else 'ies'}.")
    print(f"{'═'*62}")
    print("  Enter a choice number, the choice text, 'n' to add a new choice,")
    print("  's' to skip, 'f <N>' to skip all remaining with fewer than N occurrences,")
    print("  or 'q' to stop.\n")

    new_aliases  = defaultdict(dict)
    new_choices  = defaultdict(list)   # task -> [new choice strings]
    min_freq     = 1                   # updated interactively via 'f <N>'

    for i, ((task, raw), count) in enumerate(ranked):
        if count < min_freq:
            remaining = total - i
            print(f"  Auto-skipping {remaining} remaining pair(s) "
                  f"(all have fewer than {min_freq} occurrence(s)).\n")
            break

        choices = choices_by_task[task]
        word_m, prefix_m, substr_m = hint_matches(raw, choices)
        all_hints = set(word_m) | set(prefix_m) | set(substr_m)

        print(f"[{i+1}/{total}]  task: {task}   ×{count} occurrence{'s' if count > 1 else ''}")
        print(f"  raw : \"{raw}\"")
        if len(all_hints) > 1:
            print(f"  note: ambiguous — would have matched: "
                  f"{', '.join(f'{chr(34)}{h}{chr(34)}' for h in sorted(all_hints))}")
        elif len(all_hints) == 1:
            print(f"  note: single automatic match → \"{next(iter(all_hints))}\"")

        print(f"\n  Choices:")
        for j, c in enumerate(choices, 1):
            print(f"    {j:>2}. {c}")
        print()

        while True:
            try:
                ans = input("  > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nInterrupted.")
                return dict(new_aliases), dict(new_choices)

            if ans.lower() == "q":
                print("  Stopping phase 1 early.")
                return dict(new_aliases), dict(new_choices)

            if ans.lower() in ("s", ""):
                print("  Skipped.\n")
                break

            # 'f <N>' — set a frequency floor; auto-skip all remaining below it
            if ans.lower().startswith("f"):
                parts = ans.split()
                if len(parts) == 2 and parts[1].isdigit():
                    min_freq = int(parts[1])
                    print(f"  Frequency floor set to {min_freq}. "
                          f"Pairs with fewer occurrences will be auto-skipped.\n")
                    break
                else:
                    print("  Usage: f <N>  e.g. 'f 3' to skip pairs with fewer than 3 occurrences.")
                    continue

            # Add new choice
            if ans.lower() == "n":
                try:
                    new_name = input("  New option name: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nInterrupted.")
                    return dict(new_aliases), dict(new_choices)
                if not new_name:
                    print("  Empty name, skipped.\n")
                    break
                new_name_norm = new_name.strip().lower()
                if new_name_norm in choices:
                    print(f"  \"{new_name_norm}\" already exists — treating as alias.")
                    new_aliases[task][raw] = new_name_norm
                else:
                    choices.append(new_name_norm)          # visible to later iterations
                    new_choices[task].append(new_name_norm)
                    new_aliases[task][raw] = new_name_norm
                    print(f"  → Added \"{new_name_norm}\" to [{task}] choices "
                          f"and mapped \"{raw}\" → \"{new_name_norm}\"\n")
                break

            # Number input
            try:
                idx = int(ans) - 1
                if 0 <= idx < len(choices):
                    chosen = choices[idx]
                    new_aliases[task][raw] = chosen
                    print(f"  → mapped to \"{chosen}\"\n")
                    break
                else:
                    print(f"  Please enter 1–{len(choices)}, 'n', 's', or 'q'.")
                    continue
            except ValueError:
                pass

            # Text input — exact match against choices
            match = next((c for c in choices if c.lower() == ans.lower()), None)
            if match:
                new_aliases[task][raw] = match
                print(f"  → mapped to \"{match}\"\n")
                break

            print(f"  Not recognised. Enter a number 1–{len(choices)}, the choice text, "
                  f"'n' (new option), 's', or 'q'.")

    return dict(new_aliases), dict(new_choices)


# ---------------------------------------------------------------------------
# Phase 2 — Confirm proposed aliases
# ---------------------------------------------------------------------------

def phase2_confirm(new_aliases, new_choices):
    """Print proposed aliases and new choices; ask for confirmation. Returns bool."""
    has_aliases = any(v for v in new_aliases.values())
    has_choices = any(v for v in new_choices.values())
    if not has_aliases and not has_choices:
        print("\n  No new aliases or choices were defined.\n")
        return False

    print(f"\n{'═'*62}")
    print(f"  PHASE 2 OF 3 — Confirm New Aliases and Choices")
    print(f"{'═'*62}")

    if has_aliases:
        total = sum(len(v) for v in new_aliases.values())
        print(f"\n  {total} new alias(es) proposed:")
        for task, mapping in sorted(new_aliases.items()):
            if mapping:
                print(f"  [{task}]")
                for raw, choice in mapping.items():
                    print(f"    \"{raw}\"  →  \"{choice}\"")

    if has_choices:
        print(f"\n  New choices to be added to prompts:")
        for task, clist in sorted(new_choices.items()):
            print(f"  [{task}]: {', '.join(clist)}")
    print()

    while True:
        try:
            ans = input("  Apply these changes? [Y/n]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        if ans in ("", "y", "yes"):
            return True
        if ans in ("n", "no"):
            print("  Changes discarded.")
            return False


# ---------------------------------------------------------------------------
# Phase 3 — Apply patches to all CSVs with context display
# ---------------------------------------------------------------------------

def find_csv_row_idx(csv_rows, t_start, tol=0.002):
    """Return the index of the CSV row whose t_start matches within tol."""
    for i, r in enumerate(csv_rows):
        try:
            if abs(float(r["t_start"]) - float(t_start)) <= tol:
                return i
        except (ValueError, TypeError):
            pass
    return None


def display_context(csv_rows, fieldnames, patch_idx, patches_for_row, window=2):
    """
    Print rows [patch_idx-window .. patch_idx+window].
    patches_for_row: {task: (old_val, new_val)}.
    The patched row is marked with ► and changed cells show old→new.
    """
    lo = max(0, patch_idx - window)
    hi = min(len(csv_rows) - 1, patch_idx + window)

    show_cols = ["t_start", "t_end"] + list(patches_for_row.keys())
    for f in fieldnames:
        if f not in ("video_path",) + tuple(show_cols) and len(show_cols) < 7:
            show_cols.append(f)

    W = 24
    print(f"  {'':5}" + "".join(f"{c[:W]:<{W}}" for c in show_cols))
    print("  " + "-" * (5 + W * len(show_cols)))

    for i in range(lo, hi + 1):
        r = csv_rows[i]
        marker = "► " if i == patch_idx else "  "
        vals = []
        for c in show_cols:
            if i == patch_idx and c in patches_for_row:
                old, new = patches_for_row[c]
                cell = f"{old}→{new}"
            else:
                cell = r.get(c, "")
            vals.append(f"{str(cell)[:W]:<{W}}")
        print(f"{marker}{i:<5}" + "".join(vals))
    print()


def phase3_apply_patches(all_unknowns, all_csvs_data, new_aliases, existing_aliases,
                          csv_paths):
    """
    Re-normalise all unknown entries using existing + new aliases.
    Patches are grouped by source file (src_idx) and then by CSV row.
    Shows context and asks for confirmation row by row.

    Returns:
        all_csvs_data   — list of (fieldnames, rows) with patches applied
        applied_by_src  — {src_idx: set of (round(t_start,3), task)} actually patched
    """
    # Build patch plan: {src_idx: {row_idx: {task: (old, new)}}}
    patch_plan     = defaultdict(lambda: defaultdict(dict))
    unresolved_count = 0

    for u in all_unknowns:
        src_idx = u["_src_idx"]
        task    = u["task"]
        raw     = u["raw"]
        choices = u["choices"]
        t_start = u["t_start"]

        merged = dict(existing_aliases.get(task, {}))
        merged.update(new_aliases.get(task, {}))

        new_label = normalize_to_choices(raw, choices, merged)
        if new_label == "unknown":
            unresolved_count += 1
            continue

        _, csv_rows = all_csvs_data[src_idx]
        row_idx = find_csv_row_idx(csv_rows, t_start)
        if row_idx is None:
            print(f"  [warn] No CSV row for t_start={t_start} in {csv_paths[src_idx]}")
            continue

        old_label = csv_rows[row_idx].get(task, "unknown")
        patch_plan[src_idx][row_idx][task] = (old_label, new_label)

    total_rows    = sum(len(v) for v in patch_plan.values())
    total_patches = sum(len(p) for d in patch_plan.values() for p in d.values())

    print(f"\n{'═'*62}")
    print(f"  PHASE 3 OF 3 — Review and Apply CSV Patches")
    print(f"  {total_patches} label update(s) across {total_rows} row(s) "
          f"in {len(patch_plan)} file(s).")
    if unresolved_count:
        print(f"  {unresolved_count} entr{'y' if unresolved_count==1 else 'ies'} "
              f"still unresolved (will remain 'unknown').")
    print(f"{'═'*62}")
    print("  y / Enter = accept   n = skip   a = accept all remaining\n")

    applied_by_src = defaultdict(set)
    accept_all     = False
    applied        = 0

    for src_idx in sorted(patch_plan.keys()):
        fieldnames, csv_rows = all_csvs_data[src_idx]
        src_label = os.path.basename(csv_paths[src_idx])

        for row_idx in sorted(patch_plan[src_idx].keys()):
            patches   = patch_plan[src_idx][row_idx]
            t_start   = csv_rows[row_idx].get("t_start", "?")
            tasks_str = ", ".join(patches.keys())

            if not accept_all:
                print(f"  [{src_label}]  Row {row_idx}  (t_start={t_start}, task(s): {tasks_str})")
                display_context(csv_rows, fieldnames, row_idx, patches)
                try:
                    ans = input("  Accept patch? [Y/n/a]: ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    print("\n  Interrupted — no further patches applied.")
                    return all_csvs_data, dict(applied_by_src)

                if ans == "a":
                    accept_all = True
                    print("  Accepting all remaining patches.\n")
                elif ans in ("n", "no"):
                    print("  Skipped.\n")
                    continue

            for task, (_, new_val) in patches.items():
                csv_rows[row_idx][task] = new_val
                applied_by_src[src_idx].add((round(float(t_start), 3), task))
            applied += 1

    print(f"\n  Applied {applied} of {total_rows} row patch(es).")
    return all_csvs_data, dict(applied_by_src)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Interactively resolve unknown annotation labels and patch CSV(s).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--unknowns", nargs="+", required=True,
                    help="One or more *_unknowns.jsonl files (paired with --csv by position)")
    ap.add_argument("--csv", nargs="+", required=True,
                    help="One or more annotation CSVs (paired with --unknowns by position)")
    ap.add_argument("--prompts", default=None,
                    help="Prompts JSON used during annotation (loads existing aliases; "
                         "also used to write an updated copy with new aliases)")
    out_grp = ap.add_mutually_exclusive_group()
    out_grp.add_argument("--out_dir", default=None,
                         help="Output directory for all generated files. "
                              "Mutually exclusive with --out_csv.")
    out_grp.add_argument("--out_csv", nargs="+", default=None,
                         help="Explicit resolved CSV path(s), one per --csv input. "
                              "Other outputs default to the same directory. "
                              "Mutually exclusive with --out_dir.")
    ap.add_argument("--out_prompts", default=None,
                    help="Output prompts JSON path. Defaults to "
                         "<out_dir>/<prompts_stem>_updated.json. "
                         "Only written when --prompts is provided.")
    ap.add_argument("--out_delta_prompts", default=None,
                    help="Output delta prompts JSON path (contains only tasks where new "
                         "choices were added). Defaults to <out_dir>/<prompts_stem>_delta.json. "
                         "Only written when --prompts is provided and new choices were defined.")
    args = ap.parse_args()

    # Validate pairing
    if len(args.unknowns) != len(args.csv):
        ap.error(f"--unknowns ({len(args.unknowns)}) and --csv ({len(args.csv)}) "
                 f"must have the same number of arguments.")

    n_pairs = len(args.unknowns)

    # Resolve output CSV paths and shared output directory
    if args.out_csv:
        if len(args.out_csv) != n_pairs:
            ap.error(f"--out_csv must have the same number of arguments as --csv ({n_pairs}).")
        out_csv_paths = args.out_csv
    elif args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        out_csv_paths = [
            os.path.join(args.out_dir,
                         os.path.splitext(os.path.basename(p))[0] + "_resolved.csv")
            for p in args.csv
        ]
    else:
        out_csv_paths = [
            os.path.splitext(p)[0] + "_resolved.csv"
            for p in args.csv
        ]

    # Derive shared output directory from first resolved CSV path
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(out_csv_paths[0]))

    if args.out_prompts is None and args.prompts:
        pname = os.path.basename(args.prompts)
        stem, ext = os.path.splitext(pname)
        args.out_prompts = os.path.join(out_dir, stem + "_updated" + ext)

    if args.out_delta_prompts is None and args.prompts:
        pname = os.path.basename(args.prompts)
        stem, ext = os.path.splitext(pname)
        args.out_delta_prompts = os.path.join(out_dir, stem + "_delta" + ext)

    # Load all inputs
    all_unknowns  = []
    all_csvs_data = []

    for src_idx, (unk_path, csv_path) in enumerate(zip(args.unknowns, args.csv)):
        print(f"\nLoading [{src_idx+1}/{n_pairs}]  {os.path.basename(unk_path)}")
        entries = load_unknowns(unk_path)
        for u in entries:
            u["_src_idx"] = src_idx
        all_unknowns.extend(entries)
        print(f"  {len(entries)} unknown entr{'y' if len(entries)==1 else 'ies'}")

        print(f"  Loading CSV: {os.path.basename(csv_path)}")
        fieldnames, rows = load_csv(csv_path)
        all_csvs_data.append((fieldnames, rows))
        print(f"  {len(rows)} rows")

    existing_aliases   = load_existing_aliases(args.prompts)
    full_presets       = load_full_presets(args.prompts)
    if existing_aliases:
        n_existing = sum(len(v) for v in existing_aliases.values())
        print(f"\n  {n_existing} existing alias(es) loaded from prompts.")

    if not all_unknowns:
        print("\nNo unknowns to process. Exiting.")
        return

    print(f"\n  Total: {len(all_unknowns)} unknown entries across {n_pairs} file(s).")

    # Phase 1 — collect new aliases and new choices (pooled across all sources)
    new_aliases, new_choices = phase1_collect_aliases(all_unknowns, existing_aliases)

    # Phase 2 — confirm
    confirmed = phase2_confirm(new_aliases, new_choices)
    if not confirmed:
        new_aliases  = {}
        new_choices  = {}

    # Phase 3 — patch CSVs
    all_csvs_data, applied_by_src = phase3_apply_patches(
        all_unknowns, all_csvs_data, new_aliases, existing_aliases, args.csv
    )

    # Write outputs for each source
    print()
    for src_idx, (out_csv_path, unk_path) in enumerate(zip(out_csv_paths, args.unknowns)):
        fieldnames, rows = all_csvs_data[src_idx]

        # Patched CSV
        write_csv(out_csv_path, fieldnames, rows)
        print(f"  Patched CSV      : {out_csv_path}")

        # Remaining unknowns (not resolved or patch was skipped)
        applied_keys = applied_by_src.get(src_idx, set())
        src_unknowns = [u for u in all_unknowns if u["_src_idx"] == src_idx]
        remaining = [
            {k: v for k, v in u.items() if k != "_src_idx"}
            for u in src_unknowns
            if (round(float(u["t_start"]), 3), u["task"]) not in applied_keys
        ]
        stem, _ = os.path.splitext(out_csv_path)
        out_remaining = stem + "_unknowns_remaining.jsonl"
        with open(out_remaining, "w") as f:
            for u in remaining:
                f.write(json.dumps(u) + "\n")
        print(f"  Remaining unknowns: {out_remaining}  "
              f"({len(remaining)} entr{'y' if len(remaining)==1 else 'ies'})")

    # Updated prompts (full bank with new aliases merged in)
    if confirmed and new_aliases and args.prompts and args.out_prompts:
        save_updated_prompts(args.prompts, new_aliases, args.out_prompts)

    # Delta prompts (only tasks with new choices, for re-running annotation)
    if confirmed and new_choices and args.prompts and args.out_delta_prompts:
        save_delta_prompts(args.prompts, full_presets, new_choices, new_aliases,
                           args.out_delta_prompts)

    print("\nDone.\n")


if __name__ == "__main__":
    main()
