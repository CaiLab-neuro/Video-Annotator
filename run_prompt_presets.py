# Runs a single multimodal conversation per video clip to obtain structured behavioral annotations 
# from a video–language model using predefined prompt templates.

import argparse, json, os
from copy import deepcopy
import torch
from tarsier.tasks.utils import load_model_and_processor
from tarsier.tools.conversation import Chat, conv_templates

def normalize_to_choices(raw_text, choices, aliases=None):
    """
    Map free-form model output to one of the closed-set `choices`.
    If nothing matches, return 'unknown'.
    """
    if not raw_text:
        return "unknown"

    s = raw_text.strip().lower()

    # Exact match
    if s in choices:
        return s

    # Alias-based mapping (case-insensitive, substring allowed)
    if aliases:
        for k, v in aliases.items():
            k_norm = k.strip().lower()
            v_norm = v.strip().lower()
            if not v_norm or v_norm not in choices:
                continue
            if s == k_norm or k_norm in s:
                return v_norm

    # Word-based match
    words_in_output = s.split()
    for c in choices:
        if c in words_in_output:
            return c

    # Prefix match (either direction)
    for c in choices:
        if c.startswith(s) or s.startswith(c):
            return c

    # Special handling for yes/no
    if set(choices) == {"yes", "no"}:
        if any(w in s for w in ["yes", "yeah", "ya", "correct", "true", "is"]):
            return "yes"
        if any(w in s for w in ["no", "nope", "not", "false", "isn"]):
            return "no"

    # Fuzzy substring match as last resort
    for c in choices:
        if c in s:
            return c

    return "unknown"


def ask_all_for_clip(
    chat,
    video_path,
    presets,
    system_prompt,
    gen_kwargs,
    template_name="tarsier2-7b",
    n_frames=8,
):
    """
    Run ALL preset questions for a *single* video clip inside a single conversation.

    - Attaches the video once.
    - For each preset:
        * Adds a user turn with the question (+ allowed options).
        * Calls `chat.answer` to get the model's response.
        * Stores raw text keyed by `task`.

    Returns:
        raw_map: dict { task_name -> raw_model_text }
    """
    conv = deepcopy(conv_templates[template_name])
    conv.messages.append([conv.roles[0], {"type": "video", "text": video_path}])

    raw_map = {}

    for p in presets:
        task = p["task"]
        question = p["question"]
        choices = p["choices"]

        full_system_prompt = system_prompt or ""
        if full_system_prompt:
            full_system_prompt = full_system_prompt.strip()

        allowed = f"Allowed options: {', '.join(choices)}."
        if full_system_prompt:
            full_prompt = f"{full_system_prompt}\n\n{allowed}\n\n{question}"
        else:
            full_prompt = f"{allowed}\n\n{question}"

        # Add user turn
        conv = chat.ask(full_prompt, conv)

        # Store answer for this specific question but continue the same conversation
        text, conv = chat.answer(
            conv=conv,
            n_frames=int(n_frames),
            max_new_tokens=int(gen_kwargs.get("max_new_tokens", 12)),
            num_beams=1,
            temperature=float(gen_kwargs.get("temperature", 0.0)),
            top_p=float(gen_kwargs.get("top_p", 1.0)),
        )

        raw_map[task] = text

    return raw_map


def main(args):
    model, processor = load_model_and_processor(args.model_name_or_path, args.config)

    with open(args.prompts, "r") as f:
        bank = json.load(f)

    system_prompt = bank.get("global_instructions", {}).get("system_prompt", "") or ""
    defaults = bank.get("global_instructions", {}).get("defaults", {}) or {}
    presets = bank.get("presets", []) or []

    if not presets:
        raise ValueError(f"No 'presets' found in {args.prompts}")

    if not os.path.exists(args.input_path):
        raise FileNotFoundError(f"Input video not found: {args.input_path}")

    gen_kwargs = {
        "max_new_tokens": int(defaults.get("max_new_tokens", 12)),
        "temperature": float(defaults.get("temperature", 0.0)),
        "top_p": float(defaults.get("top_p", 1.0)),
    }

    n_frames = 8

    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    chat = Chat(model, processor, device, debug=False)


    raw_map = ask_all_for_clip(
        chat=chat,
        video_path=args.input_path,
        presets=presets,
        system_prompt=system_prompt,
        gen_kwargs=gen_kwargs,
        template_name="tarsier2-7b",  # adjust if you use a different key in conv_templates
        n_frames=n_frames,
    )

    # jsonl output
    rows = []
    for p in presets:
        task = p["task"]
        question = p["question"]
        choices = p["choices"]
        aliases = p.get("aliases", {}) or {}

        raw = raw_map.get(task, "")
        label = normalize_to_choices(raw, choices, aliases)

        rows.append(
            {
                "video_path": args.input_path,
                "task": task,
                "question": question,
                "choices": choices,
                "raw": raw,
                "label": label,
            }
        )

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print(f"[inference_presets] Video: {args.input_path}")
    for r in rows:
        print(f"  - {r['task']}: {r['label']}  (raw='{r['raw'][:60].replace(chr(10), ' ')}...')")
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Run Tarsier with a preset prompt bank and return closed-set labels."
    )
    ap.add_argument("--model_name_or_path", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--input_path", required=True)
    ap.add_argument("--prompts", required=True, help="Preset bank JSON (short or long).")
    ap.add_argument(
        "--output",
        default="outputs/preset_results.jsonl",
        help="JSONL file with one line per task for this clip.",
    )
    args = ap.parse_args()
    main(args)
