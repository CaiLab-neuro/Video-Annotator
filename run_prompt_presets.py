# Runs a single multimodal conversation per video clip to obtain structured behavioral annotations
# from a video–language model using predefined prompt templates.

import argparse, json, os, sys
from pathlib import Path
from copy import deepcopy

# Add tarsier directory to sys.path for imports
_script_dir = Path(__file__).parent.resolve()
_tarsier_dir = _script_dir / 'tarsier'
if _tarsier_dir.exists() and str(_tarsier_dir) not in sys.path:
    sys.path.insert(0, str(_tarsier_dir))

# Patch transformers' dynamic module loader so that auto_map entries whose
# repo_id is a locally-importable Python package (e.g. "models" inside tarsier/)
# are resolved via importlib instead of fetching from HuggingFace Hub.
# This is needed because tarsier's config.json uses "models--module.Class" format
# where "models" refers to the local tarsier/models/ package, not a HF Hub repo.
import importlib, importlib.util
import transformers.dynamic_module_utils as _dmu
import transformers.models.auto.auto_factory as _af

_orig_get_class = _dmu.get_class_from_dynamic_module

def _patched_get_class(class_reference, pretrained_model_name_or_path, **kwargs):
    if importlib.util.find_spec(pretrained_model_name_or_path) is not None:
        mod_name, cls_name = class_reference.rsplit(".", 1)
        mod = importlib.import_module(f"{pretrained_model_name_or_path}.{mod_name}")
        return getattr(mod, cls_name)
    return _orig_get_class(class_reference, pretrained_model_name_or_path, **kwargs)

_dmu.get_class_from_dynamic_module = _patched_get_class
_af.get_class_from_dynamic_module = _patched_get_class

import torch
import yaml
from transformers.cache_utils import DynamicCache
from models.modeling_tarsier import LlavaConfig, TarsierForConditionalGeneration
from dataset.tarsier_datamodule import init_processor
from tools.conversation import Chat, conv_templates


def load_model_and_processor(model_name_or_path, data_config):
    """Load Tarsier model and processor."""
    from tools.color import Color
    print(Color.red(f"Load model and processor from: {model_name_or_path}"), flush=True)
    if isinstance(data_config, str):
        data_config = yaml.safe_load(open(data_config, 'r'))
    processor = init_processor(model_name_or_path, data_config)
    model_config = LlavaConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = TarsierForConditionalGeneration.from_pretrained(
        model_name_or_path,
        config=model_config,
        device_map='auto',
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.eval()
    return model, processor

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

    # Prefix match (either direction) — guard against empty s matching everything
    for c in choices:
        if s and (c.startswith(s) or s.startswith(c)):
            return c

    # Special handling for yes/no
    if set(choices) == {"yes", "no"}:
        if any(w in s for w in ["yes", "yeah", "ya", "correct", "true", "is"]):
            return "yes"
        if any(w in s for w in ["no", "nope", "not", "false", "isn"]):
            return "no"

    # Substring match as last resort
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


def _greedy_decode(model, first_logits, past_kv, total_seq_len, last_pos,
                   max_new_tokens, eos_ids, device):
    """Decode one token at a time using KV cache.

    Returns:
        tokens:        list of decoded token ids (including the final EOS id)
        past_kv:       updated KV cache
        total_seq_len: updated sequence length count
        last_pos:      updated last position index
    """
    tokens = []
    next_tok = first_logits[0, -1, :].argmax(-1, keepdim=True)  # [1]
    for _ in range(max_new_tokens):
        last_pos += 1
        total_seq_len += 1
        pos_ids = torch.tensor([[[last_pos, last_pos, last_pos]]],
                               dtype=torch.long, device=device)   # [1, 1, 3]
        mask = torch.ones(1, total_seq_len, dtype=torch.long, device=device)
        with torch.no_grad():
            out = model.forward(
                input_ids=next_tok.unsqueeze(0),  # [1, 1]
                attention_mask=mask,
                position_ids=pos_ids,
                past_key_values=past_kv,
                use_cache=True,
                return_dict=True,
            )
        past_kv = out.past_key_values
        tokens.append(next_tok.item())
        if next_tok.item() in eos_ids:
            break
        next_tok = out.logits[0, -1, :].argmax(-1, keepdim=True)
    return tokens, past_kv, total_seq_len, last_pos


def ask_all_with_kv_cache(
    chat,
    video_path,
    presets,
    system_prompt,
    gen_kwargs,
    template_name="tarsier2-7b",
    n_frames=8,
):
    """
    Run ALL preset questions for a single video clip, encoding the video ONCE
    and reusing the resulting LLM KV cache for every subsequent question.

    The video is encoded during a single prefill call (Step 2).  Each additional
    question only prefills the new user-turn tokens, not the video again.

    Returns:
        raw_map: dict { task_name -> raw_model_text }
    """
    model = chat.model
    device = chat.device
    tokenizer = chat.processor.processor.tokenizer
    # TarsierProcessor stores its own chat_template; the tokenizer itself does not
    # have .chat_template set, so we must pass it explicitly to apply_chat_template.
    chat_template = chat.processor.processor.chat_template

    max_new_tokens = int(gen_kwargs.get("max_new_tokens", 12))
    eos_ids = {
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|im_end|>"),
    }

    def _build_full_prompt(p):
        question = p["question"]
        choices = p["choices"]
        full_sys = (system_prompt or "").strip()
        allowed = f"Allowed options: {', '.join(choices)}."
        if full_sys:
            return f"{full_sys}\n\n{allowed}\n\n{question}"
        return f"{allowed}\n\n{question}"

    raw_map = {}

    # ------------------------------------------------------------------
    # Step 1 – Prepare first-question inputs (video path included)
    # ------------------------------------------------------------------
    conv = deepcopy(conv_templates[template_name])
    conv.messages.append([conv.roles[0], {"type": "video", "text": video_path}])
    conv = chat.ask(_build_full_prompt(presets[0]), conv)
    model_inputs, _ = chat.prepare_model_inputs(conv, n_frames)

    # Exclude labels (training-only key)
    forward_inputs = {k: v for k, v in model_inputs.items() if k != "labels"}

    # ------------------------------------------------------------------
    # Step 2 – Prefill: video encoding + Q1 context → KV cache
    #
    # IMPORTANT: pass an explicit DynamicCache() so the attention's
    # `if past_key_value is not None:` guard passes and each layer's
    # K/V states are stored.  Passing past_key_values=None (the default)
    # causes the guard to skip the cache update, returning an empty cache.
    # ------------------------------------------------------------------
    with torch.no_grad():
        prefill_out = model.forward(
            **forward_inputs,
            past_key_values=DynamicCache(),
            use_cache=True,
            return_dict=True,
        )

    past_kv = prefill_out.past_key_values
    # position_ids shape [1, seq_len, 3]; last text token has all 3 dims equal
    last_pos = prefill_out.position_ids[0, -1, 0].item()
    total_seq_len = model_inputs["input_ids"].shape[1]

    # ------------------------------------------------------------------
    # Step 3 – Greedy decode Q1
    # ------------------------------------------------------------------
    A1_tokens, past_kv, total_seq_len, last_pos = _greedy_decode(
        model, prefill_out.logits, past_kv, total_seq_len, last_pos,
        max_new_tokens, eos_ids, device,
    )
    A1_text = tokenizer.decode(A1_tokens, skip_special_tokens=True)
    raw_map[presets[0]["task"]] = A1_text

    A_prev_text = A1_text

    # ------------------------------------------------------------------
    # Step 4 – Subsequent questions Q2–QN (text-only, reuse KV cache)
    # ------------------------------------------------------------------
    for p in presets[1:]:
        Qn_full_prompt = _build_full_prompt(p)

        # 4a. Compute delta token IDs for the new user turn only.
        #     "x" is a throw-away placeholder; it cancels out in the diff.
        prior_msgs = [
            {"role": "user",      "content": "x"},
            {"role": "assistant", "content": A_prev_text},
        ]
        next_msgs = [
            {"role": "user",      "content": "x"},
            {"role": "assistant", "content": A_prev_text},
            {"role": "user",      "content": Qn_full_prompt},
        ]
        toks_prior = tokenizer.apply_chat_template(
            prior_msgs, tokenize=True, add_generation_prompt=False,
            chat_template=chat_template,
        )
        toks_next = tokenizer.apply_chat_template(
            next_msgs, tokenize=True, add_generation_prompt=True,
            chat_template=chat_template,
        )
        delta_ids = torch.tensor(
            [toks_next[len(toks_prior):]],
            dtype=torch.long, device=device,
        )

        # 4b. Prefill delta + decode Qn answer (no video re-encoding)
        delta_len = delta_ids.shape[1]
        pos_start = last_pos + 1
        pos_ids = torch.arange(pos_start, pos_start + delta_len, device=device)
        pos_ids = pos_ids.unsqueeze(0).unsqueeze(-1).expand(1, delta_len, 3).clone()  # [1, T, 3]

        # Pass attention_mask=None: all tokens are valid (no padding), so the mask is trivially
        # all-ones.  Passing the full-length mask would trigger the `else` branch in
        # transformers' _upad_input which calls flash_attn.bert_padding.unpad_input — a
        # function that returns 5 values in recent flash_attn but the installed transformers
        # expects only 4, causing a ValueError.  With mask=None flash attention falls back to
        # its built-in causal masking path which handles q_len < kv_seq_len correctly.
        with torch.no_grad():
            prefill_n = model.forward(
                input_ids=delta_ids,
                attention_mask=None,
                position_ids=pos_ids,
                past_key_values=past_kv,
                pixel_values=None,
                num_images=torch.tensor([0]).to(device),  # required for multi-token prefill
                use_cache=True,
                return_dict=True,
            )

        past_kv = prefill_n.past_key_values
        last_pos = pos_ids[0, -1, 0].item()
        total_seq_len += delta_len

        An_tokens, past_kv, total_seq_len, last_pos = _greedy_decode(
            model, prefill_n.logits, past_kv, total_seq_len, last_pos,
            max_new_tokens, eos_ids, device,
        )
        An_text = tokenizer.decode(An_tokens, skip_special_tokens=True)
        raw_map[p["task"]] = An_text
        A_prev_text = An_text

    return raw_map


def run_clip_inference(chat, clip_path, presets, system_prompt, gen_kwargs,
                       n_frames=8, no_kv_cache=False):
    """
    Run all preset questions for a single clip. Primary API for in-process use.
    Model stays loaded between calls (caller is responsible for loading it once).

    Returns:
        label_dict: dict { task -> normalized_label }
        raw_rows:   list of dicts { video_path, task, question, choices, raw, label }
    """
    _ask_fn = ask_all_for_clip if no_kv_cache else ask_all_with_kv_cache
    raw_map = _ask_fn(
        chat=chat, video_path=clip_path, presets=presets,
        system_prompt=system_prompt, gen_kwargs=gen_kwargs,
        template_name="tarsier2-7b", n_frames=n_frames,
    )
    label_dict = {}
    raw_rows = []
    for p in presets:
        task = p["task"]
        choices = p["choices"]
        aliases = p.get("aliases", {}) or {}
        raw = raw_map.get(task, "")
        label = normalize_to_choices(raw, choices, aliases)
        label_dict[task] = label
        raw_rows.append({
            "video_path": clip_path, "task": task, "question": p["question"],
            "choices": choices, "raw": raw, "label": label,
        })
    return label_dict, raw_rows


def main(args):
    # Load config and apply optional CLI overrides before building the processor.
    # This allows callers to tune n_frames / max_pixels without editing the YAML.
    with open(args.config, "r") as f:
        data_config = yaml.safe_load(f)
    if args.n_frames is not None:
        data_config["n_frames"] = int(args.n_frames)
    if args.max_pixels is not None:
        data_config["max_pixels"] = int(args.max_pixels)

    model, processor = load_model_and_processor(args.model_name_or_path, data_config)

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

    n_frames = data_config.get("n_frames", 8)

    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    chat = Chat(model, processor, device, debug=False)

    _ask_fn = ask_all_for_clip if args.no_kv_cache else ask_all_with_kv_cache
    raw_map = _ask_fn(
        chat=chat,
        video_path=args.input_path,
        presets=presets,
        system_prompt=system_prompt,
        gen_kwargs=gen_kwargs,
        template_name="tarsier2-7b",
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
    ap.add_argument(
        "--n_frames", type=int, default=None,
        help="Frames sampled per clip (overrides config n_frames). "
             "Note: use_multi_images_for_video=true in config doubles this for the model.",
    )
    ap.add_argument(
        "--max_pixels", type=int, default=None,
        help="Max pixels per frame (overrides config max_pixels). "
             "Lower values reduce memory and speed up inference (e.g. 200704 = 448x448).",
    )
    ap.add_argument(
        "--no_kv_cache", action="store_true",
        help="Disable KV-cache reuse; re-encode video for every question (original behavior).",
    )
    args = ap.parse_args()
    main(args)
