"""Render slot-filling rows into per-model chat format + label masking.

Neutral row -> messages -> ids. Prompt/completion split via prefix property:
  prompt_ids = apply_chat_template([sys, user], add_generation_prompt=True)
  full_ids   = apply_chat_template([sys, user, assistant])
  labels     = [-100]*len(prompt_ids) + full_ids[len(prompt_ids):]
(verified prefix-consistent for qwen2.5/qwen3/llama3.2/gemma3 templates)
"""
from __future__ import annotations

import json

SYSTEM_PROMPT = (
    "You extract function parameters from user messages for an in-car navigation assistant. "
    "Given a function schema and a user message, output ONLY a JSON object with one key per "
    "function parameter. Set a key to the value found in the user message, or null when the "
    "message does not state it. Output the JSON object and nothing else."
)

TAG_SYSTEM_PROMPT = (
    "You tag the words of user messages with function parameters for an in-car navigation "
    "assistant. Given a function schema and a user message, output exactly one tag per "
    "whitespace-separated word of the user message, in word order, tags separated by single "
    "spaces. Use O for words that are not part of any parameter value. Use B-<parameter> for "
    "the first word of a parameter value and I-<parameter> for the following words of the same "
    "value; after a gap, start again with B-<parameter>. For parameters with an enum, append "
    "the canonical enum value to the tag, like B-<parameter>:<value>, and tag only the first "
    "word of the phrase that expresses it. If the schema has a lang parameter, first output "
    "lang:<value> or lang:null as a single leading tag before the word tags. Output the tags "
    "and nothing else."
)


def user_content(row: dict) -> str:
    schema = json.dumps(row["function"], ensure_ascii=False, separators=(",", ":"))
    base = f"FUNCTION: {schema}\n\nUSER_MESSAGE: {row['utterance']}"
    spans = row.get("ner_spans")
    if spans:  # NER-hint variant (Experiment B); absent field -> byte-identical prompt
        hint = json.dumps(list(spans)[:12], ensure_ascii=False, separators=(",", ":"))
        base += f"\nCANDIDATE_SPANS: {hint}"
    return base


def assistant_content(row: dict) -> str:
    tags = row.get("bio_tags")
    if tags:  # tagging formulation (round 4); absent field -> JSON rendering
        return " ".join(tags)
    return json.dumps(row["target"], ensure_ascii=False, separators=(",", ":"))


def build_messages(row: dict) -> list[dict]:
    system = TAG_SYSTEM_PROMPT if "bio_tags" in row else SYSTEM_PROMPT
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content(row)},
        {"role": "assistant", "content": assistant_content(row)},
    ]


def render_ids(tokenizer, row: dict, template_kwargs: dict | None = None):
    """Return (full_ids, labels, prompt_len).

    String-level split: prompt_text = template(sys,user,add_generation_prompt=True);
    completion = full_text minus prompt_text prefix. Tokenized separately and
    concatenated — this matches inference (generate() tokenizes the prompt string
    once, then decodes tokens), unlike single-pass full tokenization where BPE may
    merge across the boundary.
    """
    kw = dict(template_kwargs or {})
    msgs = build_messages(row)
    prompt_text = tokenizer.apply_chat_template(
        msgs[:2], tokenize=False, add_generation_prompt=True, **kw)
    full_text = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=False, **kw)
    if not full_text.startswith(prompt_text):
        raise ValueError(f"chat template string-prefix mismatch for {type(tokenizer).__name__}")
    completion_text = full_text[len(prompt_text):]
    p_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    c_ids = tokenizer(completion_text, add_special_tokens=False)["input_ids"]
    ids = p_ids + c_ids
    labels = [-100] * len(p_ids) + c_ids
    return ids, labels, len(p_ids)


def render_prompt(tokenizer, row: dict, template_kwargs: dict | None = None) -> list[int]:
    kw = dict(template_kwargs or {})
    out = tokenizer.apply_chat_template(
        build_messages(row)[:2], tokenize=True, add_generation_prompt=True, **kw)
    if hasattr(out, "keys") and "input_ids" in out:  # transformers 5.x returns BatchEncoding
        return out["input_ids"]
    return list(out)
