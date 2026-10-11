# 2026-10-11 — BIO tagging & the BERT encoder tagger explained

Educational note covering the two Track-B/round-5 concepts (see
`2026-10-11-tagging-formulation-round4.md` for the experiment plan and RESULTS.md
Round 5 for numbers).

## BIO labeling scheme

**BIO** is the classic sequence-labeling scheme for tagging spans in text (from NER).
Each token gets exactly one label:

- **`O`** — *Outside*: this word is not part of any parameter value
- **`B-X`** — *Begin*: first word of a value for parameter X
- **`I-X`** — *Inside*: continuation word of the value for parameter X

Example from this task:

```
"Route from Central London to Paris avoiding tolls"

Route   → O            (not a value)
from    → O
Central → B-origin     (origin value begins)
London  → I-origin     (…continues)
to      → O            (gap ends the span)
Paris   → B-destination
avoiding→ O
tolls   → B-avoid_features:tollRoad   (enum label carries the canonical value)
```

Decoding back to JSON is just merging runs:

- `B-origin I-origin` → `"origin": "Central London"` (slice the tagged words)
- `B-destination` alone → `"destination": "Paris"`
- any parameter with no B/I tags → `null`

Design details:

1. **Why B and I at all, instead of one label per param?** B marks the *boundary* — it
   distinguishes two separate values of the same parameter from one multi-word value.
   E.g. `B-q B-q` = two segments (decoded "Św. Kierka, Chicago"-style), while
   `B-q I-q` = one two-word span.
2. **Restart after a gap**: since a tag is emitted per word, `O` words inside a gap
   naturally terminate a span — the next value word needs a fresh `B-`.

This is the user's idea 2 realized: the "empty token" is `O`, the "parameter name" is
the `B-`/`I-` part.

## BERT encoder tagger (Track B)

Literal implementation of user's idea 1: parameter names go **into the input**, and the
model's attention aligns utterance words to them. Instead of generating anything, every
word just gets a classification label. (JointBERT-style joint slot filling.)

### Input (`train/train_tagger.py` schema_string/encode)

```
[CLS] q, lang, limit [SEP] Find me a gas station in Lyon [SEP]
 └── schema string ──┘ └──────── user utterance ────────┘
```

- Schema string: parameter names, enums spelled out —
  `"avoid_features=ferry|motorway|tollRoad, destination, origin"` — so the encoder sees
  both the slot names and the canonical enum choices.
- One input = schema + utterance concatenated; BERT's bidirectional self-attention lets
  each utterance word attend directly to the parameter names (the "attention detects
  which parts relate to parameters" mechanism, made explicit).

### Labels

Each whitespace word of the utterance gets exactly one BIO tag, placed on the word's
**first subtoken** (rest are `-100`, ignored by the loss). Label vocabulary (~70 classes,
built from training tags): `O`, `B-<param>`, `I-<param>`, and for enum slots the
canonical value is part of the label itself — `B-avoid_features:tollRoad` — because the
French word "péages" never contains "tollRoad".

### Model

```
bert-base-multilingual (110M, wikineural init)
   ├── tag_head:  Linear(768 → ~70)   on every token   → BIO label per word
   └── lang_head: Linear(768 → 6)     on [CLS]         → en/de/pl/fr/es/null
```

The `lang` param isn't a span (inferred from dialog language), so it can't be a word
tag — it's a small classification head on the pooled `[CLS]`. Total loss =
token CE + 0.5 · lang CE.

### Training / inference

- 5 epochs, AdamW 5e-5, bs 16 (~23 min on a 3090); best by val word-tag accuracy
  = 98.34%.
- Inference (`eval/eval_tagger.py`): argmax per first-subtoken → tag per word → the
  shared `decode_tags` contract (B..I runs = segments, slice original words, enum labels
  emit their canonical value) → the same JSON as every other system → identical scoring.

### Why 98% tag accuracy → only 0.7152 F1

- **Tag accuracy is word-level and O-dominated**: ~95% of words are `O`, so a single
  boundary mistake barely moves 98% but destroys a whole slot (a slot is all-or-nothing
  in scoring).
- **Structural ceiling 91.1%**: slicing words can never produce gold `"Seattle, WA"`
  when the user said "Seattle", nor canonical "Kraków" from inflected "Krakowa" — the
  JSON model memorizes these normalizations, the tagger mathematically can't.
- Same dominant confusions as the LLMs (lang 150 errors, q boundary), plus pl collapse
  (−9.9 F1) from morphology slicing.

Net: REJECT as primary (3.6 pts below the 0.7515 JSON winner) but attractive as a
5 ms/row GPU / 97 ms/row CPU edge fallback — 8.6× faster than the 1B decoder and it can
never fabricate a value or emit unparseable output.
