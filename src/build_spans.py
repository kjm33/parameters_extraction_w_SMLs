"""Build span-list targets for the span-decoding formulation (round 9).

For every row of core_{train,val,test} / aug_{train,val}: emit a copy with
`span_target` — one entry per parameter of the gold target:
  null param            -> null
  lang/language         -> literal value
  enum param            -> literal value (+ side-field `enum_spans[param]` with
                           the phrase location, does not affect decoding)
  free/number param     -> 1-based inclusive [first, last] whitespace-word
                           pairs into row["utterance"]; whole param falls back
                           to the literal value when any item is unaligned

Depth contract (decode_spans, reused by the span evaluator):
  null                          -> null
  str / number                  -> literal
  list of [int, int] pairs      -> scalar: slices joined with join_sep
  list of lists of pairs        -> array: one string per item (deduped)

Two variants per split: data/<name>_spans.jsonl (plain) and
data/<name>_spansn.jsonl ("span_numbering": true added; prompt-side numbering
is handled elsewhere).

Usage:
  python3 src/build_spans.py            # all five splits
  python3 src/build_spans.py core_test  # subset, comma-separated
"""
from __future__ import annotations

import json
import pathlib
import re
import sys
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from build_tags import (  # noqa: E402
    find_spans,
    enum_item_span,
    vparts,
    wfold,
    props_of,
    norm_eq,
)

FILES = ("core_train", "core_val", "core_test", "aug_train", "aug_val")
EDGE_PUNCT = re.compile(r"^\W+|\W+$")
ELISION = re.compile(r"(?i)^[ldjnstcm]'")


def idxs_to_pairs(idxs: list[int]) -> list[list[int]]:
    """0-based word idxs -> 1-based inclusive contiguous pairs ([2,3,7] ->
    [[3,4],[8,8]])."""
    pairs: list[list[int]] = []
    for i in idxs:
        if pairs and i == pairs[-1][1]:
            pairs[-1][1] = i + 1
        else:
            pairs.append([i + 1, i + 1])
    return pairs


def decode_pairs(pairs: list, words: list[str], join_sep: str) -> str:
    segs = []
    for a, b in pairs:
        seg = " ".join(words[a - 1:b])
        seg = EDGE_PUNCT.sub("", seg)
        seg = ELISION.sub("", seg)
        segs.append(seg)
    return join_sep.join(segs)


def _is_pair(x) -> bool:
    return (isinstance(x, (list, tuple)) and len(x) == 2
            and all(isinstance(i, int) for i in x))


def _is_scalar(x) -> bool:
    return isinstance(x, str) or (isinstance(x, (int, float))
                                  and not isinstance(x, bool))


def decode_spans(span_target: dict, words: list[str], props: dict,
                 join_sep: str = ", ") -> dict:
    """Pure inverse of the span_target depth contract (assume valid input)."""
    out: dict = {}
    for param, v in span_target.items():
        if v is None:
            out[param] = None
        elif _is_scalar(v):
            out[param] = v
        elif isinstance(v, list) and v and all(_is_pair(x) for x in v):
            out[param] = decode_pairs(v, words, join_sep)
        elif isinstance(v, list) and v and all(
                isinstance(x, (list, tuple)) and x and all(_is_pair(p) for p in x)
                for x in v):
            vals, seen = [], set()
            for item in v:
                s = decode_pairs(item, words, join_sep)
                if s not in seen:
                    seen.add(s)
                    vals.append(s)
            out[param] = vals
        elif isinstance(v, list) and v and all(_is_scalar(x) for x in v):
            out[param] = v  # literal enum array (e.g. avoid_features)
        # anything malformed: skip the param (absent)
    return out


def enum_items(value) -> list:
    return value if isinstance(value, list) else [value]


def enum_item_pairs(item, wfolds) -> list[list[int]] | None:
    if not isinstance(item, str) or not item:
        return None
    i = enum_item_span(item, wfolds)
    if i is not None:
        return [[i + 1, i + 1]]
    hit = find_spans(vparts(item), wfolds)
    if hit:
        return idxs_to_pairs(hit[0])
    return None


def check_pairs(value, n_words: int):
    """Safety assert: every produced index is a valid 1-based word index."""
    if not isinstance(value, list):
        return
    if value and _is_pair(value[0]):
        lists = [value]
    else:
        lists = [sub for sub in value if isinstance(sub, list)]
    for pairs in lists:
        for a, b in pairs:
            assert 1 <= a <= b <= n_words, (a, b, n_words)


def build_row_spans(row, stats: Counter, examples: list) -> tuple[dict, dict]:
    words = row["utterance"].split()
    wfolds = [wfold(w) for w in words]
    props = props_of(row)
    span_target: dict = {}
    enum_spans: dict = {}
    for param, value in row["target"].items():
        if value is None:
            span_target[param] = None
            continue
        ps = props.get(param) or {}
        is_enum = "enum" in ps or "enum" in (ps.get("items") or {})
        if param in ("lang", "language"):
            span_target[param] = value
            continue
        if is_enum:
            stats["enum_literals"] += 1
            span_target[param] = value
            items = enum_items(value)
            found = [enum_item_pairs(it, wfolds) for it in items]
            enum_spans[param] = found if isinstance(value, list) else found[0]
            stats["enum_span_found"] += sum(f is not None for f in found)
            stats["enum_span_missed"] += sum(f is None for f in found)
            continue
        items = enum_items(value)
        if any(not isinstance(it, str) or not it for it in items):
            span_target[param] = value  # number / non-str: literal
            continue
        hits = [find_spans(vparts(it), wfolds) for it in items]
        good = [h for h in hits if h is not None]
        if len(good) != len(items):
            stats["unaligned_fallback"] += 1
            span_target[param] = value
            if len(examples) < 20:
                examples.append({"param": param, "value": value,
                                 "utterance": row["utterance"][:120]})
            continue
        per_item = []
        for idxs, cls in good:
            stats[cls] += 1
            per_item.append(idxs_to_pairs(idxs))
        stats["free_span_params"] += 1
        span_target[param] = per_item if isinstance(value, list) else per_item[0]
    for param, v in enum_spans.items():
        check_pairs(v, len(words))
    for param, v in span_target.items():
        check_pairs(v, len(words))
    return span_target, enum_spans


def main():
    inputs = FILES
    if len(sys.argv) > 1:
        inputs = tuple(sys.argv[1].split(","))
    tags_stats_path = ROOT / "data" / "tags_build_stats.json"
    join_sep = ", "
    if tags_stats_path.exists():
        recorded = json.loads(tags_stats_path.read_text()).get("join_sep")
        if recorded:
            join_sep = recorded
    all_stats: dict = {}
    unaligned_examples: list = []
    for name in inputs:
        stats: Counter = Counter()
        examples: list = []
        rows_out = []
        with open(ROOT / "data" / f"{name}.jsonl") as f:
            rows = [json.loads(ln) for ln in f if ln.strip()]
        for row in rows:
            span_target, enum_spans = build_row_spans(row, stats, examples)
            for with_numbering in (False, True):
                row2 = dict(row)
                row2["span_target"] = span_target
                if enum_spans:
                    row2["enum_spans"] = enum_spans
                if with_numbering:
                    row2["span_numbering"] = True
                rows_out.append(row2)
            props = props_of(row)
            words = row["utterance"].split()
            dec = decode_spans(span_target, words, props, join_sep=join_sep)
            ok = all(norm_eq(dec.get(k), v) for k, v in row["target"].items())
            stats["roundtrip_ok"] += int(ok)
        unaligned_examples.extend(examples)
        plain = rows_out[0::2]
        numbered = rows_out[1::2]
        for suffix, rws in (("_spans", plain), ("_spansn", numbered)):
            with open(ROOT / "data" / f"{name}{suffix}.jsonl", "w") as f:
                for r in rws:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        n = len(rows)
        stats["rows"] = n
        file_stats: dict = dict(stats)
        file_stats["roundtrip_rate"] = round(stats["roundtrip_ok"] / max(1, n), 4)
        all_stats[name] = file_stats
        print(f"{name}: rows={n} rt={file_stats['roundtrip_rate']} stats={file_stats}")
    report = {"join_sep": join_sep, "files": all_stats,
              "unaligned_examples": unaligned_examples[:20]}
    (ROOT / "data" / "spans_build_stats.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2))
    print(f"join_sep={join_sep!r}")
    print("-> data/spans_build_stats.json")


if __name__ == "__main__":
    main()
