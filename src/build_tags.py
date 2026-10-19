"""Build BIO-tag targets for the tagging formulation (round 4, efforts/2026-10-11-bio-tagging-formulation/).

For every row of core_{train,val,test} / aug_{train,val}: emit a copy with `bio_tags` —
one tag per whitespace word of the utterance, plus an optional leading `lang:<v>`
pseudo-tag (present iff the schema has a `lang` param; v = target value or "null").

Tag forms:
  O                     word is not part of any parameter value
  B-<param>[:<enum>]    word starts a value span; enum params carry the CANONICAL value
  I-<param>             word continues the current span of <param>
  lang:<v>              leading pseudo-tag (not aligned to any word)

Decode contract (decode_tags, reused by eval_tags/eval_tagger):
  tag i (after the lang tag) <-> utterance.split()[i]; maximal B..I runs are segments;
  enum segments use the tag's canonical value, others slice the original words
  (edge punctuation stripped); scalar = first segment (enum) or JOIN_SEP.join(segments);
  array params (schema type array) -> list of segment values; params never tagged -> null.
  JOIN_SEP (", " vs " ") is chosen by round-trip majority and recorded in
  data/tags_build_stats.json.

Value->span alignment order: contiguous word window -> enum lexicon stem word ->
monotonic subsequence (discontinuous; segments joined at decode) -> canonical-appendage
prefix (remaining parts all <=2 chars, e.g. "Seattle, WA") -> unaligned (value nulled
in tags, logged).

Usage:
  python3 src/build_tags.py            # all five splits -> data/<name>_tags.jsonl
"""
from __future__ import annotations

import json
import pathlib
import re
import sys
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from build_extraction import ENUM_STEMS, fold, tok_match  # noqa: E402

EDGE_PUNCT = re.compile(r"^\W+|\W+$")
FILES = ("core_train", "core_val", "core_test", "aug_train", "aug_val")


def wfold(w: str) -> str:
    return EDGE_PUNCT.sub("", fold(w))


def vparts(v) -> list[str]:
    fv = fold(str(v))
    return [p for p in (EDGE_PUNCT.sub("", s) for s in re.split(r"[\s,]+", fv)) if p]


def part_matches(part: str, wf: str) -> bool:
    if not wf or not part:
        return False
    if part == wf:
        return True
    if part in re.split(r"[-'’]", wf) or wf in part.split("-"):
        return True
    return tok_match(part, wf)


def enum_stems_for(value) -> list[str]:
    key = fold(value).rstrip("s")
    return list((ENUM_STEMS.get(key) or []))


def enum_item_span(value, wfolds: list[str]) -> int | None:
    """First word matching a lexicon stem. Space-padded stems ("car ") keep their
    word-boundary semantics; bare stems match as substrings ("maut" in "mautpflicht"),
    mirroring enum_supported's `st in utt_f` on whole utterances."""
    for stem in enum_stems_for(value):
        for i, wf in enumerate(wfolds):
            if not wf:
                continue
            padded = f" {wf} "
            if (stem in padded) or wf == stem or tok_match(stem, wf) \
                    or stem in re.split(r"[-'’]", wf):
                return i
    return None


def _contiguous(parts, wfolds):
    n = len(wfolds)
    for start in range(n):
        w, pi, idxs = start, 0, []
        while w < n and pi < len(parts):
            took = 0
            while pi < len(parts) and took < 4 and part_matches(parts[pi], wfolds[w]):
                idxs.append(w)
                pi += 1
                took += 1
            if not took:
                break
            w += 1
        if pi == len(parts):
            seen, dedup = set(), []
            for i in idxs:
                if i not in seen:
                    seen.add(i)
                    dedup.append(i)
            return dedup, "contig"
    return None


def _subsequence(parts, wfolds, limit: int | None = None):
    """Greedy monotonic match of parts[:limit] (limit=None -> all)."""
    want = parts if limit is None else parts[:limit]
    idxs, pi = [], 0
    for w, wf in enumerate(wfolds):
        if pi < len(want) and part_matches(want[pi], wf):
            idxs.append(w)
            pi += 1
    matched_all = pi == len(want)
    return idxs, matched_all


def find_spans(parts, wfolds):
    """-> (word_idxs, class) or None ('unaligned')."""
    hit = _contiguous(parts, wfolds)
    if hit:
        return hit
    idxs, ok = _subsequence(parts, wfolds)
    if ok:
        return idxs, "disc"
    for k in range(len(parts) - 1, 0, -1):  # longest alignable prefix
        idxs, ok = _subsequence(parts, wfolds, limit=k)
        if ok and all(len(p) <= 2 for p in parts[k:]):
            return idxs, "appendage"
    return None


def place_segments(tags, param, idxs, enum_value=None) -> bool:
    """Tag word indices as one or more B..I segments; True if any placed."""
    placed = False
    run = []
    for i in idxs + [None]:
        if i is not None and (not run or i == run[-1] + 1):
            run.append(i)
            continue
        if run:
            if tags[run[0]] == "O":
                tags[run[0]] = f"B-{param}" + (f":{enum_value}" if enum_value else "")
                for k in run[1:]:
                    if tags[k] == "O":
                        tags[k] = f"I-{param}"
                placed = True
        if i is not None:
            run = [i]
    return placed


def props_of(row) -> dict:
    return ((row.get("function") or {}).get("parameters") or {}).get("properties") or {}


def build_row_tags(row, stats: Counter, examples: list):
    words = row["utterance"].split()
    wfolds = [wfold(w) for w in words]
    tags = ["O"] * len(words)
    props = props_of(row)
    for param, value in row["target"].items():
        if value is None:
            continue
        ps = props.get(param) or {}
        is_enum = "enum" in ps or "enum" in (ps.get("items") or {})
        if param == "lang":
            stats["lang"] += 1
            continue
        items = value if isinstance(value, list) else [value]
        if not isinstance(value, list) and isinstance(value, (dict, bool)):
            stats["null-by-policy"] += 1
            continue
        any_placed = False
        for item in items:
            if not isinstance(item, str) or not item:
                continue
            if is_enum:
                i = enum_item_span(item, wfolds)
                if i is not None:
                    any_placed |= place_segments(tags, param, [i], enum_value=item)
                    stats["enum-lexicon"] += 1
                    continue
                hit = find_spans(vparts(item), wfolds)
                if hit:
                    idxs, cls = hit
                    any_placed |= place_segments(tags, param, idxs, enum_value=item)
                    stats[f"enum-{cls}"] += 1
                    continue
                stats["enum-unaligned"] += 1
                continue
            hit = find_spans(vparts(item), wfolds)
            if hit:
                idxs, cls = hit
                any_placed |= place_segments(tags, param, idxs)
                stats[cls] += 1
            else:
                stats["unaligned"] += 1
                if len(examples) < 20:
                    examples.append({"param": param, "value": item,
                                     "utterance": row["utterance"][:120]})
        if isinstance(value, list) and not any_placed:
            pass  # whole array lost; decode -> null (logged via item classes)
    if "lang" in row["target"]:
        lv = row["target"]["lang"]
        tags = [f"lang:{lv if lv is not None else 'null'}"] + tags
    return tags


def decode_tags(tags, words, props, join_sep=", "):
    """Inverse of build_row_tags (contract above). Returns dict param->value."""
    out: dict = {}
    tags = list(tags)
    if tags and str(tags[0]).startswith("lang:"):
        v = str(tags[0])[5:]
        out["lang"] = None if v == "null" else v
        tags = tags[1:]
    segments: dict[str, list] = {}
    cur = None  # [param, start, enum_value, last_i]
    for i, t in enumerate(tags):
        t = str(t)
        if cur is not None and not (i == cur[3] + 1 and t == f"I-{cur[0]}"):
            segments.setdefault(cur[0], []).append((cur[1], cur[3], cur[2]))
            cur = None
        if t.startswith("B-"):
            body = t[2:]
            param, _, ev = body.partition(":")
            cur = [param, i, ev or None, i]
        elif t.startswith("I-") and cur is not None and t[2:] == cur[0]:
            cur[3] = i
    if cur is not None:
        segments.setdefault(cur[0], []).append((cur[1], cur[3], cur[2]))
    for param, segs in segments.items():
        vals = []
        for start, end, ev in segs:
            if ev is not None:
                vals.append(ev)
            else:
                seg = " ".join(words[start:end + 1])
                seg = EDGE_PUNCT.sub("", seg)
                seg = re.sub(r"(?i)^[ldjnstcm]'", "", seg)  # fr elisions (l'aéroport)
                vals.append(seg)
        ps = props.get(param) or {}
        is_array = ps.get("type") == "array" or "enum" in (ps.get("items") or {})
        has_ev = any(ev is not None for _, _, ev in segs)
        if is_array:
            seen, uniq = set(), []
            for v in vals:
                if v not in seen:
                    seen.add(v)
                    uniq.append(v)
            out[param] = uniq
        elif has_ev:
            out[param] = vals[0]
        elif len(vals) > 1:
            out[param] = join_sep.join(vals)
        else:
            out[param] = vals[0]
    return out


def norm_key(v):
    if isinstance(v, list):
        return sorted(re.sub(r"\s*,\s*", ",", fold(str(x))) for x in v)
    if v is None:
        return None
    return re.sub(r"\s*,\s*", ",", fold(str(v)))


def norm_eq(a, b) -> bool:
    return norm_key(a) == norm_key(b)


def main():
    inputs = FILES
    if len(sys.argv) > 1:
        inputs = tuple(sys.argv[1].split(","))
    all_stats = {}
    joiner_votes = {", ": 0, " ": 0}
    unaligned_examples: list = []
    for name in inputs:
        stats: Counter = Counter()
        examples: list = []
        rows_out, rt = [], {", ": 0, " ": 0}
        with open(ROOT / "data" / f"{name}.jsonl") as f:
            rows = [json.loads(ln) for ln in f if ln.strip()]
        for row in rows:
            tags = build_row_tags(row, stats, examples)
            row2 = dict(row)
            row2["bio_tags"] = tags
            rows_out.append(row2)
            props = props_of(row)
            words = row["utterance"].split()
            for sep in rt:
                dec = decode_tags(tags, words, props, join_sep=sep)
                ok = all(norm_eq(dec.get(k), v) for k, v in row["target"].items())
                rt[sep] += int(ok)
        unaligned_examples.extend(examples)
        joiner_votes[", "] += rt[", "]
        joiner_votes[" "] += rt[" "]
        out = ROOT / "data" / f"{name}_tags.jsonl"
        with open(out, "w") as f:
            for r in rows_out:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        n = len(rows)
        stats["rows"] = n
        all_stats[name] = {"classes": dict(stats),
                           "roundtrip_ok_comma": rt[", "],
                           "roundtrip_ok_space": rt[" "],
                           "roundtrip_rate_comma": round(rt[", "] / max(1, n), 4),
                           "roundtrip_rate_space": round(rt[" "] / max(1, n), 4)}
        print(f"{name}: rows={n} rt[,]={rt[', ']} rt[' ']={rt[' ']} "
              f"classes={dict(stats)}")
    join_sep = ", " if joiner_votes[", "] >= joiner_votes[" "] else " "
    report = {"join_sep": join_sep, "joiner_votes": joiner_votes,
              "unaligned_examples": unaligned_examples[:20], "files": all_stats}
    (ROOT / "data" / "tags_build_stats.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2))
    print(f"join_sep={join_sep!r} votes={joiner_votes}")
    print("-> data/tags_build_stats.json")


if __name__ == "__main__":
    main()
