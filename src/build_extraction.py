"""Build slot-filling extraction rows from fcft verified dialogs + gold sets.

For every assistant tool_call: emit (utterance=last user msg before the call,
function schema, raw args, target). Target policy (per efforts/2026-10-07-baseline-pipeline/tasks.md):
  - every schema param gets a key (schema order), value = arg value if extractable
    from the utterance, else null
  - verbatim strings: case/diacritic-folded substring OR all >=3-char tokens present
  - enum values: kept only if a multilingual keyword stem for that value appears
  - `lang` param: kept if == dialog language (inference signal)
  - numbers: word-boundary match of the literal in folded utterance
  - arrays: filter item-wise (enum stems / verbatim); empty -> null
  - objects: always null (coords etc. come from tool results, not the user)

Usage:
  ../function_calling_edge_fine_tuning/train/venv/bin/python -m src.build_extraction
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import re
import sys
import unicodedata

ROOT = pathlib.Path(__file__).resolve().parent.parent
FCFT = ROOT.parent / "function_calling_edge_fine_tuning"
OUT = ROOT / "data"

ENUM_STEMS: dict[str, list[str]] = {
    # canonical enum value (lowercased, trailing 's' stripped) -> multilingual stems (folded)
    "tollroad": ["toll", "platn", "oplat", "maut", "peage", "peaje", "pedaggio", "betal"],
    "ferry": ["ferr", "prom", "fahre", "trajekt", "transbord", " bac", "bac "],
    "motorway": ["motorway", "autobahn", "autostrad", "autopist", "autopista", "autoroute", "highway"],
    "tunnel": ["tunnel", "tunel"],
    "car": ["autem", "samochod", "pojazd", "voiture", "coche", "wagen", " m auto", "car "],
    "truck": ["truck", "ciezarow", "lkw", "camion", "lastwagen", "poids"],
    "pedestrian": ["piesz", "piechot", "walk", "fuss", "pied", "caminand", "foot", "pie "],
    "bicycle": ["rower", "bicycl", "bike", "fahrrad", "velo", "bici", "cycl"],
    "scooter": ["skuter", "scooter", "roller", "trottinett"],
    "fast": ["fast", "najszybsz", "schnellst", "rapide", "rapido"],
    "fastest": ["fast", "najszybsz", "schnellst", "rapide", "rapido"],
    "short": ["short", "najkrotsz", "kurzest", "court", "corto", "krotk"],
    "shortest": ["short", "najkrotsz", "kurzest", "court", "corto", "krotk"],
    "thrilling": ["scenic", "malownicz", "panoram", "thrilling", "aussichtsreich"],
    "tollroads": ["toll", "platn", "oplat", "maut", "peage", "peaje", "pedaggio"],
    "motorways": ["motorway", "autobahn", "autostrad", "autopist", "autopista", "autoroute", "highway"],
    "ferries": ["ferr", "prom", "fahre", "trajekt", "transbord"],
}


def fold(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", s.lower()).strip()


def tokens(s: str) -> set[str]:
    return set(re.findall(r"\w{3,}", fold(s)))


def tok_match(vt: str, ut: str) -> bool:
    """Token equality tolerant to inflection (pl/de) and clipping: torun->torunia."""
    if vt == ut:
        return True
    if len(vt) >= 4 and ut.startswith(vt) and len(ut) - len(vt) <= 3:
        return True
    if len(ut) >= 4 and vt.startswith(ut) and len(vt) - len(ut) <= 3:
        return True
    return False


def verbatim(value: str, utt_f: str, utt_tokens: set[str]) -> bool:
    v = fold(value)
    if not v:
        return False
    if v in utt_f:
        return True
    vt = tokens(value)
    if not vt:
        return False
    return all(any(tok_match(t, u) for u in utt_tokens) for t in vt)


NUM_RE_CACHE: dict[str, re.Pattern] = {}


def number_found(value, utt_f: str) -> bool:
    lit = fold(str(value))
    if lit not in NUM_RE_CACHE:
        NUM_RE_CACHE[lit] = re.compile(rf"(?<![\w.]){re.escape(lit)}(?![\w.])")
    return bool(NUM_RE_CACHE[lit].search(utt_f))


def enum_supported(value: str, utt_f: str) -> bool:
    key = fold(value).rstrip("s")
    stems = ENUM_STEMS.get(key)
    if stems is None:
        # unknown enum: fall back to verbatim
        return verbatim(value, utt_f, tokens(utt_f))
    return any(st in utt_f for st in stems)


def clean_schema(fn: dict) -> dict:
    params = {k: v for k, v in (fn.get("parameters") or {}).items() if k != "$schema"}
    return {"name": fn.get("name"), "description": fn.get("description", ""), "parameters": params}


def param_target(pname: str, pschema: dict, value, utt_f: str, utt_tokens: set[str], lang: str):
    """Return (value|None, reason)."""
    if value is None:
        return None, "absent"
    if isinstance(value, dict):
        return None, "object"
    if isinstance(value, bool):
        return None, "bool"
    if isinstance(value, list):
        kept = []
        for item in value:
            if isinstance(item, str) and item:
                if pschema and "enum" in (pschema or {}):
                    ok = enum_supported(item, utt_f)
                elif (pschema or {}).get("items", {}).get("enum"):
                    ok = enum_supported(item, utt_f)
                else:
                    ok = verbatim(item, utt_f, utt_tokens)
                if ok:
                    kept.append(item)
        return (kept or None), "array-filtered"
    if isinstance(value, (int, float)):
        if pname == "lang":
            return None, "numeric-lang"
        return (value if number_found(value, utt_f) else None), "number"
    # string
    if pname == "lang":
        # round 7: lang dropped entirely — optional resolver-side param, set/null is
        # dialog-context-decided (216 test utterances appear with lang both set and null)
        return None, "lang-dropped"
    if pschema and "enum" in pschema:
        return (value if enum_supported(value, utt_f) else None), "enum"
    if verbatim(value, utt_f, utt_tokens):
        return value, "verbatim"
    # comma-segment rewrite: teacher args often append ', Country' the user never said;
    # keep the utterance-grounded segment ("Lyon, France" + "...vers Lyon..." -> "Lyon")
    if any(re.search(r"[^\W\d_]", s) for s in str(value).split(",")):
        for seg in str(value).split(","):
            seg = seg.strip()
            if seg and re.search(r"[^\W\d_]", seg) and verbatim(seg, utt_f, utt_tokens):
                return seg, "segment"
    return None, "no-match"


def schema_param_order(params: dict, args: dict) -> list[str]:
    props = list((params.get("properties") or {}).keys())
    extra = [k for k in args.keys() if k not in props]
    return props + extra


def process_dialog(row: dict) -> list[dict]:
    out = []
    tools_by_name = {}
    for t in row.get("tools") or []:
        f = t.get("function") or t
        if f.get("name"):
            tools_by_name.setdefault(f["name"], f)
    # name-aware exec_log queue (args parsed at execution time — most reliable)
    queue = list(row.get("exec_log") or [])
    qi = 0
    last_user = None
    for m in row.get("messages") or []:
        role = m.get("role")
        if role == "user":
            last_user = m.get("content")
            continue
        if role != "assistant" or not m.get("tool_calls"):
            continue
        for tc in m["tool_calls"]:
            fref = tc.get("function") or {}
            fname = fref.get("name")
            schema = tools_by_name.get(fname)
            if not schema:
                continue
            args = None
            while qi < len(queue):
                e = queue[qi]
                qi += 1
                if e.get("name") == fname:
                    args = e.get("args")
                    break
            if args is None:
                raw = fref.get("arguments")
                if isinstance(raw, str):
                    try:
                        args = json.loads(raw)
                    except Exception:
                        args = None
                elif isinstance(raw, dict):
                    args = raw
            if not isinstance(args, dict) or not args:
                continue
            utt = (last_user or "").strip()
            if not utt:
                continue
            utt_f, utt_t = fold(utt), tokens(utt)
            props = (schema.get("parameters") or {}).get("properties") or {}
            target = {}
            for p in schema_param_order(schema.get("parameters") or {}, args):
                target[p] = param_target(p, props.get(p, {}), args.get(p), utt_f, utt_t, row.get("lang", ""))[0]
            out.append({
                "id": "xtr-" + hashlib.sha1(f"{row['id']}|{fname}|{json.dumps(args, sort_keys=True, default=str)}".encode()).hexdigest()[:12],
                "source": "fcft",
                "lang": row.get("lang"),
                "function": clean_schema(schema),
                "utterance": utt,
                "args": args,
                "target": target,
            })
    return out


def dedup(rows: list[dict]) -> list[dict]:
    seen, out = set(), []
    for r in rows:
        k = (fold(r["utterance"])[:220], r["function"]["name"])
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out


def stats(rows: list[dict], label: str) -> dict:
    per_fn, per_lang, kept_total, gold_total = {}, {}, 0, 0
    param_kept: dict[str, int] = {}
    for r in rows:
        per_fn[r["function"]["name"]] = per_fn.get(r["function"]["name"], 0) + 1
        per_lang[r["lang"]] = per_lang.get(r["lang"], 0) + 1
        for p, v in r["target"].items():
            gold_total += 1
            if v is not None:
                kept_total += 1
                param_kept[f"{r['function']['name']}.{p}"] = param_kept.get(f"{r['function']['name']}.{p}", 0) + 1
    return {
        "label": label, "rows": len(rows), "per_function": per_fn, "per_lang": per_lang,
        "slot_kept_rate": round(kept_total / max(1, gold_total), 4),
        "top_kept_params": dict(sorted(param_kept.items(), key=lambda kv: -kv[1])[:20]),
    }


def read_jsonl(p: pathlib.Path):
    with p.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main():
    OUT.mkdir(exist_ok=True)
    train_rows = dedup([r for d in read_jsonl(FCFT / "data/final/train.jsonl") for r in process_dialog(d)])
    val_rows = dedup([r for d in read_jsonl(FCFT / "data/final/val.jsonl") for r in process_dialog(d)])
    # gold (merged superset) — only tool_use rows with reference tool_calls
    gold_rows, skipped_gold = [], 0
    seen_gold = set()
    for g in read_jsonl(FCFT / "data/test_gold_merged.jsonl"):
        gid = g.get("gold_id")
        if gid in seen_gold:
            continue
        seen_gold.add(gid)
        ref = g.get("reference") or {}
        calls = ref.get("tool_calls") or []
        if not calls:
            skipped_gold += 1
            continue
        turns = g.get("user_turns") or []
        utt = turns[-1].strip() if turns else ""
        if not utt:
            skipped_gold += 1
            continue
        tools_by_name = {}
        for t in g.get("tools") or []:
            f = t.get("function") or t
            if f.get("name"):
                tools_by_name.setdefault(f["name"], f)
        for c in calls[:4]:
            fname = c.get("name")
            schema = tools_by_name.get(fname)
            if not schema:
                skipped_gold += 1
                continue
            args = c.get("args") or c.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = None
            if not isinstance(args, dict) or not args:
                skipped_gold += 1
                continue
            utt_f, utt_t = fold(utt), tokens(utt)
            props = (schema.get("parameters") or {}).get("properties") or {}
            target = {p: param_target(p, props.get(p, {}), args.get(p), utt_f, utt_t, g.get("language", ""))[0]
                      for p in schema_param_order(schema.get("parameters") or {}, args)}
            gold_rows.append({
                "id": "gold-" + hashlib.sha1(f"{gid}|{fname}".encode()).hexdigest()[:12],
                "source": "gold", "lang": g.get("language"),
                "function": clean_schema(schema), "utterance": utt, "args": args, "target": target,
            })
    gold_rows = dedup(gold_rows)

    for name, rows in (("core_train", train_rows), ("core_val", val_rows), ("core_test", gold_rows)):
        with (OUT / f"{name}.jsonl").open("w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        st = stats(rows, name)
        (OUT / f"{name}_stats.json").write_text(json.dumps(st, ensure_ascii=False, indent=2))
        print(f"[{name}] rows={st['rows']} langs={st['per_lang']} kept_rate={st['slot_kept_rate']}")
        print("  top fns:", dict(list(st['per_function'].items())[:8]))
        print("  top kept params:", dict(list(st['top_kept_params'].items())[:12]))
    print(f"[gold] skipped rows (no call/no schema): {skipped_gold}")


if __name__ == "__main__":
    sys.exit(main())
