"""Augment slot-filling data from ../instruction_extraction (heuristic, teacher-free).

Sources:
  - instructions_v3.jsonl (21,959): text + origin/destination/constraints/language fields
  - dialogues.jsonl (2,029): driver turns, OD via per-language FROM-TO regex

Emits rows for the REAL here_geocode / here_directions schemas (canonical copies from
core_train). Target policy identical to build_extraction (same param_target machinery):
  - geocode rows: q = origin/destination when the text supports it (verbatim/segment)
  - directions rows: only text-supported slots (avoid_features / routingMode / transportMode
    / lang); origin/destination stay null (they are lat,lng slots -> resolver's job),
    consistent with core supervision
  - constraints field is NOT trusted blindly: enum values kept only if keyword stems
    appear in the text

Dedup vs core corpus via 5-gram shingle overlap; capped and lang-balanced (en boosted,
it is underrepresented in core). Cap: aug <= ~40% of final mix.

Usage:
  python3 src/build_augment.py
"""
from __future__ import annotations

import json
import pathlib
import random
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from build_extraction import ENUM_STEMS, enum_supported, fold, number_found, param_target, schema_param_order, tokens, verbatim  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
INSTR = ROOT.parent / "instruction_extraction"
OUT = ROOT / "data"

FROMTO_RE = {
    "de": re.compile(r"\bvon\s+(?P<origin>[\w .'-]{1,30}?)\s+nach\s+(?P<dest>[\w .'-]{1,30}?)(?=[,.;:!?\n]|\s(?:und|oder|über|ohne|via|mit)\b|$)", re.I),
    "fr": re.compile(r"\bde\s+(?P<origin>[\w .'-]{1,30}?)\s+à\s+(?P<dest>[\w .'-]{1,30}?)(?=[,.;:!?\n]|\s(?:en|par|sans|puis|et|avec|pour)\b|$)", re.I),
    "es": re.compile(r"\bde\s+(?P<origin>[\w .'-]{1,30}?)\s+a\s+(?P<dest>[\w .'-]{1,30}?)(?=[,.;:!?\n]|\s(?:sin|evitando|por|y|con|para)\b|$)", re.I),
    "pl": re.compile(r"\bz\s+(?P<origin>[\w .'-]{1,30}?)\s+do\s+(?P<dest>[\w .'-]{1,30}?)(?=[,.;:!?\n]|\s(?:i|bez|przez|unikaj|z)\b|$)", re.I),
    "en": re.compile(r"\bfrom\s+(?P<origin>[\w .'-]{1,30}?)\s+to\s+(?P<dest>[\w .'-]{1,30}?)(?=[,.;:!?\n]|\s(?:via|without|avoiding|and|with|for)\b|$)", re.I),
}

CONSTRAINT_TO_ARGS = {
    "avoid_tolls": ("avoid_features", ["tollRoad"]),
    "no_ferries": ("avoid_features", ["ferry"]),
    "avoid_highways": ("avoid_features", ["motorway"]),
    "fastest": ("routingMode", "fast"),
    "avoid_cities": None, "scenic": None, "breaks": None, "ev_charging": None,
    "rest_stop": None, "via_waypoint": None, "border": None, "cheap": None, "winter": None,
}

TRANSPORT_CANDIDATES = ["car", "truck", "pedestrian", "bicycle", "scooter"]


def shingles(text: str, n: int = 5) -> set[int]:
    ws = re.findall(r"\w+", fold(text))
    return {hash(" ".join(ws[i:i + n])) for i in range(max(0, len(ws) - n + 1))}


def load_canonical_schemas() -> dict[str, dict]:
    schemas = {}
    for line in (OUT / "core_train.jsonl").open():
        r = json.loads(line)
        fn = r["function"]["name"]
        if fn not in schemas:
            schemas[fn] = r["function"]
        if len(schemas) >= 2 and {"here_geocode", "here_directions"} <= set(schemas):
            break
    for need in ("here_geocode", "here_directions"):
        assert need in schemas, f"canonical schema {need} missing from core_train"
    return schemas


def make_row(source: str, lang: str, fn_schema: dict, utterance: str, args: dict) -> dict | None:
    props = (fn_schema.get("parameters") or {}).get("properties") or {}
    utt_f, utt_t = fold(utterance), tokens(utterance)
    target = {p: param_target(p, props.get(p, {}), args.get(p), utt_f, utt_t, lang)[0]
              for p in schema_param_order(fn_schema.get("parameters") or {}, args)}
    if all(v is None for v in target.values()):
        return None
    import hashlib
    return {
        "id": f"{source}-" + hashlib.sha1(f"{utterance}|{fn_schema['name']}".encode()).hexdigest()[:12],
        "source": source, "lang": lang, "function": fn_schema,
        "utterance": utterance, "args": args, "target": target,
    }


def text_ok(t: str) -> bool:
    if not t or len(t) < 30 or len(t) > 600:
        return False
    if t.lstrip()[:1] in "{[<#|-":  # structured/format artifacts
        return False
    letters = re.findall(r"[A-Za-zÀ-ž]", t)
    return len(letters) >= 0.6 * max(1, len(t.replace(" ", "")))


def v3_candidates():
    for line in (INSTR / "data/instructions_v3.jsonl").open():
        r = json.loads(line)
        t = (r.get("text") or "").strip()
        if not text_ok(t):
            continue
        lang = r.get("language") or "en"
        yield lang, t, r.get("origin"), r.get("destination"), r.get("constraints") or []


def dialogue_candidates():
    for line in (INSTR / "data/dialogues.jsonl").open():
        d = json.loads(line)
        lang = d.get("language") or "en"
        rx = FROMTO_RE.get(lang)
        for turn in d.get("turns") or []:
            if turn.get("speaker") != "driver":
                continue
            t = (turn.get("text") or "").strip()
            if not text_ok(t):
                continue
            origin = dest = None
            if rx:
                m = rx.search(t)
                if m:
                    origin, dest = (m.group("origin") or "").strip(), (m.group("dest") or "").strip()
            yield lang, t, origin, dest, []


def directions_args(lang: str, t: str, constraints: list) -> dict:
    args: dict = {"lang": lang}
    for c in constraints:
        key = fold(str(c).split(":")[0])
        mapping = CONSTRAINT_TO_ARGS.get(key)
        if not mapping:
            continue
        pname, val = mapping
        if pname in args and isinstance(args[pname], list):
            args[pname] = args[pname] + val
        else:
            args[pname] = val
    for mode in TRANSPORT_CANDIDATES:
        if enum_supported(mode, fold(t)):
            args["transportMode"] = mode
            break
    return args


def main():
    random.seed(42)
    schemas = load_canonical_schemas()
    geo_s, dir_s = schemas["here_geocode"], schemas["here_directions"]

    # core corpus shingles for near-dup filtering
    core_sh: set[int] = set()
    for line in (OUT / "core_train.jsonl").open():
        core_sh |= shingles(json.loads(line)["utterance"])
    print(f"core corpus shingles: {len(core_sh)}")

    seen_self: set[str] = set()
    pool: list[dict] = []
    n_src = n_dup_self = n_dup_core = n_no_target = 0
    for source, gen in (("aug_i", v3_candidates()), ("aug_d", dialogue_candidates())):
        for lang, t, origin, dest, constraints in gen:
            n_src += 1
            key = fold(t)[:160]
            if key in seen_self:
                n_dup_self += 1
                continue
            seen_self.add(key)
            sh = shingles(t)
            if sh and len(sh & core_sh) / len(sh) > 0.55:
                n_dup_core += 1
                continue
            # directions row (only when a mappable constraint or transport mode exists)
            dargs = directions_args(lang, t, constraints)
            if len(dargs) > 1:  # more than just lang
                row = make_row(source, lang, dir_s, t, dargs)
                if row:
                    pool.append(row)
                else:
                    n_no_target += 1
            # geocode rows for supported locations
            for loc in (origin, dest):
                if loc and 2 <= len(loc) <= 60:
                    row = make_row(source, lang, geo_s, t, {"q": loc, "lang": lang})
                    if row:
                        pool.append(row)
                    else:
                        n_no_target += 1
    print(f"source rows read={n_src} self_dup={n_dup_self} core_dup={n_dup_core} no_target={n_no_target} pool={len(pool)}")

    # lang-balanced sampling: en boosted (565 core), directions preferred ~35% per lang
    core_langs = {}
    for line in (OUT / "core_train.jsonl").open():
        l = json.loads(line)["lang"]
        core_langs[l] = core_langs.get(l, 0) + 1
    quotas = {"en": 2500, "de": 1200, "pl": 1200, "fr": 1000, "es": 1000}
    by_lang: dict[str, list[dict]] = {}
    for r in pool:
        by_lang.setdefault(r["lang"], []).append(r)
    train_rows: list[dict] = []
    for lang, rows in by_lang.items():
        random.shuffle(rows)
        dirs = [r for r in rows if r["function"]["name"] == "here_directions"]
        geos = [r for r in rows if r["function"]["name"] == "here_geocode"]
        q = quotas.get(lang, 800)
        n_dir = min(len(dirs), int(q * 0.35))
        n_geo = min(len(geos), q - n_dir)
        train_rows += dirs[:n_dir] + geos[:n_geo]
    # hard cap: aug <= 40% of final mix
    cap = int(0.40 / 0.60 * sum(core_langs.values()))
    if len(train_rows) > cap:
        random.shuffle(train_rows)
        train_rows = train_rows[:cap]

    random.shuffle(train_rows)
    n_val = max(50, int(0.05 * len(train_rows)))
    val_rows, train_rows = train_rows[:n_val], train_rows[n_val:]
    for name, rows in (("aug_train", train_rows), ("aug_val", val_rows)):
        with (OUT / f"{name}.jsonl").open("w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    from build_extraction import stats
    for name, rows in (("aug_train", train_rows), ("aug_val", val_rows)):
        st = stats(rows, name)
        (OUT / f"{name}_stats.json").write_text(json.dumps(st, ensure_ascii=False, indent=2))
        print(f"[{name}] rows={st['rows']} langs={st['per_lang']} kept_rate={st['slot_kept_rate']} fns={st['per_function']}")


if __name__ == "__main__":
    sys.exit(main())
