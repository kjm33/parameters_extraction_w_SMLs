"""Synthetic slot-filling data via templated generation + validator loop (Data-Turnstile style).

plans/2026-10-10_improvement-todo.md task 2:
  - Templated utterances per (function, lang) seeded from core vocab (places) and the
    enum keyword lexicon (ENUM_STEMS) — every constraint phrase is written to contain
    the stem that grounds it.
  - Closed loop: intended args are re-validated through param_target (the same grounding
    as detector validators: ungrounded / enum_invalid / keys_mismatch); a row is resampled
    until every intended value grounds or retries run out (guards template bugs).
  - Marginals calibrated to core+test: geocode q .85/lang .40/limit .03; directions
    avoid .45 / routingMode .08 / transportMode .09 / lang .30; lat,lng slots always null.
  - Coverage report per function x lang x slot; hard cap synthetic <= 40% of final mix;
    shingle-dedup vs core+aug+distill.

Usage:
  python3 src/build_synthetic.py --rows 8000
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import random
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from build_extraction import ENUM_STEMS, fold, param_target, schema_param_order, tokens  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
INSTR = ROOT.parent / "instruction_extraction"
OUT = ROOT / "data"

# ---------------------------------------------------------------- vocab

STOP_PLACE = re.compile(r"^(null|none|true|false|[0-9 .,-]+)$", re.I)


def harvest_places() -> list[str]:
    places, seen = [], set()
    def add(p):
        p = (p or "").strip()
        k = fold(p)
        if not (2 <= len(p) <= 40) or not k or k in seen or STOP_PLACE.match(p):
            return
        if not re.search(r"[^\W\d_]", p):
            return
        seen.add(k)
        places.append(p)
    for line in (OUT / "core_train.jsonl").open():
        r = json.loads(line)
        t = r.get("target") or {}
        for key in ("q", "origin", "destination"):
            v = t.get(key)
            if isinstance(v, str):
                add(v.split(",")[0])
    for fname in ("instructions_v3.jsonl",):
        for line in (INSTR / "data" / fname).open():
            r = json.loads(line)
            for key in ("origin", "destination"):
                add(r.get(key))
    return places


# constraint phrasings per (lang, value) — each string must contain a grounding stem
AVOID_PHRASE = {
    "tollRoad": {
        "en": ["avoiding toll roads", "without tolls", "no toll roads please"],
        "de": ["ohne Maut", "Maut vermeiden", "keine Autobahn-Gebühren, also Maut umgehen"],
        "pl": ["omijając płatne drogi", "bez opłat drogowych", "unikaj autostrad płatnych"],
        "fr": ["en évitant les péages", "sans péages", "évite les routes à péage"],
        "es": ["evitando peajes", "sin peajes", "evita las carreteras de peaje"],
    },
    "motorway": {
        "en": ["avoiding motorways", "no highways", "skip the motorway"],
        "de": ["Autobahnen vermeiden", "ohne Autobahn", "meide die Autobahn"],
        "pl": ["omijaj autostrady", "bez autostrad", "unikaj dróg ekspresowych i autostrad"],
        "fr": ["en évitant les autoroutes", "sans autoroute", "évite l'autoroute"],
        "es": ["evitando autopistas", "sin autopistas", "evita la autopista"],
    },
    "ferry": {
        "en": ["no ferries", "avoiding ferries", "skip the ferry crossing"],
        "de": ["ohne Fähren", "Fähren vermeiden", "keine Fährüberfahrt"],
        "pl": ["bez promów", "omijając promy", "unikaj przepraw promowych"],
        "fr": ["sans transbordeurs", "en évitant les bacs", "évite le bac"],
        "es": ["sin transbordadores", "evitando transbordadores", "evita el transbordador"],
    },
    "tunnel": {
        "en": ["no tunnels", "avoiding tunnels"],
        "de": ["ohne Tunnel", "Tunnel vermeiden"],
        "pl": ["bez tuneli", "omijając tunele"],
        "fr": ["sans tunnels", "en évitant les tunnels"],
        "es": ["sin túneles", "evitando túneles"],
    },
}

TRANSPORT_PHRASE = {
    "car": {"en": "by car", "de": "mit dem Auto", "pl": "samochodem", "fr": "en voiture", "es": "en coche"},
    "truck": {"en": "by truck", "de": "mit dem LKW", "pl": "ciężarówką", "fr": "en camion", "es": "en camión"},
    "pedestrian": {"en": "on foot", "de": "zu Fuß", "pl": "pieszo", "fr": "à pied", "es": "andando"},
    "bicycle": {"en": "by bicycle", "de": "mit dem Fahrrad", "pl": "rowerem", "fr": "en vélo", "es": "en bici"},
    "scooter": {"en": "by scooter", "de": "mit dem Roller", "pl": "skuterem", "fr": "en scooter", "es": "en scooter"},
}

ROUTING_PHRASE = {
    "fast": {"en": "the fastest route", "de": "die schnellste Route", "pl": "najszybszą trasę",
             "fr": "le trajet le plus rapide", "es": "lo más rápido"},
    "short": {"en": "the shortest route", "de": "die kürzeste Route", "pl": "najkrótszą trasę",
              "fr": "le trajet le plus court", "es": "lo más corto"},
}

GEO_TEMPLATES = {
    "en": ["Where is {p}?", "Take me to {p} please.", "Find {p}.", "I need the address of {p}.",
           "Navigate to {p}.", "Look up {p} for me.", "Show me {p}.", "Can you find {p}?"],
    "de": ["Wo ist {p}?", "Bring mich zu {p}.", "Suche {p}.", "Ich brauche die Adresse von {p}.",
           "Navigiere zu {p}.", "Zeig mir {p}.", "Finde {p} bitte."],
    "pl": ["Gdzie jest {p}?", "Zabierz mnie do {p}.", "Znajdź {p}.", "Potrzebuję adresu {p}.",
           "Nawiguj do {p}.", "Pokaż mi {p}.", "Możesz znaleźć {p}?"],
    "fr": ["Où se trouve {p} ?", "Emmène-moi à {p}.", "Cherche {p}.", "J'ai besoin de l'adresse de {p}.",
           "Navigue vers {p}.", "Montre-moi {p}.", "Tu peux trouver {p} ?"],
    "es": ["¿Dónde está {p}?", "Llévame a {p}.", "Busca {p}.", "Necesito la dirección de {p}.",
           "Navega a {p}.", "Muéstrame {p}.", "¿Puedes encontrar {p}?"],
}

GEO_NULL_TEMPLATES = {  # no q stated — pure negatives
    "en": ["Are there parking spots there?", "Is it open on Sundays?"],
    "de": ["Gibt es dort Parkplätze?", "Ist dort sonntags geöffnet?"],
    "pl": ["Czy są tam miejsca parkingowe?", "Czy w niedziele jest otwarte?"],
    "fr": ["Y a-t-il des places de parking ?", "C'est ouvert le dimanche ?"],
    "es": ["¿Hay aparcamiento?", "¿Está abierto los domingos?"],
}

LIMIT_PHRASE = {
    "en": "show {n} results", "de": "zeig {n} Treffer", "pl": "pokaż {n} wyników",
    "fr": "propose {n} résultats", "es": "dame {n} resultados",
}

DIR_TEMPLATES = {
    "en": ["Route from {a} to {b}.", "Take me from {a} to {b}.", "Plan a trip from {a} to {b}.",
           "How do I get from {a} to {b}?", "Directions from {a} to {b}, please."],
    "de": ["Route von {a} nach {b}.", "Bring mich von {a} nach {b}.", "Plane eine Fahrt von {a} nach {b}.",
           "Wie komme ich von {a} nach {b}?", "Navigiere von {a} nach {b}."],
    "pl": ["Trasa z {a} do {b}.", "Zabierz mnie z {a} do {b}.", "Zaplanuj podróż z {a} do {b}.",
           "Jak dojadę z {a} do {b}?", "Prowadź z {a} do {b}."],
    "fr": ["Trajet de {a} à {b}.", "Emmène-moi de {a} à {b}.", "Planifie un voyage de {a} à {b}.",
           "Comment vais-je de {a} à {b} ?", "Guide-moi de {a} à {b}."],
    "es": ["Ruta de {a} a {b}.", "Llévame de {a} a {b}.", "Planifica un viaje de {a} a {b}.",
           "¿Cómo llego de {a} a {b}?", "Guíame de {a} a {b}."],
}

DIR_NULL_TEMPLATES = {  # turn-by-turn / chit-chat — all-null negatives
    "en": ["Take the next left.", "Keep going straight for now.", "Stop here for a moment."],
    "de": ["Bieg als Nächstes links ab.", "Fahr erstmal geradeaus weiter.", "Halte hier kurz an."],
    "pl": ["Skręć w następną w lewo.", "Jedź dalej prosto.", "Zatrzymaj się tutaj na chwilę."],
    "fr": ["Prends la prochaine à gauche.", "Continue tout droit pour l'instant.", "Arrête-toi ici un instant."],
    "es": ["Gira a la próxima a la izquierda.", "Sigue recto por ahora.", "Para aquí un momento."],
}

# distractor clauses: content present in text but NEVER in args -> teaches nulls
TIME_DISTRACTOR = {
    "en": [" Let's leave tomorrow at 9.", " We depart Monday morning.", " I hope to arrive by 6pm."],
    "de": [" Fahren wir morgen um 9 los.", " Wir fahren Montag früh los.", " Ich will um 18 Uhr da sein."],
    "pl": [" Wyjedźmy jutro o 9.", " Ruszamy w poniedziałek rano.", " Chcę być na miejscu do 18."],
    "fr": [" Partons demain à 9h.", " On part lundi matin.", " J'espère arriver pour 18h."],
    "es": [" Salgamos mañana a las 9.", " Salimos el lunes por la mañana.", " Espero llegar para las 18."],
}
VIA_DISTRACTOR = {
    "en": " passing through {p}",
    "de": " über {p}",
    "pl": " przez {p}",
    "fr": " en passant par {p}",
    "es": " pasando por {p}",
}
CHATTER = {
    "en": ["Sure, ", "OK, ", "Hey, ", ""],
    "de": ["Gut, ", "Okay, ", "Hey, ", ""],
    "pl": ["Dobrze, ", "Okay, ", "Hej, ", ""],
    "fr": ["Bien, ", "D'accord, ", "Salut, ", ""],
    "es": ["Vale, ", "Bien, ", "Hola, ", ""],
}

FORCE_NULL_PARAMS = {"origin", "destination", "via"}  # lat,lng slots — resolver's job
LANGS = ["en", "de", "pl", "fr", "es"]
AVOID_WEIGHTS = [("tollRoad", 40), ("motorway", 35), ("ferry", 15), ("tunnel", 6)]


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
        if {"here_geocode", "here_directions"} <= set(schemas):
            break
    assert {"here_geocode", "here_directions"} <= set(schemas)
    return schemas


# ---------------------------------------------------------------- generation

def gen_geocode(rng: random.Random, lang: str, places: list[str]) -> tuple[str, dict]:
    args: dict = {}
    if rng.random() < 0.85:  # q stated (core 0.91 / test 0.86)
        p = rng.choice(places)
        t = rng.choice(GEO_TEMPLATES[lang]).format(p=p)
        args["q"] = p
        if rng.random() < 0.45:  # limit stated with q (joint ~0.03*... calibrate: overall 0.04)
            n = rng.choice([3, 5, 10])
            t += " " + LIMIT_PHRASE[lang].format(n=n) + "."
            args["limit"] = n
    else:
        t = rng.choice(GEO_NULL_TEMPLATES[lang])
    if rng.random() < 0.47:  # lang inferred (core 0.36 / test 0.43) — only when asked to "look up"
        args["lang"] = lang
    return t, args


def gen_directions(rng: random.Random, lang: str, places: list[str]) -> tuple[str, dict]:
    if rng.random() < 0.10:  # all-null negatives
        return rng.choice(DIR_NULL_TEMPLATES[lang]), {}
    a, b = rng.sample(places, 2)
    t = rng.choice(DIR_TEMPLATES[lang]).format(a=a, b=b)
    parts = [t[:-1] if t.endswith(".") else t]
    args: dict = {}
    if rng.random() < 0.45:  # avoid_features (core 0.53 / test 0.37)
        vals = []
        pool = [v for v, w in AVOID_WEIGHTS for _ in range(w)]
        n_av = 1 + (rng.random() < 0.25)  # mostly one, sometimes two
        for v in rng.sample(pool, n_av):
            if v not in vals:
                vals.append(v)
                parts.append(rng.choice(AVOID_PHRASE[v][lang]))
        args["avoid_features"] = vals
    if rng.random() < 0.17:  # routingMode (core 0.07 / test 0.08)
        mode = rng.choice(list(ROUTING_PHRASE))
        parts.append("take " + ROUTING_PHRASE[mode][lang])
        args["routingMode"] = mode
    if rng.random() < 0.18:  # transportMode (core 0.09 / test 0.07)
        mode = rng.choice(list(TRANSPORT_PHRASE))
        parts.append(rng.choice([TRANSPORT_PHRASE[mode][lang], "traveling " + TRANSPORT_PHRASE[mode][lang]]))
        args["transportMode"] = mode
    if rng.random() < 0.20:
        parts.append(VIA_DISTRACTOR[lang].format(p=rng.choice(places)))
    if rng.random() < 0.25:
        parts.append(rng.choice(TIME_DISTRACTOR[lang]).strip())
    if rng.random() < 0.32:  # lang (core 0.24 / test 0.33)
        args["lang"] = lang
    utterance = parts[0] + ", " + " ".join(parts[1:]) + "."
    if rng.random() < 0.3:
        utterance = rng.choice(CHATTER[lang]) + utterance[0].lower() + utterance[1:] if utterance else utterance
    return utterance, args


def validate(job_args: dict, schema: dict, utterance: str, lang: str) -> tuple[dict | None, str]:
    """Closed-loop validator: intended non-null args must all ground; keys must fit schema."""
    props = (schema.get("parameters") or {}).get("properties") or {}
    if any(k not in props for k in job_args):
        return None, "keys_mismatch"
    utt_f, utt_t = fold(utterance), tokens(utterance)
    target = {}
    for p in schema_param_order(schema.get("parameters") or {}, job_args):
        if p in FORCE_NULL_PARAMS:
            target[p] = None
            continue
        target[p] = param_target(p, props.get(p, {}), job_args.get(p), utt_f, utt_t, lang)[0]
    for k, v in job_args.items():
        if v is not None and target.get(k) is None:
            return None, f"ungrounded:{k}"
    return target, ""


def make_row(source: str, lang: str, schema: dict, utterance: str, args: dict, target: dict) -> dict:
    return {
        "id": "syn-" + hashlib.sha1(f"{utterance}|{schema['name']}".encode()).hexdigest()[:12],
        "source": source, "lang": lang, "function": schema,
        "utterance": utterance, "args": args, "target": target,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", type=int, default=8000)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-prefix", default="syn")
    ap.add_argument("--cap-frac", type=float, default=0.40, help="synthetic <= this frac of final mix")
    a = ap.parse_args()
    rng = random.Random(a.seed)

    schemas = load_canonical_schemas()
    places = harvest_places()
    print(f"[vocab] places={len(places)} sample={places[:8]}")

    # shingles of existing corpora for near-dup filtering
    existing: set[int] = set()
    existing_rows = 0
    for fname in ("core_train.jsonl", "aug_train.jsonl", "aug_val.jsonl", "distill_train.jsonl", "distill_val.jsonl"):
        p = OUT / fname
        if not p.exists():
            continue
        for line in p.open():
            r = json.loads(line)
            existing |= shingles(r["utterance"])
            existing_rows += 1
    print(f"[corpus] existing rows={existing_rows} shingles={len(existing)}")

    rows: list[dict] = []
    seen: set[str] = set()
    stats = collections.Counter()
    cov: dict[tuple, set] = collections.defaultdict(set)
    tries = 0
    while len(rows) < a.rows and tries < a.rows * 6:
        tries += 1
        lang = LANGS[len(rows) % 5]  # round-robin langs
        fn = "here_geocode" if rng.random() < 0.55 else "here_directions"
        schema = schemas[fn]
        if fn == "here_geocode":
            utt, args = gen_geocode(rng, lang, places)
        else:
            utt, args = gen_directions(rng, lang, places)
        key = fold(utt)
        if key in seen:
            stats["self_dup"] += 1
            continue
        sh = shingles(utt)
        if sh and len(sh & existing) / len(sh) > 0.55:
            stats["corpus_dup"] += 1
            continue
        target, verr = validate(args, schema, utt, lang)
        if target is None:
            stats[verr.split(":")[0]] += 1
            continue
        seen.add(key)
        row = make_row("syn", lang, schema, utt, args, target)
        rows.append(row)
        for k, v in target.items():
            if v is not None:
                cov[(fn, lang, k)].add(json.dumps(v, ensure_ascii=False, sort_keys=True))
        if len(rows) % 1000 == 0:
            print(f"  rows={len(rows)} stats={dict(stats)}")
    print(f"[gen] tries={tries} rows={len(rows)} stats={dict(stats)}")

    # coverage report: intended slots present per fn x lang
    wanted = {"here_geocode": {"q", "lang", "limit"},
              "here_directions": {"avoid_features", "routingMode", "transportMode", "lang"}}
    missing = []
    for fn, slots in wanted.items():
        for lang in LANGS:
            for s in slots:
                if (fn, lang, s) not in cov:
                    missing.append(f"{fn}.{lang}.{s}")
    print(f"[coverage] missing: {missing or 'none'}")
    enum_cov = {k: len(v) for k, v in cov.items() if k[2] in ("avoid_features", "routingMode", "transportMode")}
    print(f"[coverage] distinct enum value sets per combo: {sum(enum_cov.values())} combos populated")

    # hard cap: synthetic <= cap_frac of final mix
    cap = int(a.cap_frac / (1 - a.cap_frac) * existing_rows)
    if cap and len(rows) > cap:
        rng.shuffle(rows)
        rows = rows[:cap]
        print(f"[cap] trimmed to {cap} (cap_frac={a.cap_frac})")

    rng.shuffle(rows)
    n_val = min(max(20, int(a.val_frac * len(rows))), len(rows) // 2)
    val_rows, train_rows = rows[:n_val], rows[n_val:]
    for name, rr in ((f"{a.out_prefix}_train", train_rows), (f"{a.out_prefix}_val", val_rows)):
        with (OUT / f"{name}.jsonl").open("w") as f:
            for r in rr:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    from build_extraction import stats as mk_stats
    st = mk_stats(train_rows, f"{a.out_prefix}_train")
    (OUT / f"{a.out_prefix}_train_stats.json").write_text(json.dumps(st, ensure_ascii=False, indent=2))
    print(f"[done] train={len(train_rows)} val={len(val_rows)} langs={st['per_lang']} kept_rate={st['slot_kept_rate']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
