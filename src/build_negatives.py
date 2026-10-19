"""Hammer-style negative training rows (efforts/2026-10-10-related-work-improvements/tasks.md task 4).

Targets the measured true-hallucination FP modes of the baseline (128 slots on test):
  B1  avoid_features predicted where gold is null (29)  -> road-type MENTIONS with
      use/take/via phrasing, no avoid-intent verb.
  B2  q/query predicted where gold is null (42)         -> pure chatter place-mentions
      (feelings/past events/facts; NO search verb, NO route intent — route-intent sets
      q in 90-94% of gold rows, so route traps would be mostly wrong supervision).
  B3  transportMode/routingMode predicted where null (11) -> route request + incidental
      vehicle/speed clause (possession/state/past tense, not travel mode).
  A   cross-schema near-miss traps: directions-constraint phrases under geocode-family
      schemas -> only q grounded; teaches schema-conditioned extraction.

lang slot is OFF-LIMITS (label-inconsistent in core gold; negatives would conflict).

Closed loop: intended-null slots are guaranteed by lexical guards (regex of the
intent markers whose PRESENCE would justify extraction); positive slots must ground
via param_target.  Shingle-dedup vs core/aug/distill/syn + self.

Usage:
  python3 src/build_negatives.py --rows 1100
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
from build_extraction import fold, param_target, schema_param_order, tokens  # noqa: E402
from build_synthetic import (  # noqa: E402
    CHATTER, DIR_TEMPLATES, GEO_TEMPLATES, TIME_DISTRACTOR, harvest_places,
    load_canonical_schemas, shingles,
)

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "data"
LANGS = ["en", "de", "pl", "fr", "es"]

# ---------------------------------------------------------------- guards
# intent markers whose presence would justify extraction; trap utterances must NOT match
AVOID_INTENT = {
    "en": r"avoid|without|skip|\bno (toll|highway|motorway|ferr|tunnel)",
    "de": r"vermeid|meid|umgeh|ohne (maut|autobahn|f[aä]hr|tunnel)|keine (autobahn|maut|f[aä]hr)",
    "pl": r"omin|unikaj|\bbez\b",
    "fr": r"[eé]vit|sans (p[eé]age|autoroute|tunnel|transbordeur|bac)",
    "es": r"evit|sin (peaje|autopista|t[uú]nel|transbordador)",
}
SEARCH_INTENT = {
    "en": r"where|find|address|look up|show me|navigate|take me|directions|route|search|locate",
    "de": r"wo ist|such|find|adresse|navigier|zeig|bring mich|route|lokalisier",
    "pl": r"gdzie|znajd|adres|nawiguj|poka[zż]|zabierz|tras|szukaj",
    "fr": r"o[uù] se trouve|o[uù] est|cherch|trouv|adresse|navigu|montre|emm[eè]ne|trajet|route",
    "es": r"d[oó]nde|busc|encontr|direcci[oó]n|naveg|mu[eé]stra|ll[eé]vame|ruta",
}
TRAVEL_MODE = (
    r"by (car|truck|bicycle|bike|scooter|foot)|on foot|travel(l)?ing|mit dem (auto|lkw|fahrrad|roller)|"
    r"zu fu[sß]|samochodem|ci[eę][żz]ar[oó]wk|rowerem|pieszo|skuterem|en (voiture|camion|v[eé]lo|bici|scooter)|"
    r"[aà] pied|andando|en coche|en cami[oó]n|en scooter"
)
ROUTING_SUPERL = (
    r"fastest|shortest|schnellste|k[uü]rzeste|najszybsz|najkr[oó]tsz|"
    r"le plus (rapide|court)|lo m[aá]s (r[aá]pido|corto)"
)

# ---------------------------------------------------------------- phrases
# B1: road-type mention, use/take/via phrasing (contains enum stems, no avoid verb)
USE_PHRASE = {
    "tollRoad": {
        "en": ["we'll pay the tolls", "the toll road was cheap", "taking the toll road"],
        "de": ["wir zahlen die Maut", "die Mautstraße war günstig", "über die Mautstraße"],
        "pl": ["zapłacimy za opłaty", "opłaty drogowe były wysokie", "jedziemy płatną trasą"],
        "fr": ["on paiera les péages", "la route à péage était pas chère", "par la route à péage"],
        "es": ["pagaremos los peajes", "la carretera de peaje era barata", "por la carretera de peaje"],
    },
    "motorway": {
        "en": ["take the motorway", "via the highway", "the motorway is free of traffic"],
        "de": ["über die Autobahn", "nehmen wir die Autobahn", "die Autobahn ist frei"],
        "pl": ["jedziemy autostradą", "tylko przez autostradę", "autostrada jest pusta"],
        "fr": ["par l'autoroute", "prends l'autoroute", "l'autoroute est dégagée"],
        "es": ["por la autopista", "toma la autopista", "la autopista está libre"],
    },
    "ferry": {
        "en": ["we'll take the ferry", "the ferry crossing was nice"],
        "de": ["wir nehmen die Fähre", "die Fährüberfahrt war schön"],
        "pl": ["bierzemy prom", "przeprawa promowa była spokojna"],
        "fr": ["on prend le bac", "la traversée en bac était agréable"],
        "es": ["tomaremos el transbordador", "el viaje en transbordador fue agradable"],
    },
    "tunnel": {
        "en": ["through the tunnel", "the tunnel is closed anyway"],
        "de": ["durch den Tunnel", "der Tunnel ist eh zu"],
        "pl": ["przez tunel", "tunel jest zamknięty"],
        "fr": ["par le tunnel", "le tunnel est fermé"],
        "es": ["por el túnel", "el túnel está cerrado"],
    },
}

# B2: pure chatter place-mention, NO question/no search verb (uses raw place form)
CHATTER_PLACE = {
    "en": ["I love {p}.", "We visited {p} last summer.", "My sister lives in {p}.",
           "The weather in {p} was awful.", "I've never been to {p}.", "{p} is my favorite city."],
    "de": ["Ich liebe {p}.", "Wir waren letzten Sommer in {p}.", "Meine Schwester wohnt in {p}.",
           "Das Wetter in {p} war schrecklich.", "Ich war noch nie in {p}.", "{p} ist meine Lieblingsstadt."],
    "pl": ["Uwielbiam {p}.", "Odwiedźiliśmy {p} ostatniego lata.", "Moja siostra mieszka w {p}.",
           "Pogoda w {p} była okropna.", "Nigdy nie byłem w {p}.", "{p} to moje ulubione miasto."],
    "fr": ["J'adore {p}.", "Nous avons visité {p} l'été dernier.", "Ma sœur habite à {p}.",
           "Le temps à {p} était horrible.", "Je ne suis jamais allé à {p}.", "{p} est ma ville préférée."],
    "es": ["Me encanta {p}.", "Visitamos {p} el verano pasado.", "Mi hermana vive en {p}.",
           "El tiempo en {p} era horrible.", "Nunca he estado en {p}.", "{p} es mi ciudad favorita."],
}

# B3: incidental vehicle/speed clause (possession/state/past — not travel mode)
VEHICLE_INCIDENTAL = {
    "en": ["My car is in the shop though.", "That truck is blocking the street.",
           "I left my bicycle at home.", "The bus was late again today."],
    "de": ["Mein Auto ist aber in der Werkstatt.", "Der LKW blockiert gerade die Straße.",
           "Ich habe mein Fahrrad zu Hause gelassen.", "Der Bus war heute schon wieder spät."],
    "pl": ["Mój samochód jest w warsztacie.", "Ta ciężarówka blokuje ulicę.",
           "Zostawiłem rower w domu.", "Autobus dziś się spóźniał."],
    "fr": ["Ma voiture est au garage d'ailleurs.", "Le camion bloque la rue.",
           "J'ai laissé mon vélo à la maison.", "Le bus était encore en retard."],
    "es": ["Mi coche está en el taller.", "Ese camión bloquea la calle.",
           "Dejé la bici en casa.", "El autobús llegó tarde otra vez."],
}
FAST_INCIDENTAL = {
    "en": "That was a fast trip yesterday.",
    "de": "Das war gestern eine schnelle Fahrt.",
    "pl": "To była szybka przejażdżka.",
    "fr": "C'était un trajet rapide hier.",
    "es": "Fue un viaje rápido ayer.",
}

# A: cross-schema — constraint phrase under geocode-family schema (q grounds, rest null)
CROSS_CONSTRAINT = {
    "en": [" No toll roads on the way.", " I'm traveling by car.", " We'll take the ferry there.",
           " Take the motorway there.", " Show the fastest route afterwards."],
    "de": [" Keine Mautstraße auf dem Weg.", " Ich fahre mit dem Auto.", " Wir nehmen die Fähre.",
           " Über die Autobahn bitte.", " Danach die schnellste Route."],
    "pl": [" Bez płatnych dróg po drodze.", " Jadę samochodem.", " Bierzemy prom.",
           " Jedziemy autostradą.", " Potem najkrótszą trasę."],
    "fr": [" Pas de route à péage en chemin.", " Je voyage en voiture.", " On prend le bac.",
           " Par l'autoroute s'il te plaît.", " Ensuite le trajet le plus court."],
    "es": [" Sin carreteras de peaje en el camino.", " Viajo en coche.", " Tomamos el transbordador.",
           " Por la autopista por favor.", " Después lo más rápido."],
}

GEO_FNS = ["here_geocode", "nominatim_geocode", "tomtom-geocode", "here_search_places"]
FORCE_NULL_PARAMS = {"origin", "destination", "via"}


def guard_ok(utt: str, guards: list[str]) -> bool:
    u = fold(utt)
    return not any(re.search(g, u, re.I) for g in guards)


def build_target(schema: dict, args: dict, utterance: str, lang: str) -> dict | None:
    """Target with intended args grounded; None if a positive arg fails grounding."""
    props = (schema.get("parameters") or {}).get("properties") or {}
    if any(k not in props for k in args):
        return None
    utt_f, utt_t = fold(utterance), tokens(utterance)
    target = {}
    for p in schema_param_order(schema.get("parameters") or {}, args):
        if p in FORCE_NULL_PARAMS:
            target[p] = None
            continue
        target[p] = param_target(p, props.get(p, {}), args.get(p), utt_f, utt_t, lang)[0]
    for k, v in args.items():
        if v is not None and target.get(k) is None:
            return None
    return target


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", type=int, default=1100)
    ap.add_argument("--val-frac", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out-prefix", default="neg")
    a = ap.parse_args()
    rng = random.Random(a.seed)

    schemas = load_canonical_schemas()
    for line in (OUT / "core_train.jsonl").open():
        r = json.loads(line)
        schemas.setdefault(r["function"]["name"], r["function"])
    places = harvest_places()
    geo_fns = [f for f in GEO_FNS if f in schemas]
    print(f"[vocab] places={len(places)} geo_fns={geo_fns}")

    base_existing: set[int] = set()
    existing_rows = 0
    for fname in ("core_train.jsonl", "aug_train.jsonl", "aug_val.jsonl",
                  "distill_train.jsonl", "distill_val.jsonl", "syn_train.jsonl", "syn_val.jsonl"):
        p = OUT / fname
        if not p.exists():
            continue
        for line in p.open():
            r = json.loads(line)
            base_existing |= shingles(r["utterance"])
            existing_rows += 1
    print(f"[corpus] existing rows={existing_rows} shingles={len(base_existing)}")
    self_seen: set[str] = set()

    def emit(family: str, lang: str, schema: dict, utterance: str, args: dict) -> dict | None:
        utt = rng.choice(CHATTER[lang]) + utterance if rng.random() < 0.25 else utterance
        tgt = build_target(schema, args, utt, lang)
        if tgt is None:
            return None
        if shingles(utt) & base_existing:
            return None
        key = fold(utt)
        if key in self_seen:
            return None
        self_seen.add(key)
        return {"id": "neg-" + hashlib.sha1(f"{utt}|{schema['name']}".encode()).hexdigest()[:12],
                "source": f"neg-{family}", "lang": lang, "function": schema,
                "utterance": utt, "args": args, "target": tgt}

    fam_weights = [("B1_avoid_use", 28), ("B2_chatter_place", 28), ("B3_vehicle", 24), ("A_cross", 28)]
    fam_pool = [f for f, w in fam_weights for _ in range(w)]
    lang_pool = LANGS + ["de"]  # de over-weight: avoid FPs were de-heavy

    def maybe_tail(lang: str, p: float = 0.3) -> str:
        if rng.random() < p:
            return " " + rng.choice(TIME_DISTRACTOR[lang]).strip()
        return ""

    rows: list[dict] = []
    seen: set[str] = set()
    stats = collections.Counter()
    tries = 0
    while len(rows) < a.rows and tries < a.rows * 8:
        tries += 1
        fam = rng.choice(fam_pool)
        lang = rng.choice(lang_pool)
        sdir = schemas["here_directions"]
        if fam == "B1_avoid_use":
            val = rng.choice(list(USE_PHRASE))
            pa, pb = rng.sample(places, 2)
            base = rng.choice(DIR_TEMPLATES[lang]).format(a=pa, b=pb)
            phr = rng.choice(USE_PHRASE[val][lang])
            utt = base.rstrip(".?!") + ", " + phr + "." + maybe_tail(lang)
            if not guard_ok(utt, [AVOID_INTENT[lang]]):
                stats["guard_B1"] += 1
                continue
            row = emit(fam, lang, sdir, utt, {})
        elif fam == "B2_chatter_place":
            fn = rng.choice(geo_fns)
            p_ = rng.choice(places)
            utt = rng.choice(CHATTER_PLACE[lang]).format(p=p_)
            if not guard_ok(utt, [SEARCH_INTENT[lang]]):
                stats["guard_B2"] += 1
                continue
            row = emit(fam, lang, schemas[fn], utt, {})
        elif fam == "B3_vehicle":
            pa, pb = rng.sample(places, 2)
            base = rng.choice(DIR_TEMPLATES[lang]).format(a=pa, b=pb)
            clauses = [rng.choice(VEHICLE_INCIDENTAL[lang])]
            if rng.random() < 0.3:
                extra = rng.choice(VEHICLE_INCIDENTAL[lang])
                if extra != clauses[0]:
                    clauses.append(extra)
            fast = FAST_INCIDENTAL[lang] if rng.random() < 0.35 else ""
            utt = base.rstrip(".?!") + ". " + " ".join(clauses) + ((" " + fast) if fast else "")
            if not guard_ok(utt, [TRAVEL_MODE, ROUTING_SUPERL]):
                stats["guard_B3"] += 1
                continue
            row = emit(fam, lang, sdir, utt, {})
        else:  # A_cross
            fn = rng.choice(geo_fns)
            p_ = rng.choice(places)
            base = rng.choice(GEO_TEMPLATES[lang]).format(p=p_)
            c = rng.choice(CROSS_CONSTRAINT[lang])
            utt = base.rstrip(".?!") + "." + c + maybe_tail(lang, 0.2)
            props = (schemas[fn].get("parameters") or {}).get("properties") or {}
            args = {"q": p_} if "q" in props else {"query": p_}
            row = emit(fam, lang, schemas[fn], utt, args)
        if row is None:
            stats["reject"] += 1
            continue
        if row["id"] in seen:
            stats["dup"] += 1
            continue
        seen.add(row["id"])
        rows.append(row)
        stats[fam] += 1

    rng.shuffle(rows)
    n_val = min(max(20, int(a.val_frac * len(rows))), len(rows) // 2)
    val, train = rows[:n_val], rows[n_val:]

    st = {"rows_wanted": a.rows, "tries": tries, "families": dict(stats),
          "per_lang": dict(collections.Counter(r["lang"] for r in rows)),
          "per_fn": dict(collections.Counter(r["function"]["name"] for r in rows)),
          "nonnull_slots": dict(collections.Counter(
              k for r in rows for k, v in r["target"].items() if v is not None))}
    for name, part in (("train", train), ("val", val)):
        with (OUT / f"{a.out_prefix}_{name}.jsonl").open("w") as f:
            for r in part:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (OUT / f"{a.out_prefix}_train_stats.json").write_text(json.dumps(st, ensure_ascii=False, indent=2))
    print(f"[done] train={len(train)} val={len(val)} stats={json.dumps(st, ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
