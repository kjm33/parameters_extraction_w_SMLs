"""Compare NER span sources against GOLD free-text slot values on a dataset file.

Sources (--source NAME=path[:lang=XX], repeatable) come in two formats:
  - jsonl dumps (gliner/flair): rows {"id","lang","spans":[{"text","label","score",
    "start","end"}]}
  - wikineural-style json: dict id -> list of span TEXT strings (no labels/scores)
Flair sources additionally get an automatic `<name>_loc` variant restricted to
LOC-labeled spans, and (via the :lang= suffix) a per-source lang filter so the
per-language models are only scored on their language's rows.

Gold values per row: non-null STRING values at non-enum, non-lang parameters
(props via build_tags.props_of; list-valued params skipped). Matching is
fold-based (build_extraction.fold): exact = fold-word-sets equal; contained =
value words subset of ONE span's words; overlap = any word intersection. Also
reports a span-precision-ish stat: % of spans that contain/exactly-match SOME
gold value of their row.

Stdlib only (system python3):
  python3 eval/compare_ner.py --files data/core_test.jsonl --source ... \
      (writes eval/results/compare_ner.json + stdout table)
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from build_extraction import fold  # noqa: E402
from build_tags import props_of  # noqa: E402

LANGS = ["en", "de", "pl", "fr", "es"]


def rpath(part: str) -> pathlib.Path:
    p = pathlib.Path(part)
    return p if p.is_absolute() else ROOT / p


def read_jsonl(path: pathlib.Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def parse_source(spec: str) -> tuple[str, pathlib.Path, str | None]:
    name, _, rest = spec.partition("=")
    lang = None
    if ":" in rest:
        path, _, suffix = rest.rpartition(":")
        if suffix.startswith("lang="):
            lang = suffix.split("=", 1)[1]
        else:
            path = rest
    else:
        path = rest
    return name.strip(), rpath(path), lang


def load_spans(path: pathlib.Path) -> dict[str, list[dict]]:
    """jsonl dumps -> id->spans; wikineural json (id -> list[str|dict]) -> normalized."""
    if path.suffix == ".json":
        raw = json.loads(path.read_text())
        return {rid: [s if isinstance(s, dict) else {"text": s, "label": None,
                                                    "score": None, "start": None, "end": None}
                      for s in spans]
                for rid, spans in raw.items()}
    return {r["id"]: r.get("spans") or [] for r in read_jsonl(path)}


def wset(s: str) -> set[str]:
    return set(re.findall(r"\w+", fold(s or "")))


def gold_values(row: dict) -> list[str]:
    props = props_of(row)
    out = []
    for param, value in (row.get("target") or {}).items():
        if param in ("lang", "language"):
            continue
        if not isinstance(value, str) or not value:  # covers null, lists, dicts, bools
            continue
        ps = props.get(param) or {}
        if "enum" in ps or "enum" in (ps.get("items") or {}):
            continue
        out.append(value)
    return out


def new_stats() -> dict:
    return {"n_values": 0, "exact": 0, "contained": 0, "overlap": 0,
            "n_spans": 0, "spans_useful": 0}


def eval_rows(rows: list[dict], spans_by_id: dict[str, list[dict]],
              lang: str | None) -> dict[str, dict]:
    per: dict[str, dict] = defaultdict(new_stats)
    for row in rows:
        rlang = row.get("lang") or "?"
        if lang and rlang != lang:
            continue
        vals = gold_values(row)
        spans = spans_by_id.get(row["id"]) or []
        vsets = [wset(v) for v in vals if wset(v)]
        for bucket in (rlang, "all"):
            st = per[bucket]
            st["n_values"] += len(vsets)
            st["n_spans"] += len(spans)
        ssets = [wset(s.get("text") or "") for s in spans]
        for vs in vsets:
            exact = any(ss == vs for ss in ssets)
            contained = any(vs <= ss for ss in ssets)
            overlap = any(vs & ss for ss in ssets)
            for bucket in (rlang, "all"):
                st = per[bucket]
                st["exact"] += exact
                st["contained"] += contained
                st["overlap"] += overlap
        for ss in ssets:
            if not ss:
                continue
            useful = any(vs <= ss for vs in vsets)
            for bucket in (rlang, "all"):
                per[bucket]["spans_useful"] += useful
    return per


def finalize(per: dict) -> dict:
    out = {}
    for key, st in per.items():
        n = max(st["n_values"], 0)
        out[key] = {
            "n_values": n,
            "exact_pct": round(100.0 * st["exact"] / n, 2) if n else None,
            "contained_pct": round(100.0 * st["contained"] / n, 2) if n else None,
            "overlap_pct": round(100.0 * st["overlap"] / n, 2) if n else None,
            "n_spans": st["n_spans"],
            "span_useful_pct": round(100.0 * st["spans_useful"] / st["n_spans"], 2)
                               if st["n_spans"] else None,
        }
    return out


def pct(report: dict, source: str, lang: str, key: str) -> float | None:
    return (report.get(source, {}).get("per_lang", {}).get(lang) or {}).get(key)


def best_inhouse(report: dict, inhouse: list[str], lang: str, key: str):
    cands = [(pct(report, s, lang, key), s) for s in inhouse]
    cands = [(v, s) for v, s in cands if v is not None]
    return max(cands) if cands else (None, None)


def main():
    ap = argparse.ArgumentParser(
        description="Compare NER span sources vs gold free-text slot values")
    ap.add_argument("--files", default="data/core_test.jsonl",
                    help="comma-separated dataset files")
    ap.add_argument("--source", action="append", required=True,
                    help="NAME=path[:lang=XX] span source (jsonl dump or wikineural json)")
    ap.add_argument("--out", default="eval/results/compare_ner.json")
    args = ap.parse_args()

    rows: list[dict] = []
    for part in args.files.split(","):
        rows.extend(read_jsonl(rpath(part)))
    file_stems = "__".join(pathlib.Path(p).stem for p in args.files.split(","))

    report: dict[str, dict] = {}
    order: list[str] = []
    for spec in args.source:
        name, path, lang = parse_source(spec)
        spans_by_id = load_spans(path)
        labels = {(s.get("label") or "") for spans in spans_by_id.values() for s in spans}
        # _loc variants only make sense for flair-style PER/LOC/ORG/MISC tag sets;
        # GLiNER's lowercase labels ("city","location",...) have no exact LOC tag
        flair_style = any(lab.upper() in {"PER", "LOC", "ORG", "MISC"} for lab in labels)
        variants = [(name, spans_by_id, lang)]
        if flair_style:
            loc = {rid: [s for s in spans if (s.get("label") or "").upper() == "LOC"]
                   for rid, spans in spans_by_id.items()}
            variants.append((f"{name}_loc", loc, lang))
        for vname, vspans, vlang in variants:
            per = eval_rows(rows, vspans, vlang)
            report[vname] = {"path": str(path), "lang_filter": vlang,
                             "per_lang": finalize(per)}
            order.append(vname)

    flair_sources = [n for n in order if n.startswith("flair")]
    inhouse = [n for n in order if n not in flair_sources]

    langs_eval = [l for l in LANGS
                  if any(pct(report, n, l, "exact_pct") is not None for n in order)]
    table: dict[str, dict] = {}
    for l in langs_eval + ["all"]:
        table[l] = {}
        for n in order:
            v = pct(report, n, l, "exact_pct")
            if v is not None:
                table[l][n] = (pct(report, n, l, "exact_pct"),
                               pct(report, n, l, "contained_pct"))

    flair_by_lang = {}
    for l in langs_eval:
        cands = [(pct(report, n, l, "exact_pct"), n) for n in flair_sources]
        cands = [(v, n) for v, n in cands if v is not None]
        flair_by_lang[l] = max(cands) if cands else (None, None)

    per_source_wins = {}
    for n in flair_sources:
        wins_e, wins_c = [], []
        for l in langs_eval:
            v_e, v_c = pct(report, n, l, "exact_pct"), pct(report, n, l, "contained_pct")
            if v_e is None:
                continue
            bi_e, _ = best_inhouse(report, inhouse, l, "exact_pct")
            bi_c, _ = best_inhouse(report, inhouse, l, "contained_pct")
            if bi_e is not None and v_e > bi_e:
                wins_e.append(l)
            if bi_c is not None and v_c is not None and v_c > bi_c:
                wins_c.append(l)
        per_source_wins[n] = {"exact_wins": wins_e, "contained_wins": wins_c}
    combined = {"exact_wins": [], "contained_wins": [], "flair_best_by_lang": {}}
    for l in langs_eval:
        f_e, f_src = flair_by_lang[l]
        bi_e, bi_src = best_inhouse(report, inhouse, l, "exact_pct")
        bi_c, _ = best_inhouse(report, inhouse, l, "contained_pct")
        f_c = pct(report, f_src, l, "contained_pct") if f_src else None
        combined["flair_best_by_lang"][l] = {
            "flair": [f_src, f_e], "inhouse_best": [bi_src, bi_e],
            "flair_exact_win": bool(f_e is not None and bi_e is not None and f_e > bi_e),
            "flair_contained_win": bool(f_c is not None and bi_c is not None and f_c > bi_c)}
        if combined["flair_best_by_lang"][l]["flair_exact_win"]:
            combined["exact_wins"].append(l)
        if combined["flair_best_by_lang"][l]["flair_contained_win"]:
            combined["contained_wins"].append(l)
    verdict = ("ADOPT (>=2/5 langs)" if len(combined["exact_wins"]) >= 2
               else "DROP (<2/5 langs; document)")
    notes = {
        "decision_rule": "Adopt Flair as feature/arbitration source only if it beats the "
                         "best in-house span source on >=2 of 5 langs; else document+drop.",
        "inhouse_sources": inhouse,
        "best_inhouse_by_lang": {
            l: {"exact": best_inhouse(report, inhouse, l, "exact_pct"),
                "contained": best_inhouse(report, inhouse, l, "contained_pct")}
            for l in langs_eval},
        "flair_per_source_wins": per_source_wins,
        "flair_combined_vs_inhouse": combined,
        "verdict": verdict,
        "license_note": "All Flair 2.0 entity-* models (and 1.0 ner-*) are Flukes NC 1.0 "
                        "NONCOMMERCIAL per HF model cards (2026-10-12) - measurement only.",
    }

    hdr = "source".ljust(22) + "".join(l.rjust(14) for l in langs_eval) + "all".rjust(14)
    print(hdr)
    print("-" * len(hdr))
    for n in order:
        cells = []
        for l in langs_eval + ["all"]:
            ec = table.get(l, {}).get(n)
            cells.append(f"{ec[0]:.1f}/{ec[1]:.1f}".rjust(14) if ec else "-".rjust(14))
        print(n[:21].ljust(22) + "".join(cells))
    print(f"\ncell = exact%/contained% of gold free-text values ({file_stems}); "
          f"n_values per lang: "
          + json.dumps({l: report[order[0]]['per_lang'].get(l, {}).get('n_values')
                        for l in langs_eval}))
    print("flair verdict vs decision rule: " + verdict)
    print("  flair exact wins by lang: " + json.dumps(combined["exact_wins"]))
    print("  flair contained wins by lang: " + json.dumps(combined["contained_wins"]))
    print("  best in-house per lang (source exact%): "
          + json.dumps({l: notes["best_inhouse_by_lang"][l]["exact"]
                        for l in langs_eval}))

    out = rpath(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"files": args.files, "sources": report, "notes": notes},
                  f, indent=1, ensure_ascii=False, default=str)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
