"""Build a GeoNames gazetteer index for slot-value canonicalization (v2).

Reads the local GeoNames raw dumps in data/geo/ (cities15000.zip,
alternateNames.zip, admin1CodesASCII.txt, countryInfo.txt) and pickles a
compact index for eval/canonicalize_v2.py:

  gid2city     geonameid -> {name, ascii, country, admin1, pop,
                             feature_class, feature_code}
  alt2gids     fold(name | asciiname | kept altname) -> sorted tuple of gids
  key2canon    fold -> {lang -> display form, default -> name column}; only for
               folds whose gids all share the same country+admin1 AND the same
               city name (ambiguous folds are dropped entirely)
  admin1       "US.CA" -> {name, asciiname, geonameid}
  admin1_of    geonameid -> "US.CA"
  admin1_codes "US.CA" -> {"code2": "CA"} (only 2-uppercase-letter admin1 parts)
  meta         file mtimes / source + kept row counts / dropped-ambiguous /
               attribution

Alternate names are kept only for isolanguage in {de,en,es,fr,pl,int} and
geonameids present in cities15000; isHistoric=1 rows are skipped. The per-lang
display form prefers isPreferredName=1, then isShortName=1, else the name col.

GeoNames data: © GeoNames, CC-BY 4.0 — https://download.geonames.org/export/dump/

Usage:
  python3 src/build_gazetteer.py             # build from local data/geo/ files
  python3 src/build_gazetteer.py --download  # fetch missing files first
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import pickle
import re
import sys
import time
import urllib.request
import zipfile
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from build_extraction import fold  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
GEO = ROOT / "data" / "geo"
PICKLE_OUT = ROOT / "data" / "geo_gazetteer.pkl"
STATS_OUT = GEO / "gazetteer_stats.json"
BASE_URL = "https://download.geonames.org/export/dump/"
FILES = ("cities15000.zip", "alternateNames.zip", "admin1CodesASCII.txt", "countryInfo.txt")
ZIP_MEMBERS = {"cities15000.zip": "cities15000.txt", "alternateNames.zip": "alternateNames.txt"}
ALT_LANGS = ("de", "en", "es", "fr", "pl", "int")
ATTRIBUTION = "© GeoNames CC-BY 4.0"
CODE2_RE = re.compile(r"[A-Z]{2}")


def dec(b: bytes) -> str:
    """GeoNames exports are UTF-8; tolerate stray bytes via cp1252/latin-1."""
    try:
        return b.decode("utf-8")
    except UnicodeDecodeError:
        try:
            return b.decode("cp1252")
        except UnicodeDecodeError:
            return b.decode("latin-1")


def _rows(raw, skip_comments: bool):
    rdr = csv.reader((dec(ln) for ln in raw), delimiter="\t", quoting=csv.QUOTE_NONE)
    for cols in rdr:
        if not cols or (skip_comments and cols[0].startswith("#")):
            continue
        yield cols


def tsv_rows(path: pathlib.Path, member: str | None = None, skip_comments: bool = False):
    if member is not None:
        with zipfile.ZipFile(path) as zf:
            yield from _rows(zf.open(member), skip_comments)
    else:
        with open(path, "rb") as raw:
            yield from _rows(raw, skip_comments)


def col(cols: list[str], i: int) -> str:
    return cols[i] if len(cols) > i else ""


def maybe_download(do: bool) -> None:
    missing = [n for n in FILES
               if not (GEO / n).exists() or (GEO / n).stat().st_size == 0]
    if not missing:
        return
    if not do:
        sys.exit(f"ERROR: missing {missing} in {GEO}; re-run with --download to fetch them")
    GEO.mkdir(parents=True, exist_ok=True)
    for n in missing:
        url = BASE_URL + n
        print(f"downloading {url} ...")
        urllib.request.urlretrieve(url, GEO / n)


def build() -> dict:
    src, kept = Counter(), Counter()
    cities: dict[int, dict] = {}
    for cols in tsv_rows(GEO / "cities15000.zip", ZIP_MEMBERS["cities15000.zip"]):
        if len(cols) < 15:
            continue
        src["cities15000"] += 1
        gid = int(col(cols, 0))
        cities[gid] = {"name": col(cols, 1), "ascii": col(cols, 2),
                       "country": col(cols, 8), "admin1": col(cols, 10),
                       "pop": int(col(cols, 14) or 0),
                       "feature_class": col(cols, 6), "feature_code": col(cols, 7)}
    kept["cities"] = len(cities)

    alt2gids: dict[str, set] = {}
    best: dict[tuple[int, str], tuple[tuple[bool, bool], str]] = {}
    for cols in tsv_rows(GEO / "alternateNames.zip", ZIP_MEMBERS["alternateNames.zip"]):
        src["alternateNames"] += 1
        iso = col(cols, 2)
        if iso not in ALT_LANGS or col(cols, 7) == "1":
            continue
        try:
            gid = int(col(cols, 1))
        except ValueError:
            continue
        if gid not in cities:
            continue
        kept["altnames"] += 1
        alt = col(cols, 3)
        if not alt:
            continue
        alt2gids.setdefault(fold(alt), set()).add(gid)
        rank = (col(cols, 4) == "1", col(cols, 5) == "1")
        cur = best.get((gid, iso))
        if cur is None or rank > cur[0]:
            best[(gid, iso)] = (rank, alt)

    for gid, c in cities.items():
        for key in (fold(c["name"]), fold(c["ascii"])):
            if key:
                alt2gids.setdefault(key, set()).add(gid)

    key2canon: dict[str, dict] = {}
    dropped_amb = 0
    for key, gidset in alt2gids.items():
        gids = sorted(gidset)
        sig = {(cities[g]["country"], cities[g]["admin1"]) for g in gids}
        names = {fold(cities[g]["name"]) for g in gids}
        if len(sig) > 1 or len(names) > 1:
            dropped_amb += 1
            continue
        top = max(gids, key=lambda g: cities[g]["pop"])
        entry = {"default": cities[top]["name"]}
        for iso in ALT_LANGS:
            disp = None
            for g in gids:
                b = best.get((g, iso))
                if b is not None and (disp is None or b[0] > disp[0]):
                    disp = b
            if disp is not None:
                entry[iso] = disp[1]
        key2canon[key] = entry

    alt2gids_final = {k: tuple(sorted(v)) for k, v in alt2gids.items()}

    admin1_table: dict[str, dict] = {}
    for cols in tsv_rows(GEO / "admin1CodesASCII.txt", skip_comments=True):
        if len(cols) < 4:
            continue
        src["admin1CodesASCII"] += 1
        admin1_table[col(cols, 0)] = {"name": col(cols, 1), "asciiname": col(cols, 2),
                                      "geonameid": int(col(cols, 3) or 0)}
    admin1_codes = {}
    for code in admin1_table:
        part = code.split(".", 1)[1] if "." in code else ""
        if CODE2_RE.fullmatch(part):
            admin1_codes[code] = {"code2": part}

    admin1_of = {}
    for gid, c in cities.items():
        if c["country"] and c["admin1"]:
            admin1_of[gid] = f"{c['country']}.{c['admin1']}"

    for cols in tsv_rows(GEO / "countryInfo.txt", skip_comments=True):
        src["countryInfo"] += 1

    meta = {
        "attribution": ATTRIBUTION,
        "source": BASE_URL,
        "download_dates": {n: time.strftime("%Y-%m-%d %H:%M",
                                            time.localtime((GEO / n).stat().st_mtime))
                           for n in FILES},
        "source_rows": dict(src),
        "kept": {"cities": kept["cities"], "altnames": kept["altnames"],
                 "altname_keys": len(alt2gids_final), "key2canon": len(key2canon),
                 "admin1_table": len(admin1_table), "admin1_codes": len(admin1_codes)},
        "dropped_ambiguous_folds": dropped_amb,
    }
    return {"gid2city": cities, "alt2gids": alt2gids_final, "key2canon": key2canon,
            "admin1": admin1_table, "admin1_of": admin1_of, "admin1_codes": admin1_codes,
            "meta": meta}


def main():
    ap = argparse.ArgumentParser(description="Build the GeoNames gazetteer pickle")
    ap.add_argument("--download", action="store_true",
                    help=f"fetch missing files from {BASE_URL} before building")
    args = ap.parse_args()
    maybe_download(args.download)
    gaz = build()
    meta = gaz["meta"]
    with open(PICKLE_OUT, "wb") as f:
        pickle.dump(gaz, f, protocol=4)
    stats = dict(meta)
    stats["pickle"] = str(PICKLE_OUT.relative_to(ROOT))
    stats["pickle_bytes"] = PICKLE_OUT.stat().st_size
    STATS_OUT.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n")
    print(f"GeoNames gazetteer: cities={meta['kept']['cities']} "
          f"altnames_kept={meta['kept']['altnames']} "
          f"unique_keys={meta['kept']['key2canon']} "
          f"dropped_ambiguous={meta['dropped_ambiguous_folds']} "
          f"admin1_codes={meta['kept']['admin1_codes']}")
    print(f"-> {PICKLE_OUT} ({stats['pickle_bytes'] / 1e6:.1f} MB) [{meta['attribution']}]")
    print(f"-> {STATS_OUT}")


if __name__ == "__main__":
    main()
