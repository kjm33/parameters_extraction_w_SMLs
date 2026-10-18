"""Freeze gold-v3 datasets: merge grounded teacher predictions into the v2 rows.

Reproduces the deterministic pooled v2 frame (build_gold_v3.pool_all) and aligns
the full-run teacher prediction dumps POSITIONALLY (297 ids occur twice in the
pool with different content, so id-joins would cross-contaminate). Writes
data/<file>_v3.jsonl for each of the five v2 files: identical rows with only
`target` replaced and a `gold_src` field added, plus data/v3_stats.json.

Merge policy per pool row:
  t1 grounded                 -> teacher_target, gold_src=t1_haiku55
  else t4 grounded (if exist) -> teacher_target, gold_src=t4_glm53flash
  else                        -> v2 target,      gold_src=v2_fallback

Teacher targets carry exactly the schema-property keys (null-filled); where a
v2 target has extra legacy keys absent from the current schema, those are kept
as-is (teacher values overwrite every schema key).

Usage:
  python3 src/freeze_gold_v3.py                                  # t4 tier if present
  python3 src/freeze_gold_v3.py --t4-preds /nonexistent.jsonl    # t1-only freeze
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from build_extraction import read_jsonl  # noqa: E402
from build_gold_v3 import FILES, load_ids_from, pool_all  # noqa: E402

T1_PREDS = ROOT / "eval" / "results" / "goldv3" / "t1_haiku55_v3full_preds.jsonl"
T4_PREDS_DEFAULT = ROOT / "eval" / "results" / "goldv3" / "t4_glm53flash_v3full_preds.jsonl"
T1_SRC, T4_SRC, V2_SRC = "t1_haiku55", "t4_glm53flash", "v2_fallback"


def load_jsonl(path: pathlib.Path) -> list:
    return list(read_jsonl(path))


def align(lines: list, rows: list, what: str):
    if len(lines) != len(rows):
        raise SystemExit(f"{what}: got {len(lines)} lines, expected {len(rows)}")
    for k, (line, row) in enumerate(zip(lines, rows)):
        if line.get("id") != row["id"] or line.get("file") != row["file"]:
            raise SystemExit(f"{what}: positional mismatch at line {k}: preds "
                             f"(id={line.get('id')}, file={line.get('file')}) != "
                             f"pool (id={row['id']}, file={row['file']})")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--t4-preds", default=str(T4_PREDS_DEFAULT), metavar="PATH",
                   help="t4 full-run preds jsonl (skipped gracefully if missing)")
    return p.parse_args()


def main():
    args = parse_args()
    pool = pool_all()

    t1 = load_jsonl(T1_PREDS)
    align(t1, pool, f"t1 preds {T1_PREDS.name}")

    t4_path = pathlib.Path(args.t4_preds)
    t4_map: dict = {}
    if t4_path.exists():
        want_ids = load_ids_from(str(T1_PREDS))
        t4_rows = [r for r in pool if r["id"] in want_ids]
        t4_lines = load_jsonl(t4_path)
        align(t4_lines, t4_rows, f"t4 preds {t4_path.name}")
        t4_map = dict(zip((i for i, r in enumerate(pool) if r["id"] in want_ids),
                          t4_lines))
        split = [i for i, line in enumerate(t1)
                 if (i in t4_map) != (not line["grounded"])]
        if split:
            print(f"note: {len(split)} duplicate-id rows re-run by t4 despite a "
                  f"differently-grounded t1 sibling; t1 verdict keeps precedence")
    else:
        print(f"note: t4 preds not found at {t4_path} — freezing with t1 + "
              f"v2 fallback only")

    per_file, overall = {}, Counter()
    em_hit, em_n = Counter(), Counter()
    fb_fns, legacy_rows, dup_ids = Counter(), 0, Counter(row["id"] for row in pool)
    pos = 0
    for f in FILES:
        stem = pathlib.Path(f).stem
        v2_rows = load_jsonl(ROOT / f)
        chunk = pool[pos:pos + len(v2_rows)]
        if len(chunk) != len(v2_rows):
            raise SystemExit(f"{stem}: pool slice shorter than {f}")
        for j, (vrow, prow) in enumerate(zip(v2_rows, chunk)):
            if vrow["id"] != prow["id"]:
                raise SystemExit(f"{stem}: v2/pool misalignment at line {j}: "
                                 f"{vrow['id']} vs {prow['id']}")
        out_rows = []
        src_count, file_em = Counter(), [0, 0]
        for j, vrow in enumerate(v2_rows):
            i = pos + j
            tt, src = None, V2_SRC
            if t1[i]["grounded"]:
                tt = t1[i].get("teacher_target")
                if not isinstance(tt, dict):
                    raise SystemExit(f"{stem} line {j} ({vrow['id']}): grounded t1 "
                                     f"row without dict teacher_target")
                src = T1_SRC
            elif i in t4_map and t4_map[i].get("grounded"):
                tt = t4_map[i].get("teacher_target")
                if not isinstance(tt, dict):
                    raise SystemExit(f"{stem} line {j} ({vrow['id']}): grounded t4 "
                                     f"row without dict teacher_target")
                src = T4_SRC
            if tt is not None:
                if set(tt) - set(vrow["target"]):
                    raise SystemExit(f"{stem} line {j} ({vrow['id']}): teacher "
                                     f"target has keys absent from v2 target: "
                                     f"{sorted(set(tt) - set(vrow['target']))}")
                if set(tt) != set(vrow["target"]):
                    legacy_rows += 1
                new_target = {**vrow["target"], **tt}
            else:
                src = V2_SRC
                new_target = vrow["target"]
            if set(new_target) != set(vrow["target"]):
                raise SystemExit(f"{stem} line {j} ({vrow['id']}): merged target "
                                 f"key set != v2 target key set")
            row = dict(vrow)
            row["target"] = new_target
            row["gold_src"] = src
            out_rows.append(row)
            overall[src] += 1
            src_count[src] += 1
            em_n[src] += 1
            hit = int(new_target == vrow["target"])
            em_hit[src] += hit
            file_em[0] += hit
            file_em[1] += 1
            if src == V2_SRC:
                fb_fns[vrow["function"]["name"]] += 1
        pos += len(v2_rows)

        out_path = ROOT / "data" / f"{stem}_v3.jsonl"
        out_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                                    for r in out_rows))
        if len(load_jsonl(out_path)) != len(v2_rows):
            raise SystemExit(f"{out_path.name}: written line count != input")
        per_file[stem] = {"rows": len(v2_rows),
                          "by_src": {s: c for s, c in src_count.items()},
                          "em_vs_v2": round(file_em[0] / max(1, file_em[1]), 4)}

    n = pos
    em = {s: round(em_hit[s] / em_n[s], 4) for s in em_n}
    stats = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
             "t4_preds": str(t4_path) if t4_map else None,
             "per_file": per_file,
             "overall": {"rows": n, "by_src": dict(overall)},
             "em_vs_v2": {"overall": round(sum(em_hit.values()) / max(1, n), 4),
                          "by_src": em},
             "fallback_top_fns": dict(sorted(fb_fns.items(),
                                             key=lambda kv: (-kv[1], kv[0]))[:10]),
             "duplicate_id_rows": sum(c for c in dup_ids.values() if c > 1),
             "legacy_key_rows": legacy_rows}
    (ROOT / "data" / "v3_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2))

    print(f"{'file':<12}{'rows':>7}" + "".join(f"{s:>13}" for s in
                                                (T1_SRC, T4_SRC, V2_SRC)) +
          f"{'em_vs_v2':>10}")
    for stem, st in per_file.items():
        print(f"{stem:<12}{st['rows']:>7}" +
              "".join(f"{st['by_src'].get(s, 0):>13}" for s in
                      (T1_SRC, T4_SRC, V2_SRC)) + f"{st['em_vs_v2']:>10.4f}")
    print(f"{'overall':<12}{n:>7}" +
          "".join(f"{overall.get(s, 0):>13}" for s in (T1_SRC, T4_SRC, V2_SRC)) +
          f"{stats['em_vs_v2']['overall']:>10.4f}")
    print(f"-> {ROOT / 'data' / 'v3_stats.json'} "
          f"(duplicate_id_rows={stats['duplicate_id_rows']}, "
          f"legacy_key_rows={legacy_rows})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
