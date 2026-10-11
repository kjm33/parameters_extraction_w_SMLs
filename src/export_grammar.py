"""Export per-function decoding grammars for edge runtimes.

Outputs into grammars/:
  - schemas/<fn>.json  — target JSON-Schema (xgrammar / llama.cpp --json-schema /
                         lm-format-enforcer compatible; mirrors src/target_schema.py)
  - <fn>.gbnf          — llama.cpp GBNF, fixed key order (schema order = training
                         emission order), nullable values, enum-constrained slots

Usage:  python3 src/export_grammar.py [--files data/core_train.jsonl,data/core_test.jsonl]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from target_schema import target_json_schema  # noqa: E402


def _esc_enum(v: str) -> str:
    return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'


def gbnf_for(function: dict) -> str:
    props = (function.get("parameters") or {}).get("properties") or {}
    lines = [
        'root ::= "{" kvs "}"',
        "kvs ::= " + " \",\" ".join(f'kv-{i}' for i in range(len(props))),
        'ws ::= [ \\t\\n]*',
        'str ::= "\\"" ([^"\\\\\\x7F\\x00-\\x1F] | "\\\\" '
        '(["\\\\bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]))* "\\""',
        "num ::= [0-9]+",
    ]
    safe = re.compile(r"[A-Za-z0-9_-]+")
    for i, (pname, ps) in enumerate(props.items()):
        key = _esc_enum(pname)
        lines.append(f'kv-{i} ::= {key} ":" val-{i}')

        def val_rule(ps=ps, pname=pname):
            if ps.get("enum"):
                alts = " | ".join(_esc_enum(v) for v in ps["enum"])
                return f'({alts} | "null")'
            t = ps.get("type", "string")
            if t == "array":
                items = ps.get("items") or {}
                if items.get("enum"):
                    alts = " | ".join(_esc_enum(v) for v in items["enum"])
                    return (f'("[" ws ({alts}) (ws "," ws ({alts}))* ws "]" | "null")')
                return '("[" ws str (ws "," ws str)* ws "]" | "null")'
            if t in ("integer", "number"):
                return '(num | "null")'
            if t == "boolean":
                return '("true" | "false" | "null")'
            return '(str | "null")'

        r = val_rule()
        assert safe.match(pname), pname
        lines.append(f"val-{i} ::= {r}")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", default="data/core_train.jsonl,data/core_val.jsonl,data/core_test.jsonl")
    ap.add_argument("--out", default="grammars")
    a = ap.parse_args()

    fns: dict[str, dict] = {}
    for part in a.files.split(","):
        p = ROOT / part
        if not p.exists():
            continue
        for line in open(p):
            if not line.strip():
                continue
            r = json.loads(line)
            f = r["function"]
            fns.setdefault(f["name"], f)

    out_dir = ROOT / a.out
    (out_dir / "schemas").mkdir(parents=True, exist_ok=True)
    for name, f in sorted(fns.items()):
        fname = name.replace("/", "_")
        schema = target_json_schema(f)
        (out_dir / "schemas" / f"{fname}.json").write_text(
            json.dumps(schema, ensure_ascii=False, indent=2))
        (out_dir / f"{fname}.gbnf").write_text(gbnf_for(f))
        n_props = len(schema["required"])
    print(f"[export] {len(fns)} functions -> {out_dir}/{{*.gbnf,schemas/*.json}}")
    for name in sorted(fns):
        print(f"  {name}: {len(target_json_schema(fns[name])['required'])} params")
    return 0


if __name__ == "__main__":
    sys.exit(main())
