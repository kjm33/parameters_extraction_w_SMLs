"""JSON-Schema for the assistant target object (one key per parameter, honest nulls).

Shared by grammar-constrained decoding (xgrammar) and GBNF export. The schema
mirrors the supervision policy in build_extraction.param_target:
  - every schema parameter appears as a required key,
  - every value is nullable (absent evidence -> null),
  - enums are constrained to their value sets,
  - 'lang'/'language' string params are constrained to the 5 supported langs,
  - arrays need >=1 item (empty grounding result is stored as null).
"""
from __future__ import annotations

LANG_ENUM = ["de", "en", "es", "fr", "pl"]
LANG_PARAM_NAMES = {"lang", "language"}


def value_schema(pname: str, pschema: dict) -> dict:
    if pschema.get("enum"):
        return {"enum": list(pschema["enum"])}
    t = pschema.get("type", "string")
    if t == "array":
        items = pschema.get("items") or {}
        item = {"enum": list(items["enum"])} if items.get("enum") else {"type": "string"}
        return {"type": "array", "items": item, "minItems": 1}
    if t == "integer":
        return {"type": "integer", "minimum": 1}
    if t == "number":
        return {"type": "number"}
    if t == "boolean":
        return {"type": "boolean"}
    if pname in LANG_PARAM_NAMES:
        return {"enum": LANG_ENUM}
    return {"type": "string"}


def target_json_schema(function: dict) -> dict:
    """function: {"name":..., "parameters": {"properties": {...}}} -> JSON schema."""
    props_in = (function.get("parameters") or {}).get("properties") or {}
    props, required = {}, []
    for pname, pschema in props_in.items():
        props[pname] = {"anyOf": [value_schema(pname, pschema), {"type": "null"}]}
        required.append(pname)
    return {
        "type": "object",
        "properties": props,
        "required": required,
        "additionalProperties": False,
    }
