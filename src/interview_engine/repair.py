"""Shape-drift repair for model-produced JSON before contract validation.

Small/local models drift shape under pressure: a list field comes back as {},
a dict-of-lists comes back with bare scalars inside. These are mechanical
drifts with an unambiguous repair - coercion is safe because it only fires
when the value cannot match the schema as-is. Genuinely missing or wrong
CONTENT (a field invented, a wrong statement) is not repairable here and
goes to the validation-error retry loop in the distillation node.
"""
from __future__ import annotations


def _wants(spec: dict) -> set[str]:
    types = set()
    if isinstance(spec.get("type"), str):
        types.add(spec["type"])
    for s in spec.get("anyOf", []) + spec.get("oneOf", []):
        if isinstance(s, dict) and isinstance(s.get("type"), str):
            types.add(s["type"])
    return types


def coerce_for_model(data: dict, model_cls) -> dict:
    """Return a copy of data with mechanical shape drift repaired against the
    model's JSON schema. Only empty/scalar mismatches are touched."""
    props = model_cls.model_json_schema().get("properties", {})
    out = dict(data)
    for name, spec in props.items():
        if name not in out:
            continue
        v = out[name]
        wants = _wants(spec)
        if "array" in wants:
            if v is None or v == {}:
                out[name] = []
            elif isinstance(v, dict):
                out[name] = list(v.values())
            elif isinstance(v, str):
                out[name] = [v]
        elif "object" in wants:
            if v is None or v == []:
                out[name] = {}
            elif isinstance(v, dict):
                ap = spec.get("additionalProperties")
                if isinstance(ap, dict) and ap.get("type") == "array":
                    out[name] = {
                        k: (val if isinstance(val, list)
                            else [val] if isinstance(val, (int, str)) else [])
                        for k, val in v.items()
                    }
    return out


def _canon(v: str) -> str:
    return v.strip().lower().replace("-", "_").replace(" ", "_")


def normalize_enums(data: dict, model_cls) -> dict:
    """Small models spell enum values loosely ('non-functional', 'Functional').
    For array-of-object fields whose item schema has enum-typed properties,
    snap each string to the closest declared enum value (case/separator
    insensitive). Only exact canonical matches are changed; unknown values
    stay put and fail validation loudly."""
    props = model_cls.model_json_schema().get("properties", {})
    defs = model_cls.model_json_schema().get("$defs", {})
    out = dict(data)
    for name, spec in props.items():
        items = spec.get("items")
        if not (isinstance(items, dict) and isinstance(out.get(name), list)):
            continue
        ref = items.get("$ref", "")
        item_schema = defs.get(ref.split("/")[-1], {}) if ref else items
        item_props = item_schema.get("properties", {})
        enum_fields = {}
        for pn, ps in item_props.items():
            if not isinstance(ps, dict):
                continue
            target = ps
            if isinstance(ps.get("$ref"), str):  # enum via $defs reference
                target = defs.get(ps["$ref"].split("/")[-1], {})
            if isinstance(target.get("enum"), list):
                enum_fields[pn] = [str(v) for v in target["enum"]]
        if not enum_fields:
            continue
        fixed_items = []
        for item in out[name]:
            if isinstance(item, dict):
                item = dict(item)
                for pn, allowed in enum_fields.items():
                    v = item.get(pn)
                    if isinstance(v, str):
                        for a in allowed:
                            if _canon(v) == _canon(a):
                                item[pn] = a
                                break
            fixed_items.append(item)
        out[name] = fixed_items
    return out



def extract_json_value(text: str) -> str:
    """Tolerant extraction (run-6, same doctrine as spec-generator): strip
    fences, else cut the first balanced {...} or [...] out of prose with a
    string-aware scan. Raises ValueError when nothing balanced exists (e.g.
    truncated output)."""
    import json as _j

    candidate = text.strip().strip("`").removeprefix("json").strip()
    try:
        _j.loads(candidate)
        return candidate
    except _j.JSONDecodeError:
        pass
    starts = [(text.find(o), o, c) for o, c in (("{", "}"), ("[", "]")) if text.find(o) >= 0]
    for start, open_c, close_c in sorted(starts):
        depth = 0
        in_str = esc = False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == open_c:
                depth += 1
            elif c == close_c:
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
    raise ValueError("no balanced JSON value found in the model output")

_ECHO_FRAGMENTS = ("input_value=", "input_type=", "validation errors", "validationerror",
                   "for further information visit https://errors.pydantic.dev")

def clean_model_patch(value, model_cls, error: Exception) -> dict:
    """Normalize and whitelist a repair against actual failed fields."""
    value = normalize_model_item(value, model_cls)
    allowed = set(model_cls.model_fields)
    requested = set()
    errors = getattr(error, "errors", lambda: [])()
    for item in errors:
        loc=item.get("loc") or ()
        if loc and isinstance(loc[-1],str) and loc[-1] in allowed:
            requested.add(loc[-1])
    if not requested:
        requested=allowed
    cleaned={}
    for key,val in value.items():
        if key not in requested:
            continue
        rendered=str(val).casefold()
        if any(fragment in rendered for fragment in _ECHO_FRAGMENTS):
            continue
        if val not in (None,"",[]):
            cleaned[key]=val
    if not cleaned:
        raise ValueError(f"repair patch is empty after cleaning; requested fields: {sorted(requested)}")
    return cleaned


def _plain_string_leaf(value, field_name: str):
    """Normalize one grounded envelope only where the schema requires str."""
    if isinstance(value,str):return value
    if isinstance(value,dict):
        # Field-level prompts often echo their requested key around the text
        # leaf, including inside a list item: {"operational_flow": "..."}.
        # Unwrap only one named envelope with one recursively unambiguous leaf.
        envelope_keys=(field_name,"statement","value","text")
        present=[k for k in envelope_keys if k in value]
        if len(value)==1 and len(present)==1:
            inner=value[present[0]]
            if isinstance(inner,list) and len(inner)==1:
                inner=inner[0]
            return _plain_string_leaf(inner,field_name)
        candidates=[k for k in dict.fromkeys((field_name,"statement","value","text"))
                    if isinstance(value.get(k),str)]
        if len(candidates)==1:return value[candidates[0]]
        # DeepSeek may expand an ordered flow step into actor + action. This is
        # richer than list[str], but its prose has one deterministic rendering;
        # numeric order and provenance remain owned by list position/item.
        actor=value.get("actor");action=value.get("action")
        if isinstance(actor,str) and actor.strip() and isinstance(action,str) and action.strip():
            text_keys={k for k,v in value.items() if isinstance(v,str) and v.strip()}
            if text_keys == {"actor","action"}:
                return f"{actor.strip()}: {action.strip()}"
    raise ValueError(f"field {field_name} requires one unambiguous text leaf, got {type(value).__name__}")


def normalize_model_item(value, model_cls) -> dict:
    """Unwrap one model item, whitelist its fields, and fix trivial shapes."""
    from typing import get_origin
    allowed = set(model_cls.model_fields)
    # Small models commonly retain the collection wrapper from the prompt.
    # Unwrap repeatedly, but only when it encloses exactly one object.
    for _ in range(4):
        if isinstance(value, list) and len(value) == 1 and isinstance(value[0], dict):
            value = value[0]
            continue
        if isinstance(value, dict) and not (set(value) & allowed) and len(value) == 1:
            only = next(iter(value.values()))
            if isinstance(only, list) and len(only) == 1 and isinstance(only[0], dict):
                value = only[0]
                continue
            if isinstance(only, dict):
                value = only
                continue
        break
    if not isinstance(value, dict):
        raise ValueError(f"model item must be one JSON object, got {type(value).__name__}")
    out={k:v for k,v in value.items() if k in allowed}
    for key,field in model_cls.model_fields.items():
        if key not in out:
            continue
        origin=get_origin(field.annotation)
        val=out[key]
        if origin is list:
            if isinstance(val,str):
                out[key]=[val]
            elif isinstance(val,dict):
                out[key]=list(val.values())
            elif val is None:
                out[key]=[]
            # DeepSeek systematically reuses grounded-statement envelopes for
            # plain string list fields. Provenance belongs to the owning item;
            # normalize the leaf value before contract validation.
            args=getattr(field.annotation,"__args__",())
            if args and args[0] is str and isinstance(out[key],list):
                out[key]=[_plain_string_leaf(item,key) for item in out[key]]
        elif field.annotation is str:
            out[key]=_plain_string_leaf(val,key)
        elif origin is dict and val is None:
            out[key]={}
    from vmodel_contracts.normalize import normalize_list_string_fields
    return normalize_list_string_fields(out, model_cls)
