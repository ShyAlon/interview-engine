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
    """Reduce a model-authored subtree when the contract requires one string.

    A provider may decompose one text field into coherent named facts. Preserve
    those facts as deterministic ``key: value`` lines. Structural metadata is
    omitted. Competing aliases and mixed actor/action prose remain ambiguous.
    """
    if isinstance(value,str):return value
    if isinstance(value,list) and len(value)==1:
        return _plain_string_leaf(value[0],field_name)
    if not isinstance(value,dict):
        raise ValueError(f"field {field_name} requires text, got {type(value).__name__}")

    aliases=tuple(dict.fromkeys((field_name,"statement","value","text")))
    alias_text=[(key,value[key].strip()) for key in aliases
                if isinstance(value.get(key),str) and value[key].strip()]
    if len(alias_text)>1 and len({text for _,text in alias_text})>1:
        paths=", ".join(key for key,_ in alias_text)
        raise ValueError(f"field {field_name} requires one unambiguous text leaf; conflicting paths: {paths}")
    if len(value)==1:
        return _plain_string_leaf(next(iter(value.values())),field_name)

    # Ordered flow actor/action has a specific rendering. Extra prose would
    # make ordering and meaning uncertain, so do not fold it in silently.
    actor=value.get("actor");action=value.get("action")
    if isinstance(actor,str) and actor.strip() and isinstance(action,str) and action.strip():
        prose=[key for key,item in value.items() if isinstance(item,str) and item.strip()]
        if set(prose)=={"actor","action"}:
            return f"{actor.strip()}: {action.strip()}"
        raise ValueError(f"field {field_name} requires one unambiguous text leaf; actor/action conflicts at: {', '.join(prose)}")

    metadata={"source_turns","step","order","index","confidence"}
    lines=[];unsupported=[]
    def collect(node,path):
        if isinstance(node,str) and node.strip():
            lines.append((path,node.strip()));return
        if node is None:return
        if isinstance(node,dict):
            for key,item in node.items():
                child=f"{path}.{key}" if path else str(key)
                if key in metadata and (not isinstance(item,str)):
                    continue
                collect(item,child)
            return
        if isinstance(node,list):
            if path.rsplit(".",1)[-1] in metadata and all(isinstance(x,(int,float,bool)) for x in node):return
            if all(isinstance(x,str) and x.strip() for x in node):
                lines.append((path,"; ".join(x.strip() for x in node)));return
            unsupported.append(path);return
        if path.rsplit(".",1)[-1] in metadata and isinstance(node,(int,float,bool)):return
        # Semantic numeric/bool leaves are preserved rather than mistaken for
        # metadata merely because of their type.
        if isinstance(node,(int,float,bool)):
            lines.append((path,str(node).lower() if isinstance(node,bool) else str(node)));return
        unsupported.append(path)
    collect(value,"")
    if unsupported or not lines:
        detail=", ".join(unsupported) if unsupported else "none"
        raise ValueError(f"field {field_name} has no coherent text rendering; unsupported paths: {detail}")
    # A lone leaf is an envelope regardless of its provider-invented key.
    if len(lines)==1:return lines[0][1]
    return "\n".join(f"{path}: {text}" for path,text in lines)


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
