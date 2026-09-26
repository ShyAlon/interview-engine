#!/usr/bin/env python3
"""Replay stored categorization inputs through Jev ONLY, once per rubric variant.

No baseline, no interview, no LLM. Each stored input package keeps its exact
state (question + founder answer); only the rubric under test is swapped:
intent variants replace the intent question, relevance variants replace the
coverage relevance question. The 11 covers_* questions stay as stored unless
the variant carries its own (relevance-v3-se*: new per-area statements, the
SE preamble prepended to the state, and - for option B - the question clause
only on the stored target area).

Inputs: enriched categorization-shadow JSONL (v2) and/or an adjudication CSV
(labeled or not - input_state and rubric columns are enough).
Output: one v2-shape shadow JSONL per variant in OUT_DIR, plus a summary.

  .venv/bin/python scripts/replay-jev-rubric-variants.py INPUT... -o OUT_DIR
      [--intent-variants all|id,id] [--relevance-variants all|id,id]
      [--repeats N] [--workers 4] [--keep-going] [--resume-partial] [--dry-run]

Needs TYPESAFE_API_KEY (env or the repo .env). Exits after the first Jev
error unless --keep-going. A failing case is retried twice (15s, 45s) before it
counts as an error - transient provider 5xx should not strand a leg.
Refuses to write into a non-empty OUT_DIR, unless --resume-partial: then the
one partial replay-<variant>.jsonl in OUT_DIR is read, its completed ok
records (matching variant, repeat 1, current case IDs and input_refs) are
kept, and only the missing cases are called - a transient abort no longer
forces a full re-spend. The finished file validates like a fresh one.
"""
from __future__ import annotations
import argparse, csv, dataclasses, hashlib, json, sys, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from interview_engine.categorization import ChoiceQuestion, ClassificationRequest, NoulQuestion, ScoreQuestion  # noqa: E402
from interview_engine.categorization_rubrics import COMPONENT_VARIANTS, INTENT_VARIANTS, RELEVANCE_VARIANTS, VARIANT_ALIASES, RubricVariant, resolve_variant_id  # noqa: E402
from interview_engine.categorization_shadow import RECORD_SCHEMA_VERSION, _side, input_package  # noqa: E402

REPLAY_SLOTS = {"intent": "intent", "coverage": "relevance"}


def _question(d):
    if "criteria" in d:
        crit = d["criteria"]
        if isinstance(crit, dict): return ChoiceQuestion(d["instructions"], crit, bool(d.get("multiple", False)))
        return ScoreQuestion(d["instructions"], list(crit))
    return NoulQuestion(d["instructions"])


def load_cases(paths):
    """Unique (input_ref, slot) cases with their stored state + rubric."""
    cases, seen, skipped = [], set(), Counter()
    for path in map(Path, paths):
        if path.suffix == ".csv":
            with path.open(encoding="utf-8-sig", newline="") as f:
                raw = [{"slot": r["slot"], "input_ref": r["input_ref"], "session_id": r.get("session_id") or None,
                        "state": r.get("input_state") or "", "context": json.loads(r.get("input_context") or "{}"),
                        "questions": json.loads(r.get("rubric") or "{}"), "rubric_version": r.get("rubric_version") or "",
                        "case_id": r.get("case_id")} for r in csv.DictReader(f)]
        else:
            raw = []
            with path.open() as f:
                for line in f:
                    if not line.strip(): continue
                    r = json.loads(line); inp = r.get("input") or {}
                    raw.append({"slot": r.get("slot"), "input_ref": r.get("input_ref"), "session_id": r.get("session_id"),
                                "state": inp.get("state") or "", "context": inp.get("context") or {},
                                "questions": inp.get("questions") or {}, "rubric_version": inp.get("rubric_version") or ""})
        for c in raw:
            if c["slot"] not in REPLAY_SLOTS: skipped["slot:" + str(c["slot"])] += 1; continue
            if not c["state"]: skipped["no-input-package"] += 1; continue
            key = (c["input_ref"], c["slot"])
            if key in seen: skipped["duplicate"] += 1; continue
            seen.add(key); c["source_file"] = str(path)
            c["case_id"] = c.get("case_id") or hashlib.sha256((c["input_ref"] + "|" + c["slot"]).encode()).hexdigest()[:16]
            cases.append(c)
    return cases, skipped


def build_request(case, variant: RubricVariant) -> ClassificationRequest:
    stored = {k: _question(v) for k, v in case["questions"].items()}
    state = case["state"]
    if case["slot"] == "intent":
        questions = dict(variant.questions)
    elif variant.build is not None:  # component trees: every statement comes from the variant + context
        questions = variant.build(case["context"] or {})
        state = variant.state(state)
    else:  # coverage: swap relevance; covers_* from the variant when it has them, else as stored
        covers = variant.covers_questions((case["context"] or {}).get("target"))
        if covers is None: covers = {k: v for k, v in stored.items() if k.startswith("covers_")}
        questions = {**dict(variant.questions), **covers}
        state = variant.state(state)
    return ClassificationRequest(case["slot"], state, questions, case["context"], rubric_version=variant.rubric_version)


def pick(table, spec, opt_in=None):
    """'all' = the table; opt-in variants (component trees) only when named."""
    if spec in ("", "none"): return []
    if spec == "all": return list(table.values())
    table = {**table, **(opt_in or {})}
    out = []
    for vid in spec.split(","):
        rid = resolve_variant_id(vid)
        if rid not in table:
            aliases = [a for a, t in VARIANT_ALIASES.items() if t in table]
            raise SystemExit(f"unknown variant {vid!r}; known: {', '.join([*table, *aliases])}")
        if rid != vid.strip(): print(f"note: {vid.strip()} is an alias of {rid}")
        if table[rid] not in out: out.append(table[rid])  # an alias plus its target runs once
    return out



def load_partial(path, cases, variant_id, slot):
    """Completed records to keep from a partial replay, and the cases still to run.

    A record is kept only if it is an ok record for this variant, repeat 1,
    this slot, a known case ID and the current input_ref; everything else is
    dropped and counted, so the finished file validates against the current
    bench. Duplicates abort: a partial file must never go in twice.
    """
    by_id = {c["case_id"]: c for c in cases}
    kept, dropped = [], 0
    for line in path.open(encoding="utf-8"):
        if not line.strip(): continue
        r = json.loads(line); cid = r.get("replay", {}).get("case_id")
        ok = (r.get("slot") == slot and r.get("jev", {}).get("status") == "ok"
              and r.get("replay", {}).get("variant_id") == variant_id and r.get("replay", {}).get("repeat") == 1
              and cid in by_id and r.get("input_ref") == by_id[cid]["input_ref"])
        (kept.append(r) if ok else None); dropped += 0 if ok else 1
    if len({r["replay"]["case_id"] for r in kept}) != len(kept):
        raise SystemExit(f"partial replay has duplicate case IDs: {path}")
    done = {r["replay"]["case_id"] for r in kept}
    return kept, [c for c in cases if c["case_id"] not in done], dropped


def ordered_results(pool, run, todo, progress_every, started, variant_id, wait_seconds=30):
    """Keep replay order while reporting if a slow result blocks progress."""
    futures = [pool.submit(run, case) for case in todo]
    for done, future in enumerate(futures, 1):
        while True:
            try:
                yield done, future.result(timeout=wait_seconds)
                break
            except FutureTimeout:
                if progress_every:
                    stamp = datetime.now().astimezone().isoformat(timespec="seconds")
                    print(f"{stamp} | [{variant_id}] waiting | {done-1}/{len(todo)} cases written; elapsed={time.monotonic()-started:.0f}s", flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="+"); p.add_argument("-o", "--out-dir", required=True)
    p.add_argument("--intent-variants", default="all"); p.add_argument("--relevance-variants", default="all")
    p.add_argument("--repeats", type=int, default=1, help="run each variant N times to measure Jev run-to-run variance")
    p.add_argument("--workers", type=int, default=4); p.add_argument("--keep-going", action="store_true")
    p.add_argument("--resume-partial", action="store_true", help="complete a partial replay in OUT_DIR, calling only the missing cases")
    p.add_argument("--progress-every", type=int, default=0, help="print timestamped progress every N completed cases (0=off)")
    p.add_argument("--dry-run", action="store_true", help="build every request and print counts; no Jev calls, no key needed")
    a = p.parse_args(argv)
    if a.progress_every < 0: p.error("--progress-every must be >= 0")

    cases, skipped = load_cases(a.inputs)
    variants = pick(INTENT_VARIANTS, a.intent_variants) + pick(RELEVANCE_VARIANTS, a.relevance_variants, COMPONENT_VARIANTS)
    by_slot = Counter(c["slot"] for c in cases)
    print(f"cases: {len(cases)} {dict(by_slot)}  skipped: {dict(skipped)}")
    cov = [c for c in cases if c["slot"] == "coverage"]
    if cov:  # option B depends on the stored target; say how many cases have one
        print(f"coverage cases with a stored target area: {sum(bool((c['context'] or {}).get('target')) for c in cov)}/{len(cov)}")
    plan = [(v, r) for v in variants for r in range(a.repeats)]
    calls = sum(by_slot["intent" if v.slot_key == "intent" else "coverage"] for v, _ in plan)
    print(f"variants: {', '.join(v.id for v in variants)}  repeats: {a.repeats}  Jev calls: {calls}")
    for v in variants:  # every request must build before any money or time is spent
        for c in cases:
            if REPLAY_SLOTS[c["slot"]] == v.slot_key: build_request(c, v)
    if a.dry_run:
        print("dry run: all requests built; nothing called"); return 0

    if a.resume_partial and a.repeats != 1: raise SystemExit("--resume-partial supports repeats=1 only")
    out = Path(a.out_dir)
    if out.exists() and any(out.iterdir()) and not a.resume_partial:
        raise SystemExit(f"{out} is not empty; pick a fresh directory (or --resume-partial to complete a partial replay)")
    out.mkdir(parents=True, exist_ok=True)

    from interview_engine import model_config
    from interview_engine.categorization import TypeSafeProvider
    key = model_config._env("TYPESAFE_API_KEY")
    print(f"TYPESAFE_API_KEY present: {bool(key)}  typesafe-sdk installed: {TypeSafeProvider.available()}")
    if not key or not TypeSafeProvider.available(): raise SystemExit("Jev unavailable: need TYPESAFE_API_KEY and typesafe-sdk in this venv")
    rate = model_config._env("TYPESAFE_USD_PER_MTOK")
    jev = TypeSafeProvider(api_key=key, usd_per_mtok=float(rate) if rate else 0.0323)

    summary = []
    for v, rep in plan:
        slot = "intent" if v.slot_key == "intent" else "coverage"
        todo = [c for c in cases if c["slot"] == slot]
        path = out / (f"replay-{v.id}.jsonl" if a.repeats == 1 else f"replay-{v.id}-r{rep + 1}.jsonl")

        def run(case):
            req = build_request(case, v); t = time.monotonic(); err = None
            for wait in (0, 15, 45):  # two retries: transient provider 5xx should not strand a leg
                if wait:
                    print(f"  retry {v.id} {case['case_id']} in {wait}s after {type(err).__name__}", flush=True); time.sleep(wait)
                try:
                    res = jev.classify(req)
                    res = dataclasses.replace(res, answers=v.apply(res.answers, case["context"]))
                    return case, req, _side(res, elapsed=time.monotonic() - t), None
                except Exception as e:  # noqa: BLE001
                    err = e
            return case, req, _side(error=err, elapsed=time.monotonic() - t), err

        kept = []
        if a.resume_partial:
            if not path.is_file(): raise SystemExit(f"--resume-partial but no partial replay at {path}; nothing to complete")
            kept, todo, dropped = load_partial(path, todo, v.id, slot)
            print(f"resume-partial: keeping {len(kept)} completed records (dropped {dropped}), {len(todo)} cases to call", flush=True)
        total = len(kept) + len(todo)
        labels, errors, toks, lat = Counter(), 0, 0, 0.0
        started = time.monotonic()
        if a.progress_every:
            stamp=datetime.now().astimezone().isoformat(timespec="seconds")
            print(f"{stamp} | [{v.id}] start | 0/{len(todo)} cases; workers={max(1,a.workers)}",flush=True)
        with path.open("w") as f, ThreadPoolExecutor(max_workers=max(1, a.workers)) as pool:
            for r in kept: f.write(json.dumps(r, separators=(",", ":"), default=str) + "\n")
            for r in kept:
                side = r["jev"]; labels[str(side["verdict"].get(v.slot_key))] += 1
                toks += side.get("input_tokens", 0) + side.get("output_tokens", 0); lat += side.get("latency_s") or 0
            for done,(case, req, side, err) in ordered_results(pool, run, todo, a.progress_every, started, v.id):
                rec = {"record_schema_version": RECORD_SCHEMA_VERSION, "session_id": case.get("session_id"),
                       "timestamp": datetime.now(timezone.utc).isoformat(), "slot": slot, "input_ref": case["input_ref"],
                       "input": input_package(req), "baseline": {"status": "not_rerun"}, "jev": side, "agreement": None,
                       "jev_fallback_count": 0, "jev_error_count": int(err is not None),
                       "replay": {"variant_id": v.id, "repeat": rep + 1, "case_id": case["case_id"],
                                  "source_file": case["source_file"], "source_rubric_version": case["rubric_version"]}}
                f.write(json.dumps(rec, separators=(",", ":"), default=str) + "\n"); f.flush()
                if err is not None:
                    errors += 1
                if a.progress_every and (done % a.progress_every == 0 or done == len(todo)):
                    stamp=datetime.now().astimezone().isoformat(timespec="seconds")
                    print(f"{stamp} | [{v.id}] progress | {done}/{len(todo)} cases; errors={errors}; elapsed={time.monotonic()-started:.0f}s",flush=True)
                if err is not None:
                    print(f"  ERROR {v.id} {case['case_id']}: {type(err).__name__}: {str(err)[:300]}")
                    if not a.keep_going:
                        pool.shutdown(cancel_futures=True)
                        raise SystemExit(f"stopped after first Jev error (partial file kept: {path}); rerun with --resume-partial to complete it, or --keep-going to continue past errors")
                    continue
                labels[str(side["verdict"].get(v.slot_key))] += 1
                toks += side["input_tokens"] + side["output_tokens"]; lat += side["latency_s"] or 0
        line = f"{v.id:28s} r{rep + 1}  n={total:3d}  errors={errors}  tokens={toks}  latency={lat:.1f}s  labels={dict(labels)}"
        print(line); summary.append(line)
    (out / "SUMMARY.txt").write_text("\n".join(summary) + "\n")
    print(f"\nwrote {len(plan)} files to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
