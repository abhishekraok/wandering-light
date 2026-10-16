"""Final-review independent reconstruction (#183). Does not use runner helpers.

Uses only: raw gzip JSON, hashlib, low-level parse_stack / run_program,
value_key, CATALOGUE and the registered step3 check(); grade() only as a
cross-check of the own hidden replay.
"""

import gzip
import hashlib
import json
import math
import statistics
import sys
import time
from fractions import Fraction
from pathlib import Path

from experiments.step3_verify_solves import check
from wandering_light.stack_machine import parse_stack, value_key
from wandering_light.stack_pbe import BASIS, CATALOGUE, load, run_program
from wandering_light.stack_pbe_eval import grade

C = Path("/home/abhishekrao/repos/wandering-light/checkpoints/gumbel-retention-screen-20261015")
out = {}
t0 = time.time()
payload = C / "comparison/inputs/stack_pbe_v4_train.jsonl.gz"
raw = payload.read_bytes()
assert hashlib.sha256(raw).hexdigest() == "13450717aadfce2ec16e890f3e9509514cf4c82bdff3d1199006854fe80c683b"
rows = [json.loads(line) for line in gzip.decompress(raw).decode().splitlines() if line.strip()]
assert all(r["split"] == "train" and isinstance(r["skeleton"], str) and r["skeleton"] for r in rows)
first = {}
for i, r in enumerate(rows):
    first.setdefault(r["skeleton"], i)
dig = {s: hashlib.sha256(b"gumbel-retention-v1\x00" + s.encode("utf-8")).hexdigest() for s in first}
assert len(set(dig.values())) == len(dig), "digest collision"
rank = sorted(first, key=lambda s: (dig[s], s))
panel = [first[s] for s in rank[:24]]
out["rows"] = len(rows)
out["skeleton_groups"] = len(first)
for phase in ("baseline", "comparison"):
    sel = json.loads((C / phase / "selection.json").read_text())
    shown = json.loads((C / phase / "shown.json").read_text())
    assert [d["source_index"] for d in sel] == panel, phase
    for d, e, i in zip(sel, shown, panel, strict=True):
        r = rows[i]
        assert d["skeleton"] == r["skeleton"] and d["behaviour"] == r["behaviour"]
        assert set(e) == {"source_index", "inputs", "outputs"} and e["source_index"] == i
        assert e["inputs"] == r["inputs"] and e["outputs"] == r["outputs"]
        h = hashlib.sha256(json.dumps({"inputs": r["inputs"], "outputs": r["outputs"]}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        assert h == d["shown_sha256"]
out["panel"] = panel
out["panel_behaviours"] = [rows[i]["behaviour"] for i in panel]
assert len(set(out["panel_behaviours"])) == 24
catalogue = {b.name: b for b in CATALOGUE}


def fits(program, ins, outs):
    for a, b in zip(ins, outs, strict=True):
        res = run_program(tuple(program), parse_stack(a, basis=BASIS))
        if isinstance(res, str):
            return False
        stack, _ = res
        want = parse_stack(b, basis=BASIS)
        if len(stack) != len(want) or any(value_key(x) != value_key(y) for x, y in zip(stack, want)):
            return False
    return True


verdicts = {}
cells = {}
ORDER = {"baseline": ["baseline-17", "baseline-29", "baseline-43"],
         "comparison": ["baseline-17", "candidate-17", "candidate-29", "baseline-29", "baseline-43", "candidate-43"]}
raw_cells = {}
for phase, names in ORDER.items():
    for name in names:
        cell = json.loads((C / phase / f"{name}.json").read_text())
        raw_cells[phase, name] = cell
        arm, seed = name.split("-")
        assert cell["complete"] is True and cell["arm"] == arm and cell["seed"] == int(seed)
        assert cell["threads"] == 1 and cell["interop"] == 1
        assert [r["source_index"] for r in cell["rows"]] == panel
        shown_fit, hidden, good, kinds = set(), set(), set(), {}
        for r in cell["rows"]:
            assert (r["program"] is not None) == (r["termination"] == "fit"), r
            assert math.isfinite(r["solve_seconds"]) and r["solve_seconds"] > 0
            if r["program"] is None:
                continue
            t = rows[r["source_index"]]
            b = t["behaviour"]
            prog = tuple(r["program"])
            assert fits(prog, t["inputs"], t["outputs"]), (name, b)
            shown_fit.add(b)
            ho = fits(prog, t["held_out_inputs"], t["held_out_outputs"])
            if ho:
                hidden.add(b)
            key = (b, prog)
            if key not in verdicts:
                v = check(prog, catalogue[b], 10000, 0, dense=True, min_tested=500)
                assert (v["verdict"] == "equals_catalogue_witness") == (prog in catalogue[b].programs)
                verdicts[key] = v
            kinds[b] = verdicts[key]["verdict"]
            if ho and kinds[b] in ("genuine", "equals_catalogue_witness"):
                good.add(b)
        cells[f"{phase}/{name}"] = {
            "shown_fit": len(shown_fit),
            "hidden": sorted(hidden),
            "verified": sorted(good),
            "verified_count": len(good),
            "witness": sorted(b for b in good if kinds[b] == "equals_catalogue_witness"),
            "draw_tested": sorted(b for b in good if kinds[b] == "genuine"),
            "shortcut": sorted(b for b, k in kinds.items() if k == "shortcut"),
            "inconclusive": sorted(b for b, k in kinds.items() if k == "inconclusive"),
            "solve_seconds": math.fsum(r["solve_seconds"] for r in cell["rows"]),
            "executions": sum(r["executions"] for r in cell["rows"]),
            "fuel": sum(r["fuel"] for r in cell["rows"]),
            "terminations": {k: sum(r["termination"] == k for r in cell["rows"]) for k in sorted({r["termination"] for r in cell["rows"]})},
        }

tasks = load(payload)
mismatch = 0
for (b, prog) in verdicts:
    i = next(i for i in panel if rows[i]["behaviour"] == b)
    t = rows[i]
    own = fits(prog, t["inputs"], t["outputs"]) and fits(prog, t["held_out_inputs"], t["held_out_outputs"])
    mismatch += grade(prog, tasks[i]).solved != own
out["grade_crosscheck_mismatches"] = mismatch

for phase in ORDER:
    s = json.loads((C / phase / "summary.json").read_text())["cells"]
    for name in ORDER[phase]:
        mine, rec = cells[f"{phase}/{name}"], s[name]
        assert mine["verified"] == rec["verified"], (phase, name)
        assert mine["hidden"] == rec["graded"]
        assert mine["witness"] == rec["verified_witness_equal"]
        assert mine["draw_tested"] == rec["verified_draw_tested"]
        assert mine["solve_seconds"] == rec["solve_seconds"]
        assert mine["executions"] == rec["executions"] and mine["fuel"] == rec["fuel"]
        assert mine["shortcut"] == rec["kinds"]["shortcut"] and mine["terminations"] == rec["terminations"]

for phase in ORDER:
    recorded = json.loads((C / phase / "verification.json").read_text())
    rec_map = {(d["behaviour"], tuple(d["program"])): {k: v for k, v in d.items() if k not in ("behaviour", "program")} for d in recorded}
    keys = {(rows[r["source_index"]]["behaviour"], tuple(r["program"])) for n in ORDER[phase] for r in raw_cells[phase, n]["rows"] if r["program"] is not None}
    assert set(rec_map) == keys, (phase, len(rec_map), len(keys))
    assert all(json.loads(json.dumps(verdicts[k])) == rec_map[k] for k in keys), phase
    out[f"distinct_{phase}_programs"] = len(keys)
out["distinct_all_programs"] = len(verdicts)
out["verdict_counts"] = {v: sum(x["verdict"] == v for x in verdicts.values()) for v in ("genuine", "equals_catalogue_witness", "shortcut", "inconclusive")}
out["verdicts"] = {f"{b}|len{len(p)}": [v["verdict"], {m: (t["tested"], t["failed"]) for m, t in v.get("draws", {}).items()}] for (b, p), v in verdicts.items()}

deltas, ratios, pairs = [], [], []
for seed in (17, 29, 43):
    B, G = cells[f"comparison/baseline-{seed}"], cells[f"comparison/candidate-{seed}"]
    d = G["verified_count"] - B["verified_count"]
    q = Fraction(G["solve_seconds"]) / Fraction(B["solve_seconds"])
    deltas.append(d)
    ratios.append(q)
    pairs.append({"seed": seed, "B": B["verified_count"], "G": G["verified_count"], "TB": B["solve_seconds"], "TG": G["solve_seconds"],
                  "ratio": [q.numerator, q.denominator, float(q)], "lost": sorted(set(B["verified"]) - set(G["verified"])), "new": sorted(set(G["verified"]) - set(B["verified"]))})
med = sorted(ratios)[1]
assert med == statistics.median(ratios)
out["pairs"] = pairs
out["median_gain"] = statistics.median(deltas)
out["median_ratio"] = [med.numerator, med.denominator, float(med)]
out["components"] = {"median_gain_ge_2": statistics.median(deltas) >= 2, "all_nonneg": min(deltas) >= 0, "median_ratio_le_11_10": med <= Fraction(11, 10)}
out["cells"] = cells
out["elapsed"] = time.time() - t0
json.dump(out, sys.stdout, indent=1, sort_keys=True, default=str)
