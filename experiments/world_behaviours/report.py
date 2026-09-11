"""Aggregate the world-behaviours probe (#118) into one report.

Reads every ``programs_*.jsonl`` beside ``behaviours.jsonl``, re-verifies each
program against every test (the authors' statuses are not trusted), and writes
``README.md`` and ``report.json`` in the same directory: verified fraction per
source, witness lengths against ``stack_pbe_v4``, macro compression, macro
usage, and overlap with v4's witnesses.

Usage:
    uv run python experiments/world_behaviours/report.py reports/world-behaviours-probe-20260909
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from verify import check, expand, load_macros

from wandering_light.stack_pbe import skeleton

V4_MANIFEST = Path("wandering_light/evals/data/stack_pbe_v4/manifest.json")
BUCKETS = ((1, 5), (6, 10), (11, 20), (21, 40), (41, 10**9))

FINDINGS = """## Findings (written with the 2026-09-09 run; the tables below regenerate)

- **Feasibility is not the gap.** 91 of 95 verified. The four misses are one
  behaviour that is CPython set-iteration order rather than a function, and
  three LeetCode problems whose tests span up to 90,002 numbers against the
  100,000-step fuel cap. Nothing failed for want of an op.
- **Raw witnesses are longer than v4's, not shorter.** Median 102 `mark-28`
  ops against v4's 43 over both splits, and 66 of 91 sit in the 41+ bucket. The same programs
  are a median 17 tokens in macro form (4.6 ops per token), so the length is
  the basis's, not the behaviours': every author ended up with a library of
  stack shuffles, packed accumulators, index-list builders and digit helpers.
  That library is the object highways (#96) proposes to mine; here it was
  written top-down in an afternoon.
- **Loops are capped by nesting, not fuel.** `MAX_QUOTATION_DEPTH = 32` and a
  self-applying quotation nests ~4 levels per iteration, so the recursive
  `range` in the shared macros refuses past n = 7. All four authors found this
  independently and rebuilt ranges as recursion-free doubling folds.
- **Six solves replay the reference's quirks, not the stated behaviour**
  (`mbpp-34`, `mbpp-733`, `mbpp-466` binary searches on unsorted input,
  `mbpp-824` mutate-while-iterating, `mbpp-846` merge replay, `mbpp-799`
  bit-width bound). They pass every test because the tests were drawn from
  that reference. A corpus built this way inherits the reference as the
  behaviour, which is the right reading of world-authored data but the wrong
  one of the description.
- **Overlap with v4 is small**: 5 of 91 expanded programs equal a v4 witness
  (`add`, `product`, `max2`, `min2`, and the range of a list), and none of the
  rest share a skeleton with any of v4's 220 behaviours.
"""


def bucket(n: int) -> str:
    for lo, hi in BUCKETS:
        if lo <= n <= hi:
            return f"{lo}-{hi}" if hi < 10**9 else f"{lo}+"
    return "0"


def v4_witnesses() -> dict[str, tuple[str, str]]:
    """behaviour -> (program text, skeleton) over both pinned v4 splits.

    The payload is fetched, digest-checked, from the tracked manifest; a
    missing or unreachable payload is an error, not an empty comparison.
    """
    from wandering_light.corpus_hub import fetch_corpus

    directory = fetch_corpus(V4_MANIFEST)
    manifest = json.loads(V4_MANIFEST.read_text())
    out: dict[str, tuple[str, str]] = {}
    for split in ("train", "eval"):
        path = directory / manifest["splits"][split]["path"]
        if not path.exists():
            raise FileNotFoundError(f"v4 {split} payload missing at {path}")
        with gzip.open(path, "rt") as fh:
            for line in fh:
                row = json.loads(line)
                out.setdefault(
                    row["behaviour"], (" ".join(row["program"]), row["skeleton"])
                )
    expected = sum(manifest["splits"][s]["behaviours"] for s in ("train", "eval"))
    if len(out) != expected:
        raise ValueError(
            f"v4 catalogue has {len(out)} behaviours, manifest says {expected}"
        )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("directory", type=Path)
    parser.add_argument(
        "--no-findings",
        action="store_true",
        help="omit the prose findings written for the 2026-09-09 run",
    )
    parser.add_argument(
        "--findings-file",
        type=Path,
        help="markdown inserted after the intro (a hand-written findings section that survives regeneration)",
    )
    parser.add_argument(
        "--title",
        default="World-behaviours probe: can MBPP and LeetCode-easy behaviours get `mark-28` witnesses?",
    )
    parser.add_argument(
        "--intro",
        default=(
            "training-data-sources (#118). Behaviours sampled by `experiments/world_behaviours/extract.py`\n"
            "(seed 17, ints and flat int lists only, ≤ 3 arguments); programs authored by Claude in macro\n"
            "form and expanded to `mark-28`; every program re-verified on every test by `report.py`."
        ),
    )
    args = parser.parse_args(argv)
    d = args.directory

    behaviours = {
        json.loads(l)["id"]: json.loads(l) for l in (d / "behaviours.jsonl").open()
    }
    macros = load_macros()
    rows = []
    for file in sorted(d.glob("programs_*.jsonl")):
        for line in file.open():
            if line.strip():
                row = json.loads(line)
                row["author"] = file.stem.split("_")[-1]
                rows.append(row)
    seen = collections.Counter(r["id"] for r in rows)
    duplicates = [k for k, v in seen.items() if v > 1]
    missing = sorted(set(behaviours) - set(seen))

    results = []
    absent = sorted({r["id"] for r in rows} - set(behaviours))
    rows = [r for r in rows if r["id"] in behaviours]
    for row in rows:
        b = behaviours[row["id"]]
        tokens = row["program"].split()
        expanded = expand(tokens, macros) if tokens else []
        res = (
            check(expanded, b)
            if expanded
            else {"parsed": False, "passed": 0, "total": len(b["tests"])}
        )
        results.append(
            {
                "id": row["id"],
                "source": b["source"],
                "difficulty": b.get("difficulty", ""),
                "author": row["author"],
                "claimed": row.get("status"),
                "solved": bool(expanded) and res.get("passed") == res["total"],
                "passed": res.get("passed", 0),
                "total": res["total"],
                "macro_program": row["program"],
                "program": " ".join(expanded),
                "macro_length": len(tokens),
                "length": len(expanded),
                "macros_used": sorted({t for t in tokens if t in macros}),
                "note": row.get("note", ""),
            }
        )

    solved = [r for r in results if r["solved"]]
    by_source = collections.defaultdict(lambda: [0, 0])
    for r in results:
        by_source[r["source"]][1] += 1
        by_source[r["source"]][0] += r["solved"]
    mismatched = [r["id"] for r in results if (r["claimed"] == "solved") != r["solved"]]
    by_group = collections.defaultdict(lambda: [0, 0])
    for r in results:
        key = f"{r['source']}-{r['difficulty']}" if r["difficulty"] else r["source"]
        by_group[key][1] += 1
        by_group[key][0] += r["solved"]
    by_author = collections.defaultdict(lambda: [0, 0])
    for r in results:
        by_author[r["author"]][1] += 1
        by_author[r["author"]][0] += r["solved"]

    lengths = [r["length"] for r in solved]
    hist = collections.Counter(bucket(n) for n in lengths)
    v4 = v4_witnesses()
    v4_lengths = [len(p.split()) for p, _ in v4.values()]
    v4_hist = collections.Counter(bucket(n) for n in v4_lengths)
    v4_programs = {p for p, _ in v4.values()}
    v4_skeletons = {s for _, s in v4.values()}
    exact = [r["id"] for r in solved if r["program"] in v4_programs]
    same_skeleton = [
        r["id"] for r in solved if skeleton(r["program"].split()) in v4_skeletons
    ]

    macro_use = collections.Counter(m for r in solved for m in r["macros_used"])
    compression = [r["length"] / r["macro_length"] for r in solved if r["macro_length"]]

    reasons = collections.Counter()
    for r in results:
        if not r["solved"]:
            reasons[r["note"][:60] or "(no note)"] += 1

    summary = {
        "behaviours": len(behaviours),
        "authored_rows": len(rows),
        "missing": missing,
        "programs_without_behaviour": absent,
        "duplicates": duplicates,
        "claim_mismatches": mismatched,
        "solved": len(solved),
        "by_source": {k: {"solved": v[0], "total": v[1]} for k, v in by_source.items()},
        "by_source_difficulty": {
            k: {"solved": v[0], "total": v[1]} for k, v in sorted(by_group.items())
        },
        "by_author": {
            k: {"solved": v[0], "total": v[1]} for k, v in sorted(by_author.items())
        },
        "length_median": statistics.median(lengths) if lengths else None,
        "length_hist": dict(hist),
        "v4_length_median": statistics.median(v4_lengths) if v4_lengths else None,
        "v4_length_hist": dict(v4_hist),
        "v4_behaviours": len(v4),
        "exact_v4_matches": exact,
        "same_skeleton_as_v4": same_skeleton,
        "macro_length_median": statistics.median([r["macro_length"] for r in solved])
        if solved
        else None,
        "compression_median": statistics.median(compression) if compression else None,
        "macro_use": macro_use.most_common(25),
        "macros_defined": len(macros),
        "unsolved_reasons": reasons.most_common(),
    }
    (d / "report.json").write_text(
        json.dumps({"summary": summary, "results": results}, indent=1)
    )

    def pct(a: int, b: int) -> str:
        return f"{a}/{b} ({100 * a / b:.0f}%)" if b else "-"

    lines = [
        f"# {args.title}",
        "",
        args.intro,
        "",
        args.findings_file.read_text()
        if args.findings_file
        else ""
        if args.no_findings
        else FINDINGS,
        "## Verified",
        "",
        "| source | verified | median witness length (`mark-28` ops) | median macro-form length |",
        "|---|---|---|---|",
    ]
    for src, (n_solved, n_total) in sorted(by_group.items()):
        rows_ = [
            r
            for r in solved
            if (f"{r['source']}-{r['difficulty']}" if r["difficulty"] else r["source"])
            == src
        ]
        ls = [r["length"] for r in rows_]
        ms = [r["macro_length"] for r in rows_]
        lines.append(
            f"| {src} | {pct(n_solved, n_total)} | {statistics.median(ls) if ls else '-'} | {statistics.median(ms) if ms else '-'} |"
        )
    lines.append(
        f"| **all** | {pct(len(solved), len(results))} | {summary['length_median']} | {summary['macro_length_median']} |"
    )
    lines += [
        "",
        f"Claimed-vs-verified mismatches: {len(mismatched)}; missing ids: {len(missing)}; duplicates: {len(duplicates)}.",
        "",
        "By author: "
        + "; ".join(f"{k} {pct(v[0], v[1])}" for k, v in sorted(by_author.items()))
        + ".",
        "",
        "## Witness length against `stack_pbe_v4`",
        "",
        "| bucket (ops) | this probe (verified) | v4 behaviours (train + eval, 220) |",
        "|---|---|---|",
    ]
    for lo, _hi in BUCKETS:
        key = bucket(lo)
        lines.append(f"| {key} | {hist.get(key, 0)} | {v4_hist.get(key, 0)} |")
    lines += [
        f"| median | {summary['length_median']} | {summary['v4_length_median']} |",
        "",
        f"Exact matches with a v4 witness: {len(exact)} ({', '.join(exact) or 'none'}). "
        f"Same skeleton as a v4 witness: {len(same_skeleton)}.",
        "",
        "## Macros",
        "",
        f"{summary['macros_defined']} macros defined across `macros*.txt`; median expansion ratio "
        f"(`mark-28` ops per macro-form token) on verified programs: {summary['compression_median']:.1f}."
        if summary["compression_median"]
        else "",
        "",
        "| macro | verified programs using it |",
        "|---|---|",
    ]
    lines += [f"| `{m}` | {n} |" for m, n in macro_use.most_common(20)]
    lines += [
        "",
        "## Unsolved",
        "",
        "| id | source | passed | note |",
        "|---|---|---|---|",
    ]
    for r in results:
        if not r["solved"]:
            lines.append(
                f"| {r['id']} | {r['source']} | {r['passed']}/{r['total']} | {r['note'][:120]} |"
            )
    lines += [
        "",
        "## Every verified program",
        "",
        "| id | len | macro form |",
        "|---|---|---|",
    ]
    for r in sorted(solved, key=lambda r: r["length"]):
        lines.append(f"| {r['id']} | {r['length']} | `{r['macro_program']}` |")
    (d / "README.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in summary.items()
                if k not in ("macro_use", "unsolved_reasons")
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
