"""Do two routes that meet on the shown examples meet on fresh ones? (meeting-pairs-shortcut-signal, #169)

`experiments/seen_set_collisions.py` writes ``pairs.jsonl``: for every closed
state the beam reached by a second route, the retained program ``first`` and
the discarded ``second``, walked back to the op where they part. "Prefer
``op_first`` at that state" is a next-op label with no solver call, *if* the
two programs compute the same function. On k shown examples they can agree by
accident: ``[ over lt? ]`` and ``[ swap lt? ]`` met on one task (#164). That
pair taught as redundancy would be a shortcut (stack-debugger, #115).

The gate is solved-program-cache (#151)'s, per pair: draw fresh inputs from the
behaviour's sampler, run both programs on each, and keep the pair only where the
two stacks agree on every draw. A draw both programs refuse says nothing and is
not counted; a draw one refuses and the other does not is a disagreement.

    uv run python experiments/meeting_pairs_gate.py --run-dir DIR --tasks TASKS [--draws 10]

``DIR`` holds ``pairs.jsonl``; ``TASKS`` is the task file that search ran on,
so each pair's behaviour resolves to its catalogue sampler. Writes
``gate.jsonl`` (one verdict per pair), ``gate.json`` and ``gate.md``: pass rate
by meeting kind and by tail shape, and the ops the shorter route saves.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from wandering_light.corpus_hub import fetch_corpus  # noqa: E402
from wandering_light.stack_machine import parse_stack  # noqa: E402
from wandering_light.stack_pbe import (  # noqa: E402
    BASIS,
    BY_NAME,
    load,
    run_program,
    to_stack,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from wandering_light.stack_pbe import Behaviour

DRAWS = 10
MIN_INFORMATIVE = 5
SEED = 169
KINDS = ("same-layer", "earlier-layer")
SHAPES = ("commutation", "insertion", "other", "extension")
FAIL_SAMPLES = 40


def gate_pair(
    first: Sequence[str],
    second: Sequence[str],
    behaviour: Behaviour,
    rng: random.Random,
    draws: int = DRAWS,
    min_informative: int = MIN_INFORMATIVE,
    exclude: frozenset = frozenset(),
) -> dict[str, Any]:
    """``pass`` if both programs agree on every informative fresh draw."""
    informative = 0
    counterexample = None
    for _ in range(draws):
        stack = to_stack(behaviour.sample(rng))
        if stack in exclude:
            continue
        a, b = run_program(first, stack), run_program(second, stack)
        refused_a, refused_b = isinstance(a, str), isinstance(b, str)
        if refused_a and refused_b:
            continue
        informative += 1
        if refused_a != refused_b or a[0] != b[0]:
            counterexample = {
                "input": [str(v) for v in stack],
                "first": a if refused_a else [str(v) for v in a[0]],
                "second": b if refused_b else [str(v) for v in b[0]],
            }
            break
    if counterexample is not None:
        verdict = "fail"
    elif informative >= min_informative:
        verdict = "pass"
    else:
        verdict = "inconclusive"
    return {
        "verdict": verdict,
        "informative": informative,
        "counterexample": counterexample,
    }


def gate_pairs(
    pairs: Sequence[dict[str, Any]],
    tasks,
    draws: int = DRAWS,
    seed: int = SEED,
) -> list[dict[str, Any]]:
    shown_by_task = {
        i: frozenset(parse_stack(text, basis=BASIS) for text in task.inputs)
        for i, task in enumerate(tasks)
    }
    out = []
    for n, pair in enumerate(pairs):
        behaviour = BY_NAME.get(pair["behaviour"])
        if behaviour is None:
            out.append({**pair, "verdict": "no_sampler", "informative": 0})
            continue
        rng = random.Random(f"{seed}:{pair['task']}:{n}")
        verdict = gate_pair(
            tuple(pair["first"]),
            tuple(pair["second"]),
            behaviour,
            rng,
            draws=draws,
            exclude=shown_by_task[pair["task"]],
        )
        out.append({**pair, **verdict})
    return out


def _ratio(a: float, b: float) -> float | None:
    return a / b if b else None


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def _median(values: Sequence[int]) -> float | None:
    if not values:
        return None
    s = sorted(values)
    m = len(s) // 2
    return float(s[m]) if len(s) % 2 else (s[m - 1] + s[m]) / 2


def summarise(rows: Sequence[dict[str, Any]], draws: int) -> dict[str, Any]:
    def cell(subset):
        c = Counter(r["verdict"] for r in subset)
        judged = c["pass"] + c["fail"]
        return {
            "pairs": len(subset),
            "pass": c["pass"],
            "fail": c["fail"],
            "inconclusive": c["inconclusive"],
            "no_sampler": c["no_sampler"],
            "pass_rate_of_judged": _ratio(c["pass"], judged),
        }

    by_kind = {k: cell([r for r in rows if r["kind"] == k]) for k in KINDS}
    by_shape = {
        s: cell([r for r in rows if r["shape"] == s])
        for s in SHAPES
        if any(r["shape"] == s for r in rows)
    }
    by_kind_shape = {
        f"{k}/{s}": cell([r for r in rows if r["kind"] == k and r["shape"] == s])
        for k in KINDS
        for s in SHAPES
        if any(r["kind"] == k and r["shape"] == s for r in rows)
    }
    passing_longer = [
        r for r in rows if r["verdict"] == "pass" and r["kind"] == "earlier-layer"
    ]
    saved = [len(r["second"]) - len(r["first"]) for r in passing_longer]
    saved_hist = Counter(saved)
    # one label is (task, divergence prefix, preferred op, dispreferred op)
    labels = {
        (r["task"], tuple(r["first"][: r["d"]]), r["op_first"], r["op_second"])
        for r in passing_longer
    }
    fails = [r for r in rows if r["verdict"] == "fail"]
    return {
        "draws": draws,
        "all": cell(rows),
        "by_kind": by_kind,
        "by_shape": by_shape,
        "by_kind_shape": by_kind_shape,
        "ops_saved_median": _median(saved),
        "ops_saved_histogram": {str(k): saved_hist[k] for k in sorted(saved_hist)},
        "passing_earlier_layer_labels": len(labels),
        "passing_earlier_layer_tasks": len({r["task"] for r in passing_longer}),
        "tasks_with_pairs": len({r["task"] for r in rows}),
        "fail_by_behaviour": dict(Counter(r["behaviour"] for r in fails).most_common()),
        "fail_samples": [
            {
                k: r[k]
                for k in (
                    "behaviour",
                    "kind",
                    "shape",
                    "first",
                    "second",
                    "counterexample",
                )
            }
            for r in fails[:FAIL_SAMPLES]
        ],
    }


def markdown(summary: dict[str, Any], label: str) -> str:
    def row(name, c):
        return (
            f"| {name} | {c['pairs']:,} | {c['pass']:,} | {c['fail']:,} | "
            f"{c['inconclusive']:,} | **{_pct(c['pass_rate_of_judged'])}** |"
        )

    lines = [
        f"## {label}",
        "",
        f"{summary['all']['pairs']:,} closed cross-path meetings in "
        f"{summary['tasks_with_pairs']} tasks, {summary['draws']} fresh draws each.",
        "",
        "| pairs | n | pass | fail | inconclusive | pass rate of judged |",
        "|---|---:|---:|---:|---:|---:|",
        row("all", summary["all"]),
    ]
    for k, c in summary["by_kind"].items():
        lines.append(row(k, c))
    for k, c in summary["by_kind_shape"].items():
        lines.append(row(k, c))
    hist = summary["ops_saved_histogram"]
    lines += [
        "",
        f"- passing earlier-layer pairs: the shorter route saves a median of "
        f"{summary['ops_saved_median']} ops; distribution "
        + ", ".join(f"{k}: {v:,}" for k, v in hist.items()),
        f"- distinct next-op labels among them (task, divergence prefix, preferred "
        f"op, dispreferred op): {summary['passing_earlier_layer_labels']:,} in "
        f"{summary['passing_earlier_layer_tasks']} tasks",
        "- failures by behaviour: "
        + (
            ", ".join(f"{b} {n}" for b, n in summary["fail_by_behaviour"].items())
            or "none"
        ),
        "",
    ]
    if summary["fail_samples"]:
        lines += ["Failing pairs (first / second / the input that parts them):", ""]
        for f in summary["fail_samples"][:12]:
            ce = f["counterexample"]
            lines.append(
                f"- `{f['behaviour']}` ({f['kind']}, {f['shape']}): "
                f"`{' '.join(f['first'])}` vs `{' '.join(f['second'])}` on "
                f"`{' '.join(ce['input'])}`"
            )
        lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--tasks", required=True, help="the task file search ran on")
    parser.add_argument("--split", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--draws", type=int, default=DRAWS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--label", default=None)
    args = parser.parse_args(argv)

    path = Path(args.tasks)
    if path.name == "manifest.json":
        path = fetch_corpus(path)
    tasks = load(path)
    if args.split is not None:
        tasks = [t for t in tasks if t.split == args.split]
    if args.limit is not None:
        tasks = tasks[: args.limit]
    pairs = [
        json.loads(line)
        for line in (args.run_dir / "pairs.jsonl").read_text().splitlines()
    ]
    rows = gate_pairs(pairs, tasks, draws=args.draws, seed=args.seed)
    with open(args.run_dir / "gate.jsonl", "w") as handle:
        for r in rows:
            handle.write(json.dumps(r, sort_keys=True) + "\n")
    summary = summarise(rows, args.draws)
    label = args.label or f"{args.run_dir}"
    summary["label"] = label
    summary["seed"] = args.seed
    (args.run_dir / "gate.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.run_dir / "gate.md").write_text(markdown(summary, label))
    print(markdown(summary, label))


if __name__ == "__main__":
    main()
