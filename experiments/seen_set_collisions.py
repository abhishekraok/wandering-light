"""How often does the beam reach a state it has already seen? (branch-and-check, #164)

`stack_beam` keeps a visited set and gives a successor already in it the
verdict ``duplicate``; the second route is discarded and nothing records that
it existed. That hit rate is the re-convergence rate of *search-generated*
neighbours, the population the authored-alternatives probe on
closed-state-value-search (#156) could not see: over authored witness pairs,
1 of 320 re-converged on an interior closed state. A value function can only
propagate through states two routes share, so this number says whether a
Bellman-style backup would ever have anything to carry on this substrate.

The recorder is a `BeamObserver`, so the search is unchanged; only its
verdicts are read. Each duplicate is classified:

- **cycle**: the state is on the duplicating node's own path (``swap swap``,
  ``dup drop``), one route folding back, not two routes meeting;
- **same-layer**: first retained in this layer by another node, so two
  programs of the same length reach one state;
- **earlier-layer**: first retained in an earlier layer, so the second route
  is longer.

Under a depth-synchronous beam the second route is never shorter: same layer
means equal length, earlier layer means longer, and within a layer the ranked
order keeps the higher-scored twin. Never reopening a state therefore loses
no path length here; it would in a best-first search. Each class is split by
whether the state is closed (no open quotation, no code on the stack), since
the closed states are the ones a ranker over closed states would archive.

    uv run python experiments/seen_set_collisions.py --tasks TASKS [--split eval] \\
        [--policy DIR] --out DIR

``TASKS`` is a task file or a corpus manifest; ``--policy`` a local checkpoint
directory, else `solve.BEST`'s is fetched. The search runs `BEST`'s settings
with the library off (replay does not search). Writes ``tasks.jsonl`` (one
row per task), ``summary.json`` and ``summary.md``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from wandering_light import solve  # noqa: E402
from wandering_light.corpus_hub import fetch_corpus  # noqa: E402
from wandering_light.stack_beam import BeamObserver, beam_solve  # noqa: E402
from wandering_light.stack_machine import Lst, Sym, quotation_open  # noqa: E402
from wandering_light.stack_pbe import OPS, load  # noqa: E402
from wandering_light.stack_policy import PolicyScorer, load_policy  # noqa: E402
from wandering_light.stack_policy_hub import fetch_stack_policy  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from wandering_light.stack_beam import Node, Product
    from wandering_light.stack_machine import Value
    from wandering_light.stack_pbe import Task

CLASSES = ("cycle", "same-layer", "earlier-layer")
SAMPLE_LIMIT = 40


def holds_code(value: Value) -> bool:
    if isinstance(value, Lst):
        return any(holds_code(item) for item in value.items)
    return isinstance(value, Sym)


def closed(product: Product) -> bool:
    """No open quotation and no code on the stack, on the first example.

    The mark structure is the same on every example, since every example ran
    the same program; the first stack decides.
    """
    stack = product[0]
    return not quotation_open(stack) and not any(holds_code(v) for v in stack)


@dataclass
class CollisionRecorder(BeamObserver):
    """Reads the search's ``duplicate`` verdicts and says what each one was."""

    counts: Counter = field(default_factory=Counter)
    layers: int = 0
    retained_per_layer: list[int] = field(default_factory=list)
    samples: list[dict[str, Any]] = field(default_factory=list)
    _beam: Sequence[Node] = ()
    _successors: dict[tuple[int, int], Product] = field(default_factory=dict)
    # product -> (program length, program) at first retention; the root is 0.
    _first: dict[Product, tuple[int, tuple[str, ...]]] = field(default_factory=dict)
    # per beam rank, every product on the path above it, root included
    _ancestors: list[frozenset] = field(default_factory=list)
    # closed states that a second route reached, so a state hit five times is one
    _met: set = field(default_factory=set)

    def layer_started(self, depth: int, beam: Sequence[Node]) -> None:
        self._beam = beam
        self._successors = {}
        if depth == 0:
            self._first = {beam[0][0]: (0, ())}
            self._ancestors = [frozenset()]

    def expansion_attempted(
        self, depth, node, op_index, successor, reason, fuel, stepped
    ) -> None:
        if successor is not None:
            self._successors[node, op_index] = successor

    def layer_retained(
        self,
        depth: int,
        beam: Sequence[Node],
        verdicts: Mapping[tuple[int, int], str],
        parents: Sequence[tuple[int, int]],
    ) -> None:
        self.layers += 1
        self.retained_per_layer.append(len(beam))
        length = depth + 1
        # ``verdicts`` is in ranked order, so a twin retained earlier in this
        # layer is already in ``_first`` when its duplicate is reached.
        for (node, index), verdict in verdicts.items():
            product = self._successors[node, index]
            if verdict == "retained":
                self.counts["retained"] += 1
                self.counts[
                    "retained:closed" if closed(product) else "retained:open"
                ] += 1
                self._first[product] = (length, (*self._beam[node][1], OPS[index].name))
                continue
            if verdict != "duplicate":
                self.counts[verdict] += 1
                continue
            program = (*self._beam[node][1], OPS[index].name)
            if product in self._ancestors[node] or product == self._beam[node][0]:
                kind = "cycle"
            else:
                first_length, _first_program = self._first[product]
                kind = "same-layer" if first_length == length else "earlier-layer"
            state = "closed" if closed(product) else "open"
            self.counts["duplicate"] += 1
            self.counts[f"duplicate:{kind}"] += 1
            self.counts[f"duplicate:{kind}:{state}"] += 1
            if kind != "cycle" and state == "closed":
                # where along the program the two routes meet, and how much
                # longer the second one is (0 for a same-layer twin)
                self.counts[f"met:length:{length}"] += 1
                self.counts[f"met:longer-by:{length - self._first[product][0]}"] += 1
                if product not in self._met:
                    self._met.add(product)
                    self.counts["met:states"] += 1
            if (
                kind != "cycle"
                and state == "closed"
                and len(self.samples) < SAMPLE_LIMIT
            ):
                self.samples.append(
                    {
                        "kind": kind,
                        "first": " ".join(self._first[product][1]),
                        "second": " ".join(program),
                    }
                )
        self._ancestors = [
            self._ancestors[node] | {self._beam[node][0]} for node, _ in parents
        ]

    def row(self) -> dict[str, Any]:
        return {
            "layers": self.layers,
            "retained_per_layer": self.retained_per_layer,
            "counts": dict(self.counts),
        }


def run_tasks(
    tasks: Sequence[Task],
    scorer: PolicyScorer,
    system: solve.System,
    log,
) -> list[dict[str, Any]]:
    rows = []
    for index, task in enumerate(tasks):
        recorder = CollisionRecorder()
        started = time.perf_counter()
        result = beam_solve(
            task,
            scorer,
            beam_width=system.beam_width,
            max_depth=system.max_depth,
            guard=system.guard,
            max_state_atoms=system.max_state_atoms,
            observer=recorder,
        )
        row = {
            "index": index,
            "behaviour": task.behaviour,
            "termination": result.termination,
            "program": list(result.program) if result.program else None,
            "depth": result.depth,
            "expansions": result.expansions,
            "seconds": time.perf_counter() - started,
            **recorder.row(),
            "samples": recorder.samples,
        }
        rows.append(row)
        c = recorder.counts
        log(
            f"{index + 1}/{len(tasks)} {task.behaviour} {result.termination} "
            f"layers {recorder.layers} retained {c['retained']} "
            f"duplicate {c['duplicate']} (closed cross-path "
            f"{c['duplicate:same-layer:closed'] + c['duplicate:earlier-layer:closed']})"
        )
    return rows


def summarise(rows: Sequence[dict[str, Any]], beam_width: int) -> dict[str, Any]:
    total: Counter = Counter()
    layers = 0
    retained_slots = 0
    for row in rows:
        total.update(row["counts"])
        layers += row["layers"]
        retained_slots += sum(row["retained_per_layer"])
    ranked = (
        total["retained"]
        + total["duplicate"]
        + sum(v for k, v in total.items() if k.startswith("guard:"))
    )
    cross_closed = (
        total["duplicate:same-layer:closed"] + total["duplicate:earlier-layer:closed"]
    )
    cross = total["duplicate:same-layer"] + total["duplicate:earlier-layer"]
    tasks_with_cross_closed = sum(
        1
        for r in rows
        if r["counts"].get("duplicate:same-layer:closed", 0)
        + r["counts"].get("duplicate:earlier-layer:closed", 0)
    )
    return {
        "tasks": len(rows),
        "layers": layers,
        "ranked_successors": ranked,
        "counts": dict(total),
        "duplicate_share_of_ranked": _ratio(total["duplicate"], ranked),
        "cross_path_share_of_duplicates": _ratio(cross, total["duplicate"]),
        # the re-convergence rate a ranker over closed states would see
        "closed_cross_path_per_closed_retained": _ratio(
            cross_closed, total["retained:closed"]
        ),
        "tasks_with_closed_cross_path": tasks_with_cross_closed,
        "effective_width": _ratio(retained_slots, layers * beam_width),
        "closed_share_of_retained": _ratio(total["retained:closed"], total["retained"]),
        # one closed state per second route, however many routes reached it
        "closed_states_met_per_closed_retained": _ratio(
            total["met:states"], total["retained:closed"]
        ),
        "met_by_length": _histogram(total, "met:length:"),
        "met_by_longer_by": _histogram(total, "met:longer-by:"),
    }


def _histogram(counts: Counter, prefix: str) -> dict[str, int]:
    keys = sorted(int(k[len(prefix) :]) for k in counts if k.startswith(prefix))
    return {str(k): counts[f"{prefix}{k}"] for k in keys}


def _ratio(a: float, b: float) -> float | None:
    return a / b if b else None


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def markdown(summary: dict[str, Any], label: str) -> str:
    c = summary["counts"]
    ranked = summary["ranked_successors"]
    cross_closed = c.get("duplicate:same-layer:closed", 0) + c.get(
        "duplicate:earlier-layer:closed", 0
    )
    lines = [
        f"## {label}",
        "",
        f"{summary['tasks']} tasks, {summary['layers']} layers, "
        f"{ranked:,} ranked successors.",
        "",
        "| verdict | count | share of ranked |",
        "|---|---:|---:|",
        f"| retained | {c.get('retained', 0):,} | {_pct(_ratio(c.get('retained', 0), ranked))} |",
        f"| duplicate | {c.get('duplicate', 0):,} | {_pct(summary['duplicate_share_of_ranked'])} |",
    ]
    for k in sorted(k for k in c if k.startswith("guard:")):
        lines.append(f"| {k} | {c[k]:,} | {_pct(_ratio(c[k], ranked))} |")
    lines += ["", "| duplicate class | open | closed |", "|---|---:|---:|"]
    for kind in CLASSES:
        lines.append(
            f"| {kind} | {c.get(f'duplicate:{kind}:open', 0):,} | "
            f"{c.get(f'duplicate:{kind}:closed', 0):,} |"
        )
    lines += [
        "",
        f"- closed cross-path duplicates per closed retained state: "
        f"**{_pct(summary['closed_cross_path_per_closed_retained'])}** "
        f"({cross_closed:,} / {c.get('retained:closed', 0):,}), in "
        f"{summary['tasks_with_closed_cross_path']} of {summary['tasks']} tasks",
        f"- cross-path share of duplicates: "
        f"{_pct(summary['cross_path_share_of_duplicates'])}; the rest are cycles",
        f"- distinct closed states a second route reached, per closed retained state: "
        f"{_pct(summary['closed_states_met_per_closed_retained'])} "
        f"({c.get('met:states', 0):,})",
        f"- closed share of retained states: {_pct(summary['closed_share_of_retained'])}",
        f"- effective width: {_pct(summary['effective_width'])} of the beam filled per layer",
        "",
        "Where the second route meets the first (closed cross-path duplicates):",
        "",
        "| program length at meeting | count | | second route longer by | count |",
        "|---:|---:|---|---:|---:|",
    ]
    by_length = list(summary["met_by_length"].items())
    by_gap = list(summary["met_by_longer_by"].items())
    for i in range(max(len(by_length), len(by_gap))):
        left = (
            f"| {by_length[i][0]} | {by_length[i][1]:,} |"
            if i < len(by_length)
            else "| | |"
        )
        right = f" {by_gap[i][0]} | {by_gap[i][1]:,} |" if i < len(by_gap) else " | |"
        lines.append(left + " |" + right)
    lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tasks", required=True, help="task file or corpus manifest")
    parser.add_argument("--split", default=None, help="keep only tasks in this split")
    parser.add_argument(
        "--policy", default=None, help="local checkpoint dir; else BEST"
    )
    parser.add_argument("--label", default=None, help="name for the summary table")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    path = Path(args.tasks)
    if path.name == "manifest.json":
        path = fetch_corpus(path)
    tasks = load(path)
    if args.split is not None:
        tasks = [t for t in tasks if t.split == args.split]
    if args.limit is not None:
        tasks = tasks[: args.limit]
    policy_dir = (
        Path(args.policy)
        if args.policy
        else fetch_stack_policy(solve.BEST.repo_id, solve.BEST.revision)
    )
    scorer = PolicyScorer(load_policy(policy_dir, device="cpu"), "cpu")
    system = dataclasses.replace(solve.BEST, library=False)
    label = args.label or f"{policy_dir.name} on {args.tasks}" + (
        f" [{args.split}]" if args.split else ""
    )

    def log(text: str) -> None:
        sys.stderr.write(text + "\n")
        sys.stderr.flush()

    args.out.mkdir(parents=True, exist_ok=True)
    log(f"policy {policy_dir}; {system.describe()}; {len(tasks)} tasks")
    rows = run_tasks(tasks, scorer, system, log)
    with open(args.out / "tasks.jsonl", "w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    summary = summarise(rows, system.beam_width)
    summary["label"] = label
    summary["policy"] = str(policy_dir)
    summary["system"] = system.describe()
    summary["commit"] = solve.commit()
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.out / "summary.md").write_text(markdown(summary, label))
    log(markdown(summary, label))


if __name__ == "__main__":
    main()
