"""Audit a behaviour-first stack-PBE corpus.

The analysis is deliberately corpus-derived. A published corpus can outlive
or differ from the current in-tree catalogue, so group and scaffold claims are
computed from one consistency-checked representative row per behaviour.

    python experiments/stack_pbe_scope.py \
        --corpus wandering_light/evals/data/stack_pbe_v4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from wandering_light.corpus_hub import verify_corpus_files
from wandering_light.stack_machine import Int, Lst, Sym, Value, parse_stack
from wandering_light.stack_pbe import BASIS, Task, load, skeleton

COMBINATORS = {"map", "filter", "fold", "ifte"}
RUNS_CODE = COMBINATORS | {"i", "dip"}
ARITHMETIC = {"add", "sub", "mul", "div"}
PREDICATES = {"eq?", "lt?", "null?"}
BRACKETS = {"[", "]"}
REPORT_TAGS = ("branch", "fold", "map", "filter", "closure")
SCAFFOLD_WINDOW = 8

# Published figures for `stack_pbe_v2.jsonl.gz`, from #59. Keeping them here
# prevents the comparison from drifting from what STATUS records.
V2 = {
    "tasks": 1857,
    "distinct_input_stacks": 17,
    "quotation_in_input": 0.366,
    "builds_quotation": 0.0,
    "runs_code": 0.118,
    "uses_combinator": 0.057,
    "no_computation": 0.553,
}


def has_quotation(value: Value) -> bool:
    """Whether a value contains an op name -- a program rather than data."""
    if isinstance(value, Sym):
        return True
    if isinstance(value, Lst):
        return any(has_quotation(item) for item in value.items)
    return False


def _stack_shape(text: str) -> tuple[str, ...]:
    """Return the parsed top-level data shape of one displayed input stack."""
    shape = []
    for value in parse_stack(text, basis=BASIS):
        if isinstance(value, Lst):
            if any(not isinstance(item, Int) for item in value.items):
                raise ValueError(f"input list is not flat integer data: {text}")
            shape.append("list")
        elif isinstance(value, Int):
            shape.append("scalar")
        else:
            raise ValueError(f"input contains non-integer data: {text}")
    return tuple(shape)


def _shape_label(shape: tuple[str, ...]) -> str:
    counts = Counter(shape)
    pieces = []
    for name in dict.fromkeys(shape):
        count = counts[name]
        pieces.append(name if count == 1 else f"{count} {name}s")
    return " + ".join(pieces) if pieces else "empty stack"


def _task_shape(task: Task) -> tuple[str, ...]:
    shapes = {_stack_shape(text) for text in (*task.inputs, *task.held_out_inputs)}
    if len(shapes) != 1:
        rendered = ", ".join(sorted(_shape_label(shape) for shape in shapes))
        raise ValueError(f"{task.behaviour}: task mixes input shapes: {rendered}")
    return next(iter(shapes))


def _representatives(tasks: list[Task]) -> tuple[dict[str, Task], dict[str, int]]:
    """Choose and validate one canonical corpus row for every behaviour."""
    grouped: dict[str, list[Task]] = {}
    for task in tasks:
        grouped.setdefault(task.behaviour, []).append(task)

    representatives = {}
    counts = {}
    for name in sorted(grouped):
        rows = grouped[name]
        representative = rows[0]
        expected = (
            representative.program,
            representative.length,
            representative.skeleton or skeleton(representative.program),
            representative.split,
            representative.tags,
            representative.summary,
            _task_shape(representative),
        )
        for task in rows:
            if task.length != len(task.program):
                raise ValueError(
                    f"{name}: labelled length {task.length} != {len(task.program)} ops"
                )
            observed = (
                task.program,
                task.length,
                task.skeleton or skeleton(task.program),
                task.split,
                task.tags,
                task.summary,
                _task_shape(task),
            )
            if observed != expected:
                raise ValueError(
                    f"{name}: corpus rows disagree on canonical behaviour metadata"
                )
        representatives[name] = representative
        counts[name] = len(rows)
    return representatives, counts


def independent_groups_from_names(names: list[str]) -> list[list[str]]:
    """Group corpus behaviours connected by ``a_then_b`` components."""
    names = sorted(set(names))
    parent = {name: name for name in names}

    def find(node: str) -> str:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    shared: dict[str, list[str]] = {}
    for name in names:
        components = name.split("_then_") if "_then_" in name else [name]
        for component in components:
            shared.setdefault(component, []).append(name)
    for related in shared.values():
        for name in related[1:]:
            root_a, root_b = find(related[0]), find(name)
            if root_a != root_b:
                parent[root_a] = root_b

    groups: dict[str, list[str]] = {}
    for name in names:
        groups.setdefault(find(name), []).append(name)
    return sorted(
        (sorted(group) for group in groups.values()),
        key=lambda group: (-len(group), group),
    )


def _windows(program: tuple[str, ...], window: int) -> list[tuple[str, ...]]:
    return [
        program[index : index + window] for index in range(len(program) - window + 1)
    ]


def _share_summary(rows: list[dict[str, Any]], total_targets: int) -> dict[str, Any]:
    shares = [row["share"] for row in rows]
    histogram = [0] * 10
    for share in shares:
        histogram[min(int(share * 10), 9)] += 1
    matched = sum(row["matched_windows"] for row in rows)
    windows = sum(row["windows"] for row in rows)
    return {
        "targets": total_targets,
        "eligible": len(rows),
        "excluded_short": total_targets - len(rows),
        "median": statistics.median(shares) if shares else None,
        "mean": statistics.fmean(shares) if shares else None,
        "matched_windows": matched,
        "windows": windows,
        "aggregate_share": matched / windows if windows else None,
        "histogram": [
            {
                "label": f"{lo * 10}-{'100' if lo == 9 else f'<{(lo + 1) * 10}'}%",
                "count": count,
            }
            for lo, count in enumerate(histogram)
        ],
        "by_behaviour": rows,
    }


def _scaffold_analysis(
    representatives: dict[str, Task], window: int = SCAFFOLD_WINDOW
) -> dict[str, Any]:
    programs = {name: task.program for name, task in representatives.items()}
    owners: dict[tuple[str, ...], set[str]] = {}
    for name, program in programs.items():
        for gram in _windows(program, window):
            owners.setdefault(gram, set()).add(name)

    overall_rows = []
    for name in sorted(programs):
        grams = _windows(programs[name], window)
        if not grams:
            continue
        matched = sum(bool(owners[gram] - {name}) for gram in grams)
        overall_rows.append(
            {
                "behaviour": name,
                "share": matched / len(grams),
                "matched_windows": matched,
                "windows": len(grams),
            }
        )

    train_windows = {
        gram
        for name, program in programs.items()
        if representatives[name].split == "train"
        for gram in _windows(program, window)
    }
    eval_names = sorted(
        name for name, task in representatives.items() if task.split == "eval"
    )
    eval_rows = []
    for name in eval_names:
        grams = _windows(programs[name], window)
        if not grams:
            continue
        matched = sum(gram in train_windows for gram in grams)
        eval_rows.append(
            {
                "behaviour": name,
                "share": matched / len(grams),
                "matched_windows": matched,
                "windows": len(grams),
            }
        )

    return {
        "window": window,
        "overall": _share_summary(overall_rows, len(programs)),
        "eval_to_train": _share_summary(eval_rows, len(eval_names)),
    }


def _tag_length_analysis(representatives: dict[str, Task]) -> dict[str, Any]:
    exact: dict[str, dict[str, Any]] = {}
    max_length = max((task.length for task in representatives.values()), default=0)
    for length in sorted({task.length for task in representatives.values()}):
        rows = [task for task in representatives.values() if task.length == length]
        exact[str(length)] = {
            "behaviours": len(rows),
            "tags": {
                tag: sum(tag in task.tags for task in rows) for tag in REPORT_TAGS
            },
        }

    bins = []
    for start in range(1, max_length + 1, 10):
        stop = min(start + 9, max_length)
        rows = [
            task for task in representatives.values() if start <= task.length <= stop
        ]
        if not rows:
            continue
        counts = {tag: sum(tag in task.tags for task in rows) for tag in REPORT_TAGS}
        bins.append(
            {
                "start": start,
                "stop": stop,
                "label": f"{start}-{stop}",
                "behaviours": len(rows),
                "counts": counts,
                "incidence": {tag: counts[tag] / len(rows) for tag in REPORT_TAGS},
            }
        )
    return {"tags": list(REPORT_TAGS), "exact": exact, "bins": bins}


def analyze_tasks(tasks: list[Task], *, corpus: str | None = None) -> dict[str, Any]:
    """Return a JSON-serializable, catalogue-independent corpus analysis."""
    if not tasks:
        raise ValueError("corpus is empty")
    representatives, behaviour_counts = _representatives(tasks)
    n = len(tasks)

    shown = {value for task in tasks for value in task.inputs}
    hidden = {value for task in tasks for value in task.held_out_inputs}
    input_sets = {tuple(sorted(task.inputs)) for task in tasks}
    seeded = sum(
        any(
            has_quotation(value)
            for text in task.inputs
            for value in parse_stack(text, basis=BASIS)
        )
        for task in tasks
    )
    programs = [set(task.program) for task in tasks]
    lengths = Counter(task.length for task in tasks)
    effective_skeleton = {
        id(task): task.skeleton or skeleton(task.program) for task in tasks
    }

    split_names = sorted({task.split for task in tasks})
    split_behaviours = {
        split: {task.behaviour for task in tasks if task.split == split}
        for split in split_names
    }
    split_skeletons = {
        split: {effective_skeleton[id(task)] for task in tasks if task.split == split}
        for split in split_names
    }
    split_summary = {
        split: {
            "tasks": sum(task.split == split for task in tasks),
            "behaviours": len(split_behaviours[split]),
            "skeletons": len(split_skeletons[split]),
            "behaviour_names": sorted(split_behaviours[split]),
        }
        for split in split_names
    }

    behaviour_overlap = sorted(
        set.intersection(*(split_behaviours[name] for name in split_names))
        if len(split_names) > 1
        else set()
    )
    skeleton_overlap = sorted(
        set.intersection(*(split_skeletons[name] for name in split_names))
        if len(split_names) > 1
        else set()
    )

    shape_tasks: Counter[tuple[str, ...]] = Counter()
    shape_behaviours: Counter[tuple[str, ...]] = Counter()
    shape_splits: dict[tuple[str, ...], Counter[str]] = {}
    for task in tasks:
        shape = _task_shape(task)
        shape_tasks[shape] += 1
        shape_splits.setdefault(shape, Counter())[task.split] += 1
    for task in representatives.values():
        shape_behaviours[_task_shape(task)] += 1
    input_shapes = [
        {
            "key": "+".join(shape),
            "label": _shape_label(shape),
            "signature": list(shape),
            "tasks": count,
            "behaviours": shape_behaviours[shape],
            "splits": dict(sorted(shape_splits[shape].items())),
        }
        for shape, count in sorted(
            shape_tasks.items(), key=lambda item: (-item[1], _shape_label(item[0]))
        )
    ]

    groups = independent_groups_from_names(list(representatives))
    group_sizes = Counter(map(len, groups))
    scaffolds = _scaffold_analysis(representatives)
    overall_share = {
        row["behaviour"]: row["share"] for row in scaffolds["overall"]["by_behaviour"]
    }
    eval_share = {
        row["behaviour"]: row["share"]
        for row in scaffolds["eval_to_train"]["by_behaviour"]
    }

    behaviour_rows = []
    for name, task in representatives.items():
        shape = _task_shape(task)
        behaviour_rows.append(
            {
                "behaviour": name,
                "summary": task.summary,
                "split": task.split,
                "tasks": behaviour_counts[name],
                "length": task.length,
                "skeleton": task.skeleton or skeleton(task.program),
                "input_shape": _shape_label(shape),
                "input_shape_key": "+".join(shape),
                "tags": list(task.tags),
                "overall_scaffold_share": overall_share.get(name),
                "eval_to_train_scaffold_share": eval_share.get(name),
            }
        )

    bases = sorted({task.basis for task in tasks})
    return {
        "corpus": corpus,
        "tasks": n,
        "behaviours": len(representatives),
        "skeletons": len(set(effective_skeleton.values())),
        "bases": bases,
        "distinct_input_stacks": len(shown | hidden),
        "distinct_shown_input_stacks": len(shown),
        "distinct_input_sets": len(input_sets),
        "shown_examples": sum(task.examples for task in tasks),
        "held_out_examples": sum(task.held_out for task in tasks),
        "quotation_in_input": seeded / n,
        "builds_quotation": sum(bool(program & BRACKETS) for program in programs) / n,
        "runs_code": sum(bool(program & RUNS_CODE) for program in programs) / n,
        "uses_combinator": sum(bool(program & COMBINATORS) for program in programs) / n,
        "no_computation": sum(
            not (program & (ARITHMETIC | PREDICATES | RUNS_CODE))
            for program in programs
        )
        / n,
        "builds_closure": sum("closure" in task.tags for task in tasks) / n,
        "median_length": statistics.median(task.length for task in tasks),
        "min_length": min(task.length for task in tasks),
        "max_length": max(task.length for task in tasks),
        "median_fuel": statistics.median(task.fuel for task in tasks),
        "length_histogram": dict(sorted(lengths.items())),
        "splits": split_summary,
        "behaviour_overlap": behaviour_overlap,
        "skeleton_overlap": skeleton_overlap,
        "witnesses_per_task": dict(
            sorted(
                Counter(1 + len(task.alternative_programs) for task in tasks).items()
            )
        ),
        "input_shapes": input_shapes,
        "tag_length": _tag_length_analysis(representatives),
        "inseparable_groups": {
            "groups": len(groups),
            "behaviours": len(representatives),
            "largest": max(map(len, groups), default=0),
            "size_histogram": dict(sorted(group_sizes.items())),
            "members": groups,
        },
        "scaffolds": scaffolds,
        "behaviour_rows": behaviour_rows,
    }


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _expect(label: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise ValueError(f"manifest {label} says {expected!r}, corpus has {actual!r}")


def analyze_verified_corpus(corpus_dir: str | Path) -> dict[str, Any]:
    """Verify a manifest/payload directory, then analyze exactly those rows."""
    root = Path(corpus_dir)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checked = verify_corpus_files(root)

    tasks = []
    payloads = []
    for split in sorted(manifest["splits"]):
        metadata = manifest["splits"][split]
        path = root / metadata["path"]
        rows = load(path)
        if any(task.split != split for task in rows):
            raise ValueError(
                f"{path.name}: row split does not match manifest {split!r}"
            )
        effective_skeletons = {task.skeleton or skeleton(task.program) for task in rows}
        _expect(f"splits.{split}.records", len(rows), metadata.get("records"))
        _expect(
            f"splits.{split}.behaviours",
            len({task.behaviour for task in rows}),
            metadata.get("behaviours"),
        )
        _expect(
            f"splits.{split}.skeletons",
            len(effective_skeletons),
            metadata.get("skeletons"),
        )
        _expect(f"splits.{split}.bytes", path.stat().st_size, metadata.get("bytes"))
        tasks.extend(rows)
        payloads.append(
            {
                "split": split,
                "path": path.name,
                "records": len(rows),
                "bytes": path.stat().st_size,
                "sha256": checked[split],
            }
        )

    analysis = analyze_tasks(tasks, corpus=manifest.get("corpus", root.name))
    counts = manifest["counts"]
    for key in (
        "tasks",
        "shown_examples",
        "held_out_examples",
        "distinct_input_stacks",
    ):
        _expect(f"counts.{key}", analysis[key], counts.get(key))
    catalogue = manifest["catalogue"]
    _expect("catalogue.behaviours", analysis["behaviours"], catalogue.get("behaviours"))
    _expect("catalogue.skeletons", analysis["skeletons"], catalogue.get("skeletons"))
    _expect("basis", analysis["bases"], [manifest.get("basis")])
    if manifest.get("split_policy") == "skeleton" and analysis["skeleton_overlap"]:
        raise ValueError("skeleton split leaks skeletons across train and eval")

    hub = manifest.get("hub", {})
    analysis["provenance"] = {
        "hub_repo_id": hub.get("repo_id"),
        "hub_revision": hub.get("revision"),
        "manifest": manifest_path.name,
        "manifest_sha256": _sha256(manifest_path),
        "catalogue_digest": catalogue.get("digest"),
        "payloads": payloads,
    }
    return analysis


def _format_number(value: int | float) -> str:
    return f"{value:g}" if isinstance(value, float) else f"{value}"


def print_text_report(analysis: dict[str, Any], *, source: str) -> None:
    """Print the historical scope audit plus the new split/scaffold detail."""
    print(
        f"{source}: {analysis['tasks']:,} tasks over {analysis['behaviours']} "
        f"behaviours and {analysis['skeletons']} distinct skeletons\n"
    )
    rows = [
        (
            "distinct input stacks",
            analysis["distinct_input_stacks"],
            V2["distinct_input_stacks"],
        ),
        (
            "distinct input sets (one per task)",
            analysis["distinct_input_sets"],
            V2["distinct_input_stacks"],
        ),
    ]
    print(f"{'':<38}{'current':>12}{'v2 (#59)':>12}")
    for label, got, was in rows:
        print(f"{label:<38}{got:>12,}{was:>12,}")
    for key in (
        "quotation_in_input",
        "builds_quotation",
        "runs_code",
        "uses_combinator",
        "no_computation",
    ):
        print(f"{key.replace('_', ' '):<38}{analysis[key]:>11.1%}{V2[key]:>12.1%}")
    print(
        f"{'builds a closure over task data':<38}"
        f"{analysis['builds_closure']:>11.1%}{'--':>12}"
    )

    print(
        "\nwitness length: "
        f"median {_format_number(analysis['median_length'])}, "
        f"range {analysis['min_length']}-{analysis['max_length']}; "
        f"median fuel {_format_number(analysis['median_fuel'])}"
    )
    print(
        f"examples: {analysis['shown_examples']:,} shown + "
        f"{analysis['held_out_examples']:,} held out"
    )
    print(f"length histogram: {analysis['length_histogram']}")

    print("\nsplit is on the skeleton, so no program shape spans both halves:")
    for name, split in analysis["splits"].items():
        print(
            f"  {name:<6}{split['tasks']:>6,} tasks  "
            f"{split['skeletons']:>3} skeletons  "
            f"{split['behaviours']:>3} behaviours  "
            f"{', '.join(split['behaviour_names'])}"
        )
    print(f"  behaviour overlap: {analysis['behaviour_overlap']}")
    print(f"  skeleton overlap:  {analysis['skeleton_overlap']}")
    print(f"\nwitnesses per task: {analysis['witnesses_per_task']}")

    groups = analysis["inseparable_groups"]
    largest = groups["largest"]
    print(
        f"\ninseparable groups: {groups['groups']} across "
        f"{groups['behaviours']} behaviours; largest holds {largest} "
        f"({largest / max(groups['behaviours'], 1):.1%})"
    )
    print(f"  group size histogram: {groups['size_histogram']}")

    scaffolds = analysis["scaffolds"]
    overall = scaffolds["overall"]
    if overall["median"] is not None:
        print(
            f"\nshared scaffolding: among {overall['eligible']}/"
            f"{overall['targets']} witnesses with at least "
            f"{scaffolds['window']} ops, the median share of windows another "
            f"witness contains is {overall['median']:.1%}"
        )
    eval_to_train = scaffolds["eval_to_train"]
    if eval_to_train["median"] is not None:
        print(
            f"  eval to train: among {eval_to_train['eligible']}/"
            f"{eval_to_train['targets']} eligible eval witnesses, median "
            f"{eval_to_train['median']:.1%} of windows occur in train"
        )
    print(
        "  This is the basis, not the catalogue -- it bounds what a held-out "
        "score means rather than invalidating one."
    )

    if largest > groups["behaviours"] * 0.2:
        print(
            "  WARNING: the largest group exceeds a fifth of the corpus, so "
            "no split can hold it out without moving that whole share."
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="wandering_light/evals/data/stack_pbe_v4")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    corpus = Path(args.corpus)
    if corpus.is_dir() and not any(corpus.glob("*.jsonl.gz")):
        print(
            f"{corpus} holds a manifest but no payload -- the corpus lives on the "
            f"Hub, not in git. Fetch it with:\n\n"
            f'  python -c "from wandering_light.corpus_hub import '
            f"fetch_corpus; fetch_corpus('{corpus}/manifest.json')\"\n\n"
            f"or point --corpus at a directory you generated yourself.",
            file=sys.stderr,
        )
        return 1

    analysis = (
        analyze_verified_corpus(corpus)
        if corpus.is_dir()
        else analyze_tasks(load(corpus), corpus=str(corpus))
    )
    print_text_report(analysis, source=str(corpus))

    if args.out:
        with Path(args.out).open("w", encoding="utf-8") as handle:
            json.dump(analysis, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
