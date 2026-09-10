"""Build `stack_pbe_world_v1` from the verified world-behaviour witnesses (#120).

Reads a probe directory (``behaviours.jsonl`` plus ``report.json`` from
``report.py``), keeps every behaviour whose program verified on every test,
and writes v4-schema ``Task`` rows: shown examples are the probe's shown tests,
held-out examples the rest, the witness is the expanded ``mark-28`` program and
the macro form travels in ``Task.macro_program``. Every row is labelled by
execution, as ``stack_pbe._label`` does, and re-graded with
``stack_pbe_eval.grade`` before anything is written.

Usage:
    uv run python experiments/world_behaviours/build_corpus.py \
        reports/world-behaviours-probe-20260909 [--publish]

``--publish`` uploads the payload and manifest to the Hub, verifies the
published revision, and records it in the tracked manifest.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

from wandering_light import corpus_hub
from wandering_light.stack_machine import Config, MachineError, format_stack, value_key
from wandering_light.stack_pbe import (
    BASIS,
    SCHEMA_VERSION,
    Task,
    assign_task_hash_split,
    run_program,
    skeleton,
    to_stack,
    to_value,
)
from wandering_light.stack_pbe_eval import grade

sys.path.insert(0, str(Path(__file__).parent))
from verify import expand, load_macros

CORPUS_NAME = "stack_pbe_world_v1"
CORPUS_DIR = Path("wandering_light/evals/data") / CORPUS_NAME
HUB_REPO_ID = f"abhishekraok/wandering-light-{CORPUS_NAME.replace('_', '-')}"
SUMMARY_CHARS = 200


def summary_of(behaviour: dict) -> str:
    """The upstream description, cut to its first sentence or so."""
    text = " ".join(behaviour["description"].split())
    for stop in (". ", ".\n"):
        if stop in text[:SUMMARY_CHARS]:
            text = text[: text.index(stop) + 1]
            break
    return text[:SUMMARY_CHARS]


def label(
    program: tuple[str, ...], tests: list[dict]
) -> tuple[tuple[str, ...], tuple[str, ...], int]:
    """Run the witness on every test; refuse to write a row it does not produce."""
    inputs, outputs, fuel = [], [], 0
    for test in tests:
        stack = to_stack(tuple(test["args"]))
        outcome = run_program(program, stack)
        if isinstance(outcome, str):
            raise MachineError(f"{test['args']}: {outcome}")
        result, spent = outcome
        if len(result) != 1 or value_key(result[0]) != value_key(to_value(test["out"])):
            raise MachineError(f"{test['args']}: witness disagrees with the reference")
        inputs.append(format_stack(Config(stack)))
        outputs.append(format_stack(Config(result)))
        fuel += spent
    return tuple(inputs), tuple(outputs), fuel


def split_tests(behaviour: dict) -> tuple[list[dict], list[dict]]:
    """Shown and held-out tests with no input appearing in both.

    Upstream test lists repeat inputs (three LeetCode behaviours in the probe),
    and an index split would show an example and then grade on it. Keep the
    first occurrence of each serialised input, in the probe's order.
    """
    tests = behaviour["tests"]
    seen: set[str] = set()
    unique: list[int] = []
    for index in [*behaviour["shown"], *behaviour["held_out"]]:
        key = json.dumps(tests[index]["args"], sort_keys=True)
        if key not in seen:
            seen.add(key)
            unique.append(index)
    shown_count = min(len(behaviour["shown"]), len(unique))
    shown = [tests[i] for i in unique[:shown_count]]
    held = [tests[i] for i in unique[shown_count:]]
    if not held:
        raise ValueError(f"{behaviour['id']}: no distinct held-out inputs")
    return shown, held


def build_task(behaviour: dict, result: dict, macros: dict[str, list[str]]) -> Task:
    program = tuple(expand(result["macro_program"].split(), macros))
    if " ".join(program) != result["program"]:
        raise ValueError(f"{behaviour['id']}: macro expansion differs from the report")
    shown, held = split_tests(behaviour)
    inputs, outputs, fuel = label(program, shown)
    hidden_in, hidden_out, _ = label(program, held)
    tags = (
        behaviour["source"],
        *(t.lower().replace(" ", "-") for t in behaviour.get("tags", [])),
    )
    task = Task(
        behaviour=behaviour["id"],
        summary=summary_of(behaviour),
        split="train",
        task_hash_split=assign_task_hash_split(inputs, outputs),
        inputs=inputs,
        outputs=outputs,
        held_out_inputs=hidden_in,
        held_out_outputs=hidden_out,
        program=program,
        length=len(program),
        examples=len(shown),
        held_out=len(held),
        fuel=fuel,
        tags=tags,
        skeleton=skeleton(program),
        witness_source="translated",
        macro_program=result["macro_program"],
    )
    score = grade(task.program, task)
    if not score.solved:
        raise MachineError(f"{behaviour['id']}: witness does not grade as solved")
    return task


def build(probe_dir: Path) -> list[Task]:
    behaviours = {
        json.loads(l)["id"]: json.loads(l)
        for l in (probe_dir / "behaviours.jsonl").open()
    }
    report = json.loads((probe_dir / "report.json").read_text())
    macros = load_macros()
    tasks = [
        build_task(behaviours[r["id"]], r, macros)
        for r in sorted(report["results"], key=lambda r: r["id"])
        if r["solved"]
    ]
    if len({t.behaviour for t in tasks}) != len(tasks):
        raise ValueError("duplicate behaviour ids")
    return tasks


def write(tasks: list[Task], corpus_dir: Path, probe_dir: Path) -> Path:
    corpus_dir.mkdir(parents=True, exist_ok=True)
    splits = {}
    for split in sorted({t.split for t in tasks}):
        rows = [t for t in tasks if t.split == split]
        path = corpus_dir / f"{CORPUS_NAME}_{split}.jsonl.gz"
        with gzip.open(path, "wt") as handle:
            for task in rows:
                handle.write(json.dumps(asdict(task), sort_keys=True) + "\n")
        splits[split] = {
            "path": path.name,
            "sha256": corpus_hub._sha256_file(path),
            "bytes": path.stat().st_size,
            "records": len(rows),
            "skeletons": len({t.skeleton for t in rows}),
            "behaviours": len({t.behaviour for t in rows}),
        }
    catalogue = hashlib.sha256(
        "\n".join(f"{t.behaviour}\t{' '.join(t.program)}" for t in tasks).encode()
    ).hexdigest()
    inputs = {i for t in tasks for i in (*t.inputs, *t.held_out_inputs)}
    sources = sorted({t.tags[0] for t in tasks})
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "corpus": CORPUS_NAME,
        "basis": BASIS,
        "generator": "experiments/world_behaviours/build_corpus.py",
        "generator_description": (
            "Behaviours are human-written programming problems with tests "
            "(MBPP, LeetCode-easy) restricted to ints and flat int lists; "
            "witnesses were authored by Claude sessions in macro form over "
            "experiments/world_behaviours/macros*.txt and expanded to the basis; "
            "examples are the upstream tests (MBPP extended by shape-mutated "
            "inputs run through the reference solution). Train-only."
        ),
        "config": {
            "probe": probe_dir.as_posix(),
            "sources": sources,
            "shown_per_task": max(t.examples for t in tasks),
            "witness_source": "translated",
            "issue": "https://github.com/abhishekraok/wandering-light/issues/120",
        },
        "split_policy": "train-only",
        "catalogue": {
            "behaviours": len(tasks),
            "skeletons": len({t.skeleton for t in tasks}),
            "digest": f"sha256:{catalogue}",
        },
        "counts": {
            "tasks": len(tasks),
            "shown_examples": sum(t.examples for t in tasks),
            "held_out_examples": sum(t.held_out for t in tasks),
            "distinct_input_stacks": len(inputs),
        },
        "splits": splits,
    }
    path = corpus_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", "utf-8")
    readme = corpus_dir / "README.md"
    readme.write_text(
        "---\nconfigs:\n- config_name: default\n  data_files:\n"
        + "".join(f"  - split: {s}\n    path: {m['path']}\n" for s, m in splits.items())
        + f"---\n\n# `{CORPUS_NAME}`\n\n"
        "World-authored stack-programming-by-example corpus: MBPP and LeetCode-easy\n"
        "behaviours with `mark-28` witnesses authored in macro form. Train-only; eval\n"
        "stays `stack_pbe_v4`. The tracked `manifest.json` pins the payload and records\n"
        "its provenance. See stack-pbe-world-v1 (#120).\n"
    )
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("probe_dir", type=Path)
    parser.add_argument("--corpus-dir", type=Path, default=CORPUS_DIR)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args(argv)

    tasks = build(args.probe_dir)
    manifest = write(tasks, args.corpus_dir, args.probe_dir)
    corpus_hub.verify_corpus_files(args.corpus_dir)
    print(f"{len(tasks)} tasks -> {manifest}")
    if args.publish:
        revision = corpus_hub.publish_corpus(
            args.corpus_dir,
            repo_id=HUB_REPO_ID,
            extra_files=[args.corpus_dir / "README.md"],
            commit_message=f"Publish {CORPUS_NAME}",
        )
        corpus_hub.verify_published_corpus(
            args.corpus_dir, repo_id=HUB_REPO_ID, revision=revision
        )
        corpus_hub.record_hub_location(
            args.corpus_dir, repo_id=HUB_REPO_ID, revision=revision
        )
        print(f"published {HUB_REPO_ID} @ {revision}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
