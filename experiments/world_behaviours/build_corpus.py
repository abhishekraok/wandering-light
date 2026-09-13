"""Build a `stack_pbe_world_*` corpus from verified world-behaviour witnesses (#120, #127).

Reads one or more report directories (``behaviours.jsonl`` plus ``report.json``
from ``report.py``), keeps every behaviour whose program verified on every
test, and writes v4-schema ``Task`` rows: shown examples are the report's shown
tests, held-out examples the rest, the witness is the expanded ``mark-28``
program and the macro form travels in ``Task.macro_program``. Every row is
labelled by execution, as ``stack_pbe._label`` does, and re-graded with
``stack_pbe_eval.grade`` before anything is written. With ``--eval-share`` a
deterministic slice of behaviours (by id hash) goes to an ``eval`` split, so
in-distribution gain can be measured; v1 was train-only.

Usage:
    uv run python experiments/world_behaviours/build_corpus.py \
        reports/world-behaviours-library-20260910 reports/world-authoring-20260910 \
        --corpus-name stack_pbe_world_v2 --eval-share 0.1 --issue 127 [--publish]

``--publish`` uploads the payload and manifest to the Hub, verifies the
published revision, and records it in the tracked manifest.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
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
DATA_DIR = Path("wandering_light/evals/data")
SUMMARY_CHARS = 200


def hub_repo_id(corpus_name: str) -> str:
    return f"abhishekraok/wandering-light-{corpus_name.replace('_', '-')}"


def eval_bucket(behaviour_id: str) -> float:
    """A stable point in [0, 1) per behaviour id; ids below ``eval_share`` are eval."""
    digest = hashlib.sha256(behaviour_id.encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


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


def build_task(
    behaviour: dict,
    result: dict,
    macros: dict[str, list[str]],
    eval_share: float = 0.0,
) -> Task:
    program = tuple(expand(result["macro_program"].split(), macros))
    if " ".join(program) != result["program"]:
        raise ValueError(f"{behaviour['id']}: macro expansion differs from the report")
    shown, held = split_tests(behaviour)
    inputs, outputs, fuel = label(program, shown)
    hidden_in, hidden_out, _ = label(program, held)
    tags = (
        behaviour["source"],
        *([behaviour["difficulty"]] if behaviour.get("difficulty") else []),
        *(t.lower().replace(" ", "-") for t in behaviour.get("tags", [])),
    )
    task = Task(
        behaviour=behaviour["id"],
        summary=summary_of(behaviour),
        split="eval" if eval_bucket(behaviour["id"]) < eval_share else "train",
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


def as_dirs(probe_dirs: Path | list[Path]) -> list[Path]:
    return [probe_dirs] if isinstance(probe_dirs, Path) else list(probe_dirs)


def build(
    probe_dirs: Path | list[Path],
    eval_share: float = 0.0,
    exclude: frozenset[str] = frozenset(),
) -> list[Task]:
    """Tasks for every verified behaviour, minus ``exclude``.

    ``exclude`` removes a behaviour from the corpus outright rather than moving
    it between splits, and a behaviour's split depends only on its own id, so
    excluding eval behaviours leaves the train split and its payload digest
    unchanged.
    """
    macros = load_macros()
    tasks: list[Task] = []
    for probe_dir in as_dirs(probe_dirs):
        behaviours = {
            json.loads(l)["id"]: json.loads(l)
            for l in (probe_dir / "behaviours.jsonl").open()
        }
        report = json.loads((probe_dir / "report.json").read_text())
        tasks += [
            build_task(behaviours[r["id"]], r, macros, eval_share)
            for r in sorted(report["results"], key=lambda r: r["id"])
            if r["solved"] and r["id"] not in exclude
        ]
    if len({t.behaviour for t in tasks}) != len(tasks):
        raise ValueError("duplicate behaviour ids")
    return tasks


def write(
    tasks: list[Task],
    corpus_dir: Path,
    probe_dirs: Path | list[Path],
    corpus_name: str = CORPUS_NAME,
    eval_share: float = 0.0,
    issue: int = 120,
    exclude: frozenset[str] = frozenset(),
    exclude_reason: str = "",
) -> Path:
    corpus_dir.mkdir(parents=True, exist_ok=True)
    splits = {}
    for split in sorted({t.split for t in tasks}):
        rows = [t for t in tasks if t.split == split]
        path = corpus_dir / f"{corpus_name}_{split}.jsonl.gz"
        # mtime pinned so the payload digest depends on the rows alone
        with (
            gzip.GzipFile(path, "wb", mtime=0) as raw,
            io.TextIOWrapper(raw, "utf-8") as handle,
        ):
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
    policy = (
        f"by-behaviour-hash eval_share={eval_share}" if eval_share else "train-only"
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "corpus": corpus_name,
        "basis": BASIS,
        "generator": "experiments/world_behaviours/build_corpus.py",
        "generator_description": (
            "Behaviours are human-written programming problems with tests "
            f"({', '.join(sources)}) restricted to ints and flat int lists; "
            "witnesses were authored by Claude and Codex sessions in macro form "
            "over experiments/world_behaviours/macros*.txt and expanded to the "
            "basis; examples are the upstream tests extended by inputs that keep "
            "or deliberately break the properties the given inputs share, run "
            f"through the reference solution. Split policy: {policy}."
        ),
        "config": {
            "probe": [d.as_posix() for d in as_dirs(probe_dirs)],
            "sources": sources,
            "shown_per_task": max(t.examples for t in tasks),
            "witness_source": "translated",
            "eval_share": eval_share,
            "excluded_behaviours": sorted(exclude),
            "exclusion_reason": exclude_reason,
            "issue": f"https://github.com/abhishekraok/wandering-light/issues/{issue}",
        },
        "split_policy": policy,
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
        + f"---\n\n# `{corpus_name}`\n\n"
        f"World-authored stack-programming-by-example corpus: {', '.join(sources)}\n"
        f"behaviours with `mark-28` witnesses authored in macro form. Split policy: {policy}.\n"
        "The tracked `manifest.json` pins the payload and records its provenance.\n"
        f"See https://github.com/abhishekraok/wandering-light/issues/{issue}.\n"
    )
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("probe_dirs", type=Path, nargs="+")
    parser.add_argument("--corpus-name", default=CORPUS_NAME)
    parser.add_argument("--corpus-dir", type=Path)
    parser.add_argument("--eval-share", type=float, default=0.0)
    parser.add_argument("--issue", type=int, default=120)
    parser.add_argument(
        "--exclude-behaviours",
        type=Path,
        help="file of behaviour ids to leave out; its '#' lines are recorded as the reason",
    )
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args(argv)
    corpus_dir = args.corpus_dir or DATA_DIR / args.corpus_name
    repo_id = hub_repo_id(args.corpus_name)

    exclude: frozenset[str] = frozenset()
    reason = ""
    if args.exclude_behaviours:
        lines = args.exclude_behaviours.read_text().splitlines()
        exclude = frozenset(
            line.strip() for line in lines if line.strip() and not line.startswith("#")
        )
        reason = " ".join(
            line.lstrip("#").strip() for line in lines if line.startswith("#")
        )
    tasks = build(args.probe_dirs, args.eval_share, exclude)
    if exclude:
        present = {t.behaviour for t in build(args.probe_dirs, args.eval_share)}
        if missing := sorted(exclude - present):
            parser.error(f"--exclude-behaviours names unknown behaviours: {missing}")
    manifest = write(
        tasks,
        corpus_dir,
        args.probe_dirs,
        args.corpus_name,
        args.eval_share,
        args.issue,
        exclude,
        reason,
    )
    corpus_hub.verify_corpus_files(corpus_dir)
    print(f"{len(tasks)} tasks -> {manifest}")
    if args.publish:
        revision = corpus_hub.publish_corpus(
            corpus_dir,
            repo_id=repo_id,
            extra_files=[corpus_dir / "README.md"],
            commit_message=f"Publish {args.corpus_name}",
        )
        corpus_hub.verify_published_corpus(
            corpus_dir, repo_id=repo_id, revision=revision
        )
        corpus_hub.record_hub_location(corpus_dir, repo_id=repo_id, revision=revision)
        print(f"published {repo_id} @ {revision}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
