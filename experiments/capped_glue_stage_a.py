"""Synthetic-tested selection, provenance and Stage A for #178, D1 + A.

No command entry point and no checkpoint loading. The final diagnostic must
still add encoder diagnostics and terminal/scoring completeness before review.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import TYPE_CHECKING

from experiments import capped_glue_loss as core
from wandering_light.stack_machine import (
    DEFAULT_FUEL_LIMIT,
    MAX_QUOTATION_DEPTH,
    Config,
    format_stack,
)
from wandering_light.stack_pbe import Task
from wandering_light.stack_policy import validation_skeleton

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

CHECKPOINT_ORDER = tuple(
    (encoding, seed) for encoding in ("machine", "difference") for seed in range(3)
)
CHECKPOINT_FILES = frozenset(
    ("model.safetensors", "policy_config.json", "training_manifest.json")
)
GROUP_CAP = 64
BEHAVIOUR_SALT = "capped-glue-D1-behaviour:"
GROUP_SALT = "capped-glue-D1-group:"
HIDDEN_FIELDS = frozenset(("held_out_inputs", "held_out_outputs"))


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def hash_order(salt: str, name: str) -> tuple[str, str]:
    return hashlib.sha256((salt + name).encode("utf-8")).hexdigest(), name


def sha256_file(path: Path, *, deadline: float | None = None) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1 << 20):
            core.check_deadline(deadline)
            digest.update(block)
    core.check_deadline(deadline)
    return digest.hexdigest()


def verify_provenance(
    spec: Mapping, repo: Path, *, deadline: float | None = None
) -> dict:
    """Hash the exact pinned bytes; do not deserialize any checkpoint weights."""
    if set(spec["inputs"]) != {"train", "side"}:
        raise core.InvalidDiagnosticError(
            "expected exactly train and capped side inputs"
        )
    checkpoints = spec["checkpoints"]
    if [(row["encoding"], row["seed"]) for row in checkpoints] != list(
        CHECKPOINT_ORDER
    ):
        raise core.InvalidDiagnosticError("checkpoint order/population differs from D1")
    checked = {"inputs": {}, "checkpoints": []}

    def check(pin):
        path = repo / pin["path"]
        digest = sha256_file(path, deadline=deadline)
        if digest != pin["sha256"]:
            raise core.InvalidDiagnosticError(f"input hash mismatch: {path}")
        return {"path": str(path.resolve()), "sha256": digest}

    for role in ("train", "side"):
        checked["inputs"][role] = check(spec["inputs"][role])
    for row in checkpoints:
        if set(row["hashes"]) != CHECKPOINT_FILES:
            raise core.InvalidDiagnosticError("missing/extra checkpoint hash pins")
        path = repo / row["path"]
        hashes = {}
        for name in sorted(CHECKPOINT_FILES):
            hashes[name] = check(
                {"path": str(path / name), "sha256": row["hashes"][name]}
            )["sha256"]
        checked["checkpoints"].append(
            {
                "encoding": row["encoding"],
                "seed": row["seed"],
                "path": str(path.resolve()),
                "hashes": hashes,
            }
        )
    return checked


def shown_task_from_record(record: Mapping) -> Task:
    """Retain original metadata while neither accessing nor retaining hidden pairs."""
    values = {
        field.name: record[field.name]
        for field in fields(Task)
        if field.name not in HIDDEN_FIELDS and field.name in record
    }
    values.update(held_out_inputs=(), held_out_outputs=())
    for name in ("inputs", "outputs", "program", "tags"):
        values[name] = tuple(values[name])
    if "alternative_programs" in values:
        values["alternative_programs"] = tuple(
            tuple(program) for program in values["alternative_programs"]
        )
    task = Task(**values)
    if task.split != "train":
        raise core.InvalidDiagnosticError("non-train row in pinned training payload")
    return task


def read_train(path: Path, *, deadline: float | None = None) -> list[Task]:
    # Deliberately takes one payload, never a directory/corpus manifest which
    # might open the eval split as well. Real callers must verify its pin first.
    opener = gzip.open if path.suffix == ".gz" else Path.open
    tasks = []
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            core.check_deadline(deadline)
            if line.strip():
                tasks.append(shown_task_from_record(json.loads(line)))
    return tasks


def side_programs(records: Sequence[Mapping]) -> dict[str, str]:
    programs = {}
    for row in records:
        name, program = row.get("behaviour"), row.get("program")
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(program, str)
            or not program.split()
        ):
            raise core.InvalidDiagnosticError(
                "side row requires behaviour and program strings"
            )
        if name in programs:
            raise core.InvalidDiagnosticError(
                f"duplicate capped side behaviour: {name}"
            )
        programs[name] = program
    return programs


def read_side(path: Path, *, deadline: float | None = None) -> dict[str, str]:
    records = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            core.check_deadline(deadline)
            if line.strip():
                records.append(json.loads(line))
    return side_programs(records)


@dataclass(frozen=True)
class Selected:
    original_index: int
    original_program: tuple[str, ...]
    task: Task


def select_panel(
    tasks: Sequence[Task], programs: Mapping[str, str], *, deadline: float | None = None
) -> tuple[tuple[Selected, ...], dict]:
    """Frozen hash ranking; original groups/holdout; first file occurrence only."""
    first = {}
    identities = {}
    groups = defaultdict(set)
    for index, task in enumerate(tasks):
        core.check_deadline(deadline)
        if task.split != "train":
            raise core.InvalidDiagnosticError("non-train task in selection population")
        group = task.skeleton or task.behaviour
        if identities.setdefault(task.behaviour, group) != group:
            raise core.InvalidDiagnosticError(
                "behaviour has inconsistent ORIGINAL skeletons"
            )
        first.setdefault(task.behaviour, (index, task))
        if task.behaviour in programs and not validation_skeleton(task):
            groups[group].add(task.behaviour)
    selected = []
    for group in sorted(groups, key=lambda name: hash_order(GROUP_SALT, name))[
        :GROUP_CAP
    ]:
        core.check_deadline(deadline)
        behaviour = min(
            groups[group], key=lambda name: hash_order(BEHAVIOUR_SALT, name)
        )
        index, original = first[behaviour]
        # Only program changes. In particular the original length/skeleton are
        # metadata, never relabelled from the capped shape during selection.
        capped = replace(original, program=tuple(programs[behaviour].split()))
        selected.append(Selected(index, original.program, capped))
    summary = {
        "train_tasks": len(tasks),
        "train_behaviours": len(first),
        "side_behaviours": len(programs),
        "eligible_groups": len(groups),
        "selected_groups": len(selected),
        "group_cap": GROUP_CAP,
        "side_behaviours_absent_from_train": sorted(set(programs) - set(first)),
    }
    return tuple(selected), summary


def selected_metadata(item: Selected) -> dict:
    task = item.task
    metadata = {
        field.name: getattr(task, field.name)
        for field in fields(Task)
        if field.name not in HIDDEN_FIELDS and field.name != "program"
    }
    return {
        "original_index": item.original_index,
        "original_metadata": metadata,
        "original_program": item.original_program,
        "capped_program": task.program,
        "group": task.skeleton or task.behaviour,
        "hidden_pairs_retained": False,
    }


def duplicate_programs(selected: Sequence[Selected]) -> list[dict]:
    copies = defaultdict(list)
    for item in selected:
        copies[" ".join(item.task.program)].append(item.task.behaviour)
    return [
        {"program": program, "behaviours": sorted(behaviours)}
        for program, behaviours in sorted(copies.items())
        if len(behaviours) > 1
    ]


def row_summary(
    rows: Sequence[core.Occurrence], cells: Sequence[core.MatchedCell]
) -> dict:
    matched = {key for cell in cells for key in (*cell.top, *cell.body)}
    counts = Counter(
        (
            row.side,
            "structural" if row.structural else "content",
            row.position,
            row.op,
            row.depth,
        )
        for row in rows
    )
    return {
        "all_rows": len(rows),
        "content_rows": sum(not row.structural for row in rows),
        "structural_rows": sum(row.structural for row in rows),
        "matched_rows": len(matched),
        "matched_top_rows": sum(
            row.key in matched and row.side == "top" for row in rows
        ),
        "matched_body_rows": sum(
            row.key in matched and row.side == "body" for row in rows
        ),
        "omitted_content_rows": sum(
            not row.structural and row.key not in matched for row in rows
        ),
        "by_label": [
            {
                "side": side,
                "class": role,
                "position": position,
                "op": op,
                "depth": depth,
                "count": count,
            }
            for (side, role, position, op, depth), count in sorted(counts.items())
        ],
    }


def build_stage_a(
    tasks: Sequence[Task],
    programs: Mapping[str, str],
    provenance: Mapping,
    *,
    source_revision: str,
    deadline: float | None = None,
) -> dict:
    """Selection/replay/support first; only supported panels enumerate alternatives.

    Encoder width/truncation diagnostics will be added before the real pre-trial
    freeze. This builder currently has synthetic callers only.
    """
    panel, selection = select_panel(tasks, programs, deadline=deadline)
    all_rows, replays = [], []
    for item in panel:
        core.check_deadline(deadline)
        replay = core.replay_shown(item.task, deadline=deadline)
        all_rows.extend(core.occurrences(item.task, replay))
        replays.append((item, replay))
    cells = core.matched_cells(all_rows)
    gate = core.support(cells)
    matched = {key for cell in cells for key in (*cell.top, *cell.body)}
    by_key = {row.key: row for row in all_rows}
    sealed_rows = []
    refusal_counts = Counter()
    for item, replay in replays:
        for step, product in enumerate(replay.states[:-1]):
            core.check_deadline(deadline)
            occurrence = by_key[item.task.behaviour, step]
            row = {
                **asdict(occurrence),
                "side": occurrence.side,
                "structural": occurrence.structural,
                "assembly": occurrence.op in core.ASSEMBLY,
                "matched": occurrence.key in matched,
                "original_index": item.original_index,
                "product": [format_stack(Config(stack=stack)) for stack in product],
                "target": item.task.outputs,
                "witness_fuel": replay.fuel[step],
                "witness_executions": replay.executions[step],
            }
            if gate["passes"]:
                legality = core.legal_alternatives(
                    product, occurrence.op, deadline=deadline
                )
                row["legality"] = legality
                refusal_counts.update(legality["first_refusal_counts"])
            sealed_rows.append(row)
    core.check_deadline(deadline)
    return {
        "schema": "capped-glue-D1-A-stage-a",
        "source_revision": source_revision,
        "provenance": dict(provenance),
        "complete": True,
        "kind": "supported" if gate["passes"] else "support_only",
        "limits": {
            "group_cap": GROUP_CAP,
            "call_fuel": DEFAULT_FUEL_LIMIT,
            "initial_depth": 0,
            "max_quotation_depth": MAX_QUOTATION_DEPTH,
            "deadline_seconds": 900,
            "batch_size": 32,
            "torch_threads": 1,
        },
        "selection": selection,
        "panel": [selected_metadata(item) for item in panel],
        "duplicate_capped_programs": duplicate_programs(panel),
        "support": gate,
        "counts": row_summary(all_rows, cells),
        "cells": [asdict(cell) for cell in cells],
        "rows": sealed_rows,
        "alternative_first_refusal_counts": dict(refusal_counts),
        "models_loaded": 0,
        "logit_rows": 0,
    }


def validate_stage_a(stage: Mapping) -> None:
    """Structural seal completeness, independent of the model and its losses."""
    if (
        stage["complete"] is not True
        or stage["models_loaded"] != 0
        or stage["logit_rows"] != 0
    ):
        raise core.InvalidDiagnosticError(
            "Stage A is incomplete or contains model work"
        )
    panel = stage["panel"]
    behaviours = [item["original_metadata"]["behaviour"] for item in panel]
    if len(set(behaviours)) != len(behaviours):
        raise core.InvalidDiagnosticError("Stage A panel repeats a behaviour")
    if any(
        item["original_metadata"]["split"] != "train"
        or HIDDEN_FIELDS & set(item["original_metadata"])
        for item in panel
    ):
        raise core.InvalidDiagnosticError(
            "Stage A contains eval or hidden-example metadata"
        )
    expected = {
        (item["original_metadata"]["behaviour"], step)
        for item in panel
        for step in range(len(item["capped_program"]))
    }
    rows = stage["rows"]
    actual = [(row["behaviour"], row["step"]) for row in rows]
    if (
        len(set(actual)) != len(actual)
        or set(actual) != expected
        or len(rows) != stage["counts"]["all_rows"]
    ):
        raise core.InvalidDiagnosticError(
            "Stage A row identities/counts are incomplete"
        )
    occurrence_fields = tuple(field.name for field in fields(core.Occurrence))
    occurrences = [
        core.Occurrence(**{name: row[name] for name in occurrence_fields})
        for row in rows
    ]
    cells = core.matched_cells(occurrences)
    if canonical_bytes([asdict(cell) for cell in cells]) != canonical_bytes(
        stage["cells"]
    ):
        raise core.InvalidDiagnosticError("Stage A cells disagree with row membership")
    if canonical_bytes(core.support(cells)) != canonical_bytes(
        stage["support"]
    ) or canonical_bytes(row_summary(occurrences, cells)) != canonical_bytes(
        stage["counts"]
    ):
        raise core.InvalidDiagnosticError("Stage A support/counts disagree with rows")
    matched = {key for cell in cells for key in (*cell.top, *cell.body)}
    by_behaviour = {item["original_metadata"]["behaviour"]: item for item in panel}
    for row, occurrence in zip(rows, occurrences, strict=True):
        item = by_behaviour[row["behaviour"]]
        metadata = item["original_metadata"]
        if row["op"] != item["capped_program"][row["step"]] or row["group"] != (
            metadata["skeleton"] or metadata["behaviour"]
        ):
            raise core.InvalidDiagnosticError(
                "Stage A row disagrees with selected program/group"
            )
        if (
            row["side"] != occurrence.side
            or row["structural"] != occurrence.structural
            or row["matched"] != (occurrence.key in matched)
        ):
            raise core.InvalidDiagnosticError("Stage A derived row labels disagree")
        if canonical_bytes(row["target"]) != canonical_bytes(
            metadata["outputs"]
        ) or len(row["product"]) != len(metadata["inputs"]):
            raise core.InvalidDiagnosticError(
                "Stage A shown-example row identity disagrees"
            )
    supported = stage["support"]["passes"]
    if stage["kind"] != ("supported" if supported else "support_only"):
        raise core.InvalidDiagnosticError("Stage A kind/support disagree")
    for row in rows:
        if supported:
            legal = row.get("legality")
            if (
                legal is None
                or len(legal["mask"]) != len(core.OPS)
                or not all(type(value) is bool for value in legal["mask"])
            ):
                raise core.InvalidDiagnosticError(
                    "supported Stage A lacks complete legal mask"
                )
            names = [op.name for op in core.OPS]
            if not legal["mask"][names.index(row["op"])]:
                raise core.InvalidDiagnosticError("sealed witness is illegal")
        elif "legality" in row:
            raise core.InvalidDiagnosticError(
                "unsupported panel performed alternative enumeration"
            )


def atomic_new_json(path: Path, value: object) -> str:
    """Publish a complete file atomically and refuse to overwrite a prior seal."""
    payload = canonical_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        # link is atomic and fails if the destination already exists.
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class StageSeal:
    path: str
    sha256: str
    rows: int
    kind: str


def seal_stage_a(stage: Mapping, path: Path) -> StageSeal:
    validate_stage_a(stage)
    digest = atomic_new_json(path, stage)
    return StageSeal(str(path.resolve()), digest, len(stage["rows"]), stage["kind"])


def read_sealed_stage_a(seal: StageSeal) -> dict:
    payload = Path(seal.path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != seal.sha256:
        raise core.InvalidDiagnosticError("Stage A seal hash mismatch")
    stage = json.loads(payload)
    validate_stage_a(stage)
    if len(stage["rows"]) != seal.rows or stage["kind"] != seal.kind:
        raise core.InvalidDiagnosticError("Stage A seal metadata mismatch")
    return stage
