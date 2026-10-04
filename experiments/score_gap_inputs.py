"""Reserved, pinned shown-input stage for #180; synthetic invocations only.

No model loader or trial entrypoint. A future reviewed runner must reserve the
one real attempt before calling this stage. Companion reading proves storage
and projection consistency, not independent original-source selection.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from experiments.score_gap_profiles import HALVES, digest, project_shown, select_panel
from experiments.score_gap_storage import (
    DiagnosticProvenanceError,
    _publish,
    beam_chunks,
    canonical_json,
    decode_beam,
    write_chunks,
)
from wandering_light.stack_machine import parse_stack
from wandering_light.stack_pbe import BASIS

if TYPE_CHECKING:
    from collections.abc import Mapping

    from experiments.score_gap_storage import AttemptReceipts, PayloadBudget

CHECKPOINT_FILES = frozenset(
    ("model.safetensors", "policy_config.json", "training_manifest.json")
)
INPUT_SCHEMA = "SG-INPUT1"
PANEL_SIZE = 48


def _sha(value):
    if (
        type(value) is not str
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise DiagnosticProvenanceError("invalid lowercase SHA256 pin")


def _path(value, repo):
    if type(value) is not str or not value:
        raise DiagnosticProvenanceError("invalid pinned path")
    return (repo / value).resolve()


def validate_spec(spec: Mapping, repo: Path) -> dict:
    """Validate the entire population before opening any pinned file."""
    if set(spec["inputs"]) != {"train", "side"}:
        raise DiagnosticProvenanceError("expected exactly train and side pins")
    checkpoints = spec["checkpoints"]
    if (
        type(checkpoints) is not list
        or len(checkpoints) != 3
        or any(
            row.get("encoding") != "difference"
            or type(row.get("seed")) is not int
            or row["seed"] != seed
            for seed, row in enumerate(checkpoints)
        )
    ):
        raise DiagnosticProvenanceError("expected difference seeds0/1/2 in order")
    result = {"inputs": {}, "checkpoints": []}
    for role in ("train", "side"):
        pin = spec["inputs"][role]
        if set(pin) != {"path", "sha256"}:
            raise DiagnosticProvenanceError("invalid input pin fields")
        _sha(pin["sha256"])
        result["inputs"][role] = {
            "path": str(_path(pin["path"], repo)),
            "sha256": pin["sha256"],
        }
    for row in checkpoints:
        if (
            set(row) != {"path", "hashes", "encoding", "seed"}
            or set(row["hashes"]) != CHECKPOINT_FILES
        ):
            raise DiagnosticProvenanceError("invalid checkpoint pin fields")
        for sha in row["hashes"].values():
            _sha(sha)
        result["checkpoints"].append(
            {
                "encoding": "difference",
                "seed": row["seed"],
                "path": str(_path(row["path"], repo)),
                "hashes": dict(row["hashes"]),
            }
        )
    paths = [pin["path"] for pin in result["inputs"].values()] + [
        str((Path(row["path"]) / name).resolve())
        for row in result["checkpoints"]
        for name in sorted(CHECKPOINT_FILES)
    ]
    if len(paths) != len(set(paths)):
        raise DiagnosticProvenanceError("pinned roles alias a file")
    return result


def hash_file(path: Path, budget: PayloadBudget) -> dict:
    digest_, size = hashlib.sha256(), 0
    budget.check()
    with path.open("rb") as stream:
        while block := stream.read(1 << 20):
            budget.check()
            digest_.update(block)
            size += len(block)
    budget.check()
    return {"bytes": size, "sha256": digest_.hexdigest()}


def check_pins(pins: Mapping, budget: PayloadBudget) -> dict:
    """Hash every byte, including checkpoint bytes; never deserialize weights."""
    files = {}
    expected = [(pin["path"], pin["sha256"]) for pin in pins["inputs"].values()]
    expected += [
        (str(Path(row["path"]) / name), row["hashes"][name])
        for row in pins["checkpoints"]
        for name in sorted(CHECKPOINT_FILES)
    ]
    for name, sha in expected:
        got = hash_file(Path(name), budget)
        if got["sha256"] != sha:
            raise DiagnosticProvenanceError("input/checkpoint byte pin mismatch")
        files[name] = got
    return files


def _file_pin(pin):
    if (
        type(pin) is not dict
        or set(pin) != {"bytes", "sha256"}
        or type(pin["bytes"]) is not int
        or pin["bytes"] < 0
    ):
        raise DiagnosticProvenanceError("invalid file size/hash pin")
    _sha(pin["sha256"])


def _records(path, budget):
    opener = gzip.open if path.suffix == ".gz" else Path.open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            budget.check()
            if line.strip():
                row = json.loads(line)
                if type(row) is not dict:
                    raise DiagnosticProvenanceError("input row must be an object")
                yield row
    budget.check()


def side_eligibility(records):
    """Only identity contributes; never access program, length or other fields."""
    eligible, rows = set(), 0
    for row in records:
        name = row["behaviour"]
        if type(name) is not str or not name:
            raise DiagnosticProvenanceError("invalid side behaviour identity")
        eligible.add(name)
        rows += 1
    return frozenset(eligible), rows


@dataclass(frozen=True)
class SolverShown:
    """Only the two fields imported ordinary beam actually consumes."""

    inputs: tuple[str, ...]
    outputs: tuple[str, ...]


@dataclass(frozen=True)
class SelectedInput:
    position: int
    half: str
    group: str
    behaviour: str
    source_index: int
    solver: SolverShown
    product: tuple
    target: tuple


@dataclass(frozen=True)
class InputSeal:
    directory: Path
    sha256: str
    kind: str


def seal_inputs(
    spec: Mapping,
    repo: Path,
    receipts: AttemptReceipts,
    budget: PayloadBudget,
    directory: Path,
) -> InputSeal:
    """Pin -> whitelist -> select -> type -> fsync -> atomic seal, before models.

    Insufficient selection is a sealed metadata-only outcome, with no model
    load. Resource/provenance failures leave no seal and cannot be analysed.
    All files and the input seal itself charge the shared payload ledger.
    """
    receipts._active()  # Before even resolving or hashing payload paths.
    budget.check()
    if receipts.selected:
        raise DiagnosticProvenanceError("input stage after profile lock")
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(directory)
    attempt_path = receipts.root / "attempt.json"
    attempt = _read_json(attempt_path)
    if set(attempt) != {"kind", "metadata"} or attempt["kind"] != "attempt":
        raise DiagnosticProvenanceError("missing valid on-disk attempt reservation")
    attempt_pin = hash_file(attempt_path, budget)
    pins = validate_spec(spec, Path(repo))
    checked = check_pins(pins, budget)
    eligible, side_rows = side_eligibility(
        _records(Path(pins["inputs"]["side"]["path"]), budget)
    )
    tasks = [
        project_shown(row, index)
        for index, row in enumerate(
            _records(Path(pins["inputs"]["train"]["path"]), budget)
        )
    ]
    budget.check()
    try:
        panel = select_panel(tasks, eligible, size=PANEL_SIZE)
    except ValueError as error:
        if str(error) != "INSUFFICIENT_SELECTION":
            raise DiagnosticProvenanceError("invalid selection population") from error
        panel = ()
    budget.check()
    kind = "selected" if panel else "INSUFFICIENT_SELECTION"
    # Recheck all source/model bytes after streaming read, before a model can
    # consume the seal. A changed path never becomes a valid empty panel.
    if check_pins(pins, budget) != checked:
        raise DiagnosticProvenanceError("input bytes changed during projection")
    directory.mkdir(parents=True, exist_ok=False)
    selected = []
    with (directory / "shown.bin").open("xb") as stream:
        for item in panel:
            budget.check()
            product = tuple(parse_stack(text, basis=BASIS) for text in item.task.inputs)
            target = tuple(parse_stack(text, basis=BASIS) for text in item.task.outputs)
            budget.check()
            state = write_chunks(
                stream,
                beam_chunks(((product, (), 0.0), (target, (), 0.0))),
                budget,
            )
            selected.append(
                {
                    "position": item.position,
                    "half": item.half,
                    "group": item.task.group,
                    "shown": asdict(item.task),
                    "typed": state,
                }
            )
        stream.flush()
        os.fsync(stream.fileno())
    manifest = {
        "schema": INPUT_SCHEMA,
        "kind": kind,
        "attempt": attempt_pin,
        "pins": pins,
        "checked_files": checked,
        "population": {
            "train_rows": len(tasks),
            "train_behaviours": len({task.behaviour for task in tasks}),
            "original_groups": len({task.group for task in tasks}),
            "side_rows": side_rows,
            "side_behaviours": len(eligible),
        },
        "selected": selected,
    }
    with (directory / "inputs.json").open("xb") as stream:
        budget.write(stream, canonical_json(manifest))
        stream.flush()
        os.fsync(stream.fileno())
    files = {
        name: hash_file(directory / name, budget)
        for name in ("inputs.json", "shown.bin")
    }
    seal = {"schema": INPUT_SCHEMA, "kind": kind, "files": files}
    payload = canonical_json(seal)
    budget.check(len(payload))
    _publish(directory, "input-seal.json", seal)
    budget.bytes_written += len(payload)
    budget.check()
    return InputSeal(directory, hashlib.sha256(payload).hexdigest(), kind)


def _read_json(path):
    payload = path.read_bytes()

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise DiagnosticProvenanceError("duplicate stored JSON key")
            result[key] = value
        return result

    value = json.loads(payload, object_pairs_hook=pairs)
    if type(value) is not dict or canonical_json(value) != payload:
        raise DiagnosticProvenanceError("noncanonical stored JSON object")
    return value


def load_inputs(seal: InputSeal, budget: PayloadBudget) -> tuple[dict, tuple]:
    """Hash both complete streams before decoding; validate all typed pairs.

    This companion is not an independent metadata-selection oracle. The full
    after-run verifier must reconstruct selection from original pinned input.
    """
    directory = seal.directory
    budget.check()
    if hash_file(directory / "input-seal.json", budget)["sha256"] != seal.sha256:
        raise DiagnosticProvenanceError("input seal hash mismatch")
    control = _read_json(directory / "input-seal.json")
    if (
        set(control) != {"schema", "kind", "files"}
        or control["schema"] != INPUT_SCHEMA
        or control["kind"] != seal.kind
        or set(control["files"]) != {"inputs.json", "shown.bin"}
    ):
        raise DiagnosticProvenanceError("invalid input seal schema")
    for name in ("inputs.json", "shown.bin"):
        _file_pin(control["files"][name])
        if hash_file(directory / name, budget) != control["files"][name]:
            raise DiagnosticProvenanceError("sealed input stream hash mismatch")
    manifest = _read_json(directory / "inputs.json")
    if (
        set(manifest)
        != {
            "schema",
            "kind",
            "attempt",
            "pins",
            "checked_files",
            "population",
            "selected",
        }
        or manifest["schema"] != INPUT_SCHEMA
        or manifest["kind"] != seal.kind
        or seal.kind not in ("selected", "INSUFFICIENT_SELECTION")
    ):
        raise DiagnosticProvenanceError("invalid input manifest schema")
    _file_pin(manifest["attempt"])
    pins = validate_spec(manifest["pins"], Path("/"))
    if pins != manifest["pins"]:
        raise DiagnosticProvenanceError("sealed input pins must have resolved paths")
    expected_pins = {pin["path"]: pin["sha256"] for pin in pins["inputs"].values()}
    expected_pins.update(
        {
            str(Path(row["path"]) / name): row["hashes"][name]
            for row in pins["checkpoints"]
            for name in CHECKPOINT_FILES
        }
    )
    if type(manifest["checked_files"]) is not dict or set(
        manifest["checked_files"]
    ) != set(expected_pins):
        raise DiagnosticProvenanceError("sealed input check population mismatch")
    for name, sha in expected_pins.items():
        pin = manifest["checked_files"][name]
        _file_pin(pin)
        if pin["sha256"] != sha:
            raise DiagnosticProvenanceError(
                "sealed input check differs from declared pin"
            )
    population = manifest["population"]
    if (
        type(population) is not dict
        or set(population)
        != {
            "train_rows",
            "train_behaviours",
            "original_groups",
            "side_rows",
            "side_behaviours",
        }
        or any(type(n) is not int or n < 0 for n in population.values())
        or not population["original_groups"]
        <= population["train_behaviours"]
        <= population["train_rows"]
        or population["side_behaviours"] > population["side_rows"]
        or (
            seal.kind == "selected"
            and min(population["original_groups"], population["side_behaviours"])
            < PANEL_SIZE
        )
    ):
        raise DiagnosticProvenanceError("invalid input population counts")
    rows = manifest["selected"]
    if type(rows) is not list or len(rows) != (
        PANEL_SIZE if seal.kind == "selected" else 0
    ):
        raise DiagnosticProvenanceError("invalid sealed panel size")
    selected, groups, indices, offset = [], [], set(), 0
    digest_ = hashlib.sha256()
    with (directory / "shown.bin").open("rb") as stream:
        for position, row in enumerate(rows):
            budget.check()
            if set(row) != {"position", "half", "group", "shown", "typed"} or set(
                row["shown"]
            ) != {"source_index", "behaviour", "skeleton", "inputs", "outputs"}:
                raise DiagnosticProvenanceError("extra/missing selected fields")
            shown = project_shown(
                {"split": "train", **row["shown"]}, row["shown"]["source_index"]
            )
            if (
                type(row["position"]) is not int
                or row["position"] != position
                or row["half"] != HALVES[position % 2]
                or row["group"] != shown.group
                or shown.group in groups
                or shown.source_index in indices
                or shown.source_index >= population["train_rows"]
                or int(hashlib.sha256(shown.group.encode("utf-8")).hexdigest(), 16) % 10
                == 0
            ):
                raise DiagnosticProvenanceError("invalid selected identity/order")
            groups.append(shown.group)
            indices.add(shown.source_index)
            state = row["typed"]
            if (
                set(state) != {"offset", "bytes", "sha256"}
                or type(state["offset"]) is not int
                or state["offset"] != offset
                or type(state["bytes"]) is not int
                or state["bytes"] <= 0
                or state["bytes"] > control["files"]["shown.bin"]["bytes"] - offset
            ):
                raise DiagnosticProvenanceError("invalid typed shown frame")
            payload = stream.read(state["bytes"])
            budget.check()
            if (
                len(payload) != state["bytes"]
                or hashlib.sha256(payload).hexdigest() != state["sha256"]
            ):
                raise DiagnosticProvenanceError("typed shown frame hash mismatch")
            digest_.update(payload)
            offset += len(payload)
            nodes = decode_beam(payload)
            product = tuple(parse_stack(text, basis=BASIS) for text in shown.inputs)
            target = tuple(parse_stack(text, basis=BASIS) for text in shown.outputs)
            expected = ((product, (), 0.0), (target, (), 0.0))
            if nodes != expected or payload != b"".join(beam_chunks(expected)):
                raise DiagnosticProvenanceError("typed shown pairs disagree with text")
            selected.append(
                SelectedInput(
                    position,
                    row["half"],
                    shown.group,
                    shown.behaviour,
                    shown.source_index,
                    SolverShown(shown.inputs, shown.outputs),
                    product,
                    target,
                )
            )
        if (
            stream.read(1)
            or offset != control["files"]["shown.bin"]["bytes"]
            or digest_.hexdigest() != control["files"]["shown.bin"]["sha256"]
        ):
            raise DiagnosticProvenanceError("unconsumed/changed typed shown payload")
    if groups != sorted(
        groups, key=lambda group: (digest("score-gap-D1-group:", group), group)
    ):
        raise DiagnosticProvenanceError("sealed group hash order mismatch")
    if hash_file(directory / "inputs.json", budget) != control["files"]["inputs.json"]:
        raise DiagnosticProvenanceError("input metadata changed during reading")
    budget.check()
    return manifest, tuple(selected)
