"""Behavior-level data for program / input-output embeddings (v4 only).

Split whole connected components before sampling any views. Components join
explicit families and every witness skeleton, including alternative styles.
The official v4 eval stays test; additional train components supply cross-style
validation and test. This loader only reads existing data; it executes no code.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import subprocess
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from wandering_light.stack_machine import Config, format_stack
from wandering_light.stack_pbe import load, skeleton, to_stack, to_value

if TYPE_CHECKING:
    from collections.abc import Iterable

SEED = 20261024
SCHEMA_VERSION = 1
SPLIT_POLICY = "template-and-all-witness-skeleton-components-v1"


@dataclass
class BehaviorRecord:
    """One function, with aligned unique program/style lists and disjoint IO."""

    key: str
    family: str
    split: str
    programs: list[str]
    styles: list[str]
    examples: list[tuple[str, str]]
    held_out_examples: list[tuple[str, str]]
    provenance: dict[str, Any] = field(default_factory=dict)


def _digest(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _json_rows(path: Path) -> Iterable[dict]:
    # Detect gzip by magic, as stack_pbe.load does, rather than by suffix.
    with path.open("rb") as stream:
        compressed = stream.read(2) == b"\x1f\x8b"
    opener = gzip.open if compressed else open
    with opener(path, "rt") as stream:
        for line in stream:
            if line.strip():
                if path.suffix == ".out" and not line.lstrip().startswith("{"):
                    continue
                yield json.loads(line)


def _program(value: str | list | tuple) -> str:
    return " ".join(value.split() if isinstance(value, str) else value)


def _family(key: str) -> str:
    # The distinct-programs corpus varies named parameters within dc-* templates.
    # Opaque list-functions / LambdaBeam ids do not imply one shared family.
    return re.sub(r"-k\d+$", "", key)


def _new_record(key: str, family: str | None = None) -> BehaviorRecord:
    return BehaviorRecord(key, family or _family(key), "", [], [], [], [], {})


def _add_program(
    record: BehaviorRecord, program: str | list | tuple, style: str
) -> None:
    text = _program(program)
    if text and text not in record.programs:
        record.programs.append(text)
        record.styles.append(style)


def _source(record: BehaviorRecord, path: Path) -> None:
    sources = record.provenance.setdefault("sources", [])
    if str(path) not in sources:
        sources.append(str(path))


def _caveat(record: BehaviorRecord, text: str) -> None:
    caveats = record.provenance.setdefault("caveats", [])
    if text not in caveats:
        caveats.append(text)


def _add_task(
    records: dict[str, BehaviorRecord],
    row: dict,
    path: Path,
    style: str,
    *,
    official: bool = False,
    family: str | None = None,
) -> None:
    key = row["behaviour"]
    record = records.setdefault(key, _new_record(key, family))
    if family:
        record.family = family
    _source(record, path)
    _add_program(record, row["program"], style)
    for program in row.get("alternative_programs", []):
        _add_program(record, program, "alternative")
    record.examples.extend(zip(row["inputs"], row["outputs"], strict=True))
    record.held_out_examples.extend(
        zip(
            row.get("held_out_inputs", []), row.get("held_out_outputs", []), strict=True
        )
    )
    record.provenance.setdefault("original_splits", [])
    if row.get("split") not in record.provenance["original_splits"]:
        record.provenance["original_splits"].append(row.get("split"))
    if official and row.get("split") != "train":
        record.provenance["official_eval"] = True
    record.provenance.setdefault("basis", row.get("basis", "mark-28"))


def _load_sidefile(records: dict[str, BehaviorRecord], path: Path) -> None:
    if not path.exists():
        return
    for row in _json_rows(path):
        if row["behaviour"] not in records:
            continue
        flags = [str(flag).lower() for flag in row.get("flags", [])]
        if any("contrived" in flag or "precondition" in flag for flag in flags):
            continue
        # v4.jsonl is the reviewed output of build_side_file.py, which independently
        # verified both programs; a non-staged verdict is a style judgement.
        record = records[row["behaviour"]]
        _source(record, path)
        _add_program(record, row["program"], "staged")
        if row.get("refused_program"):
            _add_program(record, row["refused_program"], "refused")
        record.provenance["sidefile_verification"] = "reviewed v4 sidefile"


def _load_independent(records: dict[str, BehaviorRecord], root: Path) -> None:
    """Read published checks and test labels without regenerating task exports."""
    base = root / "reports/distinct-programs-20260926"
    behavior_path, audit_path = base / "behaviours.jsonl", base / "audit.jsonl"
    if not behavior_path.exists() or not audit_path.exists():
        return
    kept = {row["id"] for row in _json_rows(audit_path) if row["verdict"] == "keep"}
    behaviors = {row["id"]: row for row in _json_rows(behavior_path)}
    for glob, check_name, style in (
        ("programs_w*.jsonl", "verdicts.jsonl", "sol"),
        ("programs_o*.jsonl", "opus_check.out", "opus"),
    ):
        check_path = base / check_name
        if not check_path.exists():
            continue
        checked = {
            row["id"] for row in _json_rows(check_path) if row.get("verified") is True
        }
        for path in sorted(base.glob(glob)):
            for row in _json_rows(path):
                key = row["id"]
                if key not in kept or key not in checked or not row.get("program"):
                    continue
                behavior = behaviors[key]
                record = records.setdefault(key, _new_record(key))
                _add_program(record, row["program"], style)
                for source in (behavior_path, audit_path, check_path, path):
                    _source(record, source)
                _caveat(
                    record,
                    "Independent verification report is keyed by behavior id, "
                    "without a program digest; programs were not re-executed.",
                )
                # Preserve the source's shown/hidden partition. Never execute its
                # reference solution or label examples by executing the witness.
                tests = behavior["tests"]
                shown = behavior.get("shown", list(range(min(6, len(tests)))))
                hidden = behavior.get(
                    "held_out", list(range(min(6, len(tests)), len(tests)))
                )
                for indices, target in (
                    (shown, record.examples),
                    (hidden, record.held_out_examples),
                ):
                    for index in indices:
                        test = tests[index]
                        target.append(
                            (
                                format_stack(Config(to_stack(tuple(test["args"])))),
                                format_stack(Config((to_value(test["out"]),))),
                            )
                        )


def _load_extra(records: dict[str, BehaviorRecord], path: Path) -> None:
    """Accept Task JSONL(.gz), or peer-scale accepted.jsonl containing task."""
    for row in _json_rows(path):
        family, template = None, None
        if "task" in row:
            if row.get("status", "accepted") != "accepted":
                continue
            candidate = row.get("candidate", {})
            family = candidate.get("family")
            template = candidate.get("template_id")
            task = row["task"]
        else:
            task = row
        # Peer-scale exports carry family first, template second in Task.tags.
        if family is None and len(task.get("tags", [])) == 2:
            family = task["tags"][0]
            template = task["tags"][1]
        _add_task(records, task, path, "peer", family=family)
        record = records[task["behaviour"]]
        if template:
            record.provenance["split_group"] = template
        record.provenance["peer_snapshot"] = True
        _caveat(
            record,
            "Preliminary peer-scale snapshot; verified on its own examples, not a novelty certificate.",
        )
        if row.get("signature"):
            record.provenance["upstream_signature"] = row["signature"]
        if "provenance" in row:
            record.provenance["upstream"] = row["provenance"]


def _merge_exact_programs(records: list[BehaviorRecord]) -> list[BehaviorRecord]:
    """One semantic label for exact duplicate programs, even across source ids."""
    owners: dict[str, BehaviorRecord] = {}
    redirects: dict[str, BehaviorRecord] = {}

    def resolve(record: BehaviorRecord) -> BehaviorRecord:
        while record.key in redirects:
            record = redirects[record.key]
        return record

    for record in records:
        matches = {
            resolve(owners[p]).key: resolve(owners[p])
            for p in record.programs
            if p in owners
        }
        candidates = [record, *matches.values()]
        # Keep official eval behavior names; prefer canonical v4 names next.
        destination = min(
            candidates,
            key=lambda r: (
                not r.provenance.get("official_eval", False),
                "canonical" not in r.styles,
                r.key,
            ),
        )
        for source in candidates:
            if source is destination:
                continue
            redirects[source.key] = destination
            destination.examples.extend(source.examples)
            destination.held_out_examples.extend(source.held_out_examples)
            for program, style in zip(source.programs, source.styles, strict=True):
                _add_program(destination, program, style)
            for path in source.provenance["sources"]:
                _source(destination, Path(path))
            aliases = destination.provenance.setdefault("aliases", [])
            aliases.extend([source.key, *source.provenance.get("aliases", [])])
            groups = destination.provenance.setdefault("merged_split_groups", [])
            groups.extend(
                [
                    source.provenance.get("split_group", source.family),
                    *source.provenance.get("merged_split_groups", []),
                ]
            )
            if source.provenance.get("official_eval"):
                destination.provenance["official_eval"] = True
            for caveat in source.provenance.get("caveats", []):
                _caveat(destination, caveat)
        for program in destination.programs:
            owners[program] = destination
    return [record for record in records if record.key not in redirects]


def _deduplicate(record: BehaviorRecord) -> None:
    """Hidden inputs take precedence over shown inputs across all task draws."""
    hidden, visible = {}, {}
    conflicts = set()
    for pairs, destination in (
        (record.held_out_examples, hidden),
        (record.examples, visible),
    ):
        for raw_input, raw_output in pairs:
            # Canonical source serializers already normalize stack text. Whitespace
            # normalization catches harmless formatting differences in peer files.
            given, want = " ".join(raw_input.split()), " ".join(raw_output.split())
            if given in destination and destination[given] != want:
                conflicts.add(given)
            destination[given] = want
    conflicts.update(
        given
        for given in hidden
        if given in visible and hidden[given] != visible[given]
    )
    overlap = hidden.keys() & visible.keys()
    record.provenance["shown_hidden_input_overlap_removed"] = len(overlap)
    if conflicts:
        _caveat(record, f"Conflicting outputs: excluded {len(conflicts)} input stacks.")
    record.held_out_examples = sorted(
        (given, want) for given, want in hidden.items() if given not in conflicts
    )
    record.examples = sorted(
        (given, want)
        for given, want in visible.items()
        if given not in hidden and given not in conflicts
    )
    if len(record.examples) < 4:
        _caveat(
            record,
            "Fewer than four distinct visible examples after hidden-input exclusion.",
        )


def _assign_splits(records: list[BehaviorRecord], seed: int = SEED) -> None:
    parent = {record.key: record.key for record in records}

    def find(key: str) -> str:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    owner: dict[tuple[str, str], str] = {}
    for record in records:
        features = [
            ("group", group)
            for group in (
                record.provenance.get("split_group", record.family),
                *record.provenance.get("merged_split_groups", []),
            )
        ] + [("skeleton", skeleton(program.split())) for program in record.programs]
        for feature in features:
            if feature in owner:
                left, right = find(record.key), find(owner[feature])
                parent[max(left, right)] = min(left, right)
            owner[feature] = record.key
    groups: dict[str, list[BehaviorRecord]] = defaultdict(list)
    for record in records:
        groups[find(record.key)].append(record)
    for group in groups.values():
        component = _digest("\n".join(sorted(r.key for r in group)).encode())[7:23]
        official = any(r.provenance.get("official_eval") for r in group)
        bucket = (
            int(hashlib.sha256(f"{seed}:{component}".encode()).hexdigest()[:8], 16)
            % 100
        )
        split = (
            "test"
            if official or bucket < 10
            else "validation"
            if bucket < 20
            else "train"
        )
        for record in group:
            record.split = split
            record.provenance.update(
                component=component,
                component_size=len(group),
                split_policy=SPLIT_POLICY,
                seed=seed,
                partition=(
                    "official_eval"
                    if record.provenance.get("official_eval")
                    else "official_eval_overlap"
                    if official
                    else "cross_style_holdout"
                    if split != "train"
                    else "fit"
                ),
            )
            if official and not record.provenance.get("official_eval"):
                _caveat(
                    record,
                    "Held out because template or witness skeleton connects to official v4 eval.",
                )


def load_records(
    repo_root: str | Path,
    extra_task_paths: Iterable[str | Path] = (),
) -> list[BehaviorRecord]:
    """Load local v4 + reviewed styles + independent teachers + optional exports.

    No downloads, generated programs, reference execution, or model calls occur.
    Manifest/data mismatches are reported as provenance caveats, never rejected.
    """
    root = Path(repo_root).resolve()
    corpus = root / "wandering_light/evals/data/stack_pbe_v4"
    manifest_path = corpus / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    records: dict[str, BehaviorRecord] = {}
    mismatches: list[str] = []
    for split in sorted(manifest["splits"]):
        info = manifest["splits"][split]
        path = corpus / info["path"]
        if info.get("sha256") and _digest(path.read_bytes()) != info["sha256"]:
            mismatches.append(f"Manifest digest mismatch: {path}")
        for task in load(path, verify="off"):
            _add_task(records, asdict(task), path, "canonical", official=True)
    _load_sidefile(records, root / "experiments/staged_witnesses/v4.jsonl")
    _load_independent(records, root)
    # Existing task exports are optional; raw independent witness/check artifacts
    # above remain useful even when the gitignored task exports are absent.
    for name, style in (("tasks.jsonl", "sol"), ("tasks_opus.jsonl", "opus")):
        path = root / "reports/distinct-programs-20260926" / name
        if path.exists():
            for task in load(path, verify="off"):
                _add_task(records, asdict(task), path, style)
    for value in extra_task_paths:
        _load_extra(records, Path(value).resolve())
    result = _merge_exact_programs(
        sorted(records.values(), key=lambda record: record.key)
    )
    for record in result:
        _deduplicate(record)
        for caveat in mismatches:
            _caveat(record, caveat)
    _assign_splits(result)
    paths = {path for record in result for path in record.provenance["sources"]}
    digests = {path: _digest(Path(path).read_bytes()) for path in sorted(paths)}
    digests[str(manifest_path)] = _digest(manifest_path.read_bytes())
    for record in result:
        record.provenance["source_digests"] = {
            path: digests[path]
            for path in [*record.provenance["sources"], str(manifest_path)]
        }
    return result


def manifest_for(records: list[BehaviorRecord], repo_root: str | Path) -> dict:
    """Compact aggregate provenance for the exact prepared dataset."""
    root = Path(repo_root)
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    dirty = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=False,
    )
    payload = "".join(json.dumps(asdict(r), sort_keys=True) + "\n" for r in records)
    components = {
        r.provenance["component"]: r.provenance["component_size"] for r in records
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "git_commit": commit.stdout.strip() or None,
        "dirty": bool(dirty.stdout.strip()),
        "wandb_run.url": None,
        "seed": SEED,
        "config": {
            "split_policy": SPLIT_POLICY,
            "validation_share": 0.1,
            "test_share": 0.1,
        },
        "records_digest": _digest(payload.encode()),
        "counts": dict(Counter(r.split for r in records)),
        "partitions": dict(Counter(r.provenance["partition"] for r in records)),
        "components": {
            "count": len(components),
            "largest": max(components.values(), default=0),
        },
        "components_by_split": {
            split: len({r.provenance["component"] for r in records if r.split == split})
            for split in ("train", "validation", "test")
        },
        "cross_style_by_split": dict(
            Counter(r.split for r in records if len(r.programs) > 1)
        ),
        "source_data_digests": {
            path: digest
            for record in records
            for path, digest in record.provenance["source_digests"].items()
        },
        "caveats": sorted(
            {c for r in records for c in r.provenance.get("caveats", [])}
        ),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--extra-tasks", type=Path, action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    records = load_records(args.repo_root, args.extra_tasks)
    args.out.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(asdict(r), sort_keys=True) + "\n" for r in records)
    (args.out / "records.jsonl").write_text(payload)
    manifest = manifest_for(records, args.repo_root)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: manifest[key]
                for key in (
                    "counts",
                    "partitions",
                    "components",
                    "cross_style_by_split",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
