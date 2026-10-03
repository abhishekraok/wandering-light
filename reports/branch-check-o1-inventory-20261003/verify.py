"""Independent O1-B manifest/group/positive checks; no failed replay negatives."""

import argparse
import gzip
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source-tree", type=Path, required=True)
parser.add_argument("--run-dir", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
TREE = args.source_tree.resolve()
RUN = args.run_dir.resolve()
sys.path.insert(0, str(TREE))

from experiments.staged_cycles import closed_product  # noqa: E402
from wandering_light.stack_beam import closed_cuts, key_of, step_all_quiet  # noqa: E402
from wandering_light.stack_machine import parse_stack  # noqa: E402
from wandering_light.stack_pbe import (  # noqa: E402
    BASIS,
    BY_NAME,
    OP_BY_NAME,
    assign_split,
)


def canonical(value):
    return json.loads(json.dumps(value))


def execute(start, program):
    states = [start]
    for op in program.split():
        next_state, _, _ = step_all_quiet(states[-1], OP_BY_NAME[op])
        assert next_state is not None, f"verified positive refused: {op}"
        states.append(next_state)
    return states


def main():
    assert (RUN / "inventory-run.log").read_text().splitlines()[-1] == "exit 0"
    raw = (RUN / "inventory.json").read_bytes()
    data = json.loads(raw)
    source = Path(data["input"])
    sha = hashlib.sha256()
    with source.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            sha.update(chunk)
    assert (
        sha.hexdigest()
        == data["sha256"]
        == "e358fca13e69880a48f0b3794d36769b1815f3dd249c765bc08b52fc7c1fca78"
    )
    assert data["code_revision"] == "c664a7cd4dc9397e4960555ecf01d5c7ce473bf1"
    assert not data["errors"]
    tasks = {(task["behaviour"], task["example_set"]): task for task in data["tasks"]}
    zero_rows, names, adjacency = [], set(), defaultdict(set)
    skeleton_names, witness_names = defaultdict(set), defaultdict(set)
    positive_checks = []
    complete_matrix_candidates = 0
    with gzip.open(source, "rt") as stream:
        for line in stream:
            row = json.loads(line)
            name = row["behaviour"]
            assert (
                row["split"] == "train"
                and assign_split(BY_NAME[name].skeleton) == "train"
            )
            names.add(name)
            skeleton_names[BY_NAME[name].skeleton].add(name)
            witness_names[tuple(row["witness"].split())].add(name)
            if row["example_set"] == 0:
                zero_rows.append(name)
            key = (name, row["example_set"])
            if key not in tasks:
                continue
            task = tasks[key]
            pool = {}
            for segment in row["segments"]:
                if segment["stop"] is None:
                    for record in segment["states"]:
                        pool[segment["segment"], record["program"]] = record

            def record_order(key, name=name, example_set=row["example_set"]):
                text = f"o1b-v1|{name}|{example_set}|{key[0]}|{key[1]}"
                return hashlib.sha256(text.encode()).hexdigest(), text

            expected = sorted(pool, key=record_order)[:32]
            assert expected == [
                (r["segment"], r["program"]) for r in task["selected_records"]
            ]
            for candidate in task["candidates"]:
                if candidate["exclusions"]:
                    continue
                if task["status"] == "complete":
                    assert [trial["program"] for trial in candidate["trials"]] == task[
                        "catalogue"
                    ]["programs"]
                    assert all(
                        trial["status"] in ("complete", "refused")
                        for trial in candidate["trials"]
                    )
                    minima = [
                        len(trial["program"])
                        for trial in candidate["trials"]
                        if trial["success"]
                    ]
                    assert candidate["label"] == (min(minima) if minima else None)
                    complete_matrix_candidates += 1
                successes = [trial for trial in candidate["trials"] if trial["success"]]
                if not successes:
                    continue
                start = tuple(parse_stack(text, basis=BASIS) for text in row["inputs"])
                goal = tuple(parse_stack(text, basis=BASIS) for text in row["outputs"])
                trace = execute(start, candidate["program"])
                product = trace[-1]
                assert closed_product(product) and product != goal
                assert len(candidate["program"].split()) in closed_cuts(
                    trace, candidate["program"].split()
                )
                assert canonical(key_of(product)) == candidate["product"]
                witness_states = execute(start, row["witness"])
                assert (product in witness_states) == candidate["on_witness"]
                expanded = set()
                for segment in row["segments"]:
                    if segment["segment"] > candidate["segment"]:
                        break
                    for root in segment["roots"]:
                        expanded.add(execute(start, root["program"])[-1])
                assert product not in expanded
                for trial in successes:
                    assert execute(product, " ".join(trial["program"]))[-1] == goal
                positive_checks.append(
                    {
                        "behaviour": name,
                        "digest": candidate["digest"],
                        "task_status": task["status"],
                        "successful_sequences": len(successes),
                        "label": candidate["label"],
                        "on_witness": candidate["on_witness"],
                    }
                )
    expected_manifest = [
        [name, 0]
        for name in sorted(
            zero_rows,
            key=lambda name: (
                hashlib.sha256(f"o1b-v1|{name}".encode()).hexdigest(),
                name,
            ),
        )[:64]
    ]
    assert expected_manifest == data["manifest"]
    assert data["manifest"][: len(tasks)] == [
        [t["behaviour"], t["example_set"]] for t in data["tasks"]
    ]
    assert data["manifest"][len(tasks) :] == data["unprocessed"]
    for collection in (skeleton_names, witness_names):
        for members in collection.values():
            first = min(members)
            for name in members:
                adjacency[first].add(name)
                adjacency[name].add(first)
    independently_grouped = {}
    while names:
        pending, component = [min(names)], set()
        while pending:
            name = pending.pop()
            if name not in component:
                component.add(name)
                pending.extend(adjacency[name] - component)
        for name in component:
            independently_grouped[name] = min(component)
        names -= component
    assert independently_grouped == data["group_map"]
    # Directly enumerate pairs without calling the measured builder's pair helper.
    independent_pairs, pair_groups = 0, set()
    for task in data["tasks"]:
        if task["status"] != "complete":
            assert task["pairs"] == []
            continue
        seen = set()
        candidates = task["candidates"]
        for index, a in enumerate(candidates):
            for b in candidates[index + 1 :]:
                if (
                    a["segment"] != b["segment"]
                    or a["prefix_ops"] != b["prefix_ops"]
                    or a["label"] is None
                    or b["label"] is None
                    or a["label"] == b["label"]
                    or (a["on_witness"] and b["on_witness"])
                ):
                    continue
                pair = tuple(
                    sorted((json.dumps(a["product"]), json.dumps(b["product"])))
                )
                assert pair[0] != pair[1]
                seen.add(pair)
        assert len(seen) == len(task["pairs"])
        independent_pairs += len(seen)
        if seen:
            pair_groups.add(independently_grouped[task["behaviour"]])
    assert independent_pairs == data["gate"]["pairs"] == 0
    assert sorted(pair_groups) == data["gate"]["groups"] == []
    summary = {
        "source_sha256": data["sha256"],
        "raw_output_sha256": hashlib.sha256(raw).hexdigest(),
        "code_revision": data["code_revision"],
        "verification": "PASS",
        "manifest_keys": len(data["manifest"]),
        "processed": len(tasks),
        "unprocessed": len(data["unprocessed"]),
        "complete": data["complete"],
        "stop": data["stop"],
        "statuses": dict(Counter(t["status"] for t in data["tasks"])),
        "censoring": [
            {"phase": phase, "reason": reason, "tasks": count}
            for (phase, reason), count in sorted(
                Counter(
                    (t["phase"], t["reason"])
                    for t in data["tasks"]
                    if t["status"] == "censored"
                ).items()
            )
        ],
        "counters": data["counters"],
        "elapsed_internal_seconds": data["elapsed_seconds"],
        "gate": data["gate"],
        "errors": data["errors"],
        "completed_eligible_candidates": complete_matrix_candidates,
        "positive_candidates_checked": len(positive_checks),
        "positive_sequences_checked": sum(
            p["successful_sequences"] for p in positive_checks
        ),
        "completed_labelled_candidates": sum(
            p["task_status"] == "complete" for p in positive_checks
        ),
        "positives": positive_checks,
        "tasks": [
            {
                "behaviour": t["behaviour"],
                "example_set": t["example_set"],
                "status": t["status"],
                "phase": t.get("phase"),
                "reason": t.get("reason"),
                "selected_records": len(t["selected_records"]),
                "counters": t["counters"],
                "closed_eligible_replayed": sum(
                    not c["exclusions"] for c in t["candidates"]
                ),
                "labelled_candidates": sum(
                    c["label"] is not None for c in t["candidates"]
                ),
                "pairs": len(t["pairs"]),
            }
            for t in data["tasks"]
        ],
        "unprocessed_keys": data["unprocessed"],
    }
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        {
            key: summary[key]
            for key in (
                "verification",
                "statuses",
                "completed_eligible_candidates",
                "positive_candidates_checked",
                "positive_sequences_checked",
                "gate",
            )
        }
    )


if __name__ == "__main__":
    main()
