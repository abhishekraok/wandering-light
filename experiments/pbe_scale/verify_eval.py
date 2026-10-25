"""Rescore frozen v4 solve files with the existing 10,000-draw verifier.

Executes generated stack programs: use MemoryMax=12G and MemorySwapMax=0.
Catalogue-equal programs inherit corpus verification and receive no new draws.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments import best_system_cells as cells
from experiments import step3_verify_solves as verifier
from experiments.pbe_scale import eval_compare as compare
from wandering_light import stack_machine, stack_pbe
from wandering_light.stack_pbe import catalogue_digest, catalogue_source_digest

KINDS = ("genuine", "equals_catalogue_witness", "shortcut", "inconclusive")
ACCEPTED = frozenset(KINDS[:2])


def pair_key(row):
    return f"{row['behaviour']} :: {' '.join(row['program'])}"


def verify(run: Path, old_path: Path, mixed_path: Path, output: Path) -> dict:
    frozen_dir = run / "eval-v4"
    frozen = compare.read(frozen_dir / "input_manifest.json")
    expected = set(compare.task_keys(frozen_dir / "tasks.jsonl.gz"))
    runs = {
        label: compare.read(path)
        for label, path in (("old", old_path), ("mixed", mixed_path))
    }
    for record in runs.values():
        keys = [(row["index"], row["behaviour"]) for row in record["rows"]]
        if (
            len(keys) != frozen["tasks"]
            or len(keys) != len(set(keys))
            or set(keys) != expected
        ):
            raise ValueError("incomplete or unpaired frozen v4 evaluation rows")
        if any(type(row["solved"]) is not bool for row in record["rows"]):
            raise ValueError("rows require explicit boolean held-out grades")
        if any(row["solved"] and not row["program"] for row in record["rows"]):
            raise ValueError("a graded solution is missing its program")
    if {name for _, name in expected} - {b.name for b in cells.CATALOGUE}:
        raise ValueError(
            "unresolved frozen v4 references; cannot score missing verification as zero"
        )
    paths = {
        "old": old_path,
        "mixed": mixed_path,
        "payload": frozen_dir / "tasks.jsonl.gz",
        "frozen_manifest": frozen_dir / "input_manifest.json",
    }
    sources = {}
    for label, record in runs.items():
        checkpoint = Path(record["checkpoint"])
        manifest = compare.read(checkpoint / "training_manifest.json")
        sources[label] = {
            "checkpoint": str(checkpoint),
            "model_sha256": compare.sha(checkpoint / "model.safetensors"),
            "trainer_state": compare.read(checkpoint / "trainer_state.json"),
            "training_manifest": manifest,
            "policy_config": compare.read(checkpoint / "policy_config.json"),
            "search": record["system"],
            "host": record["host"],
            "solver_commit": record["commit"],
            "wandb_run.url": manifest.get("wandb_run.url", manifest.get("wandb_url")),
        }
    origin = {
        **compare.provenance(),
        "source_sha256": {
            str(Path(module.__file__).resolve()): compare.sha(module.__file__)
            for module in (cells, verifier, compare, stack_machine, stack_pbe)
        },
        "adapter_sha256": compare.sha(Path(__file__)),
        "inputs": {
            name: {"path": str(path), "sha256": compare.sha(path)}
            for name, path in paths.items()
        },
        "frozen": frozen,
        "checkpoints": sources,
        "catalogue_digest": catalogue_digest(),
        "catalogue_source_digest": catalogue_source_digest(),
        "protocol": {
            "draws_per_mode": verifier.DRAWS,
            "seed": 134,
            "modes": ["sampler", "perturbed", "dense"],
            "min_tested_per_mode": verifier.MIN_TESTED,
            "catalogue_equal_new_draws": 0,
        },
    }
    output.mkdir(parents=True, exist_ok=False)
    compare.write(output / "input_manifest.json", origin)
    done = 0

    def progress(_message):
        nonlocal done
        done += 1
        print(f"verified {done} distinct program pairs", flush=True)

    verdicts = cells.verified_programs(output, list(runs.values()), progress)
    if not (output / "verified.json").exists():
        compare.write(output / "verified.json", verdicts)
    counts, accepted_tasks, accepted_behaviors = {}, {}, {}
    for label, record in runs.items():
        graded = [row for row in record["rows"] if row["solved"]]
        # Missing or unknown pair verdicts are incomplete, never implicit failures.
        for row in graded:
            if verdicts[pair_key(row)]["verdict"] not in KINDS:
                raise ValueError("unknown verification verdict")
        accepted = [
            row for row in graded if verdicts[pair_key(row)]["verdict"] in ACCEPTED
        ]
        accepted_tasks[label] = {(row["index"], row["behaviour"]) for row in accepted}
        accepted_behaviors[label] = {row["behaviour"] for row in accepted}
        counts[label] = {
            "tasks_attempted": len(record["rows"]),
            "graded_tasks": len(graded),
            "graded_behaviors": len({row["behaviour"] for row in graded}),
            "accepted_tasks": len(accepted),
            "accepted_behaviors": len(accepted_behaviors[label]),
            "by_verdict": {
                kind: {
                    "tasks": sum(
                        verdicts[pair_key(row)]["verdict"] == kind for row in graded
                    ),
                    "behaviors": len(
                        {
                            row["behaviour"]
                            for row in graded
                            if verdicts[pair_key(row)]["verdict"] == kind
                        }
                    ),
                }
                for kind in KINDS
            },
        }
    result = {
        "status": "complete",
        "counts": counts,
        "task_gain": len(accepted_tasks["mixed"] - accepted_tasks["old"]),
        "task_loss": len(accepted_tasks["old"] - accepted_tasks["mixed"]),
        "task_delta": len(accepted_tasks["mixed"]) - len(accepted_tasks["old"]),
        "behavior_gain": len(accepted_behaviors["mixed"] - accepted_behaviors["old"]),
        "behavior_loss": len(accepted_behaviors["old"] - accepted_behaviors["mixed"]),
        "behavior_delta": len(accepted_behaviors["mixed"])
        - len(accepted_behaviors["old"]),
        "distinct_program_pairs": len(verdicts),
        "verdicts_sha256": compare.sha(output / "verified.json"),
        "provenance": origin,
        "caveats": [
            "Accepted = draw-tested genuine or catalogue-equal; catalogue-equal programs receive zero fresh draws.",
            "Non-witness programs receive 10,000 draw attempts per mode, on the canonical witness's operating domain; this is a finite test, not a proof. Actual tested/skipped/failed counts are in verified.json.",
            "Verdict behavior counts can overlap; one behavior may have several solution programs.",
            "One training seed; fixed epochs do not match training compute. Verification does not tune search or train a model.",
        ],
    }
    if origin["inputs"]["payload"]["sha256"] != frozen["sha256"]:
        result["caveats"].append(
            "Frozen payload digest differs from its recorded digest."
        )
    compare.write(output / "summary.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--mixed", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.run_dir, args.old, args.mixed, args.output_dir)
    print(
        json.dumps(
            {key: result[key] for key in ("counts", "task_delta", "behavior_delta")}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
