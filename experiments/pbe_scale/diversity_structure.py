"""Describe saved witness structures without executing programs or references."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

from experiments.pbe_scale import generate, generate_families

PIPELINE_FAMILIES = frozenset(
    {
        "series-pipeline",
        "baseline-pipeline",
        "paired-series-pipeline",
        "original-comparison",
        "conditional-summary",
    }
)
ALGORITHM_FAMILIES = frozenset(a.family for a in generate_families.ALGORITHMS)
CATEGORIES = {s.name: s.category for s in generate.STAGES}
INTEGER = re.compile(r"^-?\d+$")


SYMMETRIC_READOUTS = frozenset(
    {"total", "count", "energy", "absolute-total", "maximum", "minimum", "span", "mean"}
)
MULTISET_CONTINUATIONS = frozenset(
    {"pointwise", "selection", "multiplicity", "ordering", "global"}
)
PIPELINE_CONTEXTS = {
    "series-pipeline": {"input"},
    "baseline-pipeline": {s.name for s in generate.STARTS if s.category == "list_int"},
    "paired-series-pipeline": {
        s.name for s in generate.STARTS if s.category == "two_lists"
    },
    "original-comparison": {
        "total-difference",
        "sample-weighted",
        "level-versus-change",
    },
    "conditional-summary": {"direction-by-total"},
}


def redundant_stages(
    stages: list[str], readout: str, *, family: str, start: str
) -> dict[str, list[int]]:
    """Sufficient output-value identities within known scalar pipeline contexts.

    Starts feed a list to the pipeline. The known wrappers only select its
    source or combine scalar summaries, never align its output with raw input.
    This concerns values, not fuel costs or duplicate functions.
    """
    flags = {"ordering": [], "absolute": []}
    categories = [CATEGORIES.get(stage) for stage in stages]
    known_categories = MULTISET_CONTINUATIONS | {"history", "neighborhood"}
    if (
        start not in PIPELINE_CONTEXTS.get(family, set())
        or readout not in SYMMETRIC_READOUTS
        or any(category not in known_categories for category in categories)
    ):
        return flags
    flags["ordering"] = [
        i
        for i, category in enumerate(categories)
        if category == "ordering"
        and all(c in MULTISET_CONTINUATIONS for c in categories[i + 1 :])
    ]
    if stages and stages[-1] == "absolute" and readout in {"energy", "absolute-total"}:
        flags["absolute"] = [len(stages) - 1]
    return flags


def source_class(candidate: dict) -> str:
    if candidate["family"] in PIPELINE_FAMILIES:
        return "pipeline"
    if candidate["family"] in ALGORITHM_FAMILIES and "algorithm" in candidate.get(
        "parameters", {}
    ):
        return "explicit_algorithm"
    return "other_authored_or_anchor"


def quantiles(values: list) -> dict:
    ordered = sorted(values)
    return {
        str(q): ordered[int((len(ordered) - 1) * q)]
        for q in (0, 0.25, 0.5, 0.75, 0.9, 0.99, 1)
    }


def summarize(accepted: Path, exclusions: Path) -> dict:
    excluded = set(json.loads(exclusions.read_text())["excluded_ids"])
    counts = {
        name: Counter()
        for name in (
            "source_class",
            "input_kind",
            "pipeline_depth",
            "pipeline_core",
            "pipeline_bag",
            "algorithm",
            "redundant_ordering",
            "redundancy_overlap",
        )
    }
    lengths = defaultdict(list)
    family_redundancy = defaultdict(Counter)
    programs, normalized, simplified = set(), set(), set()
    examples, excluded_rows, total, redundant_rows = [], 0, 0, 0
    data_hash = hashlib.sha256()
    for line in accepted.open("rb"):
        data_hash.update(line)
        row = json.loads(line)
        candidate, task = row["candidate"], row["task"]
        if candidate["id"] in excluded:
            excluded_rows += 1
            continue
        total += 1
        kind, program = candidate["input_kind"], candidate["program"]
        source = source_class(candidate)
        counts["source_class"][source] += 1
        counts["input_kind"][kind] += 1
        tokens = program.split()
        programs.add((kind, program))
        normalized.add(
            (kind, " ".join("INT" if INTEGER.fullmatch(t) else t for t in tokens))
        )
        lengths[source].append((task["length"], len(tokens)))
        if source == "explicit_algorithm":
            counts["algorithm"][candidate["parameters"]["algorithm"]] += 1
        if source != "pipeline":
            continue
        start, *stages, readout = candidate["template_id"].split("/")
        counts["pipeline_depth"][len(stages)] += 1
        counts["pipeline_core"]["/".join(stages)] += 1
        counts["pipeline_bag"]["/".join(sorted(stages))] += 1
        flags = redundant_stages(
            stages, readout, family=candidate["family"], start=start
        )
        ordering, absolute = bool(flags["ordering"]), bool(flags["absolute"])
        family_redundancy[candidate["family"]].update(
            {
                "ordering": int(ordering),
                "absolute": int(absolute),
                "union": int(ordering or absolute),
            }
        )
        overlap = (
            "both"
            if ordering and absolute
            else "ordering_only"
            if ordering
            else "absolute_only"
            if absolute
            else "neither"
        )
        counts["redundancy_overlap"][overlap] += 1
        counts["redundant_ordering"].update(stages[i] for i in flags["ordering"])
        removed = set(flags["ordering"] + flags["absolute"])
        simplified.add(
            (
                kind,
                start,
                *(s for i, s in enumerate(stages) if i not in removed),
                readout,
            )
        )
        if removed:
            redundant_rows += 1
            if len(examples) < 3:
                examples.append(
                    {
                        "id": candidate["id"],
                        "template": candidate["template_id"],
                        "expanded_length": task["length"],
                        "redundant_indices": sorted(removed),
                    }
                )
    compact = {}
    for key in ("pipeline_core", "pipeline_bag"):
        counter = counts.pop(key)
        compact[key] = {
            "unique": len(counter),
            "largest_group": max(counter.values(), default=0),
            "top10": counter.most_common(10),
        }
    repo = Path(generate.__file__).resolve().parents[2]

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(repo), *args], capture_output=True, text=True, check=False
        ).stdout.strip()

    source_paths = [
        Path(__file__),
        Path(generate.__file__),
        Path(generate_families.__file__),
    ]
    return {
        "kept_rows": total,
        "excluded_rows": excluded_rows,
        "declared_vocabulary": {
            "stages": len(generate.STAGES),
            "starts": len(generate.STARTS),
            "readouts": len(generate.READOUTS),
            "algorithms": len(generate_families.ALGORITHMS),
        },
        "counts": counts,
        "composition_groups": compact,
        "kind_macro_unique": len(programs),
        "kind_macro_integer_normalized_unique": len(normalized),
        "redundant_witness_rows": redundant_rows,
        "redundancy_rows_by_family": family_redundancy,
        "redundancy_assumptions": {
            "readouts": sorted(SYMMETRIC_READOUTS),
            "multiset_continuation_categories": sorted(MULTISET_CONTINUATIONS),
            "contexts": {
                family: sorted(starts) for family, starts in PIPELINE_CONTEXTS.items()
            },
            "scope": "Known stage semantics and scalar wrappers from generate.py; output-value identities, not fuel equivalence or duplicate functions.",
            "ordering": "Downstream stages preserve multiset dependence; the readout is invariant to permutations.",
            "absolute": "Squaring and absolute magnitude ignore a final absolute-value map.",
        },
        "redundancy_examples": examples,
        "pipeline_simplified_template_unique": len(simplified),
        "source_lengths": {
            key: {
                "rows": len(values),
                "expanded_tokens": quantiles([a for a, b in values]),
                "macro_tokens": quantiles([b for a, b in values]),
                "expansion_ratio": quantiles([a / b for a, b in values]),
            }
            for key, values in lengths.items()
        },
        "execution_performed": False,
        "limitations": [
            "Composition groups measure concentration, not semantic duplicates; order, starts and readouts can change behavior.",
            "Redundancy checks prove removable operations, not duplicate functions or minimal witnesses; checks are incomplete.",
            "Integer normalization is lexical; constants encoded by other macros are not normalized.",
        ],
        "provenance": {
            "seed": None,
            "git_commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain")),
            "config": {
                "deterministic": True,
                "pipeline_families": sorted(PIPELINE_FAMILIES),
                "algorithm_families": sorted(ALGORITHM_FAMILIES),
                "example_limit": 3,
            },
            "source_data_digests": {
                str(accepted.resolve()): data_hash.hexdigest(),
                str(exclusions.resolve()): hashlib.sha256(
                    exclusions.read_bytes()
                ).hexdigest(),
            },
            "source_file_digests": {
                str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in source_paths
            },
            "wandb_run.url": None,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accepted", type=Path, required=True)
    parser.add_argument("--exclusions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.accepted, args.exclusions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "kept_rows": result["kept_rows"],
                "redundant_witness_rows": result["redundant_witness_rows"],
                "output": str(args.output.resolve()),
            }
        )
    )


if __name__ == "__main__":
    main()
