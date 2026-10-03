"""Independent complete-result verifier; no checkpoint deserialization or rerun.

Reject unsuccessful/incomplete terminals before reading raw logits. Reconstruct
selection/replay/support via verify_stage and metrics with scalar arithmetic,
not with any experiments.capped_glue_* implementation. Real CLI is bounded and
requires the expected reviewed revision explicitly.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import runpy
import sys
import time
from collections import defaultdict
from dataclasses import fields
from pathlib import Path
from statistics import median
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from wandering_light.stack_pbe import OPS, Task  # noqa: E402 - direct-script root above

stage_check = SimpleNamespace(
    **runpy.run_path(str(Path(__file__).with_name("verify_stage.py")))
)
ORDER = [(name, seed) for name in ("machine", "difference") for seed in range(3)]
NAMES = [op.name for op in OPS]
METRICS = (
    "unmasked_nll",
    "legal_nll",
    "negative_log_legal_mass",
    "best_legal_deficit",
    "legal_rank",
    "legal_size",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_time(deadline):
    require(deadline is None or time.perf_counter() < deadline, "verification deadline")


def digest(path, deadline=None):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            check_time(deadline)
            result.update(block)
    check_time(deadline)
    return result.hexdigest()


def compare(actual, expected, context="result"):
    """Exact structure/identities; scalar floating tolerance 1e-10 absolute/relative."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), context)
        for key in expected:
            compare(actual[key], expected[key], f"{context}.{key}")
    elif isinstance(expected, (list, tuple)):
        require(
            isinstance(actual, (list, tuple)) and len(actual) == len(expected), context
        )
        for i, (a, b) in enumerate(zip(actual, expected, strict=True)):
            compare(a, b, f"{context}[{i}]")
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        require(
            isinstance(actual, (int, float))
            and not isinstance(actual, bool)
            and math.isfinite(actual)
            and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10),
            context,
        )
    else:
        require(type(actual) is type(expected) and actual == expected, context)


def read_inputs(spec, repo, deadline):
    """Hash bytes, read only TRAIN/shown fields, hash again before selection."""
    checked = {"inputs": {}, "checkpoints": []}
    require(set(spec["inputs"]) == {"train", "side"}, "input pin roles")
    for role, pin in spec["inputs"].items():
        path = (repo / pin["path"]).resolve()
        require(digest(path, deadline) == pin["sha256"], "input pin hash")
        checked["inputs"][role] = {"path": str(path), "sha256": pin["sha256"]}
    require(
        [(p["encoding"], p["seed"]) for p in spec["checkpoints"]] == ORDER,
        "checkpoint pin order",
    )
    for pin in spec["checkpoints"]:
        require(
            set(pin["hashes"])
            == {"model.safetensors", "policy_config.json", "training_manifest.json"},
            "checkpoint pin files",
        )
        path = (repo / pin["path"]).resolve()
        for name, wanted in pin["hashes"].items():
            require(digest(path / name, deadline) == wanted, "checkpoint pin hash")
        checked["checkpoints"].append({**pin, "path": str(path)})
    tasks = []
    path = Path(checked["inputs"]["train"]["path"])
    opener = gzip.open if path.suffix == ".gz" else Path.open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            check_time(deadline)
            if not line.strip():
                continue
            record = json.loads(line)
            values = {
                f.name: record[f.name]
                for f in fields(Task)
                if f.name in record
                and f.name not in {"held_out_inputs", "held_out_outputs"}
            }
            values.update(held_out_inputs=(), held_out_outputs=())
            for name in ("inputs", "outputs", "program", "tags"):
                values[name] = tuple(values[name])
            if "alternative_programs" in values:
                values["alternative_programs"] = tuple(
                    tuple(p) for p in values["alternative_programs"]
                )
            task = Task(**values)
            require(task.split == "train", "non-train input row")
            tasks.append(task)
    programs = {}
    with Path(checked["inputs"]["side"]["path"]).open() as stream:
        for line in stream:
            check_time(deadline)
            if not line.strip():
                continue
            record = json.loads(line)
            name, text = record.get("behaviour"), record.get("program")
            require(
                isinstance(name, str)
                and name
                and isinstance(text, str)
                and text.split()
                and name not in programs,
                "invalid/duplicate side row",
            )
            programs[name] = text
    for pin in checked["inputs"].values():
        require(
            digest(pin["path"], deadline) == pin["sha256"],
            "input changed during verification",
        )
    return tasks, programs, checked


def losses(logits, mask, witness):
    require(
        len(logits) == len(mask) == 28 and all(type(x) is bool for x in mask),
        "raw logit/mask shape",
    )
    require(
        all(
            isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
            for x in logits
        ),
        "nonfinite raw logits",
    )
    index = NAMES.index(witness)
    require(mask[index], "illegal witnessed op")
    legal = [v for v, keep in zip(logits, mask, strict=True) if keep]
    peak, legal_peak, wanted = max(logits), max(legal), logits[index]
    norm = math.log(math.fsum(math.exp(v - peak) for v in logits))
    legal_norm = math.log(math.fsum(math.exp(v - legal_peak) for v in legal))
    values = {
        "unmasked_nll": norm + (peak - wanted),
        "legal_nll": legal_norm + (legal_peak - wanted),
        "negative_log_legal_mass": (peak - legal_peak) + norm - legal_norm,
        "best_legal_deficit": legal_peak - wanted,
        "legal_rank": 1 + sum(v > wanted for v in legal),
        "legal_size": len(legal),
    }
    require(all(math.isfinite(v) for v in values.values()), "nonfinite derived metric")
    require(
        math.isclose(
            values["unmasked_nll"],
            values["legal_nll"] + values["negative_log_legal_mass"],
            rel_tol=1e-10,
            abs_tol=1e-10,
        ),
        "legal mass decomposition",
    )
    return values


def mean(values):
    return math.fsum(values) / len(values)


def aggregate(cells, metric):
    if not cells:
        return None
    details, behaviours = [], defaultdict(list)
    for cell in cells:
        top = mean([metric[tuple(key)]["unmasked_nll"] for key in cell["top"]])
        body = mean([metric[tuple(key)]["unmasked_nll"] for key in cell["body"]])
        delta = top - body
        behaviours[cell["behaviour"]].append(delta)
        details.append(
            {k: cell[k] for k in ("behaviour", "group", "op", "position")}
            | {
                "assembly": cell["op"] in {"quote", "cons", "cat"},
                "top_count": len(cell["top"]),
                "body_count": len(cell["body"]),
                "top_mean": top,
                "body_mean": body,
                "delta": delta,
            }
        )
    means = {name: mean(values) for name, values in sorted(behaviours.items())}
    return {
        "cells": details,
        "behaviours": means,
        "delta": mean(list(means.values())),
        "positive_behaviour_fraction": sum(v > 0 for v in means.values()) / len(means),
    }


def descriptive(rows, metric):
    buckets = defaultdict(list)
    for row in rows:
        buckets[
            row["side"],
            "structural" if row["structural"] else "content",
            row["position"],
            row["op"],
            row["depth"],
        ].append(row)
    labels = []
    for key, group in sorted(buckets.items()):
        item = dict(
            zip(("side", "class", "position", "op", "depth"), key, strict=True),
            rows=len(group),
        )
        for name in METRICS:
            values = [metric[r["behaviour"], r["step"]][name] for r in group]
            item[name] = {"sum": math.fsum(values), "mean": mean(values)}
        item.update(
            resource_affected_rows=sum(
                r["legality"]["resource_affected"] for r in group
            ),
            state_truncated_rows=sum(
                r["encoder_diagnostics"]["any_state_truncated"] for r in group
            ),
            target_truncated_rows=sum(
                r["encoder_diagnostics"]["any_target_truncated"] for r in group
            ),
        )
        labels.append(item)
    return {"rows": len(rows), "by_label": labels}


def summary(stage, metric, encoding, seed):
    shares, groups = [], defaultdict(list)
    for row in stage["rows"]:
        groups[row["behaviour"]].append(row)
    for name, rows in sorted(groups.items()):
        top = [r for r in rows if r["side"] == "top"]

        def share(kind, rows=rows, top=top):
            whole = math.fsum(metric[r["behaviour"], r["step"]][kind] for r in rows)
            return (
                math.fsum(metric[r["behaviour"], r["step"]][kind] for r in top) / whole
                if whole
                else None
            )

        shares.append(
            {
                "behaviour": name,
                "rows": len(rows),
                "top_rows": len(top),
                "top_token_share": len(top) / len(rows),
                "top_unmasked_loss_share": share("unmasked_nll"),
                "top_legal_deficit_share": share("best_legal_deficit"),
                "top_legal_nll_share": share("legal_nll"),
                "resource_affected_rows": sum(
                    r["legality"]["resource_affected"] for r in rows
                ),
            }
        )
    return {
        "encoding": encoding,
        "seed": seed,
        "primary": aggregate(stage["cells"], metric),
        "without_assembly_descriptive": aggregate(
            [c for c in stage["cells"] if c["op"] not in {"quote", "cons", "cat"}],
            metric,
        ),
        "all_rows": descriptive(stage["rows"], metric),
        "common_support_rows": descriptive(
            [r for r in stage["rows"] if r["matched"]], metric
        ),
        "behaviour_loss_and_exposure": shares,
        "per_row_metrics": [
            {
                "behaviour": r["behaviour"],
                "step": r["step"],
                **metric[r["behaviour"], r["step"]],
                "resource_affected": r["legality"]["resource_affected"],
            }
            for r in stage["rows"]
        ],
        "rank_ties": "strictly_greater_legal_logits",
        "interpretation": "descriptive_midspan_exact_op_emissions_only",
    }


def reading(summaries):
    encodings = {}
    for encoding in ("machine", "difference"):
        group = [s for s in summaries if s["encoding"] == encoding]
        deltas = [s["primary"]["delta"] for s in group]
        fractions = [s["primary"]["positive_behaviour_fraction"] for s in group]
        encodings[encoding] = {
            "seed_deltas": deltas,
            "seed_positive_behaviour_fractions": fractions,
            "median_delta": median(deltas),
            "positive_seeds": sum(v > 0 for v in deltas),
            "median_positive_behaviour_fraction": median(fractions),
            "passes": median(deltas) >= 0.25
            and sum(v > 0 for v in deltas) >= 2
            and median(fractions) >= 0.60,
        }
    return {
        "encodings": encodings,
        "reading": "TOP_LOSS_ROUTE"
        if all(e["passes"] for e in encodings.values())
        else "NO_MATCHED_MIDSPAN_TOP_EXCESS",
    }


def verify_result(
    final_path, log_path, spec_path, repo, expected_revision, *, deadline=None
):
    """No raw effects are read until successful terminal, input and Stage A checks."""
    final_path, log_path, spec_path, repo = map(
        Path, (final_path, log_path, spec_path, repo)
    )
    log_lines = log_path.read_text().splitlines()
    require(
        log_lines and log_lines[-1] == "exit 0",
        "not a successful logged trial",
    )
    manifest = json.loads(final_path.read_text())
    require(
        manifest["complete"] is True
        and manifest["exit_code"] == 0
        and manifest["retry_allowed"] is False,
        "incomplete terminal",
    )
    require(
        manifest["reading"]
        in {"TOP_LOSS_ROUTE", "NO_MATCHED_MIDSPAN_TOP_EXCESS", "INSUFFICIENT_SUPPORT"},
        "unsuccessful terminal; no partial reading",
    )
    require(manifest["source_revision"] == expected_revision, "wrong reviewed source")
    attempt = json.loads((final_path.parent / "attempt.json").read_text())
    require(
        attempt["source_revision"] == expected_revision
        and attempt["retry_allowed"] is False,
        "attempt provenance",
    )
    review = attempt["review"]
    require(
        all(
            review.get(k) == v
            for k, v in {
                "verdict": "PASS",
                "source_revision": expected_revision,
                "issue": 178,
                "design": "D1+AmendmentA",
                "reviewer_family": "claude",
                "effort": "high",
            }.items()
        ),
        "review receipt identity",
    )
    require(
        digest(review["report"]["path"], deadline) == review["report"]["sha256"],
        "review report hash",
    )
    seal = manifest["stage_a"]
    require(digest(seal["path"], deadline) == seal["sha256"], "stage seal hash")
    stage = json.loads(Path(seal["path"]).read_text())
    require(
        stage["source_revision"] == expected_revision
        and seal["rows"] == len(stage["rows"])
        and seal["kind"] == stage["kind"],
        "stage seal identity",
    )
    tasks, programs, pins = read_inputs(
        json.loads(spec_path.read_text()), repo, deadline
    )
    pins["spec_sha256"] = digest(spec_path, deadline)
    compare(stage["provenance"], pins, "frozen input provenance")
    reconstructed = stage_check.verify_stage(stage, tasks, programs, deadline=deadline)
    progress, outputs = manifest["progress"], manifest["checkpoints"]
    if not reconstructed["support_passes"]:
        require(
            manifest["reading"] == "INSUFFICIENT_SUPPORT"
            and not outputs
            and manifest["result"] is None,
            "support-only terminal",
        )
        require(
            all(
                progress[k] == 0
                for k in (
                    "checkpoint_load_attempts",
                    "models_loaded",
                    "logit_rows",
                    "checkpoints_complete",
                )
            ),
            "unsupported model work",
        )
        require(
            not list(final_path.parent.glob("*-raw.jsonl"))
            and not list(final_path.parent.glob("*-summary.json")),
            "unsupported model artifacts",
        )
        result = None
    else:
        require(
            [(o["encoding"], o["seed"]) for o in outputs] == ORDER,
            "exact six checkpoint order",
        )
        require(
            progress["models_loaded"]
            == progress["checkpoint_load_attempts"]
            == progress["checkpoints_complete"]
            == 6
            and progress["logit_rows"] == len(stage["rows"]) * 6,
            "exact six checkpoint progress",
        )
        summaries = []
        require(
            {p.resolve() for p in final_path.parent.glob("*-raw.jsonl")}
            == {Path(o["raw"]["path"]).resolve() for o in outputs}
            and {p.resolve() for p in final_path.parent.glob("*-summary.json")}
            == {Path(o["summary"]["path"]).resolve() for o in outputs},
            "unexpected persisted checkpoint artifacts",
        )
        for output in outputs:
            encoding, seed = output["encoding"], output["seed"]
            for role in ("raw", "summary"):
                require(
                    digest(output[role]["path"], deadline) == output[role]["sha256"],
                    "persisted output hash",
                )
            raw_path = Path(output["raw"]["path"])
            metric, ordinal = {}, 0
            with raw_path.open() as stream:
                for line in stream:
                    check_time(deadline)
                    require(ordinal < len(stage["rows"]), "extra raw row")
                    raw, row = json.loads(line), stage["rows"][ordinal]
                    require(
                        (
                            raw["encoding"],
                            raw["seed"],
                            raw["ordinal"],
                            raw["behaviour"],
                            raw["step"],
                            raw["op"],
                        )
                        == (
                            encoding,
                            seed,
                            ordinal,
                            row["behaviour"],
                            row["step"],
                            row["op"],
                        ),
                        "ordered raw row identity",
                    )
                    value = losses(raw["logits"], row["legality"]["mask"], row["op"])
                    compare(raw["metrics"], value, "raw loss recomputation")
                    metric[row["behaviour"], row["step"]] = value
                    ordinal += 1
            require(ordinal == len(stage["rows"]) == output["rows"], "missing raw rows")
            computed = summary(stage, metric, encoding, seed)
            compare(
                json.loads(Path(output["summary"]["path"]).read_text()),
                computed,
                "independent saved summary",
            )
            summaries.append(computed)
        result = reading(summaries)
        compare(manifest["result"], result, "independent reading")
        require(manifest["reading"] == result["reading"], "terminal reading")
    check_time(deadline)
    return {
        "verified": True,
        "source_revision": expected_revision,
        "final_sha256": digest(final_path, deadline),
        "stage_sha256": seal["sha256"],
        "reconstruction": reconstructed,
        "reading": manifest["reading"],
        "result": result,
        "models_deserialized": 0,
        "numeric_tolerance": {"relative": 1e-10, "absolute": 1e-10},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("final", "log", "spec", "repo", "source-revision", "out"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    deadline = time.perf_counter() + 900
    result = verify_result(
        args.final,
        args.log,
        args.spec,
        args.repo,
        args.source_revision,
        deadline=deadline,
    )
    with Path(args.out).open("x") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
