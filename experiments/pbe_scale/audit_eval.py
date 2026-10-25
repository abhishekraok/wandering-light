"""Blind exclusion audit against frozen v4 eval references; never load world-44.

Only candidate IDs/hashes and aggregate coverage leave the audit. Sampler-valid
inputs establish sampled agreement, not proof that related task families are
uncontaminated. Inconclusive references/candidates remain explicitly counted.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import multiprocessing
import random
import resource
from concurrent.futures import ProcessPoolExecutor
from itertools import islice
from pathlib import Path

from experiments.distinct_programs.audit import plain_args, shape, v4_reference
from experiments.pbe_scale.validate import (
    SAFE_BUILTINS,
    _valid_result,
    atomic_text,
    canonical,
    check_reference,
    provenance,
)
from wandering_light.stack_pbe import CATALOGUE, _name_seed

V4 = (
    Path(__file__).resolve().parents[2]
    / "wandering_light/evals/data/stack_pbe_v4/manifest.json"
)
BANKS: list[dict] = []


def frozen_references() -> tuple[list, dict]:
    """Read only the pinned v4 eval payload, locally cached or beside its manifest."""
    from huggingface_hub import hf_hub_download

    manifest = json.loads(V4.read_text())
    split = manifest["splits"]["eval"]
    payload = V4.parent / split["path"]
    if not payload.exists():
        payload = Path(
            hf_hub_download(
                **manifest["hub"], filename=split["path"], local_files_only=True
            )
        )
    with gzip.open(payload, "rt") as stream:
        names = {json.loads(line)["behaviour"] for line in stream}
    references = [b for b in CATALOGUE if b.name in names]
    return references, {
        "manifest_sha256": hashlib.sha256(V4.read_bytes()).hexdigest(),
        "payload_sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
        "expected_payload_sha256": split["sha256"],
        "frozen_behaviours": len(names),
        "references_resolved": len(references),
    }


def banks_for(references: list, seed: int = 188) -> tuple[list[dict], int]:
    """Choose distinct, nonconstant, sampler-valid cases including drawn extrema."""
    banks, inconclusive = [], 0
    for behaviour in references:
        rng = random.Random(_name_seed(behaviour.name) + seed)
        unique = {}
        for _ in range(512):
            try:
                args = plain_args(behaviour.sample(rng))
                value = v4_reference(behaviour, args)
                if _valid_result(value):
                    unique[canonical(args)] = (args, value)
            except Exception:
                continue
        pool = list(unique.values())
        if len(pool) < 32 or len({canonical(out) for _, out in pool}) < 2:
            inconclusive += 1
            continue

        def lengths(case):
            return sum(len(x) if isinstance(x, list) else 1 for x in case[0])

        extrema = [min(pool, key=lengths), max(pool, key=lengths), pool[0]]
        extrema += [case for case in pool if case[1] != extrema[0][1]][:1]
        chosen = list(
            {canonical(args): (args, out) for args, out in extrema + pool}.values()
        )[:64]
        banks.append(
            {
                "shape": shape(chosen[0][0]),
                "inputs": [x for x, _ in chosen],
                "outputs": [y for _, y in chosen],
            }
        )
    return banks, inconclusive


def _reference_child(writer, source: str, inputs: list) -> None:
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (3, 3))
        memory = (
            int(Path("/proc/self/statm").read_text().split()[0])
            * resource.getpagesize()
            + 512 * 1024**2
        )
        resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
        namespace = {"__builtins__": SAFE_BUILTINS}
        exec(compile(source, "<blind-audit-reference>", "exec"), namespace)
        results = []
        for args in inputs:
            try:
                value = namespace["f"](*json.loads(canonical(args)))
                results.append(value if _valid_result(value) else None)
            except Exception:
                results.append(None)
        writer.send(results)
    except BaseException:
        writer.send(None)
    finally:
        writer.close()


def evaluate(source: str, inputs: list) -> list:
    reader, writer = multiprocessing.get_context("fork").Pipe(duplex=False)
    process = multiprocessing.get_context("fork").Process(
        target=_reference_child, args=(writer, source, inputs)
    )
    process.start()
    writer.close()
    try:
        result = reader.recv() if reader.poll(5) else None
        return result if result is not None else [None] * len(inputs)
    except EOFError:
        return [None] * len(inputs)
    finally:
        reader.close()
        if process.is_alive():
            process.kill()
        process.join()


def init_worker(banks: list[dict]) -> None:
    BANKS.clear()
    BANKS.extend(banks)


def audit_candidate(row: dict) -> tuple[bool, bool]:
    candidate = row["candidate"]
    kind = (
        candidate["input_kind"]
        .replace("two_lists", "list_list")
        .replace("two_ints", "int_int")
    )
    matched = [b for b in BANKS if b["shape"] == tuple(kind.split("_"))]
    if not matched:
        return False, True
    source = candidate["reference"]
    try:
        check_reference(source)
    except (ValueError, SyntaxError):
        return False, True
    # One bounded child handles all same-shape prefilters.
    outputs = evaluate_banks(source, matched, 4)
    survivors = [b for b in matched if agrees(b, outputs, 4)]
    failed = any(value is None for value in outputs.values())
    if survivors:
        outputs = evaluate_banks(source, survivors)
        failed |= any(value is None for value in outputs.values())
        if any(agrees(b, outputs) for b in survivors):
            return True, False
    return False, failed


def evaluate_banks(source, banks, limit=None):
    pooled = list(
        {canonical(a): a for b in banks for a in b["inputs"][:limit]}.values()
    )
    return dict(zip(map(canonical, pooled), evaluate(source, pooled), strict=True))


def agrees(bank, outputs, limit=None):
    return all(
        outputs[canonical(a)] == out
        for a, out in zip(bank["inputs"][:limit], bank["outputs"][:limit], strict=True)
    )


def run(
    accepted: Path,
    output: Path,
    *,
    workers: int = 1,
    seed: int = 188,
    filtered: Path | None = None,
    references: list | None = None,
) -> dict:
    if workers < 1:
        raise ValueError("workers must be positive")
    output.mkdir(parents=True, exist_ok=True)
    frozen = {"synthetic_fixture": True}
    if references is None:
        references, frozen = frozen_references()
    banks, inconclusive = banks_for(references, seed)
    inconclusive += frozen.get("frozen_behaviours", len(references)) - len(references)
    excluded, hashes, total, failures = [], [], 0, 0
    origin = provenance([accepted, V4, Path(__file__)], seed, 0)
    origin["benchmark"] = frozen
    origin["config"].update(
        audit="blind-v4-sampled-exact",
        min_distinct=32,
        max_cases=64,
        sampler_draws=512,
        workers=workers,
    )
    temporary = filtered.with_suffix(filtered.suffix + ".tmp") if filtered else None
    if temporary:
        temporary.parent.mkdir(parents=True, exist_ok=True)
    with (
        accepted.open() as stream,
        temporary.open("wb") if temporary else io.BytesIO() as raw,
        gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as packed,
        io.TextIOWrapper(packed, encoding="utf-8") as tasks,
        ProcessPoolExecutor(
            max_workers=workers, initializer=init_worker, initargs=(banks,)
        ) as executor,
    ):
        rows = (json.loads(line) for line in stream if line.strip())
        while batch := list(islice(rows, 64)):
            for row, (exclude, failure) in zip(
                batch, executor.map(audit_candidate, batch), strict=True
            ):
                total += 1
                failures += failure
                if exclude:
                    excluded.append(row["candidate"]["id"])
                    hashes.append(row["candidate_digest"])
                elif filtered:
                    tasks.write(canonical(row["task"]) + "\n")
    if temporary:
        temporary.replace(filtered)
    report = {
        "excluded_ids": excluded,
        "excluded_hashes": hashes,
        "audited_candidates": total,
        "excluded_candidates": len(excluded),
        "kept_candidates": total - len(excluded),
        "covered_eval_references": len(banks),
        "inconclusive_eval_references": inconclusive,
        "inconclusive_candidates": failures,
        "provenance": origin,
        "limitation": "Sampled exact agreement is conservative exclusion, not a proof of no systematic family contamination. Insufficient-domain references and failed calls are inconclusive.",
    }
    atomic_text(
        output / "exclusions.json", json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accepted", required=True, type=Path)
    parser.add_argument("--output-dir", dest="output", required=True, type=Path)
    parser.add_argument("--filtered-tasks", dest="filtered", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=188)
    report = run(**vars(parser.parse_args()))
    keys = (
        "audited_candidates",
        "excluded_candidates",
        "covered_eval_references",
        "inconclusive_eval_references",
        "inconclusive_candidates",
    )
    print(canonical({k: report[k] for k in keys}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
