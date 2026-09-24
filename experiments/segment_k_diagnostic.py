"""The K diagnostic for short-segment-campaign (#168): does cumulative log-probability keep the witness's closed state among the top K roots?

For each witness, `segment_witness_ranks` runs the segment-reset beam with the
witness's own closed state forced back in as a root at every cut, and records
the rank of its next closed state among every merged closed state, by
cumulative raw log-probability. K must cover that rank in most segments. The
share within 16 is the registered number; the full distribution sets K.

Rows are appended per program to ``k-<set>.jsonl`` as they finish, so a run
resumes and can be read while it goes; the summary is written at the end.

    uv run python experiments/segment_k_diagnostic.py --set staged-train --out DIR [--checkpoint PATH] [--device cuda] [--limit 40]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from experiments.frame_gate import eval_witnesses  # noqa: E402
from experiments.staged_witnesses import check as staged  # noqa: E402
from wandering_light.solve import BEST  # noqa: E402
from wandering_light.stack_beam import segment_witness_ranks  # noqa: E402
from wandering_light.stack_policy import PolicyScorer, load_policy  # noqa: E402
from wandering_light.stack_policy_hub import fetch_stack_policy  # noqa: E402

SIDE = REPO / "experiments" / "staged_witnesses" / "v4.jsonl"
CAPPED = REPO / "reports" / "short-segment-20260923" / "train_capped.jsonl"
YARDSTICK_167 = REPO / "reports" / "short-body-20260923" / "scores.json"
YARDSTICK_168 = REPO / "reports" / "short-segment-20260923" / "yardstick.jsonl"
SETS = ("staged-train", "capped-train", "fused-train", "yardstick-167", "yardstick-168")
KS = (1, 4, 16, 64)


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def programs(which: str) -> list:
    """(task, program) pairs: the tasks' first example sets, the programs to rank."""
    tasks = staged.v4_tasks()
    if which == "staged-train":
        return [
            (tasks[r["behaviour"]][0], tuple(r["program"].split()))
            for r in _jsonl(SIDE)
            if r["verdict"] == "staged"
        ]
    if which == "capped-train":
        return [
            (tasks[r["behaviour"]][0], tuple(r["program"].split()))
            for r in _jsonl(CAPPED)
        ]
    if which == "fused-train":
        # The same behaviours as capped-train, along their fused witnesses.
        return [
            (tasks[r["behaviour"]][0], tasks[r["behaviour"]][0].program)
            for r in _jsonl(CAPPED)
        ]
    authored = {t.behaviour: t for t in eval_witnesses("stack_pbe_v4")}
    if which == "yardstick-167":
        rows = json.loads(YARDSTICK_167.read_text())["rows"]
        return [
            (authored[r["behaviour"]], tuple(r["program"].split()))
            for r in rows
            if r.get("draw_verdict") in ("genuine", "equals_catalogue_witness")
        ]
    if which == "yardstick-168":
        return [
            (authored[r["behaviour"]], tuple(r["program"].split()))
            for r in _jsonl(YARDSTICK_168)
        ]
    raise ValueError(which)


def summarise(rows: list[dict]) -> dict:
    ranks = [r["rank"] for r in rows]
    produced = [r for r in ranks if r is not None]
    return {
        "programs": len({r["behaviour"] for r in rows}),
        "segments": len(rows),
        "not_produced": len(ranks) - len(produced),
        "within": {
            str(k): sum(r <= k for r in produced) / len(ranks) if ranks else None
            for k in KS
        },
        "median_rank_when_produced": statistics.median(produced) if produced else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint", default="best")
    parser.add_argument("--set", dest="which", choices=SETS, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--limit", type=int, default=None, help="every k-th program, about this many"
    )
    args = parser.parse_args(argv)
    torch.set_num_threads(8)
    path = (
        fetch_stack_policy(BEST.repo_id, BEST.revision)
        if args.checkpoint == "best"
        else Path(args.checkpoint)
    )
    scorer = PolicyScorer(load_policy(path, device=args.device), args.device)
    pairs = programs(args.which)
    if args.limit and len(pairs) > args.limit:
        stride = -(-len(pairs) // args.limit)
        pairs = pairs[::stride]
    args.out.mkdir(parents=True, exist_ok=True)
    stream = args.out / f"k-{args.which}.jsonl"
    done = {r["behaviour"] for r in _jsonl(stream)} if stream.exists() else set()
    started = time.time()
    for number, (task, program) in enumerate(pairs, 1):
        if task.behaviour in done:
            continue
        clock = time.time()
        mine = dataclasses.replace(task, program=program)
        rows = [
            {"behaviour": task.behaviour, **row}
            for row in segment_witness_ranks(
                mine, scorer, program=program, max_state_atoms=BEST.max_state_atoms
            )
        ]
        with stream.open("a") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        print(
            f"{number}/{len(pairs)} {task.behaviour} ops {len(program)} "
            f"segments {len(rows)} ranks {[r['rank'] for r in rows]} "
            f"{time.time() - clock:.0f}s",
            flush=True,
        )
    rows = _jsonl(stream)
    summary = {
        "checkpoint": str(path),
        "encoding": scorer.encoding,
        "set": args.which,
        "limit": args.limit,
        "max_state_atoms": BEST.max_state_atoms,
        **summarise(rows),
        "seconds_this_run": round(time.time() - started, 1),
    }
    (args.out / f"k-{args.which}.json").write_text(
        json.dumps({"summary": summary}, indent=1) + "\n"
    )
    print("RESULT", json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
