"""Run one rollout under the debugger and print or serve what it did.

python -m wandering_light.rollout_debugger.cli trace --distance 5 --seed 7
python -m wandering_light.rollout_debugger.cli trace --task-id TASK --json out.json
python -m wandering_light.rollout_debugger.cli serve --port 8000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from wandering_light.rollout_debugger.context import DebugContext
from wandering_light.rollout_debugger.session import (
    DEFAULT_BEAM_WIDTH,
    DEFAULT_BUDGET,
    DEFAULT_CORPUS_DIR,
    DEFAULT_MAX_DEPTH,
    DEFAULT_SPLIT,
    SearchSettings,
    build_session,
    select_task,
)


def add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Local checkpoint directory; defaults to the published one.",
    )
    parser.add_argument("--revision", default=None)
    parser.add_argument("--device", default="cpu")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    trace = subparsers.add_parser("trace", help="Run one task and dump its trace")
    add_shared_arguments(trace)
    trace.add_argument("--task-id", default=None)
    trace.add_argument("--distance", type=int, default=None)
    trace.add_argument("--input-type", default=None)
    trace.add_argument("--seed", type=int, default=None)
    trace.add_argument("--beam-width", type=int, default=DEFAULT_BEAM_WIDTH)
    trace.add_argument("--max-depth", type=int, default=DEFAULT_MAX_DEPTH)
    trace.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    trace.add_argument("--capture-logits", action="store_true")
    trace.add_argument("--json", type=Path, default=None)

    serve = subparsers.add_parser("serve", help="Serve the interactive debugger")
    add_shared_arguments(serve)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    args = parser.parse_args(argv)

    context = DebugContext.load(
        corpus_dir=args.corpus_dir,
        split=args.split,
        checkpoint=args.checkpoint,
        revision=args.revision,
        device=args.device,
    )

    if args.command == "serve":
        from wandering_light.rollout_debugger.server import serve_debugger

        serve_debugger(context, host=args.host, port=args.port)
        return 0

    record, provenance = select_task(
        context.records,
        task_id=args.task_id,
        certified_distance=args.distance,
        input_type=args.input_type,
        seed=args.seed,
    )
    session = build_session(
        context.policy,
        record,
        context.available_functions,
        settings=SearchSettings(
            beam_width=args.beam_width,
            max_depth=args.max_depth,
            budget=args.budget,
        ),
        provenance={**context.provenance, "selection": provenance},
        capture_logits=args.capture_logits,
    )
    payload = session.to_dict()

    if args.json is not None:
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"task            {session.task_id}")
    print(f"input           {session.input_state['value']}")
    print(f"target          {session.target_state['value']}")
    print(f"certified       {session.certified_distance} ops: {session.witness}")
    print(f"termination     {payload['termination']}")
    print(f"executions      {payload['executions']}")
    print(f"solution        {payload['solution']}")
    for layer, overlay in zip(
        payload["trace"]["layers"], payload["witness_overlay"], strict=True
    ):
        print(
            f"  depth {layer['depth']}: "
            f"{layer['expansions_attempted']} attempted, "
            f"{layer['unique_successors']} unique, "
            f"{len(layer['retained'])} retained, "
            f"witness {'alive' if overlay['survives'] else 'gone'}"
        )
    if args.json is not None:
        size = args.json.stat().st_size
        print(f"wrote           {args.json} ({size / 1024:.1f} KiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
