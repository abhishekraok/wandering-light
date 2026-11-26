"""Run one `mark-28` beam under the debugger and print or serve what it did.

python -m wandering_light.stack_debugger.cli trace --task 359
python -m wandering_light.stack_debugger.cli trace --behaviour min_pair --seed 3 --json out.json
python -m wandering_light.stack_debugger.cli serve --port 8010
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from wandering_light.debugger_http import is_loopback_host
from wandering_light.stack_debugger.context import (
    DEFAULT_CORPUS_MANIFEST,
    DEFAULT_THREADS,
    DebugContext,
)
from wandering_light.stack_debugger.session import (
    DEFAULT_BEAM_WIDTH,
    DEFAULT_GUARD,
    DEFAULT_MAX_DEPTH,
    SearchSettings,
    build_session,
    select_task,
)


def add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--corpus-manifest", type=Path, default=DEFAULT_CORPUS_MANIFEST)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Local stack-policy checkpoint; defaults to the #101 2m-s1 weights "
        "on this machine, else the archived W&B artifact.",
    )
    parser.add_argument("--artifact", default=None, help="W&B model artifact to fetch")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    trace = subparsers.add_parser("trace", help="Run one task and dump its trace")
    add_shared_arguments(trace)
    trace.add_argument("--task", type=int, default=None, help="eval split index")
    trace.add_argument("--behaviour", default=None)
    trace.add_argument("--bucket", default=None, help="1-5, 6-10, 11-20, 21-40, 41+")
    trace.add_argument(
        "--cohort", action="store_true", help="sample from the #101 cohort"
    )
    trace.add_argument("--seed", type=int, default=None)
    trace.add_argument("--beam-width", type=int, default=DEFAULT_BEAM_WIDTH)
    trace.add_argument("--max-depth", type=int, default=DEFAULT_MAX_DEPTH)
    trace.add_argument("--guard", default=DEFAULT_GUARD)
    trace.add_argument("--budget", type=int, default=None)
    trace.add_argument("--json", type=Path, default=None)

    serve = subparsers.add_parser("serve", help="Serve the interactive debugger")
    add_shared_arguments(serve)
    serve.add_argument("--host", default="127.0.0.1", help="Loopback address")
    serve.add_argument("--port", type=int, default=8010)

    args = parser.parse_args(argv)
    if args.command == "serve" and not is_loopback_host(args.host):
        serve.error("--host must be a loopback address")
    context = DebugContext.load(
        corpus_manifest=args.corpus_manifest,
        checkpoint=args.checkpoint,
        artifact=args.artifact,
        device=args.device,
        threads=args.threads,
    )

    if args.command == "serve":
        from wandering_light.stack_debugger.server import serve_debugger

        serve_debugger(context, host=args.host, port=args.port)
        return 0

    index, selection = select_task(
        context.tasks,
        task=args.task,
        behaviour=args.behaviour,
        bucket=args.bucket,
        cohort=args.cohort,
        seed=args.seed,
    )
    session = build_session(
        context.scorer,
        context.tasks,
        index,
        settings=SearchSettings(
            beam_width=args.beam_width,
            max_depth=args.max_depth,
            guard=args.guard,
            budget=args.budget,
        ),
        provenance={**context.provenance, "selection": selection},
    )
    payload = session.to_dict()
    if args.json is not None:
        args.json.write_text(json.dumps(payload), encoding="utf-8")

    task = payload["task"]
    print(
        f"task            eval {task['index']} · {task['behaviour']} · {task['summary']}"
    )
    for given, wanted in zip(task["inputs"], task["outputs"], strict=True):
        print(f"  {given}  ->  {wanted}")
    print(f"witness         {task['length']} ops: {' '.join(payload['witnesses'][0])}")
    result = payload["result"]
    print(
        f"termination     {result['termination']} at depth {result['depth']} in {result['seconds']} s"
    )
    print(
        f"executions      {result['executions']}  fuel {result['fuel']}  expansions {result['expansions']}"
    )
    if result["program"]:
        verdict = (
            "solved"
            if result["solved"]
            else ("memorised" if result["shown"] else "fits nothing")
        )
        print(f"program         {' '.join(result['program'])}  ({verdict})")
    for layer, overlay in zip(
        payload["trace"]["layers"], payload["witness_overlay"], strict=True
    ):
        first = overlay["witnesses"][0]
        rank = (
            f"witness next {first['next']!r} ranked {first['rank']}/{first['legal']}"
            if first["rank"]
            else "off the witness"
        )
        print(
            f"  depth {layer['depth']:2d}: {len(layer['nodes']):3d} nodes, "
            f"{layer['counts']['executed']:5d} executed, "
            f"{len(layer['retained']):3d} retained, "
            f"witness {'alive' if first['survives'] else 'gone'}, {rank}"
        )
    if args.json is not None:
        print(
            f"wrote           {args.json} ({args.json.stat().st_size / 1024:.0f} KiB)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
