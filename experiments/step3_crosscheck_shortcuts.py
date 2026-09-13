"""Cross-check this study's solve verdicts against an independent shortcut list (#134).

Another session regraded every stored v4 held-out solve against the behaviour's
Python reference on its own sampler draws and wrote the programs that disagree,
each with a counterexample, to `known_shortcuts_v4.json`. Two verifiers built
separately should agree wherever they looked at the same program. For each
listed program this study also solved:

- my verdict `shortcut` agrees;
- my verdict `genuine` disagrees, and is settled by running the listed
  counterexample: if the program disagrees with the reference there, the
  shortcut is proven and my verdict was wrong.

Programs the list never saw are out of scope; a program absent from it was not
necessarily checked there.

Usage:
    uv run python experiments/step3_crosscheck_shortcuts.py \
        --verified reports/world-plus-v4-policy-20260912/verified_solves_epochs.json \
        --known-shortcuts ../busy-curie-2bb6c3/.claude/drafts/audit-1000/known_shortcuts_v4.json \
        --out reports/world-plus-v4-policy-20260912/shortcut_crosscheck.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from wandering_light.stack_pbe import CATALOGUE, check

BEHAVIOURS = {behaviour.name: behaviour for behaviour in CATALOGUE}


def settle(name: str, program: str, counterexample: dict[str, Any]) -> dict[str, Any]:
    """Run the listed counterexample; a mismatch proves the shortcut."""
    behaviour = BEHAVIOURS[name]
    args = tuple(counterexample["args"])
    try:
        mismatch = check(behaviour, args, tuple(program.split()))
        got = None if mismatch is None else mismatch.got
    except Exception as error:
        got = f"raised {error!r}"
    return {
        "args": list(args),
        "reference": behaviour.reference(*args),
        "program_returned": got,
        "shortcut_proven": got is not None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--verified", type=Path, action="append", required=True)
    parser.add_argument("--known-shortcuts", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    mine: dict[tuple[str, str], str] = {}
    for path in args.verified:
        for item in json.loads(path.read_text())["programs"]:
            mine[(item["behaviour"], item["program"])] = item["verdict"]
    listed = json.loads(args.known_shortcuts.read_text())

    agree, disagree, unseen = [], [], []
    for entry in listed:
        key = (entry["behaviour"], entry["program"])
        if key not in mine:
            unseen.append({"behaviour": key[0], "ops": entry["ops"]})
        elif mine[key] == "shortcut":
            agree.append({"behaviour": key[0], "ops": entry["ops"]})
        else:
            disagree.append(
                {
                    "behaviour": key[0],
                    "program": key[1],
                    "my_verdict": mine[key],
                    "settled": settle(key[0], key[1], entry["counterexample"]),
                }
            )
    listed_keys = {(e["behaviour"], e["program"]) for e in listed}
    mine_only = sorted(
        k for k, v in mine.items() if v == "shortcut" and k not in listed_keys
    )
    result = {
        "listed_programs": len(listed),
        "my_programs": len(mine),
        "overlap": len(agree) + len(disagree),
        "agree": agree,
        "disagree": disagree,
        "listed_but_not_solved_here": len(unseen),
        "my_shortcuts_not_in_list": [
            {"behaviour": b, "ops": len(p.split())} for b, p in mine_only
        ],
    }
    print(
        f"listed {result['listed_programs']}, mine {result['my_programs']}, overlap "
        f"{result['overlap']}: agree {len(agree)}, disagree {len(disagree)}; "
        f"listed but not solved here {len(unseen)}; my shortcuts not in the list "
        f"{len(mine_only)}"
    )
    for item in disagree:
        s = item["settled"]
        outcome = (
            f"returns {s['program_returned']!r} where the reference gives "
            f"{s['reference']!r}: the shortcut is proven and my verdict was wrong"
            if s["shortcut_proven"]
            else f"agrees with the reference ({s['reference']!r}): not a proven shortcut"
        )
        print(
            f"  DISAGREE {item['behaviour']} ({len(item['program'].split())} ops): "
            f"mine {item['my_verdict']}; on their counterexample the program {outcome}"
        )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(result, indent=2, sort_keys=True, default=str) + "\n"
        )
        print(f"wrote {args.out}")
    return 1 if any(d["settled"]["shortcut_proven"] for d in disagree) else 0


if __name__ == "__main__":
    raise SystemExit(main())
