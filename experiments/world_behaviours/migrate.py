"""Rewrite probe programs from the authors' macro dialects to the canonical library (#124).

Reads the 2026-09-09 probe's ``programs_*.jsonl``, replaces every author-file
macro name with its canonical tokens from ``aliases.txt`` (recursively, since
an alias body may name another alias), inlines the one-behaviour helpers
marked ``@inline`` from the author files kept under ``<out>/authors/``, and
writes one ``programs_canonical.jsonl`` that the canonical ``macros.txt``
alone can expand. ``report.py`` then re-verifies every program.

Usage:
    uv run python experiments/world_behaviours/migrate.py \
        reports/world-behaviours-probe-20260909 reports/world-behaviours-library-20260910
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from verify import load_macros

HERE = Path(__file__).parent
ALIASES = HERE / "aliases.txt"


def read_table(path: Path) -> dict[str, list[str]]:
    table: dict[str, list[str]] = {}
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            name, body = line.split(":", 1)
            table[name.strip()] = body.split()
    return table


def author_bodies(authors_dir: Path) -> dict[str, list[str]]:
    bodies: dict[str, list[str]] = {}
    for file in sorted(authors_dir.glob("macros*.txt")):
        bodies.update(read_table(file))
    return bodies


def rewrite(
    tokens: list[str],
    aliases: dict[str, list[str]],
    bodies: dict[str, list[str]],
    depth: int = 0,
) -> list[str]:
    if depth > 32:
        raise ValueError("alias recursion too deep")
    out: list[str] = []
    for tok in tokens:
        if tok not in aliases:
            out.append(tok)
            continue
        replacement = aliases[tok]
        if replacement == ["@inline"]:
            replacement = bodies[tok]
        out.extend(rewrite(replacement, aliases, bodies, depth + 1))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("probe_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args(argv)

    aliases = read_table(ALIASES)
    bodies = author_bodies(args.out_dir / "authors")
    canonical = load_macros()
    rows = []
    for file in sorted(args.probe_dir.glob("programs_*.jsonl")):
        for line in file.open():
            if line.strip():
                rows.append(json.loads(line))
    unresolved = set()
    out = args.out_dir / "programs_canonical.jsonl"
    changed = 0
    with out.open("w") as fh:
        for row in sorted(rows, key=lambda r: r["id"]):
            tokens = row["program"].split()
            rewritten = rewrite(tokens, aliases, bodies)
            for tok in rewritten:
                if tok.startswith("k") and "-" in tok and tok not in canonical:
                    unresolved.add(tok)
            changed += rewritten != tokens
            fh.write(
                json.dumps(
                    {
                        "id": row["id"],
                        "program": " ".join(rewritten),
                        "status": row.get("status"),
                        "note": row.get("note", ""),
                        "author_program": row["program"],
                    }
                )
                + "\n"
            )
    print(f"{len(rows)} programs, {changed} rewritten -> {out}")
    if unresolved:
        print("UNRESOLVED author names:", sorted(unresolved))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
