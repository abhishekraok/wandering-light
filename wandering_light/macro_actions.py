"""The authoring macro library as actions the beam can apply (macro-action-space #162).

clear-world-v3 (#161) closed every search-side term of the budget law on world
data: a beam of width `W` tolerates about `ln W` nats of deficit over a whole
witness, so a witness of length `L` needs its per-step deficit under `ln W / L`,
and depth, width and policy size all leave that unchanged. Only `L` is left, and
`experiments/world_behaviours/macros.txt` is the library the corpus was authored
in -- world-v3's witnesses are a median 149.5 raw `mark-28` ops and 28.5 macro
tokens.

**These are top-level actions only, and that is a real limit rather than a
simplification.** 77.6% of the macro tokens the authors wrote sit *inside* a
quotation, where a name is captured under the mark as a `Sym` and has to resolve
when the body runs -- and `run_body` resolves against the module-global
`OP_BY_NAME`, which cannot see a macro. Giving it an ops table is
[#149](../../issues/149), and until that exists a macro action reaches only the
22% written at top level, which expands the median witness from 28.5 macro
tokens to **71.5**: 2.1x compression rather than 5.2x. A macro op here therefore
**refuses while a quotation is open**, so the search can never write one into a
body and silently produce a program the machine would not run.

Cost is the expansion's cost. A macro applies its body one op at a time through
`step_raw` and charges every unit of fuel it spends, so a program that uses
macros is not cheaper on `docs/metrics.md`'s clock than the same program written
out -- only shorter to choose. Anything else would let a library hide work
inside itself and make cost to clear meaningless.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

from wandering_light.stack_machine import Op, quotation_open, step_raw
from wandering_light.stack_pbe import OP_BY_NAME, OPS

if TYPE_CHECKING:
    from collections.abc import Sequence

LIBRARY = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "world_behaviours"
    / "macros.txt"
)
MACRO_GROUP = "macro"


def read_library(path: Path | None = None) -> dict[str, list[str]]:
    """``{name: body}`` as written, one name per line, bodies may name macros."""
    out: dict[str, list[str]] = {}
    for raw in (path or LIBRARY).read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        name, body = line.split(":", 1)
        name, body = name.strip(), body.split()
        if name and body:
            out[name] = body
    return out


def expand(name: str, library: dict[str, list[str]], depth: int = 0) -> list[str]:
    """A macro's body in base ops. Raises rather than looping on a cycle."""
    if depth > 64:
        raise ValueError(f"macro expansion too deep at {name!r}; cycle?")
    out: list[str] = []
    for token in library[name]:
        if token in library:
            out.extend(expand(token, library, depth + 1))
        elif token in OP_BY_NAME:
            out.append(token)
        else:
            raise ValueError(f"macro {name!r} names {token!r}, which is not an op")
    return out


def _runner(expansion: tuple[str, ...]):
    def run(stack: tuple, budget: int, depth: int):
        # A macro is a top-level action: writing one into a quotation would
        # capture a name `run_body` cannot resolve (#149).
        if quotation_open(stack):
            return "macros are top-level actions"
        spent = 0
        for name in expansion:
            if spent >= budget:
                return "out of fuel"
            outcome = step_raw(OP_BY_NAME[name], stack, budget - spent, depth + 1)
            if isinstance(outcome, str):
                return f"{name}: {outcome}"
            stack, cost = outcome
            spent += cost
        return stack, spent

    return run


def macro_op(name: str, expansion: Sequence[str]) -> Op:
    """One macro as an `Op`, costed as its expansion.

    `_capture_under_mark` is False so the machine routes it to `_run`, which
    refuses under an open mark rather than capturing an unresolvable symbol.
    """
    body = tuple(expansion)
    return Op(
        name=name,
        group=MACRO_GROUP,
        effect=f"( -- )  {' '.join(body[:8])}{' ...' if len(body) > 8 else ''}",
        _guard=lambda config: None,
        _apply=lambda config: (config.stack, 1),
        _run=_runner(body),
        _capture_under_mark=False,
    )


@lru_cache(maxsize=1)
def macro_ops() -> tuple[Op, ...]:
    """Every macro in the canonical library, in the file's own order."""
    library = read_library()
    return tuple(macro_op(name, expand(name, library)) for name in library)


@lru_cache(maxsize=1)
def actions() -> tuple[Op, ...]:
    """The base ops followed by the macros: the action space of a macro policy.

    The base ops keep their indices, so a `mark-28` checkpoint's head is a
    prefix of this one's and the two orderings cannot silently disagree.
    """
    return (*OPS, *macro_ops())


def action_names() -> tuple[str, ...]:
    return tuple(op.name for op in actions())


def to_top_level_form(
    program: Sequence[str], library: dict[str, list[str]] | None = None
) -> list[str]:
    """A macro program with in-quotation macros expanded back to base ops.

    The form a top-level-only action space can actually search over: macros
    survive where the authors wrote them at top level and are written out
    wherever they sit inside a quotation.
    """
    library = library if library is not None else read_library()
    out: list[str] = []
    depth = 0
    for token in program:
        if token == "[":
            depth += 1
            out.append(token)
        elif token == "]":
            depth -= 1
            out.append(token)
        elif token in library and depth > 0:
            out.extend(expand(token, library))
        else:
            out.append(token)
    return out
