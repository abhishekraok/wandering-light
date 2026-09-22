"""inner-frames (#160) E1: can a search that commits to the consuming combinator
execute a partial quotation body on concrete data, and how often?

witness-trajectory-shape (#150) found 84% of v4 witness ops sit under an open
mark, where the machine state does not move and a ranker over states has
nothing to look at. #160 proposes splitting `[` into tagged variants
(`[map [filter [fold [ifte [dip [i`) so the search commits to a consumer when
it opens a quotation and can run the partial body on the frame that consumer
will feed it. E1 asks only whether that frame is **computable**, on the data a
search would actually have. It trains nothing and cannot say the frames help.

**The frame chain, not a depth counter.** A depth-1 frame comes from the real
stack below the mark. A depth-2 frame does not: the inner body runs on whatever
the *enclosing* body's prefix leaves on the enclosing frame, so the frames form
a chain and a deep frame is undefined the moment any ancestor's prefix refuses.
This module carries one virtual stack per open mark and advances it by one item
per written op, which is the incremental form of that chain (and O(n) rather
than O(n^2) in witness length). The prior text-level measurement asked whether
a quotation's consumer is the next token; nothing had executed a depth-2 frame.

**All k shown examples, in lockstep.** `stack_beam` applies an op to every
shown example and drops the candidate if any one refuses, so a frame defined on
one example is not a frame the search has. Coverage is therefore reported over
the **product**: defined on every example. The per-example rate is a diagnostic
of what the product costs, not a second definition. Consumers are attributed
per example and must agree, because an `ifte` branch can run on some examples
and not others.

    uv run python experiments/inner_frames.py --out reports/inner-frames-20260922

Corpora: `stack_pbe_v4` and `stack_pbe_world_v3` (train and eval reported
apart), plus the 143 staged witnesses of staged-witness-reauthoring (#155),
less the 12 rows that issue's own caveat says to treat as untreated. The unit
is the **distinct witness program**, each replayed on one task's shown inputs
(the first in corpus order), so the numbers describe witnesses, not the task
cohort.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from wandering_light import stack_machine  # noqa: E402
from wandering_light.corpus_hub import fetch_corpus  # noqa: E402
from wandering_light.stack_machine import (  # noqa: E402
    DEFAULT_FUEL_LIMIT,
    Bool,
    Int,
    Lst,
    Mark,
    Sym,
    parse_stack,
    quotation_open,
    run_body,
    step_raw,
)
from wandering_light.stack_pbe import BASIS, OP_BY_NAME, load  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import Sequence

    from wandering_light.stack_machine import Value
    from wandering_light.stack_pbe import Task

DATA = REPO / "wandering_light" / "evals" / "data"
STAGED = REPO / "experiments" / "staged_witnesses" / "v4.jsonl"

# Arity in quotations, from the ops table: `ifte` consumes three, the rest one.
COMBINATORS = {"i": 1, "dip": 1, "map": 1, "filter": 1, "fold": 1, "ifte": 3}
TAGS = tuple(COMBINATORS)
# The tag of a quotation with no attributable consumer. Deliberately not a
# combinator name, so `by_tag` cannot confuse it with one.
NO_TAG = "<no consumer>"
# A body previewed under a combinator gets the combinator's own budget in the
# machine; 2,000 is what the brainstorm's probe used and is far above any
# witness body's cost, so a refusal here is a refusal of the body, not of fuel.
BUDGET = 2_000
# `run_body`'s depth argument as the real combinators pass it: `step_raw` is
# called at depth 0 on a witness, and `_run_map` and friends pass `depth + 1`.
PREVIEW_DEPTH = 1


# ---------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------


def frame_of(tag: str, below: tuple, role: int) -> tuple | str:
    """The stack a `tag` body is first invoked on, given the stack below its mark.

    Read straight off the combinator implementations in `stack_machine.py`,
    where the quotation itself is on top and `below` is everything under it:

    - `i` (`_run_i:508`) runs the body on `stack[:-1]`, which is `below`;
    - `dip` (`_run_dip:519`) on `stack[:-2]`, so `below[:-1]`, keeping `below[-1]`;
    - `map` / `filter` (`_run_map:551`, `_run_filter:570`) need `stack[-2]` to be
      a list, so `below[-1]`, and run the body on `(item,)` -- a **fresh
      one-value stack**, which is why the design's recursive claim is plausible;
    - `fold` (`_run_fold:592`) needs `stack[-3]` to be a list, so `below[-2]`,
      with the accumulator at `below[-1]`, and runs on `(acc, item)`;
    - `ifte` (`_run_ifte:613`) takes three quotations and runs each on
      `stack[:-3]`, so the body at role r (0 condition, 1 then, 2 else) sees
      `below[:len(below) - r]`: the quotations already written sit above it.

    A string is the reason no frame exists. `"empty"` is its own answer: the
    frame is well formed but the list has no items, so there is nothing to run
    the body on -- the search would have no data even though the tag is legal.
    """
    if tag == NO_TAG:
        return "no consumer to commit to"
    if tag == "i":
        return below
    if tag == "dip":
        if not below:
            return "dip needs a value under the quotation"
        return below[:-1]
    if tag in ("map", "filter"):
        if not below or not isinstance(below[-1], Lst):
            return f"{tag} needs a list under the quotation"
        if not below[-1].items:
            return "empty"
        return (below[-1].items[0],)
    if tag == "fold":
        if len(below) < 2 or not isinstance(below[-2], Lst):
            return "fold needs a list and an accumulator under the quotation"
        if not below[-2].items:
            return "empty"
        return (below[-1], below[-2].items[0])
    if tag == "ifte":
        if role and len(below) < role:
            return "ifte role deeper than the stack"
        base = below[: len(below) - role] if role else below
        return base
    raise ValueError(f"not a combinator tag: {tag}")


def frame_with_captures(
    tag: str, below: tuple, role: int, captures: int
) -> tuple | str:
    """The frame when the quotation is consed into a closure before it runs.

    `x [ lt? ] swap cons filter` leaves `[ x lt? ]` for `filter`, so the body
    the author wrote (`lt?`) runs on the combinator's frame **with the captured
    values already pushed**: `(item, x)`. The captured values sit under the
    mark when `[` opens, so the combinator's own operands are found below them.
    Each `swap cons` prepends one value, and the pair runs innermost-first, so
    two captures over `(lst, x, y)` build `[ x y body ]`: the body sees `x` then
    `y`, which is `below[-captures:]` in stack order and **not** reversed. The
    machine check below settles this -- a reversed slice matches at one capture,
    where reversing is a no-op, and mismatches at every capture of two or more.
    """
    if captures <= 0:
        return frame_of(tag, below, role)
    if len(below) < captures:
        return "capture deeper than the stack"
    base = frame_of(tag, below[:-captures], role)
    if isinstance(base, str):
        return base
    return (*base, *below[-captures:])


def legal_tags(below: tuple) -> tuple[str, ...]:
    """The tags whose precondition holds on ``below`` at `[` time.

    This is what a search could rule out from the state alone, which is what
    the issue's `ln 6` branching cost should be measured against. `ifte` is
    counted in its **condition** role, the one a search commits to first:
    nothing about the state forbids starting an `ifte` condition, so `i` and
    `ifte` are legal everywhere and the floor on ambiguity is two, not one.
    """
    return tuple(
        tag for tag in TAGS if not isinstance(frame_of(tag, below, 0), str)
    ) + tuple(
        tag
        for tag in TAGS
        if isinstance(frame_of(tag, below, 0), str)
        and frame_of(tag, below, 0) == "empty"
    )


def shape_of(below: tuple) -> str:
    """A cheap descriptor of the stack under the mark, for H(tag | shape).

    The peer session's `lst_top` / `lst_second` / `other` split is the coarse
    version of this; the top three types plus list emptiness is as much as a
    tag could plausibly be predicted from without learning.
    """

    def kind(value: Value) -> str:
        if isinstance(value, Lst):
            return (
                "quot"
                if any(isinstance(i, Sym) for i in value.items)
                else ("lst" if value.items else "nil")
            )
        if isinstance(value, Int):
            return "int"
        if isinstance(value, Bool):
            return "bool"
        if isinstance(value, Sym):
            return "sym"
        if isinstance(value, Mark):
            return "mark"
        return "other"

    top = [kind(v) for v in below[-3:]]
    return f"{len(below)}:{'.'.join(top)}"


def entropy(counts: Counter) -> float:
    total = sum(counts.values())
    if not total:
        return 0.0
    return -sum((n / total) * math.log(n / total) for n in counts.values() if n)


def marginal_tags(joint: Counter) -> Counter:
    """Tag counts summed over shapes.

    A dict comprehension over ``joint`` would keep only the last shape's count
    for each tag, making H(tag) both wrong and dependent on iteration order,
    and the comparison with H(tag | shape) meaningless.
    """
    out: Counter = Counter()
    for (_shape, tag), n in joint.items():
        out[tag] += n
    return out


def conditional_entropy(joint: Counter) -> float:
    """H(tag | shape) in nats, from a counter over (shape, tag)."""
    by_shape: dict[str, Counter] = defaultdict(Counter)
    for (shape, tag), n in joint.items():
        by_shape[shape][tag] += n
    total = sum(joint.values())
    if not total:
        return 0.0
    return sum(
        (sum(tags.values()) / total) * entropy(tags) for tags in by_shape.values()
    )


# ---------------------------------------------------------------------------
# Replay and hindsight attribution
# ---------------------------------------------------------------------------


@contextmanager
def recording_bodies():
    """Record every ``run_body`` call the machine makes, with the stack it passes.

    `_run_map` and its siblings call the module-global `run_body`, so replacing
    that global observes the machine rather than re-deriving it. The wrapper is
    outside the `lru_cache`, so a repeated body is still recorded. This module's
    own preview calls bind `run_body` directly at import and are unaffected.
    """
    calls: list[tuple[int, tuple, tuple]] = []
    original = stack_machine.run_body

    def recorder(body, stack, budget, depth=0):
        calls.append((depth, body, stack))
        return original(body, stack, budget, depth)

    stack_machine.run_body = recorder
    try:
        yield calls
    finally:
        stack_machine.run_body = original


def observed_frames(program: Sequence[str], root: tuple) -> dict[int, tuple]:
    """``{step of a combinator: the stack it first handed a body}``.

    For `map`, `filter` and `fold` that is the fresh one- or two-value stack of
    the first item; for `i` and `dip` the outer stack less the quotations; for
    `ifte` the `rest` that all three of its quotations are run on. An empty list
    makes no call at all and simply does not appear.
    """
    out: dict[int, tuple] = {}
    stack = root
    with recording_bodies() as calls:
        for step, name in enumerate(program):
            calls.clear()
            outcome = step_raw(OP_BY_NAME[name], stack, DEFAULT_FUEL_LIMIT, 0)
            if isinstance(outcome, str):
                return out
            stack = outcome[0]
            first = next((c for c in calls if c[0] == 1), None)
            if first is not None:
                out[step] = (first[1], first[2])  # (body, stack)
    return out


def replay(program: Sequence[str], root: tuple) -> list[tuple] | None:
    """Every state along the witness, ``s_0`` to ``s_n``, or None if it refuses."""
    states, stack = [root], root
    for name in program:
        out = step_raw(OP_BY_NAME[name], stack, DEFAULT_FUEL_LIMIT, 0)
        if isinstance(out, str):
            return None
        stack = out[0]
        states.append(stack)
    return states


def marks(stack: tuple) -> list[int]:
    return [i for i, value in enumerate(stack) if isinstance(value, Mark)]


def consumer_of(
    states: Sequence[tuple], program: Sequence[str], close_step: int, m: int
) -> tuple[str, int | None]:
    """Which combinator consumes the quotation closed at ``close_step``, in hindsight.

    Follows the finished quotation sitting at index ``m`` through the later
    states: while it stays there, the first combinator that takes it as one of
    its top ``arity`` values is the consumer, and the returned int is its role
    (0 for a one-quotation combinator; 0/1/2 for `ifte`'s condition/then/else).
    The three other outcomes are the cases a committed search cannot preview,
    and E1 counts them rather than hiding them:

    - ``mutated_or_folded``: the value at ``m`` changes, so the quotation was
      composed (`cat`) or consumed as data rather than run;
    - ``deeper``: a combinator runs while the quotation is buried below its
      arity, so this quotation is not the one being consumed;
    - ``unused``: the witness ends with the quotation still on the stack.
    """
    quot = states[close_step + 1][m]
    for k in range(close_step + 1, len(program)):
        stack = states[k]
        if len(stack) <= m or stack[m] != quot:
            return "mutated_or_folded", None
        name = program[k]
        if name in COMBINATORS and not quotation_open(stack):
            arity = COMBINATORS[name]
            if m >= len(stack) - arity:
                return name, m - (len(stack) - arity)
            return "deeper", None
    return "unused", None


def attribute(program: Sequence[str]) -> dict[int, tuple]:
    """``{step that opened a quotation: (kind, role)}``, by bracket structure.

    **Structural, not value-tracking, and that is the whole point.** Following a
    finished quotation through later stack states, which is what
    `consumer_of` does, can only ever attribute a *top-level* quotation: an
    inner one is absorbed into its parent when the parent's `]` collapses the
    stack, and its consuming combinator is a token that never runs as a
    top-level op, so value-tracking reports every nested quotation as
    `mutated_or_folded`. Since #160 is a claim about the 84% of ops that sit
    inside a body, attribution has to work at depth, so it is done on brackets:
    the consumer is the first token after the matching `]` that is not a
    sibling quotation.

    `ifte` takes three quotations, so a quotation that skips ``s`` siblings
    before reaching it sits at role ``arity - 1 - s`` (condition, then, else).
    `consumer_of` is kept as the depth-1 cross-check.
    """
    out: dict[int, tuple] = {}
    match: dict[int, int] = {}
    stack: list[int] = []
    for step, name in enumerate(program):
        if name == "[":
            stack.append(step)
        elif name == "]":
            if not stack:
                continue
            match[stack.pop()] = step
    for opened in stack:
        out[opened] = ("never_closed", None, 0, None)

    for opened, closed in match.items():
        j, skipped = closed + 1, 0
        while j < len(program) and program[j] == "[":
            if j not in match:
                break
            j = match[j] + 1
            skipped += 1
        # The closure idiom: `x [ body ] swap cons filter` conses a captured
        # value into the quotation before the combinator runs it, so the
        # consumer is not the next token although the body is still executed
        # promptly. Count the captures; the frame gains those values.
        captures = 0
        while (
            j + 1 < len(program) and program[j] == "swap" and program[j + 1] == "cons"
        ):
            j += 2
            captures += 1
        if j >= len(program):
            out[opened] = ("unused", None, 0, None)
        elif program[j] == "]":
            out[opened] = ("left_on_stack", None, 0, None)
        elif program[j] in COMBINATORS:
            arity = COMBINATORS[program[j]]
            role = arity - 1 - skipped
            out[opened] = (
                (program[j], role, captures, j)
                if 0 <= role < arity
                else ("role_mismatch", None, 0, None)
            )
        else:
            out[opened] = ("other_op_first", None, 0, None)
    return out


def crosscheck(
    states: Sequence[tuple], program: Sequence[str], structural: dict[int, tuple]
) -> Counter:
    """Agreement between the structural tag and the replayed one, at depth 1 only.

    Where both are defined they should name the same combinator; a disagreement
    means a quotation was composed or stored rather than run promptly, which is
    exactly the case a committed search would guess wrong.
    """
    out = Counter()
    open_at: list[tuple[int, int]] = []
    for step, name in enumerate(program):
        after = states[step + 1]
        if name == "[":
            open_at.append((step, max(marks(after))))
        elif name == "]" and open_at:
            opened, m = open_at.pop()
            if open_at:  # depth >= 2: value-tracking cannot speak
                continue
            replayed, _role = consumer_of(states, program, step, m)
            structural_kind = structural.get(opened, ("missing", None))[0]
            out["compared"] += 1
            out["agree" if replayed == structural_kind else "disagree"] += 1
            if replayed != structural_kind:
                out[f"{structural_kind}->{replayed}"] += 1
    return out


# ---------------------------------------------------------------------------
# The walk: one virtual stack per open mark, advanced one item per op
# ---------------------------------------------------------------------------


class Level:
    """One open quotation: its tag, the frame its body runs on, and its progress."""

    __slots__ = (
        "below",
        "body",
        "captures",
        "frame",
        "refusal",
        "role",
        "tag",
        "virtual",
    )

    def __init__(self, tag: str, role: int, below: tuple, frame: tuple | str):
        self.tag = tag
        self.role = role
        self.below = below
        self.captures = 0
        self.body: list[Value] = []
        if isinstance(frame, str):
            self.frame = ()
            self.virtual: tuple | None = None
            self.refusal = "empty_list" if frame == "empty" else "no_frame"
        else:
            self.frame = frame
            self.virtual = frame
            self.refusal = None

    def write(self, item: Value) -> None:
        """Append one item to the body and advance the virtual execution."""
        self.body.append(item)
        if self.virtual is None:
            return
        out = run_body((item,), self.virtual, BUDGET, PREVIEW_DEPTH)
        if isinstance(out, str):
            self.virtual = None
            self.refusal = "body_refused"
        else:
            self.virtual = out[0]


def item_of(value: Value) -> Value:
    """A written item as `run_body` reads it: a `Sym` is an op, anything else a push."""
    return value


def walk_one_example(
    program: Sequence[str],
    states: Sequence[tuple],
    tags: dict[int, tuple],
    *,
    capture_aware: bool = False,
) -> dict[str, Any]:
    """Per under-mark op: is a frame defined, at what depth, under what parent.

    Returns lists indexed by witness step, so the caller can intersect them
    across examples and get the product the beam would see.
    """
    defined: dict[int, bool] = {}
    reason: dict[int, str] = {}
    depth_at: dict[int, int] = {}
    parent_at: dict[int, str | None] = {}
    tag_at: dict[int, str] = {}
    capture_at: dict[int, int] = {}
    computed_at: dict[int, tuple] = {}
    legal_at: dict[int, tuple[str, ...]] = {}
    shape_at: dict[int, str] = {}
    fold_steps: dict[int, dict[str, Any]] = {}
    # What each open quotation looks like just before the step's op: its tag,
    # role, and the frame its body has reached (None where undefined), outermost
    # first. This is what a frame-conditioned policy is shown (#160 E2).
    views: dict[int, tuple[tuple[str, int, tuple | None], ...]] = {}
    open_levels: list[Level] = []

    for step, name in enumerate(program):
        before = states[step]
        under_mark = name == "[" or quotation_open(before)
        if not under_mark:
            continue
        views[step] = tuple((lv.tag, lv.role, lv.virtual) for lv in open_levels)
        if name == "[":
            kind, role, captures, _consumer = tags.get(
                step, ("never_closed", None, 0, None)
            )
            # `attribute` always looks through the closure idiom, so strict mode
            # -- the design exactly as #160 writes it, where the consumer is the
            # next token -- has to put those quotations back in the undefined
            # class rather than silently getting the capture-aware answer.
            if captures and not capture_aware:
                kind, role = "other_op_first", None
            parent = open_levels[-1] if open_levels else None
            below = before[:] if parent is None else parent.virtual
            depth = len(open_levels) + 1
            depth_at[step] = depth
            parent_at[step] = parent.tag if parent is not None else None
            tag_at[step] = kind
            # Ambiguity is measured wherever a search would actually choose a
            # tag, which is every `[`, on the stack it would have there: the
            # real one at depth 1, the parent's virtual one below.
            if below is not None:
                legal_at[step] = legal_tags(below)
                shape_at[step] = shape_of(below)
            if kind not in COMBINATORS:
                # No consumer to commit to, so there is no tag. The placeholder
                # must NOT be a real combinator name: ops written inside this
                # quotation take their level's tag, and naming it `i` made every
                # such op look like an `i` body that failed, which showed up as
                # `i` having coverage exactly zero everywhere.
                level = Level(NO_TAG, 0, (), "no_consumer")
                level.refusal = kind
                open_levels.append(level)
                defined[step] = False
                reason[step] = kind
                continue
            if below is None:
                level = Level(kind, role or 0, (), "parent_refused")
                level.refusal = "parent_refused"
                open_levels.append(level)
                defined[step] = False
                reason[step] = "parent_refused"
                continue
            frame = (
                frame_with_captures(kind, below, role or 0, captures)
                if capture_aware
                else frame_of(kind, below, role or 0)
            )
            level = Level(kind, role or 0, below, frame)
            level.captures = captures
            open_levels.append(level)
            capture_at[step] = captures
            if not isinstance(frame, str):
                computed_at[step] = (kind, role or 0, captures, _consumer, frame)
            defined[step] = level.virtual is not None
            if level.refusal:
                reason[step] = level.refusal
            continue

        if not open_levels:
            continue
        level = open_levels[-1]
        depth_at[step] = len(open_levels)
        parent_at[step] = open_levels[-2].tag if len(open_levels) > 1 else None
        tag_at[step] = level.tag

        if name == "]":
            # The finished quotation becomes a value in the enclosing body.
            closed = open_levels.pop()
            if closed.tag == "fold" and closed.virtual is not None:
                fold_steps.setdefault(step, {})["completed"] = {
                    "body": tuple(closed.body),
                    "below": closed.below,
                }
            if open_levels:
                parent = open_levels[-1]
                # The finished quotation is a value **in the parent's body**, not
                # only on its virtual stack: leaving it out of `parent.body`
                # rebuilds the grandparent's body without it, so a nested
                # program like `[ [ [ 1 add ] map ] map ] map` is reported as
                # refusing when it runs perfectly well.
                quotation = Lst(tuple(closed.body))
                parent.body.append(quotation)
                if parent.virtual is not None:
                    parent.virtual = (*parent.virtual, quotation)
            defined[step] = closed.virtual is not None
            if closed.refusal:
                reason[step] = closed.refusal
            continue

        # Take the captured item from the machine rather than rebuilding it from
        # the op name. A literal is NOT captured as a symbol -- `_capture_under_mark`
        # is false for it, so `1` under a mark pushes `Int(1)` and the quotation
        # holds a value, not `Sym("1")`. Rebuilding it as a symbol runs the same
        # way but stores the wrong thing, so a body that `uncons`es a nested
        # quotation then sees a symbol where the real witness sees an integer and
        # refuses. Under a mark an op appends exactly one item, which is it.
        after = states[step + 1]
        if len(after) == len(before) + 1:
            item = after[-1]
        elif name in OP_BY_NAME:
            item = Sym(name)
        else:
            raise ValueError(f"unknown op in witness: {name}")
        level.write(item)
        defined[step] = level.virtual is not None
        if level.refusal:
            reason[step] = level.refusal
        if level.tag == "fold" and level.virtual is not None:
            fold_steps.setdefault(step, {})["write"] = {
                "body": tuple(level.body),
                "below": level.below,
            }

    return {
        "defined": defined,
        "reason": reason,
        "depth": depth_at,
        "parent": parent_at,
        "tag": tag_at,
        "legal": legal_at,
        "shape": shape_at,
        "fold": fold_steps,
        "captures": capture_at,
        "computed": computed_at,
        "views": views,
    }


def check_against_machine(
    computed: dict[int, tuple], observed: dict[int, tuple]
) -> Counter:
    """Does the frame this module computed equal the one the machine handed the body?

    Non-circular: `observed` comes from recording the machine's own `run_body`
    calls, not from re-applying the rules in `frame_of`. Under the closure idiom
    the two are not literally equal and should not be -- the machine runs the
    *consed* body `x lt?` on `(item,)` while this module runs the *written* body
    `lt?` on `(item, x)` -- so the captured literals are run onto the machine's
    stack first and the comparison is made after that. That turns the reading of
    `swap cons`, and in particular the untested 2+ capture case, into something
    the machine adjudicates.
    """
    out = Counter()
    for tag, _role, captures, consumer, frame in computed.values():
        if consumer is None or consumer not in observed:
            out["no_observation"] += 1
            continue
        body, stack = observed[consumer]
        if captures:
            if len(body) < captures:
                out["capture_longer_than_body"] += 1
                continue
            advanced = run_body(tuple(body[:captures]), stack, BUDGET, PREVIEW_DEPTH)
            if isinstance(advanced, str):
                out["capture_prefix_refused"] += 1
                continue
            stack = advanced[0]
        out["compared"] += 1
        out[f"captures_{min(captures, 2)}"] += 1
        if frame == stack:
            out["match"] += 1
        else:
            out["mismatch"] += 1
            out[f"mismatch_{tag}_c{captures}"] += 1
    return out


def fold_disagreement(body: tuple, below: tuple) -> str | None:
    """Does the accept set of a partial fold body differ across the list's items?

    The frame is `(acc, item_0)`, so a body that refuses on item 0 but not on
    item 3 is a case where committing to the first item changes what the beam
    keeps. This is the only form of the unchained-fold objection that can move
    a search: chaining the partial body (feeding iteration 1's output as the
    next accumulator) refuses at once on arity, so no better mid-body frame
    exists. Returns None when the items agree, otherwise a label.
    """
    if len(below) < 2 or not isinstance(below[-2], Lst):
        return None
    items = below[-2].items
    acc = below[-1]
    if len(items) < 2:
        return None
    verdicts = {
        not isinstance(run_body(body, (acc, item), BUDGET, PREVIEW_DEPTH), str)
        for item in items
    }
    if len(verdicts) == 1:
        return None
    return "accept_set_differs"


def fold_terminal_shape(
    body: tuple, below: tuple, closed_state: tuple | None
) -> str | None:
    """Does the first-item preview leave the type the completed fold ends on?

    Secondary to the accept-set number: this is the gap between what a ranker
    scored mid-body and what it gets when the combinator finally runs.
    """
    if closed_state is None or len(below) < 2 or not isinstance(below[-2], Lst):
        return None
    items = below[-2].items
    if not items:
        return None
    out = run_body(body, (below[-1], items[0]), BUDGET, PREVIEW_DEPTH)
    if isinstance(out, str) or not out[0] or not closed_state:
        return None
    return "agree" if type(out[0][-1]) is type(closed_state[-1]) else "differ"


# ---------------------------------------------------------------------------
# Aggregation over a corpus
# ---------------------------------------------------------------------------


def witnesses(
    tasks: Sequence[Task], split: str, programs: dict[str, str] | None = None
) -> list[tuple[str, tuple[str, ...], tuple[str, ...]]]:
    """One entry per distinct program: (behaviour, program, shown input texts).

    ``programs`` replaces a behaviour's witness with another text, which is how
    the staged corpus is measured against the same machinery.
    """
    out = []
    seen: set[tuple[str, ...]] = set()
    for task in tasks:
        if task.split != split:
            continue
        program = task.program
        if programs is not None:
            replacement = programs.get(task.behaviour)
            if replacement is None:
                continue
            program = tuple(replacement.split())
        if program in seen:
            continue
        seen.add(program)
        out.append((task.behaviour, program, task.inputs))
    return out


def analyse(
    rows: Sequence[tuple[str, tuple[str, ...], tuple[str, ...]]],
    log,
    *,
    capture_aware: bool = False,
) -> dict[str, Any]:
    stats = Counter()
    reasons = Counter()
    by_depth: dict[int, Counter] = defaultdict(Counter)
    by_parent: dict[str, Counter] = defaultdict(Counter)
    by_tag: dict[str, Counter] = defaultdict(Counter)
    tag_counts = Counter()
    legal_counts = Counter()
    shape_tag = Counter()
    fold_disagree = Counter()
    fold_shape = Counter()
    checks = Counter()
    oracle = Counter()
    per_example_defined = 0
    per_example_total = 0
    skipped = Counter()

    for behaviour, program, inputs in rows:
        walks = []
        for text in inputs:
            root = parse_stack(text, basis=BASIS)
            states = replay(program, root)
            if states is None:
                skipped[f"{behaviour}:refused"] += 1
                walks = []
                break
            tags = attribute(program)
            checks += crosscheck(states, program, tags)
            if not walks:  # example 0 carries the machine comparison
                seen_frames = observed_frames(program, root)
            walks.append(
                walk_one_example(program, states, tags, capture_aware=capture_aware)
            )
        if not walks:
            continue
        stats["programs"] += 1
        stats["all_ops"] += len(program)
        oracle += check_against_machine(walks[0]["computed"], seen_frames)

        steps = sorted({s for w in walks for s in w["defined"]})
        for step in steps:
            stats["under_mark_ops"] += 1
            per_example_total += len(walks)
            per_example_defined += sum(bool(w["defined"].get(step)) for w in walks)
            # The beam's rule: defined only if defined on every example, and the
            # consumer attributions must agree, or the tag itself is not one
            # thing across the product.
            tags = {w["tag"].get(step) for w in walks}
            product = all(w["defined"].get(step) for w in walks) and len(tags) == 1
            depth = walks[0]["depth"].get(step, 0)
            parent = walks[0]["parent"].get(step)
            by_depth[min(depth, 3)]["total"] += 1
            by_depth[min(depth, 3)]["defined"] += product
            if depth > 1:
                kind = "fresh" if parent in ("map", "filter", "fold") else "inherited"
                by_parent[kind]["total"] += 1
                by_parent[kind]["defined"] += product
            if len(tags) == 1:
                tag = next(iter(tags))
                if tag in COMBINATORS:
                    by_tag[tag]["total"] += 1
                    by_tag[tag]["defined"] += product
            if product:
                stats["defined_product"] += 1
                if len(tags) == 1:
                    tag_counts[next(iter(tags))] += 1
            elif len(tags) != 1:
                reasons["consumer_disagrees_across_examples"] += 1
            else:
                for w in walks:
                    if not w["defined"].get(step):
                        reasons[w["reason"].get(step, "unknown")] += 1
                        break

        for w in walks[:1]:
            for step, legal in w["legal"].items():
                legal_counts[len(legal)] += 1
                tag = w["tag"].get(step)
                if tag in COMBINATORS:
                    shape_tag[w["shape"][step], tag] += 1

        # Unchained fold: measured on example 0, the frame's own reference point.
        first = walks[0]
        for kinds in first["fold"].values():
            if "write" in kinds:
                fold_disagree["steps"] += 1
                verdict = fold_disagreement(
                    kinds["write"]["body"], kinds["write"]["below"]
                )
                if verdict:
                    fold_disagree[verdict] += 1
            if "completed" in kinds:
                fold_shape["bodies"] += 1

    defined = stats["defined_product"]
    under = stats["under_mark_ops"]
    return {
        "programs": stats["programs"],
        "all_ops": stats["all_ops"],
        "under_mark_ops": under,
        "under_mark_share_of_ops": round(under / stats["all_ops"], 4)
        if stats["all_ops"]
        else 0.0,
        "frame_coverage_product": round(defined / under, 4) if under else 0.0,
        "frame_coverage_per_example": round(per_example_defined / per_example_total, 4)
        if per_example_total
        else 0.0,
        "by_depth": {
            f"depth_{d}" if d < 3 else "depth_3+": {
                "under_mark_ops": c["total"],
                "share": round(c["total"] / under, 4) if under else 0.0,
                "coverage": round(c["defined"] / c["total"], 4) if c["total"] else 0.0,
            }
            for d, c in sorted(by_depth.items())
        },
        "by_tag": {
            tag: {
                "under_mark_ops": c["total"],
                "coverage": round(c["defined"] / c["total"], 4) if c["total"] else 0.0,
                "frame_kind": "fresh"
                if tag in ("map", "filter", "fold")
                else "inherited",
            }
            for tag, c in sorted(by_tag.items(), key=lambda kv: -kv[1]["total"])
        },
        "by_parent_kind_depth_2plus": {
            kind: {
                "under_mark_ops": c["total"],
                "coverage": round(c["defined"] / c["total"], 4) if c["total"] else 0.0,
            }
            for kind, c in sorted(by_parent.items())
        },
        "undefined_reasons": dict(reasons.most_common()),
        "tag_distribution": dict(tag_counts.most_common()),
        "legal_tags_per_open": {
            "histogram": {str(k): v for k, v in sorted(legal_counts.items())},
            "mean": round(
                sum(k * v for k, v in legal_counts.items())
                / sum(legal_counts.values()),
                3,
            )
            if legal_counts
            else 0.0,
        },
        "ambiguity_nats": {
            "issue_assumption_ln6": round(math.log(6), 3),
            "ln_mean_legal_tags": round(
                math.log(
                    sum(k * v for k, v in legal_counts.items())
                    / sum(legal_counts.values())
                ),
                3,
            )
            if legal_counts
            else 0.0,
            "entropy_tag": round(entropy(marginal_tags(shape_tag)), 3),
            "entropy_tag_given_shape": round(conditional_entropy(shape_tag), 3),
        },
        "attribution_crosscheck_depth1": dict(checks.most_common()),
        "frames_checked_against_the_machine": dict(oracle.most_common()),
        "unchained_fold": {
            "write_steps_with_a_frame": fold_disagree["steps"],
            "accept_set_differs": fold_disagree["accept_set_differs"],
            "rate": round(
                fold_disagree["accept_set_differs"] / fold_disagree["steps"], 4
            )
            if fold_disagree["steps"]
            else 0.0,
            "completed_bodies": fold_shape["bodies"],
        },
        "skipped_programs": len(skipped),
    }


# The staging rules of #155, as its own report states them: no quotation over 8
# atoms, no write run over 12 steps, at most 4 top-level non-write steps between
# a quotation's `]` and the combinator that runs it.
STAGING_RULES = {"largest_quotation": 8, "longest_write_run": 12, "longest_park": 4}


def fused_within_rules(row: dict[str, Any]) -> bool:
    """Was this behaviour's **fused** witness already within the staging rules?

    #155's classifier accepts a rewrite when it is within the rules **and**
    either it adds two post-execution states or the fused witness broke a rule.
    So a row whose fused witness broke nothing was accepted only through the
    `+2 executions` branch, and that issue's own caveat says to treat those as
    untreated. The file has no field for it -- `broken_rules` is empty on all
    143, because it records the staged program's rules, not the fused one's --
    so it is derived from the recorded fused metrics, which reproduces the
    caveat's count of 12 exactly, `abs_diff_pair` (its named type case) among
    them.
    """
    fused = row["fused"]
    return all(fused[key] <= limit for key, limit in STAGING_RULES.items())


def staged_programs(keep_plus_two: bool = False) -> dict[str, str]:
    """The #155 staged witnesses, by behaviour, less contrived and +2-only rows."""
    out: dict[str, str] = {}
    for line in STAGED.read_text().splitlines():
        row = json.loads(line)
        if row["verdict"] != "staged":
            continue
        if any(flag.startswith("contrived") for flag in row["flags"]):
            continue
        if not keep_plus_two and fused_within_rules(row):
            continue
        out[row["behaviour"]] = row["program"]
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--corpora",
        default="stack_pbe_v4,stack_pbe_world_v3",
        help="manifest directory names under wandering_light/evals/data",
    )
    parser.add_argument("--splits", default="train,eval")
    parser.add_argument(
        "--limit", type=int, default=0, help="first N distinct witnesses; 0 is all"
    )
    args = parser.parse_args(argv)

    def log(line: str) -> None:
        print(line, flush=True)

    report: dict[str, Any] = {"cells": {}}
    started = time.perf_counter()
    for name in args.corpora.split(","):
        tasks = load(fetch_corpus(DATA / name / "manifest.json"))
        for split in args.splits.split(","):
            rows = witnesses(tasks, split)
            if args.limit:
                rows = rows[: args.limit]
            if not rows:
                continue
            cell = f"{name}:{split}"
            log(f"== {cell}: {len(rows)} distinct witnesses")
            report["cells"][cell] = analyse(rows, log)
            report["cells"][f"{cell}+capture"] = analyse(rows, log, capture_aware=True)
            log(json.dumps(report["cells"][cell], indent=1))
        if name == "stack_pbe_v4":
            programs = staged_programs()
            rows = witnesses(tasks, "train", programs)
            if args.limit:
                rows = rows[: args.limit]
            log(f"== staged:train: {len(rows)} distinct staged witnesses")
            report["cells"]["staged:train"] = analyse(rows, log)
            report["cells"]["staged:train+capture"] = analyse(
                rows, log, capture_aware=True
            )
            log(json.dumps(report["cells"]["staged:train"], indent=1))
    report["seconds"] = round(time.perf_counter() - started, 1)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "frames.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
        log(f"wrote {args.out / 'frames.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
