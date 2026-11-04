"""Fourteen authored lessons for the ``mark-28`` machine.

Every expected output is computed by a small Python reference written
independently of the lesson's solution, so a wrong solution cannot agree with
its own tests by construction. Each lesson shows three examples and keeps three
extra tests back for verification.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

Stack = list[Any]


def _examples(reference: Callable[..., Any], inputs: Sequence[Stack]) -> list[dict]:
    """A reference returns the single result, or a tuple for a whole stack."""
    examples = []
    for stack in inputs:
        result = reference(*stack)
        output = list(result) if isinstance(result, tuple) else [result]
        examples.append({"input": list(stack), "output": output})
    return examples


def _lesson(
    *,
    id: str,  # noqa: A002 - the lesson field's name
    chapter: str,
    title: str,
    concept: str,
    story: str,
    goal: str,
    starter: str,
    solution: str,
    reference: Callable[..., Any],
    shown: Sequence[Stack],
    heldout: Sequence[Stack],
    hints: Sequence[str],
    discovery: str,
    review: str,
    prediction: dict[str, Any],
    macros: Sequence[str] = (),
) -> dict[str, Any]:
    return {
        "id": id,
        "chapter": chapter,
        "title": title,
        "concept": concept,
        "story": story,
        "goal": goal,
        "starter": starter,
        "solution": solution,
        "examples": _examples(reference, shown),
        "heldout": _examples(reference, heldout),
        "hints": list(hints),
        "discovery": discovery,
        "review": review,
        "prediction": prediction,
        "macros": list(macros),
    }


def _abs(n: int) -> int:
    return n if n >= 0 else -n


def _largest(xs: list[int]) -> int:
    best = xs[0]
    for x in xs[1:]:
        best = max(best, x)
    return best


LESSONS: list[dict[str, Any]] = [
    _lesson(
        id="first-light",
        chapter="Signals",
        title="First light",
        concept="A program is a row of ops; each one changes the stack.",
        story=(
            "The lamp-room of Signal House holds a single number on its stack. "
            "The keeper wants the next number along."
        ),
        goal="Add one to the number on the stack.",
        starter="",
        solution="1 add",
        reference=lambda n: n + 1,
        shown=[[3], [0], [41]],
        heldout=[[-5], [100], [7]],
        hints=[
            "Only 0 and 1 are literals; `1` pushes a one onto the stack.",
            "`add` takes the top two values and pushes their sum.",
        ],
        discovery="Pushing a value and then combining is the whole rhythm.",
        review="`1` pushes, `add` pops two and pushes one. Two ops, one signal.",
        prediction={
            "question": "What is on the stack after `1 1 add` from an empty stack?",
            "program": "1 1 add",
            "input": [],
        },
    ),
    _lesson(
        id="echo",
        chapter="Signals",
        title="Echo",
        concept="`dup` copies the top value, so a value can meet itself.",
        story="The signal must be sent twice as strong as it arrived.",
        goal="Double the number on the stack.",
        starter="",
        solution="dup add",
        reference=lambda n: 2 * n,
        shown=[[3], [0], [21]],
        heldout=[[-4], [50], [1]],
        hints=[
            "There is no literal 2, but the number itself is already there.",
            "`dup` copies the top; then add the copy to the original.",
        ],
        discovery="Doubling needs no constant: the value supplies its own partner.",
        review="`dup add` is x + x. Reusing a value is a stack shuffle, not a variable.",
        prediction={
            "question": "What does `dup` leave on a stack holding just 5?",
            "program": "dup",
            "input": [5],
        },
    ),
    _lesson(
        id="turnabout",
        chapter="Shuffles",
        title="Turnabout",
        concept="`swap` exchanges the top two values; order matters to `sub`.",
        story="Two readings arrive, oldest first. The keeper wants newest minus oldest.",
        goal="Given a b (b on top), leave b - a.",
        starter="sub",
        solution="swap sub",
        reference=lambda a, b: b - a,
        shown=[[3, 10], [5, 5], [2, 1]],
        heldout=[[0, 7], [-3, 4], [100, 1]],
        hints=[
            "`sub` computes deeper minus top: `a b sub` is a - b.",
            "Exchange the two readings before subtracting.",
        ],
        discovery="The starter computes a - b; one shuffle flips the sign of the question.",
        review="`swap` reorders without computing. Non-commutative ops care.",
        prediction={
            "question": "What does `swap` do to the stack 1 2?",
            "program": "swap",
            "input": [1, 2],
        },
    ),
    _lesson(
        id="square-pair",
        chapter="Shuffles",
        title="Two squares",
        concept="Combine `dup` and `swap` to work on a buried value.",
        story="The beam's strength is the sum of the squares of its two readings.",
        goal="Given a b, leave a*a + b*b.",
        starter="dup mul",
        solution="dup mul swap dup mul add",
        reference=lambda a, b: a * a + b * b,
        shown=[[3, 4], [0, 2], [1, 1]],
        heldout=[[-3, 5], [10, 0], [6, 7]],
        hints=[
            "`dup mul` squares the top value.",
            "After squaring b, `swap` brings a to the top so you can square it too.",
        ],
        discovery="Work on the top, then swap the next value up: a stack has one hand.",
        review="Shuffles route values to the top, where every op does its work.",
        prediction={
            "question": "What is left after `dup mul swap` on 3 4?",
            "program": "dup mul swap",
            "input": [3, 4],
        },
    ),
    _lesson(
        id="skip-middle",
        chapter="Shuffles",
        title="Skip the middle",
        concept="`drop` discards the top; `over` copies the second value.",
        story="Three readings arrive; the middle one is noise.",
        goal="Given a b c, leave a + c.",
        starter="",
        solution="swap drop add",
        reference=lambda a, b, c: a + c,
        shown=[[1, 99, 2], [5, 0, 5], [10, -7, 3]],
        heldout=[[0, 0, 0], [-4, 8, 4], [20, 1, 22]],
        hints=[
            "Bring b to the top so it can be dropped.",
            "`swap drop` removes the second value and keeps the top.",
        ],
        discovery="Discarding is an op like any other, and the stack forgets for good.",
        review="`drop` removes, `swap` reaches. `over` would copy instead.",
        prediction={
            "question": "What does `over` leave on 1 2?",
            "program": "over",
            "input": [1, 2],
        },
    ),
    _lesson(
        id="sealed-orders",
        chapter="Quotations",
        title="Sealed orders",
        concept="`[ ... ]` builds a quotation as data; `i` runs it.",
        story=(
            "Orders arrive sealed in brackets. Inside a bracket nothing runs: each "
            "op is written down instead. `i` breaks the seal."
        ),
        goal="Triple the number, using a quotation and `i`.",
        starter="[ ] i",
        solution="[ dup dup add add ] i",
        reference=lambda n: 3 * n,
        shown=[[2], [0], [7]],
        heldout=[[-3], [11], [1]],
        hints=[
            "While `[` is open, ops land in the quotation as names.",
            "x + x + x needs two copies of x: `dup dup add add`.",
        ],
        discovery="Watch the stack while the bracket is open: ops become symbols.",
        review="A quotation is a list of op names. `i` runs it on the stack beneath.",
        prediction={
            "question": "What is on the stack after `[ dup add ]` on 4?",
            "program": "[ dup add ]",
            "input": [4],
        },
    ),
    _lesson(
        id="every-lamp",
        chapter="Combinators",
        title="Every lamp",
        concept="`map` runs a quotation on each item of a list.",
        story="Each lamp on the gallery must burn one step brighter.",
        goal="Add one to every number in the list.",
        starter="map",
        solution="[ 1 add ] map",
        reference=lambda xs: [x + 1 for x in xs],
        shown=[[[1, 2, 3]], [[]], [[0, -1]]],
        heldout=[[[10]], [[5, 5, 5, 5]], [[-3, 0, 3]]],
        hints=[
            "`map` wants a list and then a quotation on top.",
            "The quotation sees one item alone and must leave one value.",
        ],
        discovery="The empty list maps to the empty list: no item, no work.",
        review="`map` lifts an op on one value to an op on a whole list.",
        prediction={
            "question": "What does `[ dup mul ] map` leave on (1 2 3)?",
            "program": "[ dup mul ] map",
            "input": [[1, 2, 3]],
        },
    ),
    _lesson(
        id="clear-skies",
        chapter="Combinators",
        title="Clear skies",
        concept="`filter` keeps the items whose quotation leaves true.",
        story="Only positive readings mean clear skies; the rest are discarded.",
        goal="Keep only the numbers greater than zero.",
        starter="[ ] filter",
        solution="[ 0 swap lt? ] filter",
        reference=lambda xs: [x for x in xs if x > 0],
        shown=[[[1, -2, 3]], [[0, 5]], [[-1, -1]]],
        heldout=[[[]], [[7, 0, -7, 8]], [[2]]],
        hints=[
            "`a b lt?` asks whether a < b.",
            "You want 0 < x, so push 0 and swap it underneath.",
        ],
        discovery="Zero is the edge case: `0 lt?` alone would keep negatives instead.",
        review="`filter`'s body is a predicate: one item in, one boolean out.",
        prediction={
            "question": "What does `0 lt?` leave on -3?",
            "program": "0 lt?",
            "input": [-3],
        },
    ),
    _lesson(
        id="tally",
        chapter="Combinators",
        title="The tally",
        concept="`fold` threads an accumulator through a list.",
        story="At dawn the keeper totals the night's readings.",
        goal="Sum the list.",
        starter="[ add ] fold",
        solution="0 [ add ] fold",
        reference=sum,
        shown=[[[1, 2, 3]], [[]], [[10, -10, 5]]],
        heldout=[[[7]], [[1, 1, 1, 1]], [[-5, -6]]],
        hints=[
            "`fold` takes a list, a starting accumulator, then a quotation.",
            "The sum of nothing is 0: push it as the accumulator.",
        ],
        discovery="The starting value is the answer for the empty list.",
        review="`fold`'s body sees acc and item and leaves the new acc.",
        prediction={
            "question": "What does `1 [ mul ] fold` leave on (2 3 4)?",
            "program": "1 [ mul ] fold",
            "input": [[2, 3, 4]],
        },
    ),
    _lesson(
        id="under-the-lens",
        chapter="Quotations",
        title="Under the lens",
        concept="`dip` runs a quotation beneath the top value.",
        story="Adjust the lower reading without disturbing the one on top.",
        goal="Given a b, leave (a + 1) b.",
        starter="dip",
        solution="[ 1 add ] dip",
        reference=lambda a, b: (a + 1, b),
        shown=[[1, 9], [0, 0], [-2, 5]],
        heldout=[[10, 1], [3, -3], [99, 100]],
        hints=[
            "`dip` sets the top value aside, runs the quotation, then restores it.",
            "The quotation sees a on top.",
        ],
        discovery="`dip` is how you reach down without shuffling everything.",
        review="`x [ q ] dip` runs q under x. Shuffle macros like `rot` are built on it.",
        prediction={
            "question": "What does `[ drop ] dip` leave on 1 2?",
            "program": "[ drop ] dip",
            "input": [1, 2],
        },
    ),
    _lesson(
        id="fork-in-the-road",
        chapter="Combinators",
        title="Fork in the road",
        concept="`ifte` runs a condition, then one of two quotations.",
        story="Negative readings are mirrored; the rest pass unchanged.",
        goal="Leave the absolute value of the number.",
        starter="[ ] [ ] [ ] ifte",
        solution="[ dup 0 lt? ] [ 0 swap sub ] [ ] ifte",
        reference=_abs,
        shown=[[-3], [4], [0]],
        heldout=[[-100], [1], [-1]],
        hints=[
            "The condition's result is discarded once read; the branch sees the "
            "original stack.",
            "Negate x as `0 swap sub`.",
        ],
        discovery="An empty quotation is the identity: the else-branch does nothing.",
        review="`[ c ] [ t ] [ f ] ifte`: test, then exactly one branch runs.",
        prediction={
            "question": "What does `[ true ] [ 1 ] [ 0 ] ifte` leave on an empty stack?",
            "program": "[ true ] [ 1 ] [ 0 ] ifte",
            "input": [],
        },
    ),
    _lesson(
        id="false-dawn",
        chapter="Judgement",
        title="False dawn",
        concept="A program that fits the examples is not yet an explanation.",
        story=(
            "The keeper wants the brightest reading of the night. The three logs "
            "you are shown happen to be sorted."
        ),
        goal="Leave the largest number in a non-empty list.",
        starter="0 [ swap drop ] fold",
        solution="uncons [ [ over over lt? ] [ swap drop ] [ drop ] ifte ] fold",
        reference=_largest,
        shown=[[[1, 3, 5]], [[2, 4]], [[0, 7, 9]]],
        heldout=[[[9, 1, 4]], [[-5, -2, -8]], [[6]]],
        hints=[
            "The starter returns the last item. Does that fit? Is it the goal?",
            "Start the fold from the first item: `uncons` leaves tail then head.",
            "Keep the larger of acc and item with an `ifte`.",
        ],
        discovery=(
            "Taking the last item matches every shown example and fails the extra "
            "tests: the examples were sorted, the goal was not 'last'."
        ),
        review="Verify against tests you did not fit to. Agreement is not evidence of intent.",
        prediction={
            "question": "What does the starter leave on (9 1 4)?",
            "program": "0 [ swap drop ] fold",
            "input": [[9, 1, 4]],
        },
    ),
    _lesson(
        id="named-signals",
        chapter="Library",
        title="Named signals",
        concept="A macro is a name for a row of ops; it expands before running.",
        story="The keeper's notebook names common phrases: `sq` squares, `sum` totals.",
        goal="Sum the squares of the list.",
        starter="sum",
        solution="[ sq ] map sum",
        reference=lambda xs: sum(x * x for x in xs),
        shown=[[[1, 2, 3]], [[]], [[-2]]],
        heldout=[[[0, 0]], [[5, 1]], [[3, -3, 3]]],
        hints=[
            "`sq` is `dup mul`; `sum` is `0 [ add ] fold`.",
            "Square every item first, then total.",
        ],
        discovery="Each macro token expands to many primitives; the machine sees only those.",
        review="Macros are spelling, not semantics: they add no power, only names.",
        prediction={
            "question": "What does `sq` leave on 7?",
            "program": "sq",
            "input": [7],
        },
        macros=["sq", "sum"],
    ),
    _lesson(
        id="even-watch",
        chapter="Library",
        title="Even watch",
        concept="Macros compose: predicates, filters and counts from the library.",
        story="The even-numbered watches are counted for the log.",
        goal="Count the even numbers in the list.",
        starter="len",
        solution="[ even? ] filter len",
        reference=lambda xs: sum(1 for x in xs if x % 2 == 0),
        shown=[[[1, 2, 3, 4]], [[]], [[7, 9]]],
        heldout=[[[0]], [[-2, -3, 6, 8]], [[5, 10, 15, 20, 25]]],
        hints=[
            "`even?` leaves true for even numbers.",
            "Filter, then count with `len`.",
        ],
        discovery="`even?` uses `mod`, which floors like Python: -3 is odd.",
        review="A library turns a long program into a short sentence.",
        prediction={
            "question": "What does `even?` leave on 6?",
            "program": "even?",
            "input": [6],
        },
        macros=["even?", "len"],
    ),
]

BY_ID: dict[str, dict[str, Any]] = {lesson["id"]: lesson for lesson in LESSONS}
