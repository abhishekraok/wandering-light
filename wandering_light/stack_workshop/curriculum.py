"""Fourteen authored lessons that walk a learner through the `mark-28` machine.

Every expected output is computed by an independent Python reference, never
by running the lesson's own solution: a solution that disagreed with its
reference would fail its own lesson rather than quietly redefine it. Three
examples are shown; three more are held out, which is how a later lesson can
show that a program fitting the shown examples need not be the explanation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

Example = dict[str, list]


def _examples(reference: Callable[[list], list], inputs: list[list]) -> list[Example]:
    return [
        {"input": list(given), "output": reference(list(given))} for given in inputs
    ]


def _lesson(
    *,
    lesson_id: str,
    chapter: str,
    title: str,
    concept: str,
    story: str,
    goal: str,
    starter: str,
    solution: str,
    reference: Callable[[list], list],
    shown: list[list],
    extra: list[list],
    hints: list[str],
    discovery: str,
    review: str,
    prediction: list,
    macros: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "id": lesson_id,
        "chapter": chapter,
        "title": title,
        "concept": concept,
        "story": story,
        "goal": goal,
        "starter": starter,
        "solution": solution,
        "examples": _examples(reference, shown),
        "heldout": _examples(reference, extra),
        "hints": list(hints),
        "discovery": discovery,
        "review": review,
        "prediction": {
            "input": list(prediction),
            "prompt": "Before you run anything: what does your program leave on "
            f"the stack for the input {prediction}?",
            "answer": reference(list(prediction)),
        },
        "macros": list(macros or ()),
    }


# ---------------------------------------------------------------------------
# References: plain Python, written from the behaviour, not from the program.
# ---------------------------------------------------------------------------


def _add_one(given: list) -> list:
    (n,) = given
    return [n + 1]


def _double(given: list) -> list:
    (n,) = given
    return [2 * n]


def _difference(given: list) -> list:
    a, b = given
    return [a - b]


def _reverse_difference(given: list) -> list:
    a, b = given
    return [b - a]


def _keep_both_and_sum(given: list) -> list:
    a, b = given
    return [a, b, a + b]


def _each_plus_one(given: list) -> list:
    (items,) = given
    return [[x + 1 for x in items]]


def _negatives(given: list) -> list:
    (items,) = given
    return [[x for x in items if x < 0]]


def _total(given: list) -> list:
    (items,) = given
    return [sum(items)]


def _bump_the_deeper(given: list) -> list:
    a, b = given
    return [a + 1, b]


def _absolute(given: list) -> list:
    (n,) = given
    return [n if n >= 0 else -n]


def _larger(given: list) -> list:
    a, b = given
    return [max(a, b)]


def _add_ten(given: list) -> list:
    (n,) = given
    return [n + 10]


def _sum_of_squares(given: list) -> list:
    (items,) = given
    return [sum(x * x for x in items)]


# ---------------------------------------------------------------------------
# The lessons
# ---------------------------------------------------------------------------

LESSONS: list[dict[str, Any]] = [
    _lesson(
        lesson_id="first-light",
        chapter="Signals",
        title="First light",
        concept="A program is a line of words. Each word changes the stack; "
        "`1` pushes a one and `add` replaces the top two values with their sum.",
        story="The Signal House stands on a rock at the harbour mouth. Its "
        "machine answers every message with another, and tonight the keeper "
        "wants it to answer each number with the next one.",
        goal="Leave the input plus one on the stack.",
        starter="",
        solution="1 add",
        reference=_add_one,
        shown=[[3], [0], [7]],
        extra=[[10], [-2], [41]],
        hints=[
            "Only `0` and `1` are literals. Push the one you need first.",
            "`add` wants two integers on the stack and leaves their sum.",
        ],
        discovery="Run `1` alone and look at the stack. Then run `1 1 add`. "
        "Fuel counts every word.",
        review="Words run left to right; each one reads and rewrites the top "
        "of the stack.",
        prediction=[5],
    ),
    _lesson(
        lesson_id="echo",
        chapter="Signals",
        title="Echo",
        concept="`dup` copies the top value. A copy is how one input becomes "
        "two operands.",
        story="A ship signals a number; the keeper wants to send it back twice "
        "as loud. There is no `2` to push, so the machine must use what it has.",
        goal="Leave double the input on the stack.",
        starter="",
        solution="dup add",
        reference=_double,
        shown=[[2], [5], [0]],
        extra=[[-3], [12], [100]],
        hints=[
            "You need the input twice. One word makes a copy.",
            "After `dup` the stack holds the number twice; `add` finishes it.",
        ],
        discovery="Try `dup dup add add`. What multiple of the input is that?",
        review="`dup` is how a single value feeds a two-argument op.",
        prediction=[21],
    ),
    _lesson(
        lesson_id="two-lamps",
        chapter="Signals",
        title="Two lamps",
        concept="Inputs stack bottom to top: the first is deepest, the last is "
        "on top. `sub` computes `deeper - top`.",
        story="Two lamps burn at different heights. The keeper wants to know "
        "how much higher the first one is than the second.",
        goal="Leave the first input minus the second.",
        starter="",
        solution="sub",
        reference=_difference,
        shown=[[5, 2], [10, 4], [1, 1]],
        extra=[[0, 7], [20, 5], [-3, -3]],
        hints=[
            "Both inputs are already on the stack. One word is enough.",
            "`sub` subtracts the top from the value beneath it.",
        ],
        discovery="Run `add` on the same inputs, then `mul`. Which ops care "
        "about the order of their operands?",
        review="Order matters: the stack reads bottom to top, and `sub` and "
        "`div` read it that way too.",
        prediction=[9, 4],
    ),
    _lesson(
        lesson_id="mirror",
        chapter="Shuffles",
        title="Mirror",
        concept="`swap` exchanges the top two values. Shuffling is how you "
        "put operands in the order an op wants.",
        story="The ship's reply reads the lamps the other way round. The "
        "keeper needs the second lamp's height minus the first.",
        goal="Leave the second input minus the first.",
        starter="sub",
        solution="swap sub",
        reference=_reverse_difference,
        shown=[[2, 5], [4, 10], [7, 7]],
        extra=[[9, 1], [0, 0], [3, 12]],
        hints=[
            "The starter subtracts in the wrong order. Reorder before subtracting.",
            "`swap` first, then `sub`.",
        ],
        discovery="Run `swap swap sub`. Why does it match the starter?",
        review="`swap` costs one fuel and turns `a - b` into `b - a`.",
        prediction=[1, 8],
    ),
    _lesson(
        lesson_id="keep-both",
        chapter="Shuffles",
        title="Keep both",
        concept="`over` copies the second value to the top. Combined with "
        "`dup` and `swap`, it reaches any arrangement of a few values.",
        story="The log must keep both lamp heights and add their total "
        "beneath them, so nothing is lost in the sum.",
        goal="Leave both inputs, then their sum on top.",
        starter="",
        solution="over over add",
        reference=_keep_both_and_sum,
        shown=[[1, 2], [4, 5], [0, 0]],
        extra=[[10, 20], [-1, 1], [7, 8]],
        hints=[
            "`add` consumes its operands; copy both inputs first.",
            "`over over` copies the pair above itself; then `add` the copies.",
        ],
        discovery="Run `over` alone, then `over over`, and watch the depth grow.",
        review="`over` reaches past the top without disturbing it.",
        prediction=[3, 4],
    ),
    _lesson(
        lesson_id="lantern",
        chapter="Quotations",
        title="Lantern",
        concept="`[` puts a mark on the stack. Until `]` collects it, words are "
        "not run but captured as symbols; `]` gathers them into a quotation, "
        "a list the machine can run later with `i`.",
        story="The keeper writes an instruction on a card instead of shouting "
        "it. A card does nothing until someone reads it aloud.",
        goal="Leave the input plus one, but build the instruction as a "
        "quotation and run it with `i`.",
        starter="[ 1 add ]",
        solution="[ 1 add ] i",
        reference=_add_one,
        shown=[[4], [0], [9]],
        extra=[[-1], [100], [33]],
        hints=[
            "Run the starter and look at the stack: the quotation is sitting "
            "there as a value.",
            "`i` runs the quotation on top of the stack.",
        ],
        discovery="Step through the starter. Notice the mark, then `1` and "
        "`add` arriving as symbols, then `]` folding them into one list.",
        review="A quotation is data until `i` makes it code.",
        prediction=[41],
    ),
    _lesson(
        lesson_id="each-one",
        chapter="Quotations",
        title="Each one",
        concept="`map` runs a quotation once per list item, with that item "
        "alone on the stack, and collects what each run leaves.",
        story="A whole convoy signals at once. The keeper answers every ship "
        "in the same way, one at a time.",
        goal="Add one to every item of the input list.",
        starter="[ 1 add ]",
        solution="[ 1 add ] map",
        reference=_each_plus_one,
        shown=[[[1, 2, 3]], [[]], [[0, 5]]],
        extra=[[[7]], [[-1, -2]], [[10, 20, 30, 40]]],
        hints=[
            "The quotation from the last lesson is right; it needs a combinator.",
            "`map` takes a list and a quotation and leaves a new list.",
        ],
        discovery="Inspect the `map` step and count the nested steps. How does "
        "the count depend on the list's length?",
        review="`map` is a loop with the body written once.",
        prediction=[[2, 4, 6]],
    ),
    _lesson(
        lesson_id="sieve",
        chapter="Combinators",
        title="Sieve",
        concept="`filter` runs a quotation per item and keeps the item when the "
        "body leaves `true`. `lt?` compares the deeper value to the top.",
        story="Only the ships below the waterline mark need an answer. The "
        "keeper wants just the negative readings.",
        goal="Keep only the negative items of the input list.",
        starter="[ 0 lt? ] map",
        solution="[ 0 lt? ] filter",
        reference=_negatives,
        shown=[[[1, -2, 3, -4]], [[5, 6]], [[-1]]],
        extra=[[[0, 0, -1]], [[]], [[-5, -6, 7]]],
        hints=[
            "The starter leaves booleans. You want the items that made them true.",
            "Swap `map` for `filter`; the body is already a test.",
        ],
        discovery="Run the starter and read the list of booleans it leaves. "
        "That is exactly what `filter` consults.",
        review="`filter` keeps; `map` transforms. Same body, different combinator.",
        prediction=[[-3, 0, 2, -1]],
    ),
    _lesson(
        lesson_id="tally",
        chapter="Combinators",
        title="Tally",
        concept="`fold` walks a list with an accumulator: the body sees "
        "`acc item` and must leave the next accumulator.",
        story="At dawn the keeper totals every reading of the night.",
        goal="Leave the sum of the input list.",
        starter="[ add ] fold",
        solution="0 [ add ] fold",
        reference=_total,
        shown=[[[1, 2, 3]], [[]], [[10, -4]]],
        extra=[[[5]], [[1, 1, 1, 1]], [[-2, -3]]],
        hints=[
            "The starter is refused: `fold` needs a list, a starting value and "
            "a quotation.",
            "Start the accumulator at `0`.",
        ],
        discovery="Inspect the `fold` step on the first example and watch the "
        "accumulator grow.",
        review="`fold` turns a list into one value; the start value is what an "
        "empty list gives.",
        prediction=[[4, 4, 4]],
    ),
    _lesson(
        lesson_id="under",
        chapter="Combinators",
        title="Under",
        concept="`dip` lifts the top value off, runs a quotation beneath it, "
        "and puts the value back.",
        story="The keeper must correct the lower lamp without touching the upper one.",
        goal="Add one to the first input and leave the second untouched on top.",
        starter="[ 1 add ] i",
        solution="[ 1 add ] dip",
        reference=_bump_the_deeper,
        shown=[[1, 2], [5, 0], [0, 9]],
        extra=[[-1, 3], [10, 10], [7, 1]],
        hints=[
            "The starter changes the top value. You want the one beneath.",
            "`dip` runs the quotation under the top value.",
        ],
        discovery="Compare the trace of `[ 1 add ] i` with `[ 1 add ] dip` on "
        "the same input.",
        review="`dip` is the shuffle that reaches beneath the top for a whole "
        "program, not just a value.",
        prediction=[20, 30],
    ),
    _lesson(
        lesson_id="fork",
        chapter="Combinators",
        title="Fork",
        concept="`ifte` takes three quotations: a test, a then-branch and an "
        "else-branch. The test runs on a copy of the stack, so what it "
        "consumes is restored before a branch runs.",
        story="A reading below zero means the lamp is reflected in the water. "
        "The keeper wants its true height either way.",
        goal="Leave the absolute value of the input.",
        starter="[ dup 0 lt? ] [ 0 swap sub ] [ 1 add ] ifte",
        solution="[ dup 0 lt? ] [ 0 swap sub ] [ ] ifte",
        reference=_absolute,
        shown=[[-3], [4], [0]],
        extra=[[-10], [12], [-1]],
        hints=[
            "The starter's else-branch does something it should not. An empty "
            "quotation does nothing.",
            "`[ ]` is a valid, empty quotation.",
        ],
        discovery="Inspect the `ifte` step on a negative input and on a "
        "positive one. Which branch ran?",
        review="`ifte` chooses a program; the test's values are not kept.",
        prediction=[-7],
    ),
    _lesson(
        lesson_id="the-fit",
        chapter="Judgement",
        title="The fit",
        concept="A program that matches every shown example may still not be "
        "the behaviour. Extra tests you cannot see are how the workshop "
        "tells a fit from an explanation.",
        story="Three ships signal pairs of numbers and the log shows the larger "
        "of each. The starter keeps the top value and matches all three. Is "
        "that the behaviour?",
        goal="Leave the larger of the two inputs.",
        starter="swap drop",
        solution="[ over lt? ] [ drop ] [ swap drop ] ifte",
        reference=_larger,
        shown=[[1, 3], [2, 5], [0, 4]],
        extra=[[7, 2], [5, 5], [9, 1]],
        hints=[
            "Verify the starter. It passes the examples and fails extra "
            "tests, because every shown pair had the larger value on top.",
            "Decide with `ifte`: `[ over lt? ]` tests whether the top is less "
            "than the value beneath it.",
            "If the top is smaller, `drop` it; otherwise `swap drop`.",
        ],
        discovery="Write your own example where the first value is larger and "
        "run the starter on it.",
        review="Examples constrain a program; they do not define it. Ask what "
        "input would break your fit.",
        prediction=[8, 3],
    ),
    _lesson(
        lesson_id="shorthand",
        chapter="Macros",
        title="Shorthand",
        concept="A macro is a named word that expands to primitives before the "
        "machine sees it. The library defines constants like `10` and "
        "helpers like `inc`; the trace still shows every primitive.",
        story="Writing `1 1 add dup add 1 add dup add` every night is tiresome. "
        "The keeper's logbook has a shorthand page.",
        goal="Leave the input plus ten.",
        starter="1 1 add dup add 1 add dup add add",
        solution="10 add",
        reference=_add_ten,
        shown=[[1], [0], [-5]],
        extra=[[90], [7], [-10]],
        hints=[
            "The macro library has `10`. Use it like any other word.",
            "`10 add` expands to the starter.",
        ],
        discovery="Hover the macro's primitives in the trace: every one carries "
        "the macro's name. Try `inc` ten times instead.",
        review="Macros shorten what you write, not what the machine runs.",
        prediction=[32],
        macros=["10", "inc"],
    ),
    _lesson(
        lesson_id="signal-house",
        chapter="Macros",
        title="Signal House",
        concept="Macros compose: `sq` squares, `sum` folds with `add`. A short "
        "line in macro form can expand to a long program.",
        story="The final log of the season: every reading squared, then totalled, "
        "to be sent to the mainland as one number.",
        goal="Leave the sum of the squares of the input list.",
        starter="[ sq ] map",
        solution="[ sq ] map sum",
        reference=_sum_of_squares,
        shown=[[[1, 2, 3]], [[]], [[-2, 2]]],
        extra=[[[5]], [[1, 1, 1]], [[3, 4]]],
        hints=[
            "The starter squares each item and stops. Total the list.",
            "`sum` is `0 [ add ] fold`.",
        ],
        discovery="Compare the written word count with the primitive count.",
        review="You have seen the whole machine: literals, shuffles, "
        "quotations, combinators and the names people give them.",
        prediction=[[2, 3, 6]],
        macros=["sq", "sum"],
    ),
]

BY_ID: dict[str, dict[str, Any]] = {lesson["id"]: lesson for lesson in LESSONS}
