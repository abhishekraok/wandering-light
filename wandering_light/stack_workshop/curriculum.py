"""The authored lessons of Signal House.

Each lesson's expected outputs come from a small, independent Python reference
(``REFERENCES``), never from running the lesson's solution: a solution that
disagreed with its own lesson would otherwise pass by construction. The tests
check every solution against all six examples, and recompute every expected
output from its reference.

Inputs are JSON stacks, bottom first. ``examples`` are shown to the learner;
``heldout`` are the extra tests ``/api/verify`` adds.

Fields beyond the examples:

- ``starter``: the program the editor opens with.
- ``hints``: revealed one at a time, gentlest first.
- ``discovery``: shown once the lesson is passed -- what it was really about.
- ``review``: a question to answer from memory later, with its answer.
- ``prediction``: a program and input to predict before running; ``answer`` is
  the final stack in stack notation (top on the right).
- ``macros``: library macros the lesson introduces or relies on.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

Stack = list[Any]

REFERENCES: dict[str, Callable[..., Stack]] = {}


def _lesson(
    *,
    id: str,  # noqa: A002 -- the lesson field's name
    chapter: str,
    title: str,
    concept: str,
    story: str,
    goal: str,
    starter: str,
    solution: str,
    reference: Callable[..., Stack],
    shown: list[Stack],
    heldout: list[Stack],
    hints: list[str],
    discovery: str,
    review: dict[str, str],
    prediction: dict[str, Any],
    macros: list[str] | None = None,
) -> dict[str, Any]:
    REFERENCES[id] = reference

    def pair(given: Stack) -> dict[str, Stack]:
        return {"input": given, "output": reference(*given)}

    return {
        "id": id,
        "chapter": chapter,
        "title": title,
        "concept": concept,
        "story": story,
        "goal": goal,
        "starter": starter,
        "solution": solution,
        "examples": [pair(given) for given in shown],
        "heldout": [pair(given) for given in heldout],
        "hints": hints,
        "discovery": discovery,
        "review": review,
        "prediction": prediction,
        "macros": macros or [],
    }


def _absolute(n: int) -> Stack:
    return [n if n >= 0 else -n]


LESSONS: list[dict[str, Any]] = [
    # ------------------------------------------------------------------ signals
    _lesson(
        id="first-light",
        chapter="Signals",
        title="First light",
        concept="A program is a row of ops; each op changes the stack.",
        story=(
            "The signal house has been dark for years. When you throw the main "
            "switch a single number arrives on the wire. The old operators had a "
            "habit: answer every signal with the next number up, so the sender "
            "knows someone is listening."
        ),
        goal="Leave one more than the number you are given.",
        starter="# the number is on the stack\n",
        solution="1 add",
        reference=lambda n: [n + 1],
        shown=[[41], [0], [-5]],
        heldout=[[7], [999], [-1]],
        hints=[
            "Only 0 and 1 are literals: typing `1` pushes a one on top of the stack.",
            "`add` takes the top two numbers and leaves their sum.",
            "Push a one, then add.",
        ],
        discovery=(
            "Programs read left to right and every op sees only the stack. "
            "`1 add` is the machine's successor -- the library calls it `inc`."
        ),
        review={
            "question": "What is on the stack after `1` runs on [ 41 ]?",
            "answer": "[ 41 1 ] -- the one sits on top, waiting for an op.",
        },
        prediction={
            "question": "What does `1 add` leave?",
            "program": "1 add",
            "input": [41],
            "answer": "[ 42 ]",
        },
    ),
    _lesson(
        id="echo",
        chapter="Signals",
        title="Echo",
        concept="`dup` copies the top value, so one input can be used twice.",
        story=(
            "The second station repeats what it hears, and the repeat adds to "
            "the original on the line. Whatever number arrives leaves twice as "
            "strong."
        ),
        goal="Leave twice the number you are given.",
        starter="",
        solution="dup add",
        reference=lambda n: [2 * n],
        shown=[[4], [0], [-3]],
        heldout=[[21], [-50], [1]],
        hints=[
            "You need the number twice, but you were only given it once.",
            "`dup` copies the top of the stack.",
            "Copy it, then add the two copies.",
        ],
        discovery=(
            "`dup` is how a stack program reuses a value: there are no "
            "variables, only copies. `dup mul` would square instead."
        ),
        review={
            "question": "Why can't `add` alone double a number?",
            "answer": "`add` needs two values and the stack holds only one.",
        },
        prediction={
            "question": "What does `dup` leave?",
            "program": "dup",
            "input": [7],
            "answer": "[ 7 7 ]",
        },
    ),
    # ----------------------------------------------------------------- shuffles
    _lesson(
        id="crossing",
        chapter="Shuffles",
        title="Crossed wires",
        concept="`swap` exchanges the top two values; order matters for `sub`.",
        story=(
            "Two numbers arrive, but the wires were crossed when the house was "
            "rebuilt. The receiver expects the second number minus the first."
        ),
        goal="Given a then b (b on top), leave b - a.",
        starter="",
        solution="swap sub",
        reference=lambda a, b: [b - a],
        shown=[[3, 10], [5, 5], [0, 7]],
        heldout=[[10, 3], [-2, 4], [100, 1]],
        hints=[
            "`sub` computes (second from top) - (top).",
            "As given, `sub` would compute a - b.",
            "Exchange the two values first.",
        ],
        discovery=(
            "The stack is ordered, and every non-commutative op reads that "
            "order. `swap` is the cheapest way to change which operand is which."
        ),
        review={
            "question": "What does `sub` leave on [ 3 10 ]?",
            "answer": "[ -7 ]: deeper minus top, 3 - 10.",
        },
        prediction={
            "question": "What does `swap` leave?",
            "program": "swap",
            "input": [1, 2],
            "answer": "[ 2 1 ]",
        },
    ),
    _lesson(
        id="keep-both",
        chapter="Shuffles",
        title="Keep the receipts",
        concept="`over` copies the second value to the top without consuming it.",
        story=(
            "The ledger clerk wants the two readings kept exactly as they "
            "arrived, with their total placed on top."
        ),
        goal="Given a then b, leave a, b and a + b.",
        starter="",
        solution="over over add",
        reference=lambda a, b: [a, b, a + b],
        shown=[[2, 3], [0, 0], [-1, 5]],
        heldout=[[10, 20], [7, -7], [1, 1]],
        hints=[
            "`add` would consume both readings, so add copies instead.",
            "`over` copies the value just below the top.",
            "Two `over`s copy both readings, in order.",
        ],
        discovery=(
            "`over over` duplicates the top two values as a pair -- the stack "
            "idiom for computing from values you still need afterwards."
        ),
        review={
            "question": "How does `over` differ from `dup`?",
            "answer": "`dup` copies the top; `over` copies the value below it.",
        },
        prediction={
            "question": "What does `over` leave?",
            "program": "over",
            "input": [3, 4],
            "answer": "[ 3 4 3 ]",
        },
    ),
    # --------------------------------------------------------------- quotations
    _lesson(
        id="wrap",
        chapter="Quotations",
        title="Sealed envelope",
        concept="A list is a value; `quote` wraps one value in a list.",
        story=(
            "Messages bound for the archive travel sealed. The archivist wants "
            "each number in an envelope of its own: a list holding just it."
        ),
        goal="Leave a one-item list holding the number.",
        starter="",
        solution="quote",
        reference=lambda n: [[n]],
        shown=[[5], [0], [-2]],
        heldout=[[9], [123], [-1]],
        hints=[
            "Lists print in parentheses: (5) is a list holding 5.",
            "One op turns a value into a list holding it.",
            "It is called `quote`.",
        ],
        discovery=(
            "A quotation is just a list. Lists of numbers are data; lists of op "
            "names are programs. The machine does not tell them apart until "
            "something runs one."
        ),
        review={
            "question": "What is `nil`?",
            "answer": "The empty list.",
        },
        prediction={
            "question": "What does `quote` leave?",
            "program": "quote",
            "input": [5],
            "answer": "[ (5) ]",
        },
    ),
    _lesson(
        id="sealed-orders",
        chapter="Quotations",
        title="Sealed orders",
        concept="`[ ... ]` captures ops as a quotation; `i` runs it.",
        story=(
            "Orders arrive sealed and are opened only when the station says "
            "so. Write the order for squaring a number, then open it."
        ),
        goal="Square the number by building a quotation and running it with `i`.",
        starter="[ ] i",
        solution="[ dup mul ] i",
        reference=lambda n: [n * n],
        shown=[[3], [0], [-4]],
        heldout=[[12], [1], [-9]],
        hints=[
            "Between `[` and `]` ops are captured as names, not run.",
            "Squaring is `dup mul`.",
            "Put `dup mul` inside the brackets; `i` runs what you captured.",
        ],
        discovery=(
            "`[` leaves a MARK, and while a MARK is open each op pushes its own "
            "name. `]` gathers the names into a list. That list is a program "
            "you can hold, copy and pass on before running it -- the basis of "
            "every combinator to come. (Here `dup mul` alone would also pass: "
            "the lesson is about how, not what.)"
        ),
        review={
            "question": "What does `[ dup mul ]` leave on [ 3 ]?",
            "answer": "[ 3 (dup mul) ]: the quotation sits above the number.",
        },
        prediction={
            "question": "What is on the stack before `i` runs?",
            "program": "[ dup mul ]",
            "input": [3],
            "answer": "[ 3 (dup mul) ]",
        },
    ),
    # -------------------------------------------------------------------- lists
    _lesson(
        id="brighten",
        chapter="Lists",
        title="Brighten every lamp",
        concept="`map` runs a quotation on each item of a list.",
        story=(
            "A list of lamp levels arrives from the tower. Turn every lamp up by one."
        ),
        goal="Add one to every number in the list.",
        starter="[ ] map",
        solution="[ 1 add ] map",
        reference=lambda xs: [[x + 1 for x in xs]],
        shown=[[[1, 2, 3]], [[]], [[-1, 0, 9]]],
        heldout=[[[10]], [[5, 5, 5, 5]], [[-3, -2]]],
        hints=[
            "The quotation sees one item at a time, alone on its stack.",
            "It must leave exactly one value: the new item.",
            "The body is the program from First light.",
        ],
        discovery=(
            "`map` turns a one-number program into a whole-list program. The "
            "body sees only its item -- nothing else on the stack -- which is "
            "why it cannot accidentally disturb anything."
        ),
        review={
            "question": "What does `map` need on the stack?",
            "answer": "A list, with a quotation on top of it.",
        },
        prediction={
            "question": "What does this leave?",
            "program": "[ dup add ] map",
            "input": [[1, 2]],
            "answer": "[ (2 4) ]",
        },
    ),
    _lesson(
        id="positives",
        chapter="Lists",
        title="Static on the line",
        concept="`filter` keeps the items whose quotation leaves true.",
        story=(
            "Readings at or below zero are static. Keep only the positive "
            "readings, in the order they came."
        ),
        goal="Keep only the numbers greater than zero.",
        starter="[ ] filter",
        solution="[ 0 swap lt? ] filter",
        reference=lambda xs: [[x for x in xs if x > 0]],
        shown=[[[3, -1, 0, 2]], [[-5, -6]], [[7]]],
        heldout=[[[]], [[1, -1, 1, -1]], [[0, 0, 4]]],
        hints=[
            "The body must leave one boolean per item.",
            "`lt?` asks whether the second value is less than the top.",
            "You want 0 < x: push 0, then put it underneath x.",
        ],
        discovery=(
            "`lt?` is the only ordering test; `gt?`, `le?` and `ge?` are "
            "macros built from it with `swap` and a negation."
        ),
        review={
            "question": "What does `lt?` leave on [ 2 5 ]?",
            "answer": "[ true ]: 2 < 5.",
        },
        prediction={
            "question": "What does `0 swap` do to [ 7 ]?",
            "program": "0 swap",
            "input": [7],
            "answer": "[ 0 7 ]",
        },
    ),
    _lesson(
        id="total",
        chapter="Lists",
        title="Night total",
        concept="`fold` carries an accumulator through a list.",
        story=(
            "At the end of the night the keeper writes one number in the log: "
            "the sum of every reading."
        ),
        goal="Leave the sum of the list (0 for an empty list).",
        starter="[ ] fold",
        solution="0 [ add ] fold",
        reference=lambda xs: [sum(xs)],
        shown=[[[1, 2, 3]], [[]], [[10, -4]]],
        heldout=[[[5]], [[1, 1, 1, 1, 1]], [[-7, 7, 100]]],
        hints=[
            "`fold` needs a list, a starting accumulator, then a quotation.",
            "The body sees the accumulator and one item, and leaves the new "
            "accumulator.",
            "Start from 0 and add each item.",
        ],
        discovery=(
            "`fold` is the loop of this machine: the library builds `sum`, "
            "`len`, `reverse` and `max` from it, all without recursion."
        ),
        review={
            "question": "What does a fold body see?",
            "answer": "Exactly two values: the accumulator, then the item on top.",
        },
        prediction={
            "question": "What does this leave?",
            "program": "1 [ mul ] fold",
            "input": [[2, 3, 4]],
            "answer": "[ 24 ]",
        },
        macros=["sum"],
    ),
    # ------------------------------------------------------------------ control
    _lesson(
        id="under",
        chapter="Control",
        title="Under the counter",
        concept="`dip` runs a quotation beneath the top value.",
        story=(
            "The clerk keeps the newest slip on top of the pile. Bump the "
            "number underneath it by one without touching the top slip."
        ),
        goal="Given a then b, leave a + 1 then b.",
        starter="",
        solution="[ 1 add ] dip",
        reference=lambda a, b: [a + 1, b],
        shown=[[1, 10], [0, 0], [-3, 4]],
        heldout=[[99, 1], [5, -5], [-1, -1]],
        hints=[
            "`swap 1 add swap` would work, but there is an op for this.",
            "`dip` sets the top aside, runs a quotation, and puts it back.",
            "Quote the successor and dip it.",
        ],
        discovery=(
            "`dip` reaches below the top without shuffling. The library's "
            "`rot`, `pick2` and `dip2` are all built from it."
        ),
        review={
            "question": "What does `dip` need on the stack?",
            "answer": "Some value x, with a quotation on top of it.",
        },
        prediction={
            "question": "What does this leave?",
            "program": "[ dup ] dip",
            "input": [1, 2],
            "answer": "[ 1 1 2 ]",
        },
    ),
    _lesson(
        id="absolute",
        chapter="Control",
        title="Distance from zero",
        concept="`ifte` chooses between two quotations by a condition quotation.",
        story=(
            "The range finder reports how far each echo is from zero; direction "
            "does not matter. Negative readings must be flipped."
        ),
        goal="Leave the absolute value of the number.",
        starter="[ ] [ ] [ ] ifte",
        solution="[ dup 0 lt? ] [ 0 swap sub ] [ ] ifte",
        reference=_absolute,
        shown=[[-4], [3], [0]],
        heldout=[[-100], [7], [-1]],
        hints=[
            "`ifte` takes three quotations: condition, then-branch, else-branch.",
            "The condition's leftovers are discarded; only its top boolean "
            "counts, and each branch starts from the original stack.",
            "Test `dup 0 lt?`; negate with `0 swap sub`; otherwise do nothing.",
        ],
        discovery=(
            "Branches start from the stack as it was before the condition, so "
            "a condition can consume what it likes. An empty quotation is a "
            "perfectly good branch."
        ),
        review={
            "question": "Which stack does the chosen branch of `ifte` see?",
            "answer": "The stack below the three quotations, before the condition ran.",
        },
        prediction={
            "question": "What does this leave?",
            "program": "[ true ] [ 1 ] [ 0 ] ifte",
            "input": [],
            "answer": "[ 1 ]",
        },
        macros=["abs", "neg"],
    ),
    # ----------------------------------------------------------------- evidence
    _lesson(
        id="lucky-fit",
        chapter="Evidence",
        title="A fit is not an explanation",
        concept="Passing the shown examples does not mean the program is right.",
        story=(
            "Two beacons report their brightness and the harbour wants the "
            "brighter one. Every report so far happened to put the brighter "
            "beacon second. The extra tests will not be so kind."
        ),
        goal="Leave the larger of the two numbers.",
        starter="swap drop",
        solution="[ over lt? ] [ drop ] [ swap drop ] ifte",
        reference=lambda a, b: [max(a, b)],
        shown=[[1, 5], [2, 7], [-3, 0]],
        heldout=[[8, 3], [4, 4], [0, -6]],
        hints=[
            "The starter passes every shown example. Run Verify before trusting it.",
            "The answer depends on which number is larger: that is a condition.",
            "With a b on the stack, `over lt?` asks b < a; then keep a "
            "(`drop`), else keep b (`swap drop`).",
        ],
        discovery=(
            "`swap drop` explains the examples you were shown, not the task. "
            "Three examples are compatible with many programs; held-out tests "
            "are how the corpus tells an explanation from a fit. The library "
            "names this program `max2`."
        ),
        review={
            "question": "Why can a program pass every shown example and still be wrong?",
            "answer": "The examples may share a property the task does not, and "
            "the program may rely on it.",
        },
        prediction={
            "question": "What does the starter leave on [ 8 3 ]?",
            "program": "swap drop",
            "input": [8, 3],
            "answer": "[ 3 ]",
        },
        macros=["max2"],
    ),
    # ------------------------------------------------------------------- macros
    _lesson(
        id="named-signals",
        chapter="Macros",
        title="Named signals",
        concept="A macro is a name for a program; it expands before running.",
        story=(
            "The keepers wrote their common routines in a book and gave them "
            "names. Use the book: measure the energy of a list of readings, "
            "the sum of their squares."
        ),
        goal="Leave the sum of the squares of the list.",
        starter="",
        solution="[ sq ] map sum",
        reference=lambda xs: [sum(x * x for x in xs)],
        shown=[[[1, 2, 3]], [[]], [[-2, 5]]],
        heldout=[[[4]], [[1, 1, 1]], [[-3, -4, 0]]],
        hints=[
            "`sq` squares a number; `sum` adds up a list.",
            "Square every item first, then sum.",
            "`[ sq ] map` then `sum`.",
        ],
        discovery=(
            "Macros cost nothing at run time that their expansion does not: "
            "the machine only ever sees primitives. Expand a macro in the "
            "trace to see the fold inside `sum`."
        ),
        review={
            "question": "Does the machine know about macros?",
            "answer": "No: they expand to primitive ops before anything runs.",
        },
        prediction={
            "question": "What does `sq` leave?",
            "program": "sq",
            "input": [-6],
            "answer": "[ 36 ]",
        },
        macros=["sq", "sum"],
    ),
    _lesson(
        id="even-count",
        chapter="Macros",
        title="Even keel",
        concept="Macros compose: predicates and counts from the book combine.",
        story=(
            "A ship is on an even keel when its ballast readings balance. The "
            "harbourmaster only wants to know how many readings are even."
        ),
        goal="Leave how many numbers in the list are even.",
        starter="",
        solution="[ even? ] filter len",
        reference=lambda xs: [sum(1 for x in xs if x % 2 == 0)],
        shown=[[[1, 2, 3, 4]], [[]], [[-2, 7, 0]]],
        heldout=[[[5]], [[2, 4, 6, 8]], [[-1, -3, 10]]],
        hints=[
            "`even?` leaves true for an even number; `len` counts a list.",
            "Keep the even numbers, then count what is left.",
            "`[ even? ] filter len`.",
        ],
        discovery=(
            "`even?` is `2 mod 0 eq?`, and `mod` is `over over div mul sub`: "
            "three layers of names over five primitives. Negative numbers work "
            "because `div` floors, as Python's `//` does."
        ),
        review={
            "question": "What does `mod` leave on [ -7 2 ]?",
            "answer": "[ 1 ], as Python's -7 % 2.",
        },
        prediction={
            "question": "What does `len` leave?",
            "program": "len",
            "input": [[9, 9, 9]],
            "answer": "[ 3 ]",
        },
        macros=["even?", "mod", "len"],
    ),
]

BY_ID: dict[str, dict[str, Any]] = {lesson["id"]: lesson for lesson in LESSONS}
