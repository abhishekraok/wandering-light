"""The fourteen authored lessons of Signal House.

Every lesson is plain JSON-ready data. Its expected outputs are computed here
by a small Python reference for the rule the lesson teaches -- never by
running the lesson's solution -- so a wrong solution and a wrong example
cannot agree with each other by construction. ``REFERENCES`` keeps those
functions out of the lesson dicts and available for audit.

``examples`` are the three pairs a learner sees; ``heldout`` are three more
that only verification runs. ``prediction`` is a question to answer before
running anything: a short program, an input, and the stack it leaves.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

THE_STACK = "The stack"
LISTS = "Lists"
QUOTATIONS = "Programs as values"
EVIDENCE = "Evidence and names"

REFERENCES: dict[str, Callable[..., list]] = {}


def _lesson(
    *,
    slug: str,
    chapter: str,
    title: str,
    concept: str,
    story: str,
    goal: str,
    solution: str,
    reference: Callable[..., list],
    shown: list[list],
    extra: list[list],
    hints: list[str],
    discovery: str,
    review: dict[str, str],
    prediction: dict[str, Any],
    starter: str = "",
    macros: tuple[str, ...] = (),
) -> dict[str, Any]:
    if len(shown) != 3 or len(extra) != 3:
        raise ValueError(f"{slug}: a lesson shows three examples and holds out three")
    REFERENCES[slug] = reference

    def pairs(inputs: list[list]) -> list[dict[str, list]]:
        return [{"input": stack, "output": reference(*stack)} for stack in inputs]

    return {
        "id": slug,
        "chapter": chapter,
        "title": title,
        "concept": concept,
        "story": story,
        "goal": goal,
        "starter": starter,
        "solution": solution,
        "examples": pairs(shown),
        "heldout": pairs(extra),
        "hints": hints,
        "discovery": discovery,
        "review": review,
        "prediction": prediction,
        "macros": list(macros),
    }


def _predict(
    question: str, program: str, given: list, answer: str, *others: str
) -> dict[str, Any]:
    """``answer`` is the stack ``program`` leaves on ``given``; the tests run it."""
    return {
        "question": question,
        "program": program,
        "input": given,
        "options": sorted([answer, *others]),
        "answer": answer,
    }


def _review(question: str, answer: str) -> dict[str, str]:
    return {"question": question, "answer": answer}


LESSONS: list[dict[str, Any]] = [
    _lesson(
        slug="first-light",
        chapter=THE_STACK,
        title="First Light",
        concept="An op takes its inputs from the top of the stack and leaves "
        "its result there.",
        story="Dusk at Signal House. The engine in the lamp room, a mark-28, "
        "holds one number: tonight's signal count. The keeper wants it raised "
        "by one before the lamp is lit.",
        goal="Leave the number one larger.",
        solution="1 add",
        reference=lambda n: [n + 1],
        shown=[[3], [0], [-5]],
        extra=[[41], [-1], [999]],
        hints=[
            "`1` pushes the number one onto the top of the stack.",
            "`add` removes the top two numbers and leaves their sum.",
            "Push one, then add: `1 add`.",
        ],
        discovery="A program is read left to right, and each word acts on the "
        "stack the word before it left. There are no variables: the stack is "
        "the only place a value can be.",
        review=_review(
            "A stack holds 7. What does it hold after `1`?",
            "7 1. The new value goes on top, to the right.",
        ),
        prediction=_predict(
            "The stack starts empty. What does `1 1 add` leave?",
            "1 1 add",
            [],
            "2",
            "1 1",
            "1",
        ),
    ),
    _lesson(
        slug="echo",
        chapter=THE_STACK,
        title="Echo",
        concept="`dup` copies the top value. An op uses up what it reads.",
        story="The far shore answers every flash with the same flash. The log "
        "records the two together, as one count.",
        goal="Leave the number doubled.",
        solution="dup add",
        reference=lambda n: [2 * n],
        shown=[[4], [0], [-3]],
        extra=[[25], [-50], [1]],
        hints=[
            "`add` needs two numbers and the stack holds one.",
            "`dup` copies the top value, so the stack holds it twice.",
            "Copy, then add the copies: `dup add`.",
        ],
        discovery="Ops consume what they read. To use a value twice it has to "
        "be on the stack twice, and `dup` is how it gets there.",
        review=_review(
            "Which op leaves a second copy of the top value?",
            "`dup`.",
        ),
        prediction=_predict(
            "The stack holds 5. What does `dup dup add` leave?",
            "dup dup add",
            [5],
            "5 10",
            "15",
            "10 5",
        ),
    ),
    _lesson(
        slug="crossed-wires",
        chapter=THE_STACK,
        title="Crossed Wires",
        concept="`swap` exchanges the top two values. Order on the stack is "
        "the order of an op's arguments.",
        story="Two readings come in: last night's count, then tonight's. The "
        "ledger wants the rise, tonight's minus last night's, and `sub` "
        "takes them the other way round.",
        goal="Leave the top number minus the one beneath it.",
        solution="swap sub",
        reference=lambda a, b: [b - a],
        shown=[[3, 10], [8, 2], [-4, -4]],
        extra=[[0, 9], [100, 1], [-7, 5]],
        hints=[
            "Run `sub` alone and read the sign: it leaves the lower number "
            "minus the top one.",
            "The readings are in the wrong order for `sub`. Put tonight's "
            "underneath before subtracting.",
            "Exchange them, then subtract: `swap sub`.",
        ],
        discovery="`swap` is the smallest shuffle. It changes which value an "
        "op sees first without changing what the values are.",
        review=_review(
            "After `dup`, what does `add` compute?",
            "Twice the number: both of its arguments are the same value.",
        ),
        prediction=_predict(
            "The stack holds 10 3, with 3 on top. What does `sub` leave?",
            "sub",
            [10, 3],
            "7",
            "-7",
            "3 10",
        ),
    ),
    _lesson(
        slug="spare-copy",
        chapter=THE_STACK,
        title="Spare Copy",
        concept="`over` copies the second value onto the top. The whole stack "
        "is the output, not just its top.",
        story="The keeper wants both readings kept in the log, and their "
        "total written after them.",
        goal="Keep both numbers and leave their sum on top of them.",
        solution="over over add",
        reference=lambda a, b: [a, b, a + b],
        shown=[[2, 3], [10, -4], [0, 0]],
        extra=[[7, 7], [-1, -2], [50, 1]],
        hints=[
            "`add` would use up both readings. They need copying first.",
            "`over` copies the second value onto the top: a b becomes a b a.",
            "`over over` copies the pair, and `add` uses the copies: `over over add`.",
        ],
        discovery="`dup`, `swap`, `drop` and `over` are the whole kit for "
        "arranging values. Whatever is left on the stack is part of the "
        "answer, so a stray copy is a wrong answer.",
        review=_review(
            "The stack holds 3 10. What does `sub` leave?",
            "-7: the lower value minus the top one.",
        ),
        prediction=_predict(
            "The stack holds 1 2. What does `over` leave?",
            "over",
            [1, 2],
            "1 2 1",
            "1 2 2",
            "2 1 2",
        ),
    ),
    _lesson(
        slug="ledger",
        chapter=LISTS,
        title="The Ledger",
        concept="`nil` is the empty list and `cons` adds one item to the "
        "front of a list.",
        story="Loose readings get lost. Each night's pair is filed as one "
        "ledger entry: a list holding the two, in the order they came.",
        goal="Leave one list holding the two numbers, lower one first.",
        solution="nil swap cons swap cons",
        reference=lambda a, b: [[a, b]],
        shown=[[1, 2], [5, -3], [0, 0]],
        extra=[[9, 9], [-8, 4], [12, 30]],
        hints=[
            "`nil` pushes an empty list. `cons` wants the list below and the "
            "new item on top of it.",
            "`cons` adds at the front, so the last item goes in first: "
            "`nil swap cons` files the top number.",
            "Then `swap cons` files the other in front of it: "
            "`nil swap cons swap cons`.",
        ],
        discovery="A list is one value however much it holds. `cons` builds "
        "from the front, so a list is assembled back to front.",
        review=_review(
            "Which op copies the second value onto the top?",
            "`over`.",
        ),
        prediction=_predict(
            "The stack starts empty. What does `nil 1 cons 0 cons` leave?",
            "nil 1 cons 0 cons",
            [],
            "(0 1)",
            "(1 0)",
            "0 1",
        ),
    ),
    _lesson(
        slug="top-entry",
        chapter=LISTS,
        title="Top Entry",
        concept="`uncons` splits a list into its rest and its first item; "
        "`cons` puts them back together.",
        story="The newest entry sits at the front of the ledger, and it is "
        "one short. Correct it without disturbing the rest.",
        goal="Leave the list with its first number one larger.",
        solution="uncons 1 add cons",
        reference=lambda entries: [[entries[0] + 1, *entries[1:]]],
        shown=[[[5, 2, 3]], [[0]], [[-1, -1]]],
        extra=[[[9, 9, 9, 9]], [[41]], [[-10, 3]]],
        hints=[
            "`uncons` leaves the rest of the list, then its first item on top.",
            "On top of the stack the first item is an ordinary number: "
            "`1 add` works on it.",
            "`cons` puts it back at the front: `uncons 1 add cons`.",
        ],
        discovery="`uncons` undoes `cons`. Open, change, close is the shape "
        "of every edit to the front of a list. `uncons` refuses an empty "
        "list, because there is no first item to hand over.",
        review=_review(
            "Building (1 2) with `cons`, which number goes in first?",
            "2. `cons` adds at the front, so the list is built back to front.",
        ),
        prediction=_predict(
            "The stack holds the list (7 8). What does `uncons` leave?",
            "uncons",
            [[7, 8]],
            "(8) 7",
            "7 (8)",
            "7 8",
        ),
    ),
    _lesson(
        slug="sealed-orders",
        chapter=QUOTATIONS,
        title="Sealed Orders",
        concept="Between `[` and `]` ops are recorded instead of run. The "
        "result is a quotation, and `i` runs it.",
        story="Some orders are written down and sealed, to be carried out "
        "later. The starter seals an order to square the reading. It is "
        "sitting on the stack, unopened.",
        goal="Leave the number squared, by running the quotation.",
        starter="[ dup mul ]",
        solution="[ dup mul ] i",
        reference=lambda n: [n * n],
        shown=[[3], [-4], [0]],
        extra=[[12], [1], [-9]],
        hints=[
            "Run the starter and step through it: inside the brackets `dup` "
            "and `mul` are written down, not carried out.",
            "`]` folds what was recorded into one value, a quotation, on top "
            "of the number.",
            "`i` runs the quotation on top of the stack: `[ dup mul ] i`.",
        ],
        discovery="A quotation is a program held as a value: a list of op "
        "names. Sealing an order and running it straight away does what the "
        "bare ops would have done. The difference is that the sealed form can "
        "be handed to another op, which is what every lesson after this one "
        "does.",
        review=_review(
            "What does `uncons` leave on top: the first item or the rest?",
            "The first item. The rest of the list is beneath it.",
        ),
        prediction=_predict(
            "The stack holds 4. What does `[ dup ] i add` leave?",
            "[ dup ] i add",
            [4],
            "8",
            "4 4",
            "16",
        ),
    ),
    _lesson(
        slug="every-lamp",
        chapter=QUOTATIONS,
        title="Every Lamp",
        concept="`map` runs a quotation on each item of a list and collects "
        "the results.",
        story="Every lamp along the gallery is to burn twice as bright. The "
        "order is the same for each lamp; only the lamp changes.",
        goal="Leave the list with every number doubled.",
        solution="[ dup add ] map",
        reference=lambda lamps: [[2 * lamp for lamp in lamps]],
        shown=[[[1, 2, 3]], [[]], [[-5, 0]]],
        extra=[[[7]], [[10, 20, 30, 40]], [[0, 0, -1]]],
        hints=[
            "Write the order for one lamp first: what doubles a single number?",
            "Seal it: `[ dup add ]`. The list is beneath it.",
            "`map` applies the sealed order to each item: `[ dup add ] map`.",
        ],
        discovery="`map` runs its quotation once per item, each time on a "
        "stack holding only that item, and keeps the one value each run "
        "leaves. Step inside the `map` to watch the body run per item.",
        review=_review(
            "What does `i` do?",
            "It runs the quotation on top of the stack.",
        ),
        prediction=_predict(
            "The stack holds the list (0 5). What does `[ 1 add ] map` leave?",
            "[ 1 add ] map",
            [[0, 5]],
            "(1 6)",
            "(0 5 1)",
            "6",
        ),
    ),
    _lesson(
        slug="night-watch",
        chapter=QUOTATIONS,
        title="Night Watch",
        concept="`filter` keeps the items for which a quotation leaves `true`.",
        story="A reading below zero means frost on the lens. The watch wants "
        "only those readings, in the order they were taken.",
        goal="Leave the list holding only its numbers below zero.",
        solution="[ 0 lt? ] filter",
        reference=lambda readings: [[r for r in readings if r < 0]],
        shown=[[[3, -1, 4, -5]], [[0, 2]], [[-7]]],
        extra=[[[-1, -2, -3]], [[5, 0, -9, 0]], [[]]],
        hints=[
            "A test is a quotation that leaves `true` or `false`. The test "
            "here, for one number: is it below zero?",
            "`0 lt?` asks whether the number is less than 0.",
            "`filter` keeps the items whose test says true: `[ 0 lt? ] filter`.",
        ],
        discovery="The quotation answers a question and `filter` acts on the "
        "answer. The items come through unchanged; only which of them "
        "survive is decided.",
        review=_review(
            "How many values must a `map` quotation leave for each item?",
            "Exactly one.",
        ),
        prediction=_predict(
            "The stack holds the list (0 1 2). What does `[ 1 lt? ] filter` leave?",
            "[ 1 lt? ] filter",
            [[0, 1, 2]],
            "(0)",
            "(0 1)",
            "(true false false)",
        ),
    ),
    _lesson(
        slug="tally",
        chapter=QUOTATIONS,
        title="The Tally",
        concept="`fold` carries a running value along a list, one item at a time.",
        story="At dawn the night's readings are tallied into a single number "
        "for the ledger.",
        goal="Leave the sum of the numbers in the list.",
        solution="0 [ add ] fold",
        reference=lambda readings: [sum(readings)],
        shown=[[[1, 2, 3]], [[]], [[10, -4]]],
        extra=[[[5]], [[-1, -1, -1]], [[100, 200, 300, 400]]],
        hints=[
            "A tally starts somewhere. Push the starting total: `0`.",
            "The quotation sees the total so far and the next item, and must "
            "leave the new total.",
            "Start at zero and add each item in: `0 [ add ] fold`.",
        ],
        discovery="The `fold` quotation takes the total so far and one item, "
        "and leaves the new total. An empty list hands back the starting "
        "value untouched, which is why the start matters.",
        review=_review(
            "What must a `filter` quotation leave?",
            "A boolean. `true` keeps the item.",
        ),
        prediction=_predict(
            "The stack holds the list (2 3 4). What does `1 [ mul ] fold` leave?",
            "1 [ mul ] fold",
            [[2, 3, 4]],
            "24",
            "9",
            "(2 3 4)",
        ),
    ),
    _lesson(
        slug="under-the-lamp",
        chapter=QUOTATIONS,
        title="Under the Lamp",
        concept="`dip` sets the top value aside, runs a quotation beneath it, "
        "and puts the value back.",
        story="The correction belongs to the reading underneath. The one on "
        "top must stay exactly where it is.",
        goal="Leave the lower number one larger and the top number unchanged.",
        solution="[ 1 add ] dip",
        reference=lambda a, b: [a + 1, b],
        shown=[[1, 2], [0, -5], [9, 9]],
        extra=[[-1, 0], [41, 7], [100, -100]],
        hints=[
            "`swap 1 add swap` works. There is a way that says what is meant.",
            "`dip` runs a quotation on the stack beneath the top value.",
            "Seal the correction and run it underneath: `[ 1 add ] dip`.",
        ],
        discovery="`dip` reaches under the top value without a shuffle, and "
        "it works however much the quotation does down there.",
        review=_review(
            "In `0 [ add ] fold`, what is the 0 for?",
            "The starting total, and the answer for an empty list.",
        ),
        prediction=_predict(
            "The stack holds 1 2. What does `[ dup ] dip` leave?",
            "[ dup ] dip",
            [1, 2],
            "1 1 2",
            "1 2 2",
            "1 2 1",
        ),
    ),
    _lesson(
        slug="fork-in-the-line",
        chapter=QUOTATIONS,
        title="Fork in the Line",
        concept="`ifte` runs a test, then one of two quotations depending on "
        "its answer.",
        story="A reading can come in below zero or above it. The log records "
        "only how far from zero it is.",
        goal="Leave the number's distance from zero.",
        solution="[ 0 lt? ] [ 0 swap sub ] [ ] ifte",
        reference=lambda n: [abs(n)],
        shown=[[-5], [3], [0]],
        extra=[[-1], [-100], [42]],
        hints=[
            "`ifte` takes three quotations: a test, what to do when it says "
            "true, and what to do when it says false.",
            "`0 swap sub` negates a number. `[ ]` does nothing.",
            "Below zero, negate; otherwise leave it: "
            "`[ 0 lt? ] [ 0 swap sub ] [ ] ifte`.",
        ],
        discovery="`ifte` runs the test, keeps only its answer, and runs the "
        "chosen branch on the stack as it was before the test. The test may "
        "use up what it inspects: nothing it does survives except true or "
        "false.",
        review=_review(
            "What does `dip` leave untouched?",
            "The top value. The quotation runs beneath it.",
        ),
        prediction=_predict(
            "The stack holds 0. What does `[ 0 eq? ] [ 1 add ] [ ] ifte` leave?",
            "[ 0 eq? ] [ 1 add ] [ ] ifte",
            [0],
            "1",
            "0",
            "true",
        ),
    ),
    _lesson(
        slug="three-nights",
        chapter=EVIDENCE,
        title="Three Nights",
        concept="A program that reproduces every example shown has fitted "
        "them. Whether it is the rule is a separate question.",
        story="Three nights of the log show two readings going in and one "
        "coming out. The starter reproduces every line. The keeper who wrote "
        "the log was following a rule the three lines do not pin down.",
        goal="Find the rule the log was written by. Run the starter, then "
        "verify it, before changing anything.",
        starter="swap drop",
        solution="[ over lt? ] [ drop ] [ swap drop ] ifte",
        reference=lambda a, b: [max(a, b)],
        shown=[[1, 5], [2, 2], [-3, 0]],
        extra=[[7, 3], [0, -4], [6, 9]],
        hints=[
            "The starter keeps the top reading and matches all three lines. "
            "Verify it and read the test that fails.",
            "In every line shown the top reading is at least as large as the "
            "lower one. The failing test is one where it is not.",
            "The rule is the larger of the two: "
            "`[ over lt? ] [ drop ] [ swap drop ] ifte`.",
        ],
        discovery="Three examples agreed with two different rules. Matching "
        "what was shown is evidence, not proof. The extra tests are inputs on "
        "which rules that agree about the shown lines come apart.",
        review=_review(
            "After the test in `ifte` runs, which stack does the chosen branch see?",
            "The stack as it was before the test.",
        ),
        prediction=_predict(
            "The starter matches all three lines shown. The stack holds 7 3. "
            "What does `swap drop` leave?",
            "swap drop",
            [7, 3],
            "3",
            "7",
            "3 7",
        ),
    ),
    _lesson(
        slug="named-signals",
        chapter=EVIDENCE,
        title="Names on the Shelf",
        concept="A macro is a name for a program. It expands into ops before "
        "the engine runs anything.",
        story="Some orders are written every night, so the house keeps them "
        "on a shelf under names: `sq` is `dup mul`, and `sum` is "
        "`0 [ add ] fold`. The engine still runs only its 28 ops.",
        goal="Leave the sum of the squares of the numbers in the list, using "
        "names from the shelf.",
        starter="[ sq ] map",
        solution="[ sq ] map sum",
        reference=lambda readings: [sum(r * r for r in readings)],
        shown=[[[1, 2, 3]], [[]], [[-4]]],
        extra=[[[5, 5]], [[0, 0, 1]], [[10, -10, 2]]],
        hints=[
            "The starter squares every item. Step through it: `sq` runs as `dup mul`.",
            "The squares are in a list. One name on the shelf totals a list.",
            "Square each, then total: `[ sq ] map sum`.",
        ],
        discovery="The engine never saw a name. Step through the program: "
        "every token is one of the 28 ops, tied to the word it was written "
        "as. A name costs nothing at run time and says what is meant. Every "
        "lesson before this one was stocking the shelf.",
        review=_review(
            "Why can a program that matches every example shown still be wrong?",
            "Several rules can agree on a few examples. Only more evidence "
            "tells them apart.",
        ),
        prediction=_predict(
            "The stack holds 3. What does `sq` leave?",
            "sq",
            [3],
            "9",
            "6",
            "3 3",
        ),
        macros=("sq", "sum"),
    ),
]

BY_ID: dict[str, dict[str, Any]] = {lesson["id"]: lesson for lesson in LESSONS}
