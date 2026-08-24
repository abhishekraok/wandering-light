"""Generate the conformance fixture the stack machine explorer checks itself against.

The explorer UI is a JavaScript port of `wandering_light.stack_machine`. A port
that quietly disagrees with the machine is worse than no explorer at all, so the
port carries this fixture and replays it on load. Regenerate with:

    uv run python experiments/stack_machine_fixture.py

Every example carries one or more `inputs`: small programs that build a starting
stack. Most have a single empty one. The ones with several are
programming-by-example tasks, where the same program has to work on every input
-- which is also how a fixed-width language model looks here.
"""

import json
import random

from wandering_light.stack_machine import (
    OPS,
    Config,
    format_stack,
    legal_mask,
    parse_program,
    parse_stack,
    step,
)

EXAMPLES = [
    # ---------------------------------------------------------------- essay
    {
        "group": "Start here",
        "kind": "essay",
        "title": "How a program becomes data",
        "program": "nil 'add cons '1 cons",
        "note": "The one design decision behind everything else on this page.",
    },
    # ---------------------------------------------------- the first-order 18
    {
        "group": "The first eighteen",
        "title": "The round trip",
        "program": "nil 1 cons uncons",
        "note": "uncons undoes cons: head on top, tail below.",
    },
    {
        "group": "The first eighteen",
        "title": "Building a list",
        "program": "nil 1 cons 0 cons 1 cons",
        "note": "Every list starts as nil and grows one cons at a time.",
    },
    {
        "group": "The first eighteen",
        "title": "Where did div go?",
        "program": "1 0",
        "note": "The guard reads the divisor, not just its type. Zero, so no div.",
    },
    {
        "group": "The first eighteen",
        "title": "One is not true",
        "program": "1 true",
        "note": "Look for add. It is gone -- booleans are not numbers here.",
    },
    {
        "group": "The first eighteen",
        "title": "Start from nothing",
        "program": "",
        "note": "An empty stack. Only the pushing ops are legal.",
    },
    # ------------------------------------------------------ programs as data
    {
        "group": "Programs as data",
        "title": "A program is a list",
        "program": "nil 'add cons '1 cons",
        "note": "'add pushes the NAME of add as a value. Built back to front, "
        "because cons prepends. Nothing here is text.",
    },
    {
        "group": "Programs as data",
        "title": "Running what you built",
        "program": "1 1 nil 'add cons i",
        "note": "i takes the quotation off the stack and runs it. Build, then run.",
    },
    {
        "group": "Programs as data",
        "title": "Taking a program apart",
        "program": "nil 'add cons '1 cons uncons",
        "note": "A program comes apart with the same uncons as any other list. "
        "That is what homoiconicity buys.",
    },
    {
        "group": "Programs as data",
        "title": "Joining two programs",
        "program": "nil '1 cons nil 'add cons cat",
        "note": "cat concatenates quotations, which is exactly composing the "
        "programs they stand for.",
    },
    {
        "group": "Programs as data",
        "title": "quote is not magic",
        "program": "1 quote 1 nil swap cons eq?",
        "note": "quote is shorthand for nil swap cons. The two agree, so it is "
        "convenience rather than new power.",
    },
    # ---------------------------------------------------------- combinators
    {
        "group": "Combinators",
        "title": "Increment every element",
        "program": "nil 1 cons 0 cons 1 cons nil 'add cons '1 cons map",
        "note": "The thing eighteen ops could not express at any length: a "
        "program whose work depends on how long the input is.",
    },
    {
        "group": "Combinators",
        "title": "Sum a list",
        "program": "nil 1 cons 1 cons 1 cons 0 nil 'add cons fold",
        "note": "fold takes the list, a starting value, and a two-argument body.",
    },
    {
        "group": "Combinators",
        "title": "Fold that cares about order",
        "program": "nil 1 cons 1 cons 1 cons 0 nil 'sub cons fold",
        "note": "sub is not commutative, so this pins down which of the two "
        "values the body sees first: the accumulator, then the element.",
    },
    {
        "group": "Combinators",
        "title": "Keep the zeros",
        "program": "nil 1 cons 0 cons 1 cons nil 'eq? cons '0 cons filter",
        "note": "filter's body must leave a boolean -- the guard enforces it.",
    },
    {
        "group": "Combinators",
        "title": "Choosing a branch",
        "program": "0 1 nil 'lt? cons nil '1 cons nil '0 cons ifte",
        "note": "Three quotations: a test, a then, an else. The test's own stack "
        "effect is discarded.",
    },
    {
        "group": "Combinators",
        "title": "Reaching underneath",
        "program": "1 1 0 nil 'add cons dip",
        "note": "dip runs a quotation with the top value set aside, then puts it "
        "back. It needed quotations to exist at all.",
    },
    {
        "group": "Combinators",
        "title": "A map inside a map",
        "program": "nil  nil 1 cons 0 cons cons  nil 1 cons 1 cons cons  "
        "nil 'add cons '1 cons quote nil 'map cons cat  map",
        "note": "quote keeps the inner quotation nested instead of flattening it. "
        "Increments every element of every sublist.",
    },
    # -------------------------------------------------------- fuel, totality
    {
        "group": "Fuel and totality",
        "title": "One op, seven fuel",
        "program": "nil 1 cons 0 cons 1 cons nil 'add cons '1 cons map",
        "note": "Watch fuel jump by seven on the last step. Cost stopped being "
        "one per op, so shortest stopped meaning fastest.",
    },
    {
        "group": "Fuel and totality",
        "title": "The program that never ends",
        "program": "nil 'i cons 'dup cons dup",
        "note": "Now press i. That is [dup i] dup i, the classic loop that never "
        "returns -- and the guard simply does not offer it.",
    },
    {
        "group": "Fuel and totality",
        "title": "Self-application that stops",
        "program": "nil 'drop cons dup i",
        "note": "A quotation may call itself and still finish. Only the ones that "
        "do not are refused.",
    },
    # --------------------------------------------------- examples/prediction
    {
        "group": "Learning from examples",
        "title": "Second of two",
        "inputs": ["3 5", "2 7", "9 4"],
        "program": "swap drop",
        "note": "A programming-by-example task: one program, every input. An op "
        "is legal only if it is legal on all of them at once.",
    },
    {
        "group": "Learning from examples",
        "title": "Drop the head",
        "inputs": ["(1 0 1)", "(0 1)", "(1)"],
        "program": "uncons drop",
        "note": "Three lists of different lengths, one program. Try uncons twice "
        "and watch it fall off the shortest input.",
    },
    {
        "group": "Learning from examples",
        "title": "Add a zero in front",
        "inputs": ["(0 1)", "(1 1)", "nil"],
        "program": "0 cons",
        "note": "Works on the empty list too, which is why nil is a list and not "
        "a separate thing.",
    },
    {
        "group": "Learning from examples",
        "title": "Difference of two",
        "inputs": ["7 3", "9 2", "5 5"],
        "program": "sub",
        "note": "a b sub is a minus b, not the other way round. Three examples "
        "are enough to pin the order down; one would not be.",
    },
    {
        "group": "Learning from examples",
        "title": "Division rounds down",
        "inputs": ["7 2", "9 3", "5 4"],
        "program": "div",
        "note": "Integer division, flooring toward minus infinity. Add a "
        "negative input and the rounding direction shows.",
    },
    {
        "group": "Learning from examples",
        "title": "XOR costs three",
        "inputs": ["0 0", "0 1", "1 0", "1 1"],
        "program": "add 1 eq?",
        "note": "A complete order-2 binary language model: predict the next bit "
        "from the last two. Every context is an example.",
    },
    {
        "group": "Learning from examples",
        "title": "AND costs one",
        "inputs": ["0 0", "0 1", "1 0", "1 1"],
        "program": "mul",
        "note": "The same task shape, one op. The basis has opinions about which "
        "rules are cheap.",
    },
    {
        "group": "Learning from examples",
        "title": "OR costs four",
        "inputs": ["0 0", "0 1", "1 0", "1 1"],
        "program": "add 0 swap lt?",
        "note": "And opinions about which are dear. AND is one op, OR is four -- "
        "that asymmetry is the prior you would be learning on top of.",
    },
]


def mask_string(config: Config) -> str:
    return "".join("1" if ok else "0" for ok in legal_mask(config))


def trace(source: str, start: tuple = ()) -> list[dict]:
    """The full observable trace: stack, fuel and legal mask at every prefix."""
    config = Config(start)
    rows = [
        {"stack": format_stack(config), "fuel": config.fuel, "mask": mask_string(config)}
    ]
    for op in parse_program(source):
        config = step(config, op)
        rows.append(
            {
                "stack": format_stack(config),
                "fuel": config.fuel,
                "mask": mask_string(config),
            }
        )
    return rows


def example_record(example: dict) -> dict:
    inputs = example.get("inputs", [""])
    stacks = [parse_stack(source) for source in inputs]
    record = {
        "group": example["group"],
        "kind": example.get("kind", "program"),
        "title": example["title"],
        "program": example["program"],
        "note": example["note"],
        "inputs": inputs,
        "input_stacks": [format_stack(Config(s)) for s in stacks],
        "trace": trace(example["program"], stacks[0]),
    }
    if len(stacks) > 1:
        results = []
        for stack in stacks:
            config = Config(stack)
            for op in parse_program(example["program"]):
                config = step(config, op)
            results.append(format_stack(config))
        record["results"] = results
    return record


def random_walks(count: int, length: int, seed: int) -> list[dict]:
    """Random legal programs, recorded as program plus final observable state."""
    rng = random.Random(seed)
    walks = []
    for _ in range(count):
        config = Config()
        names = []
        for _ in range(length):
            choices = [op for op in OPS if op.guard(config) is None]
            if not choices:
                break
            op = rng.choice(choices)
            config = step(config, op)
            names.append(op.name)
        walks.append(
            {
                "program": " ".join(names),
                "stack": format_stack(config),
                "fuel": config.fuel,
                "mask": mask_string(config),
            }
        )
    return walks


def build() -> dict:
    return {
        "ops": [
            {"name": op.name, "group": op.group, "effect": op.effect} for op in OPS
        ],
        "examples": [example_record(example) for example in EXAMPLES],
        "walks": random_walks(count=250, length=10, seed=42),
    }


if __name__ == "__main__":
    print(json.dumps(build(), indent=1, sort_keys=True))
