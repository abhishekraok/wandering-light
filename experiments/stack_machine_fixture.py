"""Generate the conformance fixture the stack machine explorer checks itself against.

The explorer UI is a JavaScript port of `wandering_light.stack_machine`. A port
that quietly disagrees with the machine is worse than no explorer at all, so the
port carries this fixture and replays it on load. Regenerate with:

    uv run python experiments/stack_machine_fixture.py
"""

import json
import random

from wandering_light.stack_machine import (
    FIRST_ORDER_OPS,
    Config,
    format_stack,
    parse_program,
    step,
)

# The explorer teaches the first-order basis. Quotations and combinators are
# deliberately outside it: porting a fuel-metered quotation interpreter to
# JavaScript would be a second machine to keep in step, and the page's whole
# claim is that it does not drift from this one.
OPS = FIRST_ORDER_OPS


def legal_ops(config):
    return tuple(op for op in OPS if op.guard(config) is None)


def legal_mask(config):
    return tuple(op.guard(config) is None for op in OPS)

EXAMPLES = [
    {
        "title": "The round trip",
        "program": "nil 1 cons uncons",
        "note": "uncons undoes cons: head on top, tail below.",
    },
    {
        "title": "Building a list",
        "program": "nil 1 cons 0 cons 1 cons",
        "note": "Every list starts as nil and grows one cons at a time.",
    },
    {
        "title": "A list inside a list",
        "program": "nil nil 1 cons cons",
        "note": "Lists are values, so a list can hold another list.",
    },
    {
        "title": "Taking a list apart",
        "program": "nil 1 cons 1 cons uncons swap uncons swap drop add",
        "note": "Destructuring without variables — just uncons, swap and drop.",
    },
    {
        "title": "Exactly three to the sixty-fourth",
        "program": "1 1 add 1 add dup mul dup mul dup mul dup mul dup mul dup mul",
        "note": "Integers are arbitrary-precision. A float would round this one.",
    },
    {
        "title": "Same shape, same state",
        "program": "nil 1 cons nil 1 cons eq?",
        "note": "Two lists built separately are the same value. Identity is structural.",
    },
    {
        "title": "One is not true",
        "program": "1 true",
        "note": "Look for add. It is gone — booleans are not numbers here.",
    },
    {
        "title": "Where did div go?",
        "program": "1 0",
        "note": "The guard reads the divisor, not just its type. Zero, so no div.",
    },
    {
        "title": "Asking if it is empty",
        "program": "nil dup null?",
        "note": "null? consumes its list, so dup first if you still need it.",
    },
    {
        "title": "Reaching underneath",
        "program": "0 1 over",
        "note": "over copies the second value to the top.",
    },
    {
        "title": "Start from nothing",
        "program": "",
        "note": "An empty stack. Only the five literals are legal.",
    },
]


# Programs that pin down behaviour the handpicked examples and random walks do
# not reach. Each one exists because a plausibly-wrong port passes without it.
PROBES = [
    # Division floors toward minus infinity; a truncating port returns -3 / 0.
    "1 1 add dup mul 1 1 add add 1 add 0 swap sub 1 1 add div",
    "0 1 sub 1 1 add div",
    # Negative intermediates survive mul and lt?.
    "0 1 sub dup mul",
    "0 1 sub 0 lt?",
    # eq? across types. A port comparing payloads with JavaScript's loose `==`
    # answers true to the first two; the machine says false to all of them.
    "1 true eq?",
    "0 false eq?",
    "0 nil eq?",
    "nil false eq?",
    "nil 1 cons 1 eq?",
    # A list built two ways is one value.
    "nil 1 cons 0 cons nil 1 cons 0 cons eq?",
]


def trace(source: str) -> list[dict]:
    """The full observable trace: stack, fuel and legal ops at every prefix."""
    config = Config()
    rows = [
        {
            "stack": format_stack(config),
            "fuel": config.fuel,
            "legal": [op.name for op in legal_ops(config)],
        }
    ]
    for op in parse_program(source):
        config = step(config, op)
        rows.append(
            {
                "stack": format_stack(config),
                "fuel": config.fuel,
                "legal": [op.name for op in legal_ops(config)],
            }
        )
    return rows


def random_walks(count: int, length: int, seed: int) -> list[dict]:
    """Random legal programs, recorded as program plus final observable state."""
    rng = random.Random(seed)
    walks = []
    for _ in range(count):
        config = Config()
        names = []
        for _ in range(length):
            choices = legal_ops(config)
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
                "mask": "".join("1" if ok else "0" for ok in legal_mask(config)),
            }
        )
    return walks


def build() -> dict:
    return {
        "ops": [
            {"name": op.name, "group": op.group, "effect": op.effect} for op in OPS
        ],
        "examples": [
            {**example, "trace": trace(example["program"])} for example in EXAMPLES
        ],
        "probes": [{"program": p, "trace": trace(p)} for p in PROBES],
        "walks": random_walks(count=400, length=14, seed=42),
    }


if __name__ == "__main__":
    print(json.dumps(build(), indent=1, sort_keys=True))
