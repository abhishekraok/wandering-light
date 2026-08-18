"""Text forms of a ``TypedList``, for anything with a text box.

A state has two useful text forms: the ``TL<int>([1, 2, 3])`` repr a person
reads and types, and the JSON wire format every artifact stores. Parsing
accepts both; ``display_text`` prefers the repr but only when it round-trips,
since not every builtin does.
"""

from __future__ import annotations

from wandering_light.basis_dataset import typed_list_from_builtin_str
from wandering_light.typed_list import TypedList


def parse_typed_list(text: str) -> TypedList:
    """Accept either a ``TL<int>([1, 2])`` repr or the JSON wire format."""
    stripped = text.strip()
    if stripped.startswith("{"):
        return typed_list_from_builtin_str(stripped)
    return TypedList.parse_from_repr(stripped)


def display_text(serialized: str) -> str:
    """Prefer the readable repr, but only when it parses back to the value.

    Not every builtin round-trips through ``repr``; the JSON the record stores
    always does, so that is the fallback rather than a broken text box.
    """
    value = typed_list_from_builtin_str(serialized)
    text = repr(value)
    try:
        if TypedList.parse_from_repr(text) == value:
            return text
    except (ValueError, TypeError):
        pass
    return serialized
