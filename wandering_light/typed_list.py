import ast
import importlib
import json
import math
from typing import TypeVar

T = TypeVar("T")

_SERIALIZATION_TAGS = {
    "__bytes__",
    "__bytearray__",
    "__set__",
    "__frozenset__",
    "__tuple__",
    "__range__",
    "__complex__",
    "__dict_items__",
}


def _serialize_value(value):
    if hasattr(value, "model_dump"):
        return _serialize_value(value.model_dump())
    if hasattr(value, "dict"):
        return _serialize_value(value.dict())
    if isinstance(value, bytes):
        return {"__bytes__": list(value)}
    if isinstance(value, bytearray):
        return {"__bytearray__": list(value)}
    if isinstance(value, set | frozenset):
        items = [_serialize_value(item) for item in value]
        items.sort(key=lambda item: json.dumps(item, sort_keys=True))
        tag = "__frozenset__" if isinstance(value, frozenset) else "__set__"
        return {tag: items}
    if isinstance(value, tuple):
        return {"__tuple__": [_serialize_value(item) for item in value]}
    if isinstance(value, range):
        return {"__range__": [value.start, value.stop, value.step]}
    if isinstance(value, complex):
        return {"__complex__": [value.real, value.imag]}
    if isinstance(value, list):
        return [_serialize_value(item) for item in value]
    if isinstance(value, dict):
        use_plain_dict = all(isinstance(key, str) for key in value) and not any(
            key in _SERIALIZATION_TAGS for key in value
        )
        if use_plain_dict:
            return {key: _serialize_value(item) for key, item in value.items()}
        pairs = [
            [_serialize_value(key), _serialize_value(item)]
            for key, item in value.items()
        ]
        pairs.sort(key=lambda pair: json.dumps(pair[0], sort_keys=True))
        return {"__dict_items__": pairs}
    return value


def _deserialize_value(value):
    if isinstance(value, list):
        return [_deserialize_value(item) for item in value]
    if not isinstance(value, dict):
        return value
    if set(value) == {"__bytes__"}:
        return bytes(value["__bytes__"])
    if set(value) == {"__bytearray__"}:
        return bytearray(value["__bytearray__"])
    if set(value) == {"__set__"}:
        return {_deserialize_value(item) for item in value["__set__"]}
    if set(value) == {"__frozenset__"}:
        return frozenset(_deserialize_value(item) for item in value["__frozenset__"])
    if set(value) == {"__tuple__"}:
        return tuple(_deserialize_value(item) for item in value["__tuple__"])
    if set(value) == {"__range__"}:
        start, stop, step = value["__range__"]
        return range(start, stop, step)
    if set(value) == {"__complex__"}:
        real, imag = value["__complex__"]
        return complex(real, imag)
    if set(value) == {"__dict_items__"}:
        return {
            _deserialize_value(key): _deserialize_value(item)
            for key, item in value["__dict_items__"]
        }
    return {key: _deserialize_value(item) for key, item in value.items()}


def _canonical_value(value, *, signed_zero: bool = False):
    """Freeze serialized values with Python numeric equality and stable NaNs.

    ``signed_zero`` keeps ``-0.0`` distinct from ``0.0``.  Answer checking wants
    them equal, because Python says they are; search pruning wants them apart,
    because the basis can tell them apart.  See ``search_key``.
    """

    def freeze(item):
        if isinstance(item, float):
            if math.isnan(item):
                return ("float", "nan")
            if item == 0 and not signed_zero:
                return ("float", (0.0).hex())
            return ("float", item.hex())
        if isinstance(item, list):
            return ("list", tuple(freeze(value) for value in item))
        if isinstance(item, dict):
            return (
                "dict",
                tuple((key, freeze(value)) for key, value in sorted(item.items())),
            )
        return (type(item).__name__, item)

    return freeze(_serialize_value(value))


def _safe_repr_literal(node: ast.AST):
    """Evaluate literals plus safe constructors emitted by ``TypedList.__repr__``."""
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError):
        pass
    if isinstance(node, ast.List):
        return [_safe_repr_literal(item) for item in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_safe_repr_literal(item) for item in node.elts)
    if isinstance(node, ast.Set):
        return {_safe_repr_literal(item) for item in node.elts}
    if isinstance(node, ast.Dict):
        return {
            _safe_repr_literal(key): _safe_repr_literal(value)
            for key, value in zip(node.keys, node.values, strict=True)
        }
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"bytearray", "range"}
        and not node.keywords
    ):
        constructors = {"bytearray": bytearray, "range": range}
        return constructors[node.func.id](
            *(_safe_repr_literal(arg) for arg in node.args)
        )
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "set"
        and not node.args
        and not node.keywords
    ):
        return set()
    raise ValueError("expression is not a supported literal")


class TypedList[T]:
    """
    A typed list implementation that supports serialization and deserialization.

    Args:
        T: The type parameter for the items in the list
    """

    def __init__(self, items: list[T], item_type: type[T] | None = None):
        if not items and item_type is None:
            raise ValueError("Cannot infer type from empty list without item_type")
        self.item_type = item_type or type(items[0])
        for x in items:
            if not isinstance(x, self.item_type):
                raise TypeError(
                    f"Expected items of type {self.item_type}, got {type(x)}"
                )
        self.items = items

    def to_string(self) -> str:
        """
        Serialize the typed list as a JSON string, including a type tag and the items.
        """
        return json.dumps(
            {
                "type": f"{self.item_type.__module__}.{self.item_type.__qualname__}",
                "items": [_serialize_value(item) for item in self.items],
            }
        )

    @classmethod
    def from_str(cls, s: str) -> "TypedList":
        data = json.loads(s)
        type_str = data["type"]
        module_name, _, class_name = type_str.rpartition(".")
        mod = importlib.import_module(module_name)
        item_type = getattr(mod, class_name)
        items = [_deserialize_value(item) for item in data["items"]]
        if hasattr(item_type, "parse_obj"):
            items = [item_type.parse_obj(item) for item in items]
        return TypedList(items, item_type=item_type)

    @classmethod
    def parse_from_repr(cls, repr_str: str) -> "TypedList":
        """
        Parse a TypedList from its __repr__ string format.

        Args:
            repr_str: String like "TL<int>([1, 2, 3])"

        Returns:
            TypedList parsed from the repr string

        Raises:
            ValueError: If the string format is invalid
        """
        import re

        # Match pattern: TL<type_name>([items])
        pattern = r"TL<(\w+)>\((\[.*\])\)"
        match = re.match(pattern, repr_str.strip())

        if not match:
            raise ValueError(f"Invalid TypedList repr format: {repr_str}")

        type_name, items_str = match.groups()

        # Map common type names to actual types
        type_mapping = {
            "int": int,
            "float": float,
            "str": str,
            "bool": bool,
            "list": list,
            "dict": dict,
            "tuple": tuple,
            "set": set,
            "bytes": bytes,
            "bytearray": bytearray,
            "complex": complex,
            "range": range,
        }

        if type_name not in type_mapping:
            raise ValueError(f"Unsupported type in repr: {type_name}")

        item_type = type_mapping[type_name]

        # Parse the items list
        try:
            expression = ast.parse(items_str, mode="eval")
            items = _safe_repr_literal(expression.body)
            if not isinstance(items, list):
                raise ValueError("Items must be a list")
        except Exception as e:
            raise ValueError(f"Failed to parse items '{items_str}': {e}") from e

        return cls(items, item_type=item_type)

    def __eq__(self, other):
        return (
            isinstance(other, TypedList)
            and self.canonical_key() == other.canonical_key()
        )

    def canonical_key(self) -> tuple[type[T], object]:
        """Return a hashable structural key for this typed value sequence."""
        return (self.item_type, _canonical_value(self.items))

    def search_key(self) -> tuple[type[T], object]:
        """Return a key safe to deduplicate *search* states on.

        Two states may be merged during a search only if no basis function can
        tell them apart; otherwise the search prunes a state whose successors
        differ and can miss, or over-estimate the distance to, a target.
        ``canonical_key`` follows Python numeric equality, under which
        ``-0.0 == 0.0`` -- but ``float_to_str``, ``f_fraction`` and ``f_sin``
        all distinguish the two, so signed zero must survive here.  NaNs stay
        collapsed: no basis function observes a NaN's sign or payload, and they
        are not equal to themselves.
        """
        return (self.item_type, _canonical_value(self.items, signed_zero=True))

    def __repr__(self):
        return f"TL<{self.item_type.__name__}>({self.items})"

    def __len__(self) -> int:
        """Return the number of items in the list."""
        return len(self.items)

    def __iter__(self):
        """Iterate over the contained items."""
        return iter(self.items)

    def __getitem__(self, index: int) -> T:
        """Access an item by index."""
        return self.items[index]

    def append(self, item: T) -> None:
        """Append an item ensuring it matches the list's type."""
        if not isinstance(item, self.item_type):
            raise TypeError(f"Expected item of type {self.item_type}, got {type(item)}")
        self.items.append(item)
