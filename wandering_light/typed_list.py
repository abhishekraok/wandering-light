import importlib
import json
from typing import TypeVar

T = TypeVar("T")


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
        ser_items = []
        for x in self.items:
            if hasattr(x, "dict"):
                ser_items.append(x.dict())
            elif isinstance(x, bytes):
                ser_items.append({"__bytes__": list(x)})
            elif isinstance(x, bytearray):
                ser_items.append({"__bytearray__": list(x)})
            elif isinstance(x, set):
                ser_items.append({"__set__": list(x)})
            elif isinstance(x, tuple):
                ser_items.append({"__tuple__": list(x)})
            elif isinstance(x, range):
                ser_items.append({"__range__": [x.start, x.stop, x.step]})
            elif isinstance(x, complex):
                ser_items.append({"__complex__": [x.real, x.imag]})
            else:
                ser_items.append(x)
        return json.dumps(
            {
                "type": f"{self.item_type.__module__}.{self.item_type.__qualname__}",
                "items": ser_items,
            }
        )

    @classmethod
    def from_str(cls, s: str) -> "TypedList":
        data = json.loads(s)
        type_str = data["type"]
        module_name, _, class_name = type_str.rpartition(".")
        mod = importlib.import_module(module_name)
        item_type = getattr(mod, class_name)
        items = []
        for x in data["items"]:
            if hasattr(item_type, "parse_obj"):
                items.append(item_type.parse_obj(x))
            elif isinstance(x, dict):
                # Handle special serialized types
                if "__bytes__" in x:
                    items.append(bytes(x["__bytes__"]))
                elif "__bytearray__" in x:
                    items.append(bytearray(x["__bytearray__"]))
                elif "__set__" in x:
                    items.append(set(x["__set__"]))
                elif "__tuple__" in x:
                    items.append(tuple(x["__tuple__"]))
                elif "__range__" in x:
                    start, stop, step = x["__range__"]
                    items.append(range(start, stop, step))
                elif "__complex__" in x:
                    real, imag = x["__complex__"]
                    items.append(complex(real, imag))
                else:
                    items.append(x)
            else:
                items.append(x)
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
            items = eval(items_str)  # Safe since we're parsing our own format
            if not isinstance(items, list):
                raise ValueError("Items must be a list")
        except Exception as e:
            raise ValueError(f"Failed to parse items '{items_str}': {e}") from e

        return cls(items, item_type=item_type)

    def __eq__(self, other):
        return (
            isinstance(other, TypedList)
            and self.item_type == other.item_type
            and self.items == other.items
        )

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
