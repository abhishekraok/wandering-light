import hashlib
import math
import statistics

from wandering_light.basis_set import load_basis_set


def _stable_hash(value: str) -> int:
    return int.from_bytes(
        hashlib.sha256(value.encode()).digest()[:8], "big", signed=True
    )


# Sample inputs and expected outputs for testing
SAMPLE_INPUTS = {
    "builtins.int": [0, 1, 2, 3, 4, 5, -1, -2],
    "builtins.float": [0.5, 2.5, 4.0, -0.5, -2.5],
    "builtins.str": ["", "a", "Ab", "hello world", "123", "123.45"],
    "builtins.bool": [True, False],
    "builtins.list": [[], [3, 1, 2], [1, 2, 3]],
    "builtins.tuple": [(), (1, 2, 3), ("a", "b", "c")],
    "builtins.set": [set(), {1}, {2}],
    "builtins.dict": [{}, {"a": 1}, {"x": 1, "y": 2}],
    "builtins.bytes": [b"", b"\x00\x01", b"abc"],
    "builtins.bytearray": [bytearray(), bytearray(b"\x00\x01"), bytearray(b"abc")],
    "builtins.complex": [1 + 2j, -1 + 0j, 0 - 3j],
    "builtins.range": [range(0), range(3), range(1, 4)],
}

EXPECTED_OUTPUTS = {
    "inc": [1, 2, 3, 4, 5, 6, 0, -1],
    "dec": [-1, 0, 1, 2, 3, 4, -2, -3],
    "double": [0, 2, 4, 6, 8, 10, -2, -4],
    "half": [0, 0, 1, 1, 2, 2, -1, -1],
    "square": [0, 1, 4, 9, 16, 25, 1, 4],
    "mod2": [0, 1, 0, 1, 0, 1, 1, 0],
    "neg": [0, -1, -2, -3, -4, -5, 1, 2],
    "abs": [0, 1, 2, 3, 4, 5, 1, 2],
    "sign": [0, 1, 1, 1, 1, 1, -1, -1],
    "f_reciprocal": [2.0, 0.4, 0.25, -2.0, -0.4],
    "f_abs_sqrt": [
        0.7071067811865476,
        1.5811388300841898,
        2.0,
        0.7071067811865476,
        1.5811388300841898,
    ],
    "f_fraction": [0.5, 0.5, 0.0, -0.5, -0.5],
    "f_trunc": [0, 2, 4, 0, -2],
    "f_round": [0, 2, 4, 0, -2],
    "f_mod1": [0.5, 0.5, 0.0, 0.5, 0.5],
    "upper": ["", "A", "AB", "HELLO WORLD", "123", "123.45"],
    "lower": ["", "a", "ab", "hello world", "123", "123.45"],
    "capitalize": ["", "A", "Ab", "Hello world", "123", "123.45"],
    "title": ["", "A", "Ab", "Hello World", "123", "123.45"],
    "strip": ["", "a", "Ab", "hello world", "123", "123.45"],
    "swapcase": ["", "A", "aB", "HELLO WORLD", "123", "123.45"],
    "reverse": ["", "a", "bA", "dlrow olleh", "321", "54.321"],
    "repeat": ["", "aa", "AbAb", "hello worldhello world", "123123", "123.45123.45"],
    "duplicate": ["", "aa", "AbAb", "hello worldhello world", "123123", "123.45123.45"],
    "length": [0, 1, 2, 11, 3, 6],
    "is_digit": [False, False, False, False, True, False],
    "is_alpha": [False, True, True, False, False, False],
    "count_a": [0, 1, 0, 0, 0, 0],
    "startswith_a": [False, True, False, False, False, False],
    "endswith_z": [False, False, False, False, False, False],
    "contains_space": [False, False, False, True, False, False],
    "bool_not": [False, True],
    "bool_to_int": [1, 0],
    "bool_to_str": ["True", "False"],
    "identity_int": [0, 1, 2, 3, 4, 5, -1, -2],
    "is_even": [True, False, True, False, True, False, False, True],
    "is_odd": [False, True, False, True, False, True, True, False],
    "int_to_bool": [False, True, True, True, True, True, True, True],
    "int_to_str": ["0", "1", "2", "3", "4", "5", "-1", "-2"],
    "f_abs": [0.5, 2.5, 4.0, 0.5, 2.5],
    "f_floor": [0, 2, 4, -1, -3],
    "f_ceil": [1, 3, 4, 0, -2],
    "f_square": [0.25, 6.25, 16.0, 0.25, 6.25],
    "is_lower": [False, True, False, True, False, False],
    "is_upper": [False, False, False, False, False, False],
    "is_space": [False, False, False, False, False, False],
    "is_title": [False, False, True, False, False, False],
    "is_numeric": [False, False, False, False, True, False],
    "is_positive": [False, True, True, True, True, True, False, False],
    "is_negative": [False, False, False, False, False, False, True, True],
    "int_to_float": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, -1.0, -2.0],
    "float_to_str": ["0.5", "2.5", "4.0", "-0.5", "-2.5"],
    "first_char": ["", "a", "A", "h", "1", "1"],
    "last_char": ["", "a", "b", "d", "3", "5"],
    "bool_identity": [True, False],
    "bool_to_float": [1.0, 0.0],
    "list_length": [0, 3, 3],
    "list_reverse": [[], [2, 1, 3], [3, 2, 1]],
    "list_sorted": [[], [1, 2, 3], [1, 2, 3]],
    "list_unique": [[], [3, 1, 2], [1, 2, 3]],
    "list_sum": [0, 6, 6],
    "list_is_empty": [True, False, False],
    "list_median": [
        float(statistics.median(x)) if x else 0.0
        for x in SAMPLE_INPUTS["builtins.list"]
    ],
    "list_tail": [x[1:] for x in SAMPLE_INPUTS["builtins.list"]],
    "tuple_length": [0, 3, 3],
    "tuple_reverse": [(), (3, 2, 1), ("c", "b", "a")],
    "tuple_to_list": [[], [1, 2, 3], ["a", "b", "c"]],
    "tuple_is_empty": [True, False, False],
    "set_size": [0, 1, 1],
    "set_is_empty": [True, False, False],
    "set_to_list": [[], [1], [2]],
    "dict_keys": [[], ["a"], ["x", "y"]],
    "dict_values": [[], [1], [1, 2]],
    "dict_items": [[], [("a", 1)], [("x", 1), ("y", 2)]],
    "dict_length": [0, 1, 2],
    "dict_is_empty": [True, False, False],
    "dict_keyset": [set(), {"a"}, {"x", "y"}],
    "bytes_length": [0, 2, 3],
    "bytes_to_hex": ["", "0001", "616263"],
    "bytes_upper": [b"", b"\x00\x01", b"ABC"],
    "bytes_is_empty": [True, False, False],
    "bytearray_length": [0, 2, 3],
    "bytearray_to_bytes": [b"", b"\x00\x01", b"abc"],
    "bytearray_reverse": [bytearray(b""), bytearray(b"\x01\x00"), bytearray(b"cba")],
    "complex_real": [1.0, -1.0, 0.0],
    "complex_imag": [2.0, 0.0, -3.0],
    "complex_abs": [2.23606797749979, 1.0, 3.0],
    "range_length": [0, 3, 3],
    "range_list": [[], [0, 1, 2], [1, 2, 3]],
    "range_sum": [0, 3, 6],
    "int_bit_length": [0, 1, 2, 2, 3, 3, 1, 2],
    "int_popcount": [0, 1, 1, 2, 1, 2, 1, 1],
    "int_is_power_of_two": [False, True, True, False, True, False, False, False],
    "int_clip_0_100": [0, 1, 2, 3, 4, 5, 0, 0],
    "f_log10": [
        math.log10(x) if x > 0 else 0.0 for x in SAMPLE_INPUTS["builtins.float"]
    ],
    "f_exp": [math.exp(x) for x in SAMPLE_INPUTS["builtins.float"]],
    "f_sin": [math.sin(x) for x in SAMPLE_INPUTS["builtins.float"]],
    "f_is_integer": [x.is_integer() for x in SAMPLE_INPUTS["builtins.float"]],
    "f_frac_percent": [
        int((x - int(x)) * 100) for x in SAMPLE_INPUTS["builtins.float"]
    ],
    "str_is_palindrome": [
        x.lower() == x.lower()[::-1] for x in SAMPLE_INPUTS["builtins.str"]
    ],
    "str_count_vowels": [
        sum(1 for c in x.lower() if c in "aeiou") for x in SAMPLE_INPUTS["builtins.str"]
    ],
    "str_remove_digits": [
        "".join(c for c in x if not c.isdigit()) for x in SAMPLE_INPUTS["builtins.str"]
    ],
    "str_reverse_words": [
        " ".join(x.split()[::-1]) for x in SAMPLE_INPUTS["builtins.str"]
    ],
    "str_to_list": [list(x) for x in SAMPLE_INPUTS["builtins.str"]],
    "str_hash": [_stable_hash(x) for x in SAMPLE_INPUTS["builtins.str"]],
    "list_max": [0 if not x else max(x) for x in SAMPLE_INPUTS["builtins.list"]],
    "list_min": [0 if not x else min(x) for x in SAMPLE_INPUTS["builtins.list"]],
    "tuple_count_none": [x.count(None) for x in SAMPLE_INPUTS["builtins.tuple"]],
    "tuple_to_index_dict": [
        dict(enumerate(x)) for x in SAMPLE_INPUTS["builtins.tuple"]
    ],
    "dict_freeze": [tuple(sorted(x.items())) for x in SAMPLE_INPUTS["builtins.dict"]],
    "dict_has_duplicate_values": [
        len(list(x.values())) != len(set(x.values()))
        for x in SAMPLE_INPUTS["builtins.dict"]
    ],
    "dict_flip": [{v: k for k, v in x.items()} for x in SAMPLE_INPUTS["builtins.dict"]],
    "bytes_reverse": [x[::-1] for x in SAMPLE_INPUTS["builtins.bytes"]],
    "bytes_is_ascii": [
        all(b < 128 for b in x) for x in SAMPLE_INPUTS["builtins.bytes"]
    ],
    "complex_conjugate": [x.conjugate() for x in SAMPLE_INPUTS["builtins.complex"]],
    "complex_phase": [
        math.atan2(x.imag, x.real) for x in SAMPLE_INPUTS["builtins.complex"]
    ],
    "range_max": [x[-1] if x else 0 for x in SAMPLE_INPUTS["builtins.range"]],
    "set_hash": [
        _stable_hash(
            repr(
                sorted(
                    x,
                    key=lambda value: (
                        type(value).__module__,
                        type(value).__qualname__,
                        repr(value),
                    ),
                )
            )
        )
        for x in SAMPLE_INPUTS["builtins.set"]
    ],
}
# basic_fns remains the runtime compatibility API. The immutable manifest is
# the source of truth, and each explicit load can create an independent runtime set.
BASIC_BASIS_SET = load_basis_set("default")
basic_fns = BASIC_BASIS_SET.as_function_set()
