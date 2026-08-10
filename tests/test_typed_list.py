import pytest

from wandering_light.typed_list import TypedList


def test_int_list_roundtrip():
    tl = TypedList([1, 2, 3])
    s = tl.to_string()
    tl2 = TypedList.from_str(s)
    assert tl == tl2


def test_nested_non_json_values_roundtrip():
    original = TypedList(
        [
            {
                ("tuple", 1): {3, 2},
                4: [bytearray(b"x"), range(1, 4)],
            }
        ],
        item_type=dict,
    )

    serialized = original.to_string()

    assert TypedList.from_str(serialized) == original
    assert serialized == original.to_string()


def test_legacy_tagged_serialization_still_loads():
    serialized = (
        '{"type": "builtins.tuple", "items": '
        '[{"__tuple__": [1, {"__bytes__": [97]}]}]}'
    )

    assert TypedList.from_str(serialized) == TypedList([(1, b"a")], tuple)


def test_state_equality_handles_nan_and_dict_order():
    assert TypedList([float("nan")]) == TypedList([float("nan")])
    assert TypedList([-0.0]) == TypedList([0.0])
    assert TypedList([{"a": 1, "b": 2}], dict) == TypedList(
        [{"b": 2, "a": 1}], dict
    )


def test_mixed_type_error():
    with pytest.raises(TypeError):
        TypedList([1, "two", 3])


def test_empty_list_requires_type():
    with pytest.raises(ValueError):
        TypedList([])


def test_len_and_iteration_and_append():
    tl = TypedList([1, 2])
    assert len(tl) == 2
    collected = list(tl)
    assert collected == [1, 2]
    tl.append(3)
    assert tl.items == [1, 2, 3]


def test_append_type_mismatch():
    tl = TypedList([1])
    with pytest.raises(TypeError):
        tl.append("bad")


def test_parse_from_repr():
    """Test that TypedList can be parsed from its __repr__ string format."""
    # Test with int list
    tl = TypedList([1, 2, 3])
    repr_str = repr(tl)  # "TL<int>([1, 2, 3])"
    parsed_tl = TypedList.parse_from_repr(repr_str)
    assert parsed_tl == tl
    assert parsed_tl.items == [1, 2, 3]
    assert parsed_tl.item_type == int

    # Test with string list
    tl_str = TypedList(["hello", "world"])
    repr_str_str = repr(tl_str)  # "TL<str>(['hello', 'world'])"
    parsed_tl_str = TypedList.parse_from_repr(repr_str_str)
    assert parsed_tl_str == tl_str
    assert parsed_tl_str.items == ["hello", "world"]
    assert parsed_tl_str.item_type == str

    # Test with float list
    tl_float = TypedList([1.5, 2.5, 3.5])
    repr_str_float = repr(tl_float)  # "TL<float>([1.5, 2.5, 3.5])"
    parsed_tl_float = TypedList.parse_from_repr(repr_str_float)
    assert parsed_tl_float == tl_float
    assert parsed_tl_float.items == [1.5, 2.5, 3.5]
    assert parsed_tl_float.item_type == float

    # Test with complex list
    tl_complex = TypedList([1 + 2j, 3 - 4j, 5j])
    repr_str_complex = repr(tl_complex)  # "TL<complex>([1+2j, 3-4j, 5j])"
    parsed_tl_complex = TypedList.parse_from_repr(repr_str_complex)
    assert parsed_tl_complex == tl_complex
    assert parsed_tl_complex.items == [1 + 2j, 3 - 4j, 5j]
    assert parsed_tl_complex.item_type == complex

    # Test with empty list
    tl_empty = TypedList([], item_type=int)
    repr_str_empty = repr(tl_empty)  # "TL<int>([])"
    parsed_tl_empty = TypedList.parse_from_repr(repr_str_empty)
    assert parsed_tl_empty == tl_empty
    assert parsed_tl_empty.items == []
    assert parsed_tl_empty.item_type == int


def test_parse_from_repr_invalid_format():
    """Test that parse_from_repr raises ValueError for invalid formats."""
    import pytest

    # Test invalid format strings
    with pytest.raises(ValueError, match="Invalid TypedList repr format"):
        TypedList.parse_from_repr("invalid")

    with pytest.raises(ValueError, match="Invalid TypedList repr format"):
        TypedList.parse_from_repr("TL<int>")

    with pytest.raises(ValueError, match="Invalid TypedList repr format"):
        TypedList.parse_from_repr("not_tl([1, 2, 3])")

    # Test unsupported type
    with pytest.raises(ValueError, match="Unsupported type in repr"):
        TypedList.parse_from_repr("TL<CustomType>([1, 2, 3])")


def test_parse_from_repr_roundtrip():
    """Test that repr -> parse_from_repr -> repr produces the same result."""
    test_cases = [
        TypedList([1, 2, 3]),
        TypedList(["a", "b", "c"]),
        TypedList([1.1, 2.2]),
        TypedList([True, False]),
        TypedList([1 + 2j, 3 - 4j]),
        TypedList([], item_type=int),
    ]

    for original in test_cases:
        repr_str = repr(original)
        parsed = TypedList.parse_from_repr(repr_str)
        assert parsed == original
        assert repr(parsed) == repr_str


@pytest.mark.parametrize(
    "original",
    [
        TypedList([range(1, 5, 2)], item_type=range),
        TypedList([bytearray(b"abc")], item_type=bytearray),
        TypedList([set()], item_type=set),
    ],
)
def test_parse_from_repr_safe_constructor_roundtrip(original):
    assert TypedList.parse_from_repr(repr(original)) == original


@pytest.mark.parametrize(
    "expression",
    [
        "TL<int>([1 + 2])",
        "TL<int>([__import__('os').getpid()])",
        "TL<int>([int('3')])",
        "TL<set>([set([1])])",
    ],
)
def test_parse_from_repr_rejects_executable_expressions(expression):
    with pytest.raises(ValueError, match="Failed to parse items"):
        TypedList.parse_from_repr(expression)
