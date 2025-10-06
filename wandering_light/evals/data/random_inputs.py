"""
Generated trajectory specs data
Total specs: 100
"""

from wandering_light.function_def import FunctionDef, FunctionDefList
from wandering_light.trajectory import TrajectorySpec, TrajectorySpecList
from wandering_light.typed_list import TypedList

# Function definitions
f_ceil_func = FunctionDef(
    name="f_ceil",
    input_type="builtins.float",
    output_type="builtins.int",
    code="""return __import__('math').ceil(x)""",
    usage_count=0,
    metadata={},
)

int_to_float_func = FunctionDef(
    name="int_to_float",
    input_type="builtins.int",
    output_type="builtins.float",
    code="""return float(x)""",
    usage_count=0,
    metadata={},
)

list_tail_func = FunctionDef(
    name="list_tail",
    input_type="builtins.list",
    output_type="builtins.list",
    code="""return x[1:]""",
    usage_count=0,
    metadata={},
)

bytes_length_func = FunctionDef(
    name="bytes_length",
    input_type="builtins.bytes",
    output_type="builtins.int",
    code="""return len(x)""",
    usage_count=0,
    metadata={},
)

str_remove_digits_func = FunctionDef(
    name="str_remove_digits",
    input_type="builtins.str",
    output_type="builtins.str",
    code="""return ''.join(c for c in x if not c.isdigit())""",
    usage_count=0,
    metadata={},
)

tuple_to_index_dict_func = FunctionDef(
    name="tuple_to_index_dict",
    input_type="builtins.tuple",
    output_type="builtins.dict",
    code="""return {i:v for i,v in enumerate(x)}""",
    usage_count=0,
    metadata={},
)

bytes_reverse_func = FunctionDef(
    name="bytes_reverse",
    input_type="builtins.bytes",
    output_type="builtins.bytes",
    code="""return x[::-1]""",
    usage_count=0,
    metadata={},
)

bool_to_int_func = FunctionDef(
    name="bool_to_int",
    input_type="builtins.bool",
    output_type="builtins.int",
    code="""return int(x)""",
    usage_count=0,
    metadata={},
)

range_sum_func = FunctionDef(
    name="range_sum",
    input_type="builtins.range",
    output_type="builtins.int",
    code="""return sum(x)""",
    usage_count=0,
    metadata={},
)

duplicate_func = FunctionDef(
    name="duplicate",
    input_type="builtins.str",
    output_type="builtins.str",
    code="""return x + x""",
    usage_count=0,
    metadata={},
)

reverse_func = FunctionDef(
    name="reverse",
    input_type="builtins.str",
    output_type="builtins.str",
    code="""return x[::-1]""",
    usage_count=0,
    metadata={},
)

bool_identity_func = FunctionDef(
    name="bool_identity",
    input_type="builtins.bool",
    output_type="builtins.bool",
    code="""return x""",
    usage_count=0,
    metadata={},
)

complex_phase_func = FunctionDef(
    name="complex_phase",
    input_type="builtins.complex",
    output_type="builtins.float",
    code="""import math; return math.atan2(x.imag, x.real)""",
    usage_count=0,
    metadata={},
)

dict_flip_func = FunctionDef(
    name="dict_flip",
    input_type="builtins.dict",
    output_type="builtins.dict",
    code="""return {v:k for k,v in x.items()}""",
    usage_count=0,
    metadata={},
)

is_odd_func = FunctionDef(
    name="is_odd",
    input_type="builtins.int",
    output_type="builtins.bool",
    code="""return x % 2 == 1""",
    usage_count=0,
    metadata={},
)

list_median_func = FunctionDef(
    name="list_median",
    input_type="builtins.list",
    output_type="builtins.float",
    code="""import statistics; return float(statistics.median(x)) if x else 0.0""",
    usage_count=0,
    metadata={},
)

dict_items_func = FunctionDef(
    name="dict_items",
    input_type="builtins.dict",
    output_type="builtins.list",
    code="""return list(x.items())""",
    usage_count=0,
    metadata={},
)

list_sum_func = FunctionDef(
    name="list_sum",
    input_type="builtins.list",
    output_type="builtins.int",
    code="""return sum(x) if x and all(isinstance(v, (int, float)) for v in x) else 0""",
    usage_count=0,
    metadata={},
)

contains_space_func = FunctionDef(
    name="contains_space",
    input_type="builtins.str",
    output_type="builtins.bool",
    code="""return ' ' in x""",
    usage_count=0,
    metadata={},
)

tuple_length_func = FunctionDef(
    name="tuple_length",
    input_type="builtins.tuple",
    output_type="builtins.int",
    code="""return len(x)""",
    usage_count=0,
    metadata={},
)

is_positive_func = FunctionDef(
    name="is_positive",
    input_type="builtins.int",
    output_type="builtins.bool",
    code="""return x > 0""",
    usage_count=0,
    metadata={},
)

f_fraction_func = FunctionDef(
    name="f_fraction",
    input_type="builtins.float",
    output_type="builtins.float",
    code="""return x - int(x)""",
    usage_count=0,
    metadata={},
)

f_reciprocal_func = FunctionDef(
    name="f_reciprocal",
    input_type="builtins.float",
    output_type="builtins.float",
    code="""return float('inf') if x == 0 else 1.0 / x""",
    usage_count=0,
    metadata={},
)

dict_length_func = FunctionDef(
    name="dict_length",
    input_type="builtins.dict",
    output_type="builtins.int",
    code="""return len(x)""",
    usage_count=0,
    metadata={},
)

neg_func = FunctionDef(
    name="neg",
    input_type="builtins.int",
    output_type="builtins.int",
    code="""return -x""",
    usage_count=0,
    metadata={},
)

set_size_func = FunctionDef(
    name="set_size",
    input_type="builtins.set",
    output_type="builtins.int",
    code="""return len(x)""",
    usage_count=0,
    metadata={},
)

dict_freeze_func = FunctionDef(
    name="dict_freeze",
    input_type="builtins.dict",
    output_type="builtins.tuple",
    code="""return tuple(sorted(x.items()))""",
    usage_count=0,
    metadata={},
)

tuple_is_empty_func = FunctionDef(
    name="tuple_is_empty",
    input_type="builtins.tuple",
    output_type="builtins.bool",
    code="""return len(x) == 0""",
    usage_count=0,
    metadata={},
)

tuple_reverse_func = FunctionDef(
    name="tuple_reverse",
    input_type="builtins.tuple",
    output_type="builtins.tuple",
    code="""return x[::-1]""",
    usage_count=0,
    metadata={},
)

set_is_empty_func = FunctionDef(
    name="set_is_empty",
    input_type="builtins.set",
    output_type="builtins.bool",
    code="""return len(x) == 0""",
    usage_count=0,
    metadata={},
)

bool_not_func = FunctionDef(
    name="bool_not",
    input_type="builtins.bool",
    output_type="builtins.bool",
    code="""return not x""",
    usage_count=0,
    metadata={},
)

bool_to_float_func = FunctionDef(
    name="bool_to_float",
    input_type="builtins.bool",
    output_type="builtins.float",
    code="""return 1.0 if x else 0.0""",
    usage_count=0,
    metadata={},
)

tuple_count_none_func = FunctionDef(
    name="tuple_count_none",
    input_type="builtins.tuple",
    output_type="builtins.int",
    code="""return x.count(None)""",
    usage_count=0,
    metadata={},
)

square_func = FunctionDef(
    name="square",
    input_type="builtins.int",
    output_type="builtins.int",
    code="""return x * x""",
    usage_count=0,
    metadata={},
)

float_to_str_func = FunctionDef(
    name="float_to_str",
    input_type="builtins.float",
    output_type="builtins.str",
    code="""return str(x)""",
    usage_count=0,
    metadata={},
)

complex_real_func = FunctionDef(
    name="complex_real",
    input_type="builtins.complex",
    output_type="builtins.float",
    code="""return x.real""",
    usage_count=0,
    metadata={},
)

f_sin_func = FunctionDef(
    name="f_sin",
    input_type="builtins.float",
    output_type="builtins.float",
    code="""import math; return math.sin(x)""",
    usage_count=0,
    metadata={},
)

dec_func = FunctionDef(
    name="dec",
    input_type="builtins.int",
    output_type="builtins.int",
    code="""return x - 1""",
    usage_count=0,
    metadata={},
)

complex_abs_func = FunctionDef(
    name="complex_abs",
    input_type="builtins.complex",
    output_type="builtins.float",
    code="""return abs(x)""",
    usage_count=0,
    metadata={},
)

bytes_is_empty_func = FunctionDef(
    name="bytes_is_empty",
    input_type="builtins.bytes",
    output_type="builtins.bool",
    code="""return len(x) == 0""",
    usage_count=0,
    metadata={},
)

bool_to_str_func = FunctionDef(
    name="bool_to_str",
    input_type="builtins.bool",
    output_type="builtins.str",
    code="""return str(x)""",
    usage_count=0,
    metadata={},
)

half_func = FunctionDef(
    name="half",
    input_type="builtins.int",
    output_type="builtins.int",
    code="""return x // 2""",
    usage_count=0,
    metadata={},
)

bytes_to_hex_func = FunctionDef(
    name="bytes_to_hex",
    input_type="builtins.bytes",
    output_type="builtins.str",
    code="""return x.hex()""",
    usage_count=0,
    metadata={},
)

endswith_z_func = FunctionDef(
    name="endswith_z",
    input_type="builtins.str",
    output_type="builtins.bool",
    code="""return x.endswith('z')""",
    usage_count=0,
    metadata={},
)

range_max_func = FunctionDef(
    name="range_max",
    input_type="builtins.range",
    output_type="builtins.int",
    code="""return x[-1] if x else 0""",
    usage_count=0,
    metadata={},
)

int_to_str_func = FunctionDef(
    name="int_to_str",
    input_type="builtins.int",
    output_type="builtins.str",
    code="""return str(x)""",
    usage_count=0,
    metadata={},
)

f_abs_sqrt_func = FunctionDef(
    name="f_abs_sqrt",
    input_type="builtins.float",
    output_type="builtins.float",
    code="""return abs(x) ** 0.5""",
    usage_count=0,
    metadata={},
)

f_round_func = FunctionDef(
    name="f_round",
    input_type="builtins.float",
    output_type="builtins.int",
    code="""return round(x)""",
    usage_count=0,
    metadata={},
)

is_even_func = FunctionDef(
    name="is_even",
    input_type="builtins.int",
    output_type="builtins.bool",
    code="""return x % 2 == 0""",
    usage_count=0,
    metadata={},
)

complex_conjugate_func = FunctionDef(
    name="complex_conjugate",
    input_type="builtins.complex",
    output_type="builtins.complex",
    code="""return x.conjugate()""",
    usage_count=0,
    metadata={},
)

f_square_func = FunctionDef(
    name="f_square",
    input_type="builtins.float",
    output_type="builtins.float",
    code="""return x * x""",
    usage_count=0,
    metadata={},
)

last_char_func = FunctionDef(
    name="last_char",
    input_type="builtins.str",
    output_type="builtins.str",
    code="""return x[-1] if x else ''""",
    usage_count=0,
    metadata={},
)

length_func = FunctionDef(
    name="length",
    input_type="builtins.str",
    output_type="builtins.int",
    code="""return len(x)""",
    usage_count=0,
    metadata={},
)

int_popcount_func = FunctionDef(
    name="int_popcount",
    input_type="builtins.int",
    output_type="builtins.int",
    code="""return x.bit_count()""",
    usage_count=0,
    metadata={},
)

range_list_func = FunctionDef(
    name="range_list",
    input_type="builtins.range",
    output_type="builtins.list",
    code="""return list(x)""",
    usage_count=0,
    metadata={},
)

list_is_empty_func = FunctionDef(
    name="list_is_empty",
    input_type="builtins.list",
    output_type="builtins.bool",
    code="""return len(x) == 0""",
    usage_count=0,
    metadata={},
)

double_func = FunctionDef(
    name="double",
    input_type="builtins.int",
    output_type="builtins.int",
    code="""return x * 2""",
    usage_count=0,
    metadata={},
)

is_negative_func = FunctionDef(
    name="is_negative",
    input_type="builtins.int",
    output_type="builtins.bool",
    code="""return x < 0""",
    usage_count=0,
    metadata={},
)

bytes_upper_func = FunctionDef(
    name="bytes_upper",
    input_type="builtins.bytes",
    output_type="builtins.bytes",
    code="""return x.upper()""",
    usage_count=0,
    metadata={},
)

inc_func = FunctionDef(
    name="inc",
    input_type="builtins.int",
    output_type="builtins.int",
    code="""return x + 1""",
    usage_count=0,
    metadata={},
)

mod2_func = FunctionDef(
    name="mod2",
    input_type="builtins.int",
    output_type="builtins.int",
    code="""return x % 2""",
    usage_count=0,
    metadata={},
)

dict_values_func = FunctionDef(
    name="dict_values",
    input_type="builtins.dict",
    output_type="builtins.list",
    code="""return list(x.values())""",
    usage_count=0,
    metadata={},
)

tuple_to_list_func = FunctionDef(
    name="tuple_to_list",
    input_type="builtins.tuple",
    output_type="builtins.list",
    code="""return list(x)""",
    usage_count=0,
    metadata={},
)

int_is_power_of_two_func = FunctionDef(
    name="int_is_power_of_two",
    input_type="builtins.int",
    output_type="builtins.bool",
    code="""return x > 0 and (x & (x - 1)) == 0""",
    usage_count=0,
    metadata={},
)

set_hash_func = FunctionDef(
    name="set_hash",
    input_type="builtins.set",
    output_type="builtins.int",
    code="""return hash(frozenset(x))""",
    usage_count=0,
    metadata={},
)

bytearray_reverse_func = FunctionDef(
    name="bytearray_reverse",
    input_type="builtins.bytearray",
    output_type="builtins.bytearray",
    code="""return bytearray(x[::-1])""",
    usage_count=0,
    metadata={},
)

bytearray_to_bytes_func = FunctionDef(
    name="bytearray_to_bytes",
    input_type="builtins.bytearray",
    output_type="builtins.bytes",
    code="""return bytes(x)""",
    usage_count=0,
    metadata={},
)

int_to_bool_func = FunctionDef(
    name="int_to_bool",
    input_type="builtins.int",
    output_type="builtins.bool",
    code="""return bool(x)""",
    usage_count=0,
    metadata={},
)

sign_func = FunctionDef(
    name="sign",
    input_type="builtins.int",
    output_type="builtins.int",
    code="""return 1 if x > 0 else (-1 if x < 0 else 0)""",
    usage_count=0,
    metadata={},
)

f_trunc_func = FunctionDef(
    name="f_trunc",
    input_type="builtins.float",
    output_type="builtins.int",
    code="""return int(x)""",
    usage_count=0,
    metadata={},
)

bytes_is_ascii_func = FunctionDef(
    name="bytes_is_ascii",
    input_type="builtins.bytes",
    output_type="builtins.bool",
    code="""return all(b < 128 for b in x)""",
    usage_count=0,
    metadata={},
)

identity_int_func = FunctionDef(
    name="identity_int",
    input_type="builtins.int",
    output_type="builtins.int",
    code="""return x""",
    usage_count=0,
    metadata={},
)

list_reverse_func = FunctionDef(
    name="list_reverse",
    input_type="builtins.list",
    output_type="builtins.list",
    code="""return x[::-1]""",
    usage_count=0,
    metadata={},
)

list_min_func = FunctionDef(
    name="list_min",
    input_type="builtins.list",
    output_type="builtins.int",
    code="""return min(x) if x and all(isinstance(v, (int, float)) for v in x) else 0""",
    usage_count=0,
    metadata={},
)

dict_has_duplicate_values_func = FunctionDef(
    name="dict_has_duplicate_values",
    input_type="builtins.dict",
    output_type="builtins.bool",
    code="""vals=list(x.values()); return len(vals)!=len(set(vals))""",
    usage_count=0,
    metadata={},
)

strip_func = FunctionDef(
    name="strip",
    input_type="builtins.str",
    output_type="builtins.str",
    code="""return x.strip()""",
    usage_count=0,
    metadata={},
)

title_func = FunctionDef(
    name="title",
    input_type="builtins.str",
    output_type="builtins.str",
    code="""return x.title()""",
    usage_count=0,
    metadata={},
)

upper_func = FunctionDef(
    name="upper",
    input_type="builtins.str",
    output_type="builtins.str",
    code="""return x.upper()""",
    usage_count=0,
    metadata={},
)

lower_func = FunctionDef(
    name="lower",
    input_type="builtins.str",
    output_type="builtins.str",
    code="""return x.lower()""",
    usage_count=0,
    metadata={},
)

is_numeric_func = FunctionDef(
    name="is_numeric",
    input_type="builtins.str",
    output_type="builtins.bool",
    code="""return x.isnumeric()""",
    usage_count=0,
    metadata={},
)

list_sorted_func = FunctionDef(
    name="list_sorted",
    input_type="builtins.list",
    output_type="builtins.list",
    code="""return sorted(x)""",
    usage_count=0,
    metadata={},
)

int_bit_length_func = FunctionDef(
    name="int_bit_length",
    input_type="builtins.int",
    output_type="builtins.int",
    code="""return x.bit_length()""",
    usage_count=0,
    metadata={},
)

f_frac_percent_func = FunctionDef(
    name="f_frac_percent",
    input_type="builtins.float",
    output_type="builtins.int",
    code="""return int((x - int(x)) * 100)""",
    usage_count=0,
    metadata={},
)

dict_is_empty_func = FunctionDef(
    name="dict_is_empty",
    input_type="builtins.dict",
    output_type="builtins.bool",
    code="""return len(x) == 0""",
    usage_count=0,
    metadata={},
)

complex_imag_func = FunctionDef(
    name="complex_imag",
    input_type="builtins.complex",
    output_type="builtins.float",
    code="""return x.imag""",
    usage_count=0,
    metadata={},
)

f_abs_func = FunctionDef(
    name="f_abs",
    input_type="builtins.float",
    output_type="builtins.float",
    code="""return abs(x)""",
    usage_count=0,
    metadata={},
)

set_to_list_func = FunctionDef(
    name="set_to_list",
    input_type="builtins.set",
    output_type="builtins.list",
    code="""return list(x)""",
    usage_count=0,
    metadata={},
)

f_floor_func = FunctionDef(
    name="f_floor",
    input_type="builtins.float",
    output_type="builtins.int",
    code="""return __import__('math').floor(x)""",
    usage_count=0,
    metadata={},
)

f_log10_func = FunctionDef(
    name="f_log10",
    input_type="builtins.float",
    output_type="builtins.float",
    code="""import math; return math.log10(x) if x > 0 else 0.0""",
    usage_count=0,
    metadata={},
)

f_is_integer_func = FunctionDef(
    name="f_is_integer",
    input_type="builtins.float",
    output_type="builtins.bool",
    code="""return x.is_integer()""",
    usage_count=0,
    metadata={},
)

repeat_func = FunctionDef(
    name="repeat",
    input_type="builtins.str",
    output_type="builtins.str",
    code="""return x * 2""",
    usage_count=0,
    metadata={},
)

is_upper_func = FunctionDef(
    name="is_upper",
    input_type="builtins.str",
    output_type="builtins.bool",
    code="""return x.isupper()""",
    usage_count=0,
    metadata={},
)

f_exp_func = FunctionDef(
    name="f_exp",
    input_type="builtins.float",
    output_type="builtins.float",
    code="""import math; return math.exp(x)""",
    usage_count=0,
    metadata={},
)

bytearray_length_func = FunctionDef(
    name="bytearray_length",
    input_type="builtins.bytearray",
    output_type="builtins.int",
    code="""return len(x)""",
    usage_count=0,
    metadata={},
)

abs_func = FunctionDef(
    name="abs",
    input_type="builtins.int",
    output_type="builtins.int",
    code="""return abs(x)""",
    usage_count=0,
    metadata={},
)

dict_keys_func = FunctionDef(
    name="dict_keys",
    input_type="builtins.dict",
    output_type="builtins.list",
    code="""return list(x.keys())""",
    usage_count=0,
    metadata={},
)

list_length_func = FunctionDef(
    name="list_length",
    input_type="builtins.list",
    output_type="builtins.int",
    code="""return len(x)""",
    usage_count=0,
    metadata={},
)

int_clip_0_100_func = FunctionDef(
    name="int_clip_0_100",
    input_type="builtins.int",
    output_type="builtins.int",
    code="""return 0 if x < 0 else (100 if x > 100 else x)""",
    usage_count=0,
    metadata={},
)

# Trajectory specifications
_specs = [
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.float", "items": [-1.8919177274410792, 1.4105770474369237, -7.772782704845566, -0.2028646322123997, 9.26462706223894]}"""
        ),
        function_defs=FunctionDefList([f_ceil_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.int", "items": [6, -3, 2, -8, 8]}"""
        ),
        function_defs=FunctionDefList([int_to_float_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.list", "items": [[3], [-1, -5], [-3], [4, -4], []]}"""
        ),
        function_defs=FunctionDefList([list_tail_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytes", "items": [{"__bytes__": [173, 254]}, {"__bytes__": [186, 216, 196, 80]}, {"__bytes__": [214, 6, 59, 91]}]}"""
        ),
        function_defs=FunctionDefList([bytes_length_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.str", "items": ["Sb", "817P", "Kx"]}"""
        ),
        function_defs=FunctionDefList([str_remove_digits_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.tuple", "items": [{"__tuple__": [-1, 0]}, {"__tuple__": [-4]}, {"__tuple__": [2]}, {"__tuple__": [-2, -5]}, {"__tuple__": [-5]}]}"""
        ),
        function_defs=FunctionDefList([tuple_to_index_dict_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytes", "items": [{"__bytes__": [72, 228, 222, 183]}, {"__bytes__": [48, 237, 27, 237]}, {"__bytes__": [228, 217, 221, 242]}, {"__bytes__": [218, 173]}]}"""
        ),
        function_defs=FunctionDefList([bytes_reverse_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bool", "items": [true, false]}"""
        ),
        function_defs=FunctionDefList([bool_to_int_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.range", "items": [{"__range__": [3, 5, 1]}, {"__range__": [0, 0, 1]}, {"__range__": [3, 3, 1]}, {"__range__": [1, 2, 1]}, {"__range__": [2, 5, 1]}]}"""
        ),
        function_defs=FunctionDefList([range_sum_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.str", "items": ["lm1rL", "32", ""]}"""
        ),
        function_defs=FunctionDefList([duplicate_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.str", "items": ["aUl", "L", "", "OYuFIM"]}"""
        ),
        function_defs=FunctionDefList([reverse_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bool", "items": [false, false]}"""
        ),
        function_defs=FunctionDefList([bool_identity_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.complex", "items": [{"__complex__": [-5.0, 5.0]}, {"__complex__": [-1.0, 3.0]}, {"__complex__": [4.0, 1.0]}, {"__complex__": [1.0, -2.0]}]}"""
        ),
        function_defs=FunctionDefList([complex_phase_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{}, {"a": 1, "b": 1}, {"a": -2, "b": 1, "c": -3}, {}]}"""
        ),
        function_defs=FunctionDefList([dict_flip_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.int", "items": [7, 3, -5]}"""
        ),
        function_defs=FunctionDefList([is_odd_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bool", "items": [true, false]}"""
        ),
        function_defs=FunctionDefList([bool_to_int_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.list", "items": [[], [], [], []]}"""
        ),
        function_defs=FunctionDefList([list_median_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{"a": 4}, {"a": 1, "b": -4}, {"a": -3, "b": 4}, {"a": -1, "b": 0}, {"a": -1}]}"""
        ),
        function_defs=FunctionDefList([dict_items_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.list", "items": [[], [3, -2, 2], [4, -4]]}"""
        ),
        function_defs=FunctionDefList([list_sum_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.str", "items": ["", "FzQ74e", "2qo6OR", ""]}"""
        ),
        function_defs=FunctionDefList([contains_space_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.tuple", "items": [{"__tuple__": []}, {"__tuple__": [4, 5, 1]}, {"__tuple__": [0, -5, 3]}, {"__tuple__": [-2]}]}"""
        ),
        function_defs=FunctionDefList([tuple_length_func, is_positive_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.float", "items": [-1.2982346917927927, -5.936702321256082, 0.022005473186080593]}"""
        ),
        function_defs=FunctionDefList([f_fraction_func, f_reciprocal_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{"a": 2, "b": 0, "c": -3}, {"a": -5, "b": -4, "c": -4}, {}]}"""
        ),
        function_defs=FunctionDefList([dict_length_func, neg_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.set", "items": [{"__set__": [-5]}, {"__set__": [3, -4]}, {"__set__": [2, -3]}]}"""
        ),
        function_defs=FunctionDefList([set_size_func, neg_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{"a": -4}, {"a": -3, "b": -4, "c": 2}, {"a": 3, "b": -3, "c": -1}, {"a": 1, "b": 1, "c": 5}, {"a": -3}]}"""
        ),
        function_defs=FunctionDefList([dict_freeze_func, tuple_is_empty_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.tuple", "items": [{"__tuple__": []}, {"__tuple__": [-4, 2, 1]}, {"__tuple__": [5, 3]}, {"__tuple__": [3, -1, -3]}]}"""
        ),
        function_defs=FunctionDefList([tuple_reverse_func, tuple_is_empty_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.set", "items": [{"__set__": [5]}, {"__set__": [1]}, {"__set__": [1, 5]}, {"__set__": [3, -1]}, {"__set__": [4]}]}"""
        ),
        function_defs=FunctionDefList([set_is_empty_func, bool_not_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.set", "items": [{"__set__": [-2]}, {"__set__": [1, 5, -1]}, {"__set__": []}, {"__set__": [5, -2]}]}"""
        ),
        function_defs=FunctionDefList([set_is_empty_func, bool_to_float_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.tuple", "items": [{"__tuple__": [1, -5]}, {"__tuple__": [5]}, {"__tuple__": []}, {"__tuple__": [-5, 1, 4]}]}"""
        ),
        function_defs=FunctionDefList([tuple_count_none_func, square_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bool", "items": [false, false]}"""
        ),
        function_defs=FunctionDefList([bool_to_float_func, float_to_str_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.list", "items": [[4], [], [1, 4], [0]]}"""
        ),
        function_defs=FunctionDefList([list_sum_func, is_odd_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bool", "items": [false, false]}"""
        ),
        function_defs=FunctionDefList([bool_identity_func, bool_to_int_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.complex", "items": [{"__complex__": [3.0, -2.0]}, {"__complex__": [-3.0, 3.0]}, {"__complex__": [4.0, -5.0]}, {"__complex__": [5.0, -5.0]}, {"__complex__": [-1.0, 1.0]}]}"""
        ),
        function_defs=FunctionDefList([complex_real_func, f_sin_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.int", "items": [6, 3, 0]}"""
        ),
        function_defs=FunctionDefList([dec_func, square_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.complex", "items": [{"__complex__": [4.0, -2.0]}, {"__complex__": [4.0, -5.0]}, {"__complex__": [-1.0, 3.0]}, {"__complex__": [-3.0, 2.0]}]}"""
        ),
        function_defs=FunctionDefList([complex_abs_func, f_sin_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytes", "items": [{"__bytes__": [21]}, {"__bytes__": [244, 144]}, {"__bytes__": [44]}, {"__bytes__": [211, 73, 78]}, {"__bytes__": []}]}"""
        ),
        function_defs=FunctionDefList([bytes_is_empty_func, bool_to_str_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytes", "items": [{"__bytes__": [238, 75]}, {"__bytes__": [128, 77]}, {"__bytes__": []}]}"""
        ),
        function_defs=FunctionDefList([bytes_length_func, half_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.tuple", "items": [{"__tuple__": [-3, -5, 3]}, {"__tuple__": [4, 2, -5]}, {"__tuple__": [1]}, {"__tuple__": [5, -4, 1]}, {"__tuple__": [-1, 0, 4]}]}"""
        ),
        function_defs=FunctionDefList([tuple_to_index_dict_func, dict_flip_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytes", "items": [{"__bytes__": [70, 51]}, {"__bytes__": []}, {"__bytes__": [136, 147, 137, 138]}, {"__bytes__": [190, 211]}]}"""
        ),
        function_defs=FunctionDefList([bytes_to_hex_func, endswith_z_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.range", "items": [{"__range__": [2, 6, 1]}, {"__range__": [0, 0, 1]}, {"__range__": [2, 7, 1]}]}"""
        ),
        function_defs=FunctionDefList([range_max_func, int_to_str_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.float", "items": [-9.960142167140722, 6.643171724816675, -0.35990863945884044, 7.609055738879007, -4.030404829640444]}"""
        ),
        function_defs=FunctionDefList(
            [f_abs_sqrt_func, f_abs_sqrt_func, float_to_str_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.set", "items": [{"__set__": []}, {"__set__": [0, 1, -2]}, {"__set__": [2, 5, -2]}]}"""
        ),
        function_defs=FunctionDefList(
            [set_is_empty_func, bool_not_func, bool_not_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bool", "items": [false, true]}"""
        ),
        function_defs=FunctionDefList([bool_to_float_func, f_round_func, is_even_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.complex", "items": [{"__complex__": [1.0, 5.0]}, {"__complex__": [5.0, 4.0]}, {"__complex__": [0.0, 0.0]}]}"""
        ),
        function_defs=FunctionDefList(
            [complex_conjugate_func, complex_phase_func, f_square_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.str", "items": ["Lg", "J", "t", "xcov", "s96g"]}"""
        ),
        function_defs=FunctionDefList([last_char_func, length_func, int_popcount_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.range", "items": [{"__range__": [3, 7, 1]}, {"__range__": [1, 4, 1]}, {"__range__": [3, 5, 1]}, {"__range__": [2, 7, 1]}, {"__range__": [2, 7, 1]}]}"""
        ),
        function_defs=FunctionDefList(
            [range_list_func, list_is_empty_func, bool_to_str_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.tuple", "items": [{"__tuple__": [1, 0, -1]}, {"__tuple__": [5, 3]}, {"__tuple__": [0]}, {"__tuple__": []}, {"__tuple__": [-5, -2]}]}"""
        ),
        function_defs=FunctionDefList(
            [tuple_count_none_func, double_func, is_negative_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytes", "items": [{"__bytes__": []}, {"__bytes__": [231, 164, 29]}, {"__bytes__": []}]}"""
        ),
        function_defs=FunctionDefList(
            [bytes_upper_func, bytes_is_empty_func, bool_identity_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytes", "items": [{"__bytes__": [90, 214]}, {"__bytes__": [105]}, {"__bytes__": [159]}]}"""
        ),
        function_defs=FunctionDefList([bytes_length_func, inc_func, mod2_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.tuple", "items": [{"__tuple__": [0]}, {"__tuple__": [-1, -1]}, {"__tuple__": [4, -4, -4]}]}"""
        ),
        function_defs=FunctionDefList(
            [tuple_reverse_func, tuple_to_index_dict_func, dict_values_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.tuple", "items": [{"__tuple__": [-5]}, {"__tuple__": [-5, 3, 0]}, {"__tuple__": [0, -3, 4]}, {"__tuple__": [-4]}]}"""
        ),
        function_defs=FunctionDefList(
            [tuple_to_list_func, list_tail_func, list_is_empty_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.range", "items": [{"__range__": [0, 4, 1]}, {"__range__": [3, 8, 1]}, {"__range__": [3, 3, 1]}, {"__range__": [3, 4, 1]}, {"__range__": [2, 3, 1]}]}"""
        ),
        function_defs=FunctionDefList(
            [range_max_func, int_popcount_func, int_popcount_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.int", "items": [3, -5, 0]}"""
        ),
        function_defs=FunctionDefList(
            [double_func, int_is_power_of_two_func, bool_not_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bool", "items": [false, true]}"""
        ),
        function_defs=FunctionDefList([bool_to_int_func, dec_func, double_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.set", "items": [{"__set__": [-3]}, {"__set__": [1]}, {"__set__": [1, 3]}, {"__set__": [5]}]}"""
        ),
        function_defs=FunctionDefList(
            [set_hash_func, int_is_power_of_two_func, bool_to_float_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytearray", "items": [{"__bytearray__": [159, 11]}, {"__bytearray__": [22, 129, 42]}, {"__bytearray__": [253]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                bytearray_reverse_func,
                bytearray_to_bytes_func,
                bytes_reverse_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bool", "items": [false, true]}"""
        ),
        function_defs=FunctionDefList(
            [bool_to_int_func, is_negative_func, bool_to_str_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.float", "items": [-8.515250361120454, 2.4023248546870093, -7.8590774477726955, -2.593582305574129]}"""
        ),
        function_defs=FunctionDefList([f_abs_sqrt_func, f_square_func, f_square_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.set", "items": [{"__set__": [0]}, {"__set__": [0, -2]}, {"__set__": [0]}, {"__set__": [-5, -2, -1]}]}"""
        ),
        function_defs=FunctionDefList(
            [set_hash_func, int_popcount_func, int_to_float_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytes", "items": [{"__bytes__": [48, 153]}, {"__bytes__": [31, 161, 109]}, {"__bytes__": []}, {"__bytes__": [135, 113, 49]}, {"__bytes__": [96, 6]}]}"""
        ),
        function_defs=FunctionDefList(
            [bytes_is_empty_func, bool_to_int_func, int_popcount_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bool", "items": [false, true]}"""
        ),
        function_defs=FunctionDefList(
            [
                bool_to_float_func,
                f_ceil_func,
                int_popcount_func,
                int_to_bool_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.set", "items": [{"__set__": [0]}, {"__set__": [-1]}, {"__set__": []}]}"""
        ),
        function_defs=FunctionDefList(
            [set_hash_func, sign_func, inc_func, double_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.float", "items": [7.800126379299048, 1.5495191025355197, 7.390717531168342, -2.509114442532871]}"""
        ),
        function_defs=FunctionDefList(
            [f_sin_func, f_sin_func, f_trunc_func, sign_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytes", "items": [{"__bytes__": []}, {"__bytes__": [194, 156, 54]}, {"__bytes__": [120, 30, 181]}, {"__bytes__": [94, 143]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                bytes_is_ascii_func,
                bool_not_func,
                bool_to_int_func,
                identity_int_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{"a": -3, "b": 2, "c": -1}, {"a": -2}, {"a": -5, "b": -3}]}"""
        ),
        function_defs=FunctionDefList(
            [
                dict_items_func,
                list_reverse_func,
                list_tail_func,
                list_min_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytes", "items": [{"__bytes__": [207, 26, 27, 57]}, {"__bytes__": [168, 217, 38, 223]}, {"__bytes__": [1, 240]}, {"__bytes__": [29]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                bytes_upper_func,
                bytes_reverse_func,
                bytes_length_func,
                inc_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{"a": 5}, {"a": 5}, {"a": 1, "b": 3, "c": -4}, {"a": 1}, {}]}"""
        ),
        function_defs=FunctionDefList(
            [
                dict_has_duplicate_values_func,
                bool_to_int_func,
                is_even_func,
                bool_to_str_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.float", "items": [5.527899437184136, 2.970767530911898, 0.46412563722813616]}"""
        ),
        function_defs=FunctionDefList(
            [f_fraction_func, float_to_str_func, reverse_func, strip_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.range", "items": [{"__range__": [1, 2, 1]}, {"__range__": [0, 4, 1]}, {"__range__": [3, 7, 1]}, {"__range__": [1, 6, 1]}]}"""
        ),
        function_defs=FunctionDefList(
            [range_max_func, int_to_str_func, reverse_func, title_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bool", "items": [true, false]}"""
        ),
        function_defs=FunctionDefList(
            [bool_to_str_func, upper_func, upper_func, lower_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.list", "items": [[-5, -1, -3], [0, 0, 3], [-5]]}"""
        ),
        function_defs=FunctionDefList(
            [
                list_is_empty_func,
                bool_identity_func,
                bool_not_func,
                bool_identity_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.str", "items": ["8yhz", "t3J5", "6Br4TT", "v"]}"""
        ),
        function_defs=FunctionDefList(
            [is_numeric_func, bool_not_func, bool_not_func, bool_to_int_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{"a": -2, "b": 1}, {"a": 0, "b": -1}, {"a": -2, "b": -4, "c": -1}]}"""
        ),
        function_defs=FunctionDefList(
            [
                dict_freeze_func,
                tuple_to_list_func,
                list_sorted_func,
                list_reverse_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.list", "items": [[-4, 1], [-2, -5], [0]]}"""
        ),
        function_defs=FunctionDefList(
            [
                list_median_func,
                f_trunc_func,
                int_bit_length_func,
                is_positive_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.float", "items": [2.5678817021658205, 0.5893530348553035, -1.719271942412945, -0.5057601847225595]}"""
        ),
        function_defs=FunctionDefList(
            [f_frac_percent_func, dec_func, double_func, dec_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytearray", "items": [{"__bytearray__": [123, 249, 81]}, {"__bytearray__": [212, 114, 153]}, {"__bytearray__": [203, 204, 199]}, {"__bytearray__": [112]}, {"__bytearray__": [195, 163, 229]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                bytearray_to_bytes_func,
                bytes_length_func,
                is_odd_func,
                bool_to_float_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{"a": -1, "b": -3, "c": 5}, {"a": 4, "b": 4, "c": 0}, {}]}"""
        ),
        function_defs=FunctionDefList(
            [
                dict_is_empty_func,
                bool_identity_func,
                bool_identity_func,
                bool_not_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.complex", "items": [{"__complex__": [-3.0, -2.0]}, {"__complex__": [-2.0, 0.0]}, {"__complex__": [-2.0, 4.0]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                complex_imag_func,
                f_abs_func,
                f_round_func,
                int_is_power_of_two_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytes", "items": [{"__bytes__": [243, 0, 97, 63]}, {"__bytes__": []}, {"__bytes__": [83]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                bytes_reverse_func,
                bytes_is_ascii_func,
                bool_to_str_func,
                reverse_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{"a": 2}, {}, {}]}"""
        ),
        function_defs=FunctionDefList(
            [
                dict_flip_func,
                dict_items_func,
                list_is_empty_func,
                bool_identity_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.set", "items": [{"__set__": [4]}, {"__set__": []}, {"__set__": [0]}, {"__set__": [3]}, {"__set__": [0, 1, 3]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                set_to_list_func,
                list_median_func,
                f_floor_func,
                half_func,
                half_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.list", "items": [[-1, 1, -2], [-3, -4, -2], [0]]}"""
        ),
        function_defs=FunctionDefList(
            [
                list_sum_func,
                is_positive_func,
                bool_to_float_func,
                f_sin_func,
                f_log10_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bool", "items": [false, false]}"""
        ),
        function_defs=FunctionDefList(
            [
                bool_to_float_func,
                f_frac_percent_func,
                int_to_float_func,
                f_round_func,
                int_popcount_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bool", "items": [false, false]}"""
        ),
        function_defs=FunctionDefList(
            [
                bool_not_func,
                bool_not_func,
                bool_to_float_func,
                f_is_integer_func,
                bool_not_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.range", "items": [{"__range__": [3, 8, 1]}, {"__range__": [1, 2, 1]}, {"__range__": [0, 3, 1]}, {"__range__": [3, 5, 1]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                range_max_func,
                square_func,
                int_to_str_func,
                repeat_func,
                is_upper_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.float", "items": [3.8279487636696423, -7.384142798303744, 2.039879512491943, -1.326692581529672, -3.8831968776955916]}"""
        ),
        function_defs=FunctionDefList(
            [
                f_log10_func,
                f_square_func,
                f_fraction_func,
                f_exp_func,
                f_reciprocal_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytes", "items": [{"__bytes__": [115, 12, 21, 186]}, {"__bytes__": [65]}, {"__bytes__": [239, 89, 118]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                bytes_is_empty_func,
                bool_identity_func,
                bool_to_float_func,
                f_floor_func,
                identity_int_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.set", "items": [{"__set__": []}, {"__set__": [2, -2]}, {"__set__": [2, -5, 5]}, {"__set__": []}, {"__set__": [1, 4]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                set_is_empty_func,
                bool_not_func,
                bool_not_func,
                bool_to_float_func,
                f_sin_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytearray", "items": [{"__bytearray__": [211]}, {"__bytearray__": []}, {"__bytearray__": [71]}, {"__bytearray__": []}, {"__bytearray__": [22, 202]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                bytearray_reverse_func,
                bytearray_to_bytes_func,
                bytes_reverse_func,
                bytes_to_hex_func,
                str_remove_digits_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.str", "items": ["CD", "2Jr", "ut5s", "s"]}"""
        ),
        function_defs=FunctionDefList(
            [
                endswith_z_func,
                bool_not_func,
                bool_to_int_func,
                int_bit_length_func,
                int_bit_length_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.set", "items": [{"__set__": [5, -1]}, {"__set__": [0, -2]}, {"__set__": [1, 3, 4]}, {"__set__": [3, -3]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                set_is_empty_func,
                bool_not_func,
                bool_to_int_func,
                sign_func,
                is_positive_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytearray", "items": [{"__bytearray__": [213, 192, 161, 132]}, {"__bytearray__": [186, 92, 0, 156]}, {"__bytearray__": [135]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                bytearray_reverse_func,
                bytearray_length_func,
                neg_func,
                abs_func,
                int_to_bool_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.int", "items": [-4, 4, -8, 5]}"""
        ),
        function_defs=FunctionDefList(
            [
                dec_func,
                half_func,
                is_odd_func,
                bool_to_int_func,
                is_even_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bool", "items": [true, false]}"""
        ),
        function_defs=FunctionDefList(
            [
                bool_identity_func,
                bool_to_float_func,
                f_frac_percent_func,
                half_func,
                sign_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytes", "items": [{"__bytes__": [66, 92]}, {"__bytes__": [216, 241, 17, 61]}, {"__bytes__": []}, {"__bytes__": [152, 250, 156]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                bytes_is_empty_func,
                bool_not_func,
                bool_to_int_func,
                square_func,
                dec_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.set", "items": [{"__set__": []}, {"__set__": [-4]}, {"__set__": []}, {"__set__": [3, -5, -1]}, {"__set__": [2, -4, 5]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                set_is_empty_func,
                bool_to_float_func,
                f_fraction_func,
                f_is_integer_func,
                bool_to_float_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.float", "items": [5.641246325771391, 5.1782218518644925, -3.4730371912384257]}"""
        ),
        function_defs=FunctionDefList(
            [
                f_log10_func,
                f_is_integer_func,
                bool_to_float_func,
                f_reciprocal_func,
                float_to_str_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{"a": 2, "b": -1, "c": 5}, {}, {"a": 4}, {}]}"""
        ),
        function_defs=FunctionDefList(
            [
                dict_keys_func,
                list_length_func,
                inc_func,
                int_is_power_of_two_func,
                bool_identity_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytearray", "items": [{"__bytearray__": []}, {"__bytearray__": [13, 76]}, {"__bytearray__": [214, 235]}, {"__bytearray__": [166, 67, 225]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                bytearray_length_func,
                is_odd_func,
                bool_to_int_func,
                identity_int_func,
                mod2_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.int", "items": [8, 3, -5, 5]}"""
        ),
        function_defs=FunctionDefList(
            [
                abs_func,
                mod2_func,
                int_popcount_func,
                int_clip_0_100_func,
                mod2_func,
            ]
        ),
    ),
]

eval_trajectory_specs = TrajectorySpecList(_specs)
