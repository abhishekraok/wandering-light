"""
Generated trajectory specs data
Total specs: 500
"""

from wandering_light.function_def import FunctionDef, FunctionDefList
from wandering_light.trajectory import TrajectorySpec, TrajectorySpecList
from wandering_light.typed_list import TypedList

# Function definitions
dict_freeze_func = FunctionDef(
    name="dict_freeze",
    input_type="builtins.dict",
    output_type="builtins.tuple",
    code="""return tuple(sorted(x.items()))""",
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

tuple_length_func = FunctionDef(
    name="tuple_length",
    input_type="builtins.tuple",
    output_type="builtins.int",
    code="""return len(x)""",
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

complex_conjugate_func = FunctionDef(
    name="complex_conjugate",
    input_type="builtins.complex",
    output_type="builtins.complex",
    code="""return x.conjugate()""",
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

set_size_func = FunctionDef(
    name="set_size",
    input_type="builtins.set",
    output_type="builtins.int",
    code="""return len(x)""",
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

complex_imag_func = FunctionDef(
    name="complex_imag",
    input_type="builtins.complex",
    output_type="builtins.float",
    code="""return x.imag""",
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

set_is_empty_func = FunctionDef(
    name="set_is_empty",
    input_type="builtins.set",
    output_type="builtins.bool",
    code="""return len(x) == 0""",
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

bytes_reverse_func = FunctionDef(
    name="bytes_reverse",
    input_type="builtins.bytes",
    output_type="builtins.bytes",
    code="""return x[::-1]""",
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

f_fraction_func = FunctionDef(
    name="f_fraction",
    input_type="builtins.float",
    output_type="builtins.float",
    code="""return x - int(x)""",
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

dict_is_empty_func = FunctionDef(
    name="dict_is_empty",
    input_type="builtins.dict",
    output_type="builtins.bool",
    code="""return len(x) == 0""",
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

bool_to_int_func = FunctionDef(
    name="bool_to_int",
    input_type="builtins.bool",
    output_type="builtins.int",
    code="""return int(x)""",
    usage_count=0,
    metadata={},
)

str_hash_func = FunctionDef(
    name="str_hash",
    input_type="builtins.str",
    output_type="builtins.int",
    code="""return hash(x)""",
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

is_negative_func = FunctionDef(
    name="is_negative",
    input_type="builtins.int",
    output_type="builtins.bool",
    code="""return x < 0""",
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

dec_func = FunctionDef(
    name="dec",
    input_type="builtins.int",
    output_type="builtins.int",
    code="""return x - 1""",
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

f_floor_func = FunctionDef(
    name="f_floor",
    input_type="builtins.float",
    output_type="builtins.int",
    code="""return __import__('math').floor(x)""",
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

str_is_palindrome_func = FunctionDef(
    name="str_is_palindrome",
    input_type="builtins.str",
    output_type="builtins.bool",
    code="""s=x.lower(); return s==s[::-1]""",
    usage_count=0,
    metadata={},
)

dict_keyset_func = FunctionDef(
    name="dict_keyset",
    input_type="builtins.dict",
    output_type="builtins.set",
    code="""return set(x.keys())""",
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

f_square_func = FunctionDef(
    name="f_square",
    input_type="builtins.float",
    output_type="builtins.float",
    code="""return x * x""",
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

bytes_length_func = FunctionDef(
    name="bytes_length",
    input_type="builtins.bytes",
    output_type="builtins.int",
    code="""return len(x)""",
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

bytes_to_hex_func = FunctionDef(
    name="bytes_to_hex",
    input_type="builtins.bytes",
    output_type="builtins.str",
    code="""return x.hex()""",
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

abs_func = FunctionDef(
    name="abs",
    input_type="builtins.int",
    output_type="builtins.int",
    code="""return abs(x)""",
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

bool_not_func = FunctionDef(
    name="bool_not",
    input_type="builtins.bool",
    output_type="builtins.bool",
    code="""return not x""",
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

neg_func = FunctionDef(
    name="neg",
    input_type="builtins.int",
    output_type="builtins.int",
    code="""return -x""",
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

bytes_is_empty_func = FunctionDef(
    name="bytes_is_empty",
    input_type="builtins.bytes",
    output_type="builtins.bool",
    code="""return len(x) == 0""",
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

float_to_str_func = FunctionDef(
    name="float_to_str",
    input_type="builtins.float",
    output_type="builtins.str",
    code="""return str(x)""",
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

strip_func = FunctionDef(
    name="strip",
    input_type="builtins.str",
    output_type="builtins.str",
    code="""return x.strip()""",
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

bool_to_str_func = FunctionDef(
    name="bool_to_str",
    input_type="builtins.bool",
    output_type="builtins.str",
    code="""return str(x)""",
    usage_count=0,
    metadata={},
)

f_mod1_func = FunctionDef(
    name="f_mod1",
    input_type="builtins.float",
    output_type="builtins.float",
    code="""return x % 1.0""",
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

dict_has_duplicate_values_func = FunctionDef(
    name="dict_has_duplicate_values",
    input_type="builtins.dict",
    output_type="builtins.bool",
    code="""vals=list(x.values()); return len(vals)!=len(set(vals))""",
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

list_sum_func = FunctionDef(
    name="list_sum",
    input_type="builtins.list",
    output_type="builtins.int",
    code="""return sum(x) if x and all(isinstance(v, (int, float)) for v in x) else 0""",
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

square_func = FunctionDef(
    name="square",
    input_type="builtins.int",
    output_type="builtins.int",
    code="""return x * x""",
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

range_length_func = FunctionDef(
    name="range_length",
    input_type="builtins.range",
    output_type="builtins.int",
    code="""return len(x)""",
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

double_func = FunctionDef(
    name="double",
    input_type="builtins.int",
    output_type="builtins.int",
    code="""return x * 2""",
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

bytearray_reverse_func = FunctionDef(
    name="bytearray_reverse",
    input_type="builtins.bytearray",
    output_type="builtins.bytearray",
    code="""return bytearray(x[::-1])""",
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

f_round_func = FunctionDef(
    name="f_round",
    input_type="builtins.float",
    output_type="builtins.int",
    code="""return round(x)""",
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

bool_to_float_func = FunctionDef(
    name="bool_to_float",
    input_type="builtins.bool",
    output_type="builtins.float",
    code="""return 1.0 if x else 0.0""",
    usage_count=0,
    metadata={},
)

count_a_func = FunctionDef(
    name="count_a",
    input_type="builtins.str",
    output_type="builtins.int",
    code="""return x.count('a')""",
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

int_is_power_of_two_func = FunctionDef(
    name="int_is_power_of_two",
    input_type="builtins.int",
    output_type="builtins.bool",
    code="""return x > 0 and (x & (x - 1)) == 0""",
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

list_unique_func = FunctionDef(
    name="list_unique",
    input_type="builtins.list",
    output_type="builtins.list",
    code="""return list(dict.fromkeys(x))""",
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

f_reciprocal_func = FunctionDef(
    name="f_reciprocal",
    input_type="builtins.float",
    output_type="builtins.float",
    code="""return float('inf') if x == 0 else 1.0 / x""",
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

int_popcount_func = FunctionDef(
    name="int_popcount",
    input_type="builtins.int",
    output_type="builtins.int",
    code="""return x.bit_count()""",
    usage_count=0,
    metadata={},
)

str_reverse_words_func = FunctionDef(
    name="str_reverse_words",
    input_type="builtins.str",
    output_type="builtins.str",
    code="""return ' '.join(x.split()[::-1])""",
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

mod2_func = FunctionDef(
    name="mod2",
    input_type="builtins.int",
    output_type="builtins.int",
    code="""return x % 2""",
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

list_min_func = FunctionDef(
    name="list_min",
    input_type="builtins.list",
    output_type="builtins.int",
    code="""return min(x) if x and all(isinstance(v, (int, float)) for v in x) else 0""",
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

dict_values_func = FunctionDef(
    name="dict_values",
    input_type="builtins.dict",
    output_type="builtins.list",
    code="""return list(x.values())""",
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

list_length_func = FunctionDef(
    name="list_length",
    input_type="builtins.list",
    output_type="builtins.int",
    code="""return len(x)""",
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

range_max_func = FunctionDef(
    name="range_max",
    input_type="builtins.range",
    output_type="builtins.int",
    code="""return x[-1] if x else 0""",
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

is_space_func = FunctionDef(
    name="is_space",
    input_type="builtins.str",
    output_type="builtins.bool",
    code="""return x.isspace()""",
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

list_sorted_func = FunctionDef(
    name="list_sorted",
    input_type="builtins.list",
    output_type="builtins.list",
    code="""return sorted(x)""",
    usage_count=0,
    metadata={},
)

swapcase_func = FunctionDef(
    name="swapcase",
    input_type="builtins.str",
    output_type="builtins.str",
    code="""return x.swapcase()""",
    usage_count=0,
    metadata={},
)

capitalize_func = FunctionDef(
    name="capitalize",
    input_type="builtins.str",
    output_type="builtins.str",
    code="""return x.capitalize()""",
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

list_max_func = FunctionDef(
    name="list_max",
    input_type="builtins.list",
    output_type="builtins.int",
    code="""return max(x) if x and all(isinstance(v, (int, float)) for v in x) else 0""",
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

f_is_integer_func = FunctionDef(
    name="f_is_integer",
    input_type="builtins.float",
    output_type="builtins.bool",
    code="""return x.is_integer()""",
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

int_to_bool_func = FunctionDef(
    name="int_to_bool",
    input_type="builtins.int",
    output_type="builtins.bool",
    code="""return bool(x)""",
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

startswith_a_func = FunctionDef(
    name="startswith_a",
    input_type="builtins.str",
    output_type="builtins.bool",
    code="""return x.startswith('a')""",
    usage_count=0,
    metadata={},
)

str_to_list_func = FunctionDef(
    name="str_to_list",
    input_type="builtins.str",
    output_type="builtins.list",
    code="""return list(x)""",
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

duplicate_func = FunctionDef(
    name="duplicate",
    input_type="builtins.str",
    output_type="builtins.str",
    code="""return x + x""",
    usage_count=0,
    metadata={},
)

is_lower_func = FunctionDef(
    name="is_lower",
    input_type="builtins.str",
    output_type="builtins.bool",
    code="""return x.islower()""",
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

str_count_vowels_func = FunctionDef(
    name="str_count_vowels",
    input_type="builtins.str",
    output_type="builtins.int",
    code="""return sum(1 for c in x.lower() if c in 'aeiou')""",
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

is_numeric_func = FunctionDef(
    name="is_numeric",
    input_type="builtins.str",
    output_type="builtins.bool",
    code="""return x.isnumeric()""",
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

is_title_func = FunctionDef(
    name="is_title",
    input_type="builtins.str",
    output_type="builtins.bool",
    code="""return x.istitle()""",
    usage_count=0,
    metadata={},
)

first_char_func = FunctionDef(
    name="first_char",
    input_type="builtins.str",
    output_type="builtins.str",
    code="""return x[0] if x else ''""",
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

is_alpha_func = FunctionDef(
    name="is_alpha",
    input_type="builtins.str",
    output_type="builtins.bool",
    code="""return x.isalpha()""",
    usage_count=0,
    metadata={},
)

is_digit_func = FunctionDef(
    name="is_digit",
    input_type="builtins.str",
    output_type="builtins.bool",
    code="""return x.isdigit()""",
    usage_count=0,
    metadata={},
)

# Trajectory specifications
_specs = [
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{}, {"a": 4}, {"a": -2}, {"a": 2}, {}]}"""
        ),
        function_defs=FunctionDefList([dict_freeze_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.list", "items": [[-3], [-5, 5, -4], [2, 4], [-1, -2, 1], [-1, -3, 3]]}"""
        ),
        function_defs=FunctionDefList([list_is_empty_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.tuple", "items": [{"__tuple__": [-3, 0]}, {"__tuple__": [2]}, {"__tuple__": [-5]}, {"__tuple__": [-4, -4, 4]}]}"""
        ),
        function_defs=FunctionDefList([tuple_length_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.list", "items": [[0, 1], [], [-5, 0, 1], [-3, 0]]}"""
        ),
        function_defs=FunctionDefList([list_tail_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.complex", "items": [{"__complex__": [-3.0, -2.0]}, {"__complex__": [3.0, 0.0]}, {"__complex__": [-3.0, 4.0]}, {"__complex__": [-1.0, -4.0]}, {"__complex__": [2.0, 1.0]}]}"""
        ),
        function_defs=FunctionDefList([complex_conjugate_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.range", "items": [{"__range__": [3, 7, 1]}, {"__range__": [0, 2, 1]}, {"__range__": [3, 3, 1]}, {"__range__": [3, 5, 1]}, {"__range__": [3, 3, 1]}]}"""
        ),
        function_defs=FunctionDefList([range_list_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.set", "items": [{"__set__": [0, 3, 4]}, {"__set__": [-2]}, {"__set__": [4, -1]}, {"__set__": [-5, -3]}]}"""
        ),
        function_defs=FunctionDefList([set_size_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bool", "items": [false, false]}"""
        ),
        function_defs=FunctionDefList([bool_identity_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.complex", "items": [{"__complex__": [1.0, -1.0]}, {"__complex__": [-2.0, 0.0]}, {"__complex__": [-4.0, -4.0]}, {"__complex__": [5.0, 5.0]}, {"__complex__": [2.0, 1.0]}]}"""
        ),
        function_defs=FunctionDefList([complex_imag_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{"a": -5}, {"a": 5, "b": -3, "c": -4}, {"a": -4, "b": 5}, {"a": -1, "b": 0}]}"""
        ),
        function_defs=FunctionDefList([dict_freeze_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.tuple", "items": [{"__tuple__": []}, {"__tuple__": [3, 0]}, {"__tuple__": [-1, 2, 4]}]}"""
        ),
        function_defs=FunctionDefList([tuple_to_index_dict_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.set", "items": [{"__set__": []}, {"__set__": [2, -3]}, {"__set__": [3, 4, -1]}]}"""
        ),
        function_defs=FunctionDefList([set_is_empty_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{}, {"a": 4, "b": -3, "c": 1}, {"a": 5, "b": 0, "c": 3}, {}]}"""
        ),
        function_defs=FunctionDefList([dict_length_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytes", "items": [{"__bytes__": [236]}, {"__bytes__": []}, {"__bytes__": [57, 28, 48, 102]}]}"""
        ),
        function_defs=FunctionDefList([bytes_reverse_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.set", "items": [{"__set__": [-3, -1]}, {"__set__": [-5, 5]}, {"__set__": []}]}"""
        ),
        function_defs=FunctionDefList([set_to_list_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.float", "items": [-1.8820486280783104, 7.683448047930117, -6.317502753075434, -5.408324927631747]}"""
        ),
        function_defs=FunctionDefList([f_fraction_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{"a": 0, "b": 1}, {"a": -3, "b": -1, "c": -1}, {"a": -5, "b": -5}]}"""
        ),
        function_defs=FunctionDefList([dict_freeze_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.complex", "items": [{"__complex__": [5.0, 4.0]}, {"__complex__": [0.0, 4.0]}, {"__complex__": [2.0, 3.0]}, {"__complex__": [-2.0, -2.0]}, {"__complex__": [5.0, -1.0]}]}"""
        ),
        function_defs=FunctionDefList([complex_real_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.complex", "items": [{"__complex__": [-2.0, -3.0]}, {"__complex__": [3.0, -5.0]}, {"__complex__": [4.0, 0.0]}, {"__complex__": [-1.0, 1.0]}]}"""
        ),
        function_defs=FunctionDefList([complex_conjugate_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{"a": -5, "b": -2, "c": -1}, {"a": 3, "b": 2}, {}]}"""
        ),
        function_defs=FunctionDefList([dict_is_empty_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytearray", "items": [{"__bytearray__": []}, {"__bytearray__": [180, 90, 162, 54]}, {"__bytearray__": [37, 243]}, {"__bytearray__": []}]}"""
        ),
        function_defs=FunctionDefList([bytearray_to_bytes_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bool", "items": [true, false]}"""
        ),
        function_defs=FunctionDefList([bool_to_int_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.str", "items": ["", "", "Sc5i8W"]}"""
        ),
        function_defs=FunctionDefList([str_hash_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.range", "items": [{"__range__": [2, 4, 1]}, {"__range__": [3, 8, 1]}, {"__range__": [1, 5, 1]}, {"__range__": [3, 6, 1]}]}"""
        ),
        function_defs=FunctionDefList([range_sum_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.set", "items": [{"__set__": [5]}, {"__set__": [-3, -1, -2]}, {"__set__": [0, 1]}]}"""
        ),
        function_defs=FunctionDefList([set_to_list_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{}, {}, {"a": 0, "b": -2}]}"""
        ),
        function_defs=FunctionDefList([dict_freeze_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.int", "items": [4, 4, 9]}"""
        ),
        function_defs=FunctionDefList([is_negative_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.int", "items": [7, -3, -1, 3]}"""
        ),
        function_defs=FunctionDefList([int_clip_0_100_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.int", "items": [-7, -9, 4]}"""
        ),
        function_defs=FunctionDefList([dec_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytearray", "items": [{"__bytearray__": [167, 194, 104, 64]}, {"__bytearray__": [249, 144, 33]}, {"__bytearray__": [189]}, {"__bytearray__": [74, 15, 206, 195]}, {"__bytearray__": []}]}"""
        ),
        function_defs=FunctionDefList([bytearray_to_bytes_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.str", "items": ["veS", "LVKczn", "bu", "n", "1Na"]}"""
        ),
        function_defs=FunctionDefList([lower_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.float", "items": [-0.02489415812073581, -5.445723005824757, 0.6628189617444651]}"""
        ),
        function_defs=FunctionDefList([f_floor_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{"a": -2, "b": -3, "c": -4}, {}, {}]}"""
        ),
        function_defs=FunctionDefList([dict_flip_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.str", "items": ["D", "jX", "lpdh", ""]}"""
        ),
        function_defs=FunctionDefList([str_hash_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.set", "items": [{"__set__": [1, 3]}, {"__set__": [3, 4]}, {"__set__": [5]}, {"__set__": [2, 5]}]}"""
        ),
        function_defs=FunctionDefList([set_to_list_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.range", "items": [{"__range__": [2, 7, 1]}, {"__range__": [1, 3, 1]}, {"__range__": [3, 7, 1]}, {"__range__": [1, 5, 1]}]}"""
        ),
        function_defs=FunctionDefList([range_list_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.str", "items": ["fMF", "DzTQSR", "j", "m4"]}"""
        ),
        function_defs=FunctionDefList([str_is_palindrome_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.list", "items": [[2, 4], [2, 5, 1], [-1], [5, 1, -4], [-1, 5, -3]]}"""
        ),
        function_defs=FunctionDefList([list_tail_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.tuple", "items": [{"__tuple__": []}, {"__tuple__": [-1]}, {"__tuple__": [0, -3]}, {"__tuple__": []}]}"""
        ),
        function_defs=FunctionDefList([tuple_length_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{"a": -1, "b": 5}, {"a": 4, "b": 0}, {"a": 5}]}"""
        ),
        function_defs=FunctionDefList([dict_keyset_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.int", "items": [7, 7, -3, 9]}"""
        ),
        function_defs=FunctionDefList([dec_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.int", "items": [-9, 5, 9, -4]}"""
        ),
        function_defs=FunctionDefList([is_positive_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.str", "items": ["uVo", "", "B2y", "oJw"]}"""
        ),
        function_defs=FunctionDefList([lower_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.float", "items": [1.731252632250035, -6.892440241994276, -0.9420248019066957, 0.6111027053601923]}"""
        ),
        function_defs=FunctionDefList([f_square_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.str", "items": ["oThXi4", "Z", "uFw", "5jpxU", "n3Z"]}"""
        ),
        function_defs=FunctionDefList([endswith_z_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytes", "items": [{"__bytes__": [107]}, {"__bytes__": [254, 162, 128]}, {"__bytes__": [85]}]}"""
        ),
        function_defs=FunctionDefList([bytes_length_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.float", "items": [-7.548547915156165, 3.254146043249179, 6.117468293358371]}"""
        ),
        function_defs=FunctionDefList([f_abs_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytes", "items": [{"__bytes__": [192]}, {"__bytes__": [17, 91, 75, 231]}, {"__bytes__": []}, {"__bytes__": [92]}]}"""
        ),
        function_defs=FunctionDefList([bytes_to_hex_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.complex", "items": [{"__complex__": [1.0, 5.0]}, {"__complex__": [2.0, -4.0]}, {"__complex__": [-2.0, -2.0]}, {"__complex__": [1.0, -1.0]}]}"""
        ),
        function_defs=FunctionDefList([complex_phase_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.str", "items": ["Z2i5lT", "hapmq", "Tl", "OyZx"]}"""
        ),
        function_defs=FunctionDefList([str_is_palindrome_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{}, {"a": 4, "b": -4}, {"a": -5, "b": -1, "c": 0}, {}, {}]}"""
        ),
        function_defs=FunctionDefList([dict_flip_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.list", "items": [[1, 2, -4], [-5], [], []]}"""
        ),
        function_defs=FunctionDefList([list_is_empty_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{"a": 3}, {"a": -5, "b": 4}, {"a": 0}, {"a": -1, "b": -5, "c": -1}, {"a": 3}]}"""
        ),
        function_defs=FunctionDefList([dict_is_empty_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.float", "items": [1.1090069478293927, -0.6283547561254466, 2.900297397664586]}"""
        ),
        function_defs=FunctionDefList([f_ceil_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytes", "items": [{"__bytes__": [117, 113, 54, 156]}, {"__bytes__": [1]}, {"__bytes__": [167, 199, 62]}]}"""
        ),
        function_defs=FunctionDefList([bytes_to_hex_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.int", "items": [9, 6, -9, -7]}"""
        ),
        function_defs=FunctionDefList([int_to_float_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.int", "items": [-4, -4, 7, 4, -2]}"""
        ),
        function_defs=FunctionDefList([abs_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{"a": 2, "b": -2, "c": -1}, {"a": -1, "b": -1}, {"a": -3}, {"a": 5}]}"""
        ),
        function_defs=FunctionDefList([dict_items_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bool", "items": [true, true]}"""
        ),
        function_defs=FunctionDefList([bool_not_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.tuple", "items": [{"__tuple__": [2]}, {"__tuple__": [0, -4, 3]}, {"__tuple__": [-1, -3, 2]}]}"""
        ),
        function_defs=FunctionDefList([tuple_count_none_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.int", "items": [-10, -8, 9, -1, 4]}"""
        ),
        function_defs=FunctionDefList([neg_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bool", "items": [false, false]}"""
        ),
        function_defs=FunctionDefList([bool_not_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.int", "items": [-9, 4, 7, -3]}"""
        ),
        function_defs=FunctionDefList([is_odd_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.float", "items": [4.4947425745256275, 7.442693038663677, 2.1697496231580935, -5.526832918614428]}"""
        ),
        function_defs=FunctionDefList([f_fraction_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytes", "items": [{"__bytes__": []}, {"__bytes__": [35]}, {"__bytes__": [147, 236, 80, 127]}]}"""
        ),
        function_defs=FunctionDefList([bytes_is_empty_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bool", "items": [false, false]}"""
        ),
        function_defs=FunctionDefList([bool_not_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytes", "items": [{"__bytes__": [191]}, {"__bytes__": [253, 71]}, {"__bytes__": [23]}]}"""
        ),
        function_defs=FunctionDefList([bytes_upper_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.float", "items": [-0.19639631221363807, 0.5120454213555181, 1.5102930130478782, -9.368864634347174, -3.729003715061303]}"""
        ),
        function_defs=FunctionDefList([float_to_str_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.list", "items": [[0, -2, -3], [-4, 5, 3], [-3, 5, -5]]}"""
        ),
        function_defs=FunctionDefList([list_tail_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.int", "items": [-7, -6, 3, 4]}"""
        ),
        function_defs=FunctionDefList([inc_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.str", "items": ["w3", "GoV", "b9", "2Y"]}"""
        ),
        function_defs=FunctionDefList([strip_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytes", "items": [{"__bytes__": [221, 237, 45]}, {"__bytes__": [116, 49]}, {"__bytes__": [181]}, {"__bytes__": [174]}, {"__bytes__": [61, 217, 181, 248]}]}"""
        ),
        function_defs=FunctionDefList([bytes_is_ascii_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytes", "items": [{"__bytes__": [132, 33, 250]}, {"__bytes__": [73, 218, 137]}, {"__bytes__": [22]}, {"__bytes__": [116, 32, 244, 179]}, {"__bytes__": [219, 226]}]}"""
        ),
        function_defs=FunctionDefList([bytes_reverse_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.int", "items": [4, -3, -5, 6]}"""
        ),
        function_defs=FunctionDefList([neg_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bool", "items": [false, true]}"""
        ),
        function_defs=FunctionDefList([bool_to_str_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.float", "items": [5.106810487596611, -8.025944232835268, -2.2649369861377284]}"""
        ),
        function_defs=FunctionDefList([f_mod1_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.tuple", "items": [{"__tuple__": [-2, 4, 4]}, {"__tuple__": [-1, 5]}, {"__tuple__": [-5]}, {"__tuple__": [-2, -2]}, {"__tuple__": []}]}"""
        ),
        function_defs=FunctionDefList([tuple_is_empty_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.int", "items": [-7, -7, 8, -6]}"""
        ),
        function_defs=FunctionDefList([abs_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.tuple", "items": [{"__tuple__": []}, {"__tuple__": []}, {"__tuple__": [0, -2]}, {"__tuple__": [4, 3]}]}"""
        ),
        function_defs=FunctionDefList([tuple_to_index_dict_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{}, {"a": 3}, {"a": -3, "b": 5, "c": 5}]}"""
        ),
        function_defs=FunctionDefList([dict_has_duplicate_values_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.str", "items": ["nz2V8", "R", "4xWXn", "0Pxq"]}"""
        ),
        function_defs=FunctionDefList([upper_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.list", "items": [[2, 0], [3], [4, 2], [4, 2, 1]]}"""
        ),
        function_defs=FunctionDefList([list_sum_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytes", "items": [{"__bytes__": [217]}, {"__bytes__": [227]}, {"__bytes__": [180, 228, 235]}]}"""
        ),
        function_defs=FunctionDefList([bytes_length_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytes", "items": [{"__bytes__": []}, {"__bytes__": [145, 115, 113, 172]}, {"__bytes__": [215, 236]}, {"__bytes__": [17, 208]}, {"__bytes__": []}]}"""
        ),
        function_defs=FunctionDefList([bytes_length_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bool", "items": [false, true]}"""
        ),
        function_defs=FunctionDefList([bool_to_int_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bool", "items": [false, true]}"""
        ),
        function_defs=FunctionDefList([bool_identity_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.tuple", "items": [{"__tuple__": [-4, -2, -4]}, {"__tuple__": [-2, 4]}, {"__tuple__": []}, {"__tuple__": [1]}, {"__tuple__": [3]}]}"""
        ),
        function_defs=FunctionDefList([tuple_is_empty_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bool", "items": [false, true]}"""
        ),
        function_defs=FunctionDefList([bool_identity_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.int", "items": [8, -8, -10, 4]}"""
        ),
        function_defs=FunctionDefList([is_odd_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bool", "items": [false, true]}"""
        ),
        function_defs=FunctionDefList([bool_not_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytearray", "items": [{"__bytearray__": []}, {"__bytearray__": [33, 189, 183]}, {"__bytearray__": [163]}, {"__bytearray__": [32, 88, 242, 203]}]}"""
        ),
        function_defs=FunctionDefList([bytearray_length_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytearray", "items": [{"__bytearray__": [104, 24, 30]}, {"__bytearray__": [178]}, {"__bytearray__": [193, 85, 210]}, {"__bytearray__": []}, {"__bytearray__": []}]}"""
        ),
        function_defs=FunctionDefList([bytearray_length_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.float", "items": [-0.9883910721602831, -2.8067955464699157, 4.065881247970944, -7.874312917200785]}"""
        ),
        function_defs=FunctionDefList([f_square_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.int", "items": [-10, 4, 2, -4, -10]}"""
        ),
        function_defs=FunctionDefList([square_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.complex", "items": [{"__complex__": [-2.0, 5.0]}, {"__complex__": [-4.0, -3.0]}, {"__complex__": [1.0, 3.0]}, {"__complex__": [5.0, 5.0]}]}"""
        ),
        function_defs=FunctionDefList([complex_conjugate_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytes", "items": [{"__bytes__": [90]}, {"__bytes__": [18, 9]}, {"__bytes__": []}]}"""
        ),
        function_defs=FunctionDefList([bytes_to_hex_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytearray", "items": [{"__bytearray__": []}, {"__bytearray__": [248]}, {"__bytearray__": [97, 73]}]}"""
        ),
        function_defs=FunctionDefList([bytearray_length_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.set", "items": [{"__set__": [-5, -4]}, {"__set__": []}, {"__set__": [0, 1]}, {"__set__": []}]}"""
        ),
        function_defs=FunctionDefList([set_hash_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.range", "items": [{"__range__": [1, 6, 1]}, {"__range__": [2, 6, 1]}, {"__range__": [3, 3, 1]}]}"""
        ),
        function_defs=FunctionDefList([range_length_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.complex", "items": [{"__complex__": [3.0, -1.0]}, {"__complex__": [2.0, -1.0]}, {"__complex__": [-1.0, 2.0]}, {"__complex__": [-5.0, 2.0]}, {"__complex__": [-5.0, 2.0]}]}"""
        ),
        function_defs=FunctionDefList([complex_abs_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{"a": 2, "b": -1}, {"a": 0, "b": -3}, {"a": 2}, {"a": -4, "b": 1, "c": 5}]}"""
        ),
        function_defs=FunctionDefList([dict_length_func, double_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.set", "items": [{"__set__": [0, 1, -3]}, {"__set__": [-2]}, {"__set__": [-4, -3]}, {"__set__": [5, 4, -3]}]}"""
        ),
        function_defs=FunctionDefList([set_to_list_func, list_median_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytes", "items": [{"__bytes__": [19, 49, 206]}, {"__bytes__": []}, {"__bytes__": [210, 215]}, {"__bytes__": [39, 169]}, {"__bytes__": [161, 200, 87, 235]}]}"""
        ),
        function_defs=FunctionDefList([bytes_upper_func, bytes_to_hex_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytearray", "items": [{"__bytearray__": [245]}, {"__bytearray__": [247, 197, 65]}, {"__bytearray__": []}, {"__bytearray__": [145]}, {"__bytearray__": [134, 139]}]}"""
        ),
        function_defs=FunctionDefList([bytearray_reverse_func, bytearray_reverse_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{"a": 0, "b": -5}, {"a": -4}, {"a": 2}, {"a": 2}, {"a": -2, "b": -5}]}"""
        ),
        function_defs=FunctionDefList([dict_items_func, list_is_empty_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.float", "items": [-2.319045949155858, -8.75437589906875, 9.750516659246212, 8.591727220803755]}"""
        ),
        function_defs=FunctionDefList([f_log10_func, f_round_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.complex", "items": [{"__complex__": [1.0, -1.0]}, {"__complex__": [-1.0, -1.0]}, {"__complex__": [4.0, 4.0]}]}"""
        ),
        function_defs=FunctionDefList([complex_phase_func, f_sin_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.tuple", "items": [{"__tuple__": [3, -3]}, {"__tuple__": [0, -1, 4]}, {"__tuple__": [2, 1, 2]}, {"__tuple__": [0, 3]}]}"""
        ),
        function_defs=FunctionDefList([tuple_to_index_dict_func, dict_freeze_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.set", "items": [{"__set__": [0, 4, -3]}, {"__set__": [4, -2, -1]}, {"__set__": []}]}"""
        ),
        function_defs=FunctionDefList([set_hash_func, int_clip_0_100_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{"a": 4, "b": -1, "c": 1}, {"a": -4, "b": -5, "c": 5}, {"a": -3}, {}]}"""
        ),
        function_defs=FunctionDefList([dict_keyset_func, set_size_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytes", "items": [{"__bytes__": []}, {"__bytes__": [50, 34, 66, 202]}, {"__bytes__": []}]}"""
        ),
        function_defs=FunctionDefList([bytes_upper_func, bytes_upper_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.int", "items": [-9, -9, -8]}"""
        ),
        function_defs=FunctionDefList([is_negative_func, bool_to_float_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.str", "items": ["BE", "XYLzbc", "1NN1x", "e"]}"""
        ),
        function_defs=FunctionDefList([count_a_func, sign_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bool", "items": [false, false]}"""
        ),
        function_defs=FunctionDefList([bool_to_float_func, f_log10_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.range", "items": [{"__range__": [1, 2, 1]}, {"__range__": [0, 5, 1]}, {"__range__": [2, 6, 1]}, {"__range__": [2, 7, 1]}]}"""
        ),
        function_defs=FunctionDefList([range_length_func, int_is_power_of_two_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.complex", "items": [{"__complex__": [4.0, 5.0]}, {"__complex__": [-4.0, 0.0]}, {"__complex__": [-2.0, -1.0]}, {"__complex__": [3.0, 2.0]}]}"""
        ),
        function_defs=FunctionDefList([complex_conjugate_func, complex_abs_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{"a": -4, "b": 0}, {"a": -4}, {"a": -2}, {"a": 5}, {"a": -3, "b": 3}]}"""
        ),
        function_defs=FunctionDefList([dict_keys_func, list_unique_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytearray", "items": [{"__bytearray__": [215]}, {"__bytearray__": [76]}, {"__bytearray__": [124]}, {"__bytearray__": [132, 236, 84]}]}"""
        ),
        function_defs=FunctionDefList([bytearray_length_func, half_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytearray", "items": [{"__bytearray__": [171, 40, 179, 205]}, {"__bytearray__": [10, 149, 240, 61]}, {"__bytearray__": [110]}, {"__bytearray__": [236]}, {"__bytearray__": [103, 163, 17, 16]}]}"""
        ),
        function_defs=FunctionDefList([bytearray_to_bytes_func, bytes_upper_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.complex", "items": [{"__complex__": [2.0, -5.0]}, {"__complex__": [1.0, 0.0]}, {"__complex__": [2.0, 1.0]}]}"""
        ),
        function_defs=FunctionDefList([complex_phase_func, f_reciprocal_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.int", "items": [2, -4, 4, -4, 7]}"""
        ),
        function_defs=FunctionDefList([int_is_power_of_two_func, bool_identity_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.tuple", "items": [{"__tuple__": [3]}, {"__tuple__": [4]}, {"__tuple__": [-1, 0, -1]}, {"__tuple__": [-4, 1, 1]}, {"__tuple__": []}]}"""
        ),
        function_defs=FunctionDefList([tuple_length_func, int_bit_length_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.range", "items": [{"__range__": [1, 4, 1]}, {"__range__": [0, 3, 1]}, {"__range__": [2, 3, 1]}, {"__range__": [3, 8, 1]}, {"__range__": [2, 4, 1]}]}"""
        ),
        function_defs=FunctionDefList([range_sum_func, is_negative_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.complex", "items": [{"__complex__": [-1.0, -1.0]}, {"__complex__": [1.0, -2.0]}, {"__complex__": [-1.0, 1.0]}, {"__complex__": [-3.0, -4.0]}, {"__complex__": [2.0, 5.0]}]}"""
        ),
        function_defs=FunctionDefList([complex_imag_func, f_sin_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytearray", "items": [{"__bytearray__": [87, 117]}, {"__bytearray__": [55]}, {"__bytearray__": [186]}, {"__bytearray__": [189, 0]}, {"__bytearray__": [227, 167, 99]}]}"""
        ),
        function_defs=FunctionDefList([bytearray_length_func, int_popcount_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.str", "items": ["Hl", "gDVnY", "K"]}"""
        ),
        function_defs=FunctionDefList([str_reverse_words_func, contains_space_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.range", "items": [{"__range__": [3, 6, 1]}, {"__range__": [0, 2, 1]}, {"__range__": [3, 6, 1]}, {"__range__": [1, 3, 1]}, {"__range__": [3, 5, 1]}]}"""
        ),
        function_defs=FunctionDefList([range_length_func, mod2_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytes", "items": [{"__bytes__": [141]}, {"__bytes__": [187, 167, 12]}, {"__bytes__": [153]}, {"__bytes__": [1, 74]}, {"__bytes__": []}]}"""
        ),
        function_defs=FunctionDefList([bytes_upper_func, bytes_to_hex_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bool", "items": [true, false]}"""
        ),
        function_defs=FunctionDefList([bool_to_float_func, f_exp_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{"a": -1}, {"a": 0, "b": -5, "c": 1}, {"a": 2, "b": 3, "c": -2}, {"a": 2, "b": 4, "c": -5}, {"a": 0}]}"""
        ),
        function_defs=FunctionDefList([dict_keyset_func, set_is_empty_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.complex", "items": [{"__complex__": [0.0, 5.0]}, {"__complex__": [0.0, -3.0]}, {"__complex__": [4.0, 4.0]}, {"__complex__": [-2.0, -3.0]}, {"__complex__": [4.0, 3.0]}]}"""
        ),
        function_defs=FunctionDefList([complex_real_func, f_sin_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.float", "items": [-0.8349065522425132, 7.951263429635624, 3.7140430546246037, 4.087064396459727, 7.384337784753967]}"""
        ),
        function_defs=FunctionDefList([f_reciprocal_func, f_sin_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.list", "items": [[-3, -5], [-1, -2], [1, 3, 3], [-1, -2, 2], [5, 4, 3]]}"""
        ),
        function_defs=FunctionDefList([list_min_func, square_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.complex", "items": [{"__complex__": [3.0, 4.0]}, {"__complex__": [2.0, 4.0]}, {"__complex__": [-4.0, -5.0]}, {"__complex__": [-3.0, 2.0]}, {"__complex__": [-4.0, -5.0]}]}"""
        ),
        function_defs=FunctionDefList([complex_imag_func, f_trunc_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{}, {"a": -5, "b": 5}, {}, {"a": -5, "b": -3, "c": -1}]}"""
        ),
        function_defs=FunctionDefList([dict_keyset_func, set_is_empty_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{"a": -5, "b": 3}, {"a": 4}, {}, {"a": 2}, {"a": -4, "b": -2, "c": -3}]}"""
        ),
        function_defs=FunctionDefList([dict_values_func, list_median_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{"a": -4}, {}, {}, {"a": -5, "b": 4, "c": 5}]}"""
        ),
        function_defs=FunctionDefList([dict_values_func, list_tail_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{}, {}, {"a": 3, "b": 5, "c": 2}, {}, {"a": 2, "b": 3}]}"""
        ),
        function_defs=FunctionDefList([dict_length_func, int_clip_0_100_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.set", "items": [{"__set__": [-2]}, {"__set__": []}, {"__set__": []}]}"""
        ),
        function_defs=FunctionDefList([set_to_list_func, list_unique_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytearray", "items": [{"__bytearray__": [104, 113]}, {"__bytearray__": [91, 207]}, {"__bytearray__": []}]}"""
        ),
        function_defs=FunctionDefList([bytearray_length_func, double_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{"a": 4, "b": -2, "c": 4}, {"a": 0, "b": -1}, {}, {"a": 3}, {"a": 4, "b": 5, "c": -2}]}"""
        ),
        function_defs=FunctionDefList([dict_keyset_func, set_to_list_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{"a": -1}, {"a": 5, "b": 2, "c": 3}, {}, {}, {"a": 3}]}"""
        ),
        function_defs=FunctionDefList([dict_flip_func, dict_values_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.int", "items": [3, -6, -10, -8, -7]}"""
        ),
        function_defs=FunctionDefList([is_even_func, bool_not_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.range", "items": [{"__range__": [1, 4, 1]}, {"__range__": [3, 3, 1]}, {"__range__": [3, 5, 1]}, {"__range__": [1, 4, 1]}, {"__range__": [0, 5, 1]}]}"""
        ),
        function_defs=FunctionDefList([range_length_func, abs_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytearray", "items": [{"__bytearray__": [35, 157, 0]}, {"__bytearray__": [113, 121, 205, 2]}, {"__bytearray__": [165, 198, 151, 166]}, {"__bytearray__": [97, 239]}]}"""
        ),
        function_defs=FunctionDefList([bytearray_length_func, mod2_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.list", "items": [[], [1, -1], [1]]}"""
        ),
        function_defs=FunctionDefList([list_length_func, sign_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.float", "items": [9.957879402016086, 1.3068116316687473, -0.4677383544774081, -9.52236821860464]}"""
        ),
        function_defs=FunctionDefList([f_frac_percent_func, square_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.set", "items": [{"__set__": []}, {"__set__": [-4]}, {"__set__": [1, -4, -1]}, {"__set__": []}]}"""
        ),
        function_defs=FunctionDefList([set_hash_func, abs_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.complex", "items": [{"__complex__": [1.0, -4.0]}, {"__complex__": [3.0, -4.0]}, {"__complex__": [4.0, 4.0]}, {"__complex__": [3.0, -5.0]}]}"""
        ),
        function_defs=FunctionDefList([complex_abs_func, f_reciprocal_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.set", "items": [{"__set__": [-3, 5]}, {"__set__": [4]}, {"__set__": [4, -2]}]}"""
        ),
        function_defs=FunctionDefList([set_to_list_func, list_is_empty_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.complex", "items": [{"__complex__": [-1.0, -1.0]}, {"__complex__": [-4.0, -2.0]}, {"__complex__": [3.0, 4.0]}, {"__complex__": [4.0, -2.0]}, {"__complex__": [-4.0, -4.0]}]}"""
        ),
        function_defs=FunctionDefList([complex_real_func, f_ceil_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.range", "items": [{"__range__": [1, 1, 1]}, {"__range__": [0, 2, 1]}, {"__range__": [3, 3, 1]}, {"__range__": [3, 6, 1]}, {"__range__": [2, 5, 1]}]}"""
        ),
        function_defs=FunctionDefList([range_max_func, double_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytes", "items": [{"__bytes__": [6, 251]}, {"__bytes__": [164]}, {"__bytes__": [201]}]}"""
        ),
        function_defs=FunctionDefList([bytes_is_ascii_func, bool_not_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.list", "items": [[], [5, 3], [5, 3, -2], [-2, 0], [-4, 1]]}"""
        ),
        function_defs=FunctionDefList([list_is_empty_func, bool_identity_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{"a": -5, "b": -1, "c": 5}, {"a": -3}, {"a": 4}, {}, {"a": 5, "b": -4}]}"""
        ),
        function_defs=FunctionDefList([dict_keys_func, list_sum_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bool", "items": [true, false]}"""
        ),
        function_defs=FunctionDefList([bool_not_func, bool_to_str_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.tuple", "items": [{"__tuple__": [1, -2]}, {"__tuple__": [-5, 1]}, {"__tuple__": []}]}"""
        ),
        function_defs=FunctionDefList([tuple_to_index_dict_func, dict_is_empty_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.tuple", "items": [{"__tuple__": []}, {"__tuple__": []}, {"__tuple__": [-4, -3, -1]}]}"""
        ),
        function_defs=FunctionDefList([tuple_reverse_func, tuple_count_none_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.str", "items": ["5gPJ", "e", "b9", ""]}"""
        ),
        function_defs=FunctionDefList([is_space_func, bool_not_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.str", "items": ["k", "aUE0t9", "crxX"]}"""
        ),
        function_defs=FunctionDefList([str_reverse_words_func, count_a_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytes", "items": [{"__bytes__": [112, 186, 52, 103]}, {"__bytes__": [184]}, {"__bytes__": []}, {"__bytes__": []}, {"__bytes__": []}]}"""
        ),
        function_defs=FunctionDefList([bytes_reverse_func, bytes_is_ascii_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytes", "items": [{"__bytes__": [43, 29, 176]}, {"__bytes__": [14, 114, 170, 87]}, {"__bytes__": [96, 120, 59]}]}"""
        ),
        function_defs=FunctionDefList([bytes_is_empty_func, bool_to_float_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.int", "items": [0, -7, -7, 8, 3]}"""
        ),
        function_defs=FunctionDefList([int_bit_length_func, int_to_str_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{"a": 2, "b": 3, "c": 0}, {"a": -1, "b": -3, "c": 1}, {"a": 1, "b": 2}]}"""
        ),
        function_defs=FunctionDefList([dict_keys_func, list_sum_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.range", "items": [{"__range__": [1, 3, 1]}, {"__range__": [2, 2, 1]}, {"__range__": [0, 3, 1]}]}"""
        ),
        function_defs=FunctionDefList([range_max_func, is_positive_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.float", "items": [4.973256499378287, 9.396364813260767, 3.1728416968764]}"""
        ),
        function_defs=FunctionDefList([f_reciprocal_func, f_floor_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.complex", "items": [{"__complex__": [1.0, 1.0]}, {"__complex__": [-3.0, -5.0]}, {"__complex__": [3.0, 5.0]}]}"""
        ),
        function_defs=FunctionDefList([complex_conjugate_func, complex_conjugate_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.list", "items": [[-4], [2, -3], [5, -3, -1], [2]]}"""
        ),
        function_defs=FunctionDefList([list_tail_func, list_sorted_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bool", "items": [true, false]}"""
        ),
        function_defs=FunctionDefList([bool_not_func, bool_identity_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytearray", "items": [{"__bytearray__": [6, 202, 90, 192]}, {"__bytearray__": [20, 94, 177]}, {"__bytearray__": [155, 85, 130, 47]}, {"__bytearray__": [28]}]}"""
        ),
        function_defs=FunctionDefList([bytearray_reverse_func, bytearray_length_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.set", "items": [{"__set__": [5, -1]}, {"__set__": [0, 1, 3]}, {"__set__": [3, -4]}, {"__set__": [-5, -2]}]}"""
        ),
        function_defs=FunctionDefList([set_hash_func, mod2_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.int", "items": [-3, -8, -4, 3, -8]}"""
        ),
        function_defs=FunctionDefList([int_popcount_func, is_positive_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.complex", "items": [{"__complex__": [3.0, -4.0]}, {"__complex__": [0.0, 4.0]}, {"__complex__": [3.0, 3.0]}, {"__complex__": [5.0, -4.0]}]}"""
        ),
        function_defs=FunctionDefList([complex_real_func, f_exp_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.str", "items": ["g4ttR", "W", "kgw", ""]}"""
        ),
        function_defs=FunctionDefList([swapcase_func, str_is_palindrome_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.range", "items": [{"__range__": [1, 5, 1]}, {"__range__": [2, 5, 1]}, {"__range__": [2, 4, 1]}, {"__range__": [3, 8, 1]}, {"__range__": [2, 5, 1]}]}"""
        ),
        function_defs=FunctionDefList([range_length_func, is_even_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.float", "items": [1.546123054575455, 3.341633574242435, 7.092917455244926, 7.166518355920918]}"""
        ),
        function_defs=FunctionDefList([f_trunc_func, abs_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{"a": -5}, {"a": 0}, {"a": -3, "b": 2, "c": 3}, {"a": -2}]}"""
        ),
        function_defs=FunctionDefList([dict_keyset_func, set_to_list_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.range", "items": [{"__range__": [3, 7, 1]}, {"__range__": [1, 2, 1]}, {"__range__": [1, 2, 1]}, {"__range__": [1, 6, 1]}, {"__range__": [2, 4, 1]}]}"""
        ),
        function_defs=FunctionDefList([range_sum_func, int_to_str_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.list", "items": [[3, 1], [-2, 4], [-5, 3, 2], [-3, -1, 1], [0]]}"""
        ),
        function_defs=FunctionDefList([list_is_empty_func, bool_to_float_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytes", "items": [{"__bytes__": [52]}, {"__bytes__": []}, {"__bytes__": []}, {"__bytes__": [94, 202]}]}"""
        ),
        function_defs=FunctionDefList([bytes_upper_func, bytes_reverse_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.set", "items": [{"__set__": [0]}, {"__set__": []}, {"__set__": [1, -3]}, {"__set__": []}, {"__set__": []}]}"""
        ),
        function_defs=FunctionDefList([set_size_func, abs_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.str", "items": ["9sxQ", "9Jg", "AV", "s"]}"""
        ),
        function_defs=FunctionDefList([count_a_func, is_positive_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.str", "items": ["sY7nGT", "BgV", "LF"]}"""
        ),
        function_defs=FunctionDefList([upper_func, endswith_z_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytearray", "items": [{"__bytearray__": []}, {"__bytearray__": [28, 249]}, {"__bytearray__": [28]}, {"__bytearray__": [188]}, {"__bytearray__": [29, 169, 37]}]}"""
        ),
        function_defs=FunctionDefList([bytearray_reverse_func, bytearray_length_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.int", "items": [-7, 9, 5, 0]}"""
        ),
        function_defs=FunctionDefList([double_func, inc_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.list", "items": [[], [2], [1, -4]]}"""
        ),
        function_defs=FunctionDefList([list_is_empty_func, bool_to_int_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bool", "items": [false, false]}"""
        ),
        function_defs=FunctionDefList([bool_to_float_func, f_sin_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytearray", "items": [{"__bytearray__": [14]}, {"__bytearray__": [130, 80, 156]}, {"__bytearray__": [112]}]}"""
        ),
        function_defs=FunctionDefList(
            [bytearray_reverse_func, bytearray_to_bytes_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytes", "items": [{"__bytes__": [76, 207, 94, 43]}, {"__bytes__": [155, 19, 76, 82]}, {"__bytes__": [165, 194, 198, 111]}, {"__bytes__": [199, 58, 189, 62]}]}"""
        ),
        function_defs=FunctionDefList([bytes_is_ascii_func, bool_to_str_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bool", "items": [false, true]}"""
        ),
        function_defs=FunctionDefList([bool_identity_func, bool_to_str_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.list", "items": [[], [], []]}"""
        ),
        function_defs=FunctionDefList([list_median_func, f_trunc_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytes", "items": [{"__bytes__": [22, 182, 72]}, {"__bytes__": [104, 247]}, {"__bytes__": []}, {"__bytes__": [239]}, {"__bytes__": [71, 38, 12, 1]}]}"""
        ),
        function_defs=FunctionDefList([bytes_to_hex_func, capitalize_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.tuple", "items": [{"__tuple__": [4, 1, -5]}, {"__tuple__": []}, {"__tuple__": [-3]}, {"__tuple__": [1]}, {"__tuple__": [-3]}]}"""
        ),
        function_defs=FunctionDefList([tuple_reverse_func, tuple_reverse_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.complex", "items": [{"__complex__": [5.0, 3.0]}, {"__complex__": [5.0, 5.0]}, {"__complex__": [2.0, -4.0]}, {"__complex__": [-2.0, -4.0]}]}"""
        ),
        function_defs=FunctionDefList([complex_imag_func, f_abs_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytearray", "items": [{"__bytearray__": [145, 114, 165, 13]}, {"__bytearray__": [155, 132, 29]}, {"__bytearray__": [234, 195, 205]}, {"__bytearray__": [136]}, {"__bytearray__": [65, 237, 251]}]}"""
        ),
        function_defs=FunctionDefList(
            [bytearray_reverse_func, bytearray_to_bytes_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytearray", "items": [{"__bytearray__": [74, 30, 1, 235]}, {"__bytearray__": [132, 199]}, {"__bytearray__": [48, 252]}, {"__bytearray__": []}, {"__bytearray__": [200]}]}"""
        ),
        function_defs=FunctionDefList(
            [bytearray_reverse_func, bytearray_to_bytes_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.tuple", "items": [{"__tuple__": [-4]}, {"__tuple__": [3]}, {"__tuple__": [2]}]}"""
        ),
        function_defs=FunctionDefList([tuple_count_none_func, is_positive_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{}, {"a": -5}, {"a": 5, "b": -3}, {"a": 2}]}"""
        ),
        function_defs=FunctionDefList(
            [dict_has_duplicate_values_func, bool_to_float_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytes", "items": [{"__bytes__": [124]}, {"__bytes__": [40]}, {"__bytes__": []}, {"__bytes__": [123, 1, 212]}]}"""
        ),
        function_defs=FunctionDefList([bytes_upper_func, bytes_is_empty_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.complex", "items": [{"__complex__": [-1.0, 2.0]}, {"__complex__": [-3.0, 0.0]}, {"__complex__": [4.0, -1.0]}, {"__complex__": [2.0, -1.0]}]}"""
        ),
        function_defs=FunctionDefList([complex_abs_func, f_reciprocal_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.list", "items": [[5, 2, 2], [], [1, 5], [1, 2], [4, 4, -4]]}"""
        ),
        function_defs=FunctionDefList(
            [list_tail_func, list_unique_func, list_sorted_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.list", "items": [[], [], [-4]]}"""
        ),
        function_defs=FunctionDefList([list_reverse_func, list_max_func, double_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.tuple", "items": [{"__tuple__": []}, {"__tuple__": [-3, -1, -5]}, {"__tuple__": []}, {"__tuple__": [2, -5]}]}"""
        ),
        function_defs=FunctionDefList(
            [tuple_to_list_func, list_median_func, f_frac_percent_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytearray", "items": [{"__bytearray__": [47, 199]}, {"__bytearray__": [124, 180]}, {"__bytearray__": [93, 144, 42, 149]}, {"__bytearray__": [59, 226]}, {"__bytearray__": [87, 30]}]}"""
        ),
        function_defs=FunctionDefList([bytearray_length_func, abs_func, dec_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytearray", "items": [{"__bytearray__": [4, 254, 240]}, {"__bytearray__": [234, 199]}, {"__bytearray__": [52, 89, 235, 33]}]}"""
        ),
        function_defs=FunctionDefList(
            [bytearray_length_func, int_popcount_func, int_clip_0_100_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.range", "items": [{"__range__": [3, 7, 1]}, {"__range__": [2, 2, 1]}, {"__range__": [3, 4, 1]}]}"""
        ),
        function_defs=FunctionDefList(
            [range_sum_func, int_bit_length_func, int_to_float_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.list", "items": [[-1], [5, -2], [3, 0, 1], []]}"""
        ),
        function_defs=FunctionDefList(
            [list_sum_func, is_negative_func, bool_identity_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.range", "items": [{"__range__": [2, 3, 1]}, {"__range__": [1, 2, 1]}, {"__range__": [0, 5, 1]}, {"__range__": [0, 4, 1]}]}"""
        ),
        function_defs=FunctionDefList([range_max_func, is_even_func, bool_to_str_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.set", "items": [{"__set__": [3, 4, -3]}, {"__set__": []}, {"__set__": [-5, -3]}, {"__set__": [3, 5]}]}"""
        ),
        function_defs=FunctionDefList(
            [set_to_list_func, list_unique_func, list_sorted_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.complex", "items": [{"__complex__": [2.0, -5.0]}, {"__complex__": [-3.0, 3.0]}, {"__complex__": [-1.0, 2.0]}, {"__complex__": [5.0, -1.0]}, {"__complex__": [-2.0, -3.0]}]}"""
        ),
        function_defs=FunctionDefList(
            [complex_imag_func, f_is_integer_func, bool_not_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytearray", "items": [{"__bytearray__": []}, {"__bytearray__": [38, 168]}, {"__bytearray__": [230, 144, 211, 92]}, {"__bytearray__": [150, 250]}, {"__bytearray__": [226]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                bytearray_to_bytes_func,
                bytes_is_empty_func,
                bool_identity_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.range", "items": [{"__range__": [3, 3, 1]}, {"__range__": [3, 6, 1]}, {"__range__": [3, 3, 1]}, {"__range__": [0, 4, 1]}]}"""
        ),
        function_defs=FunctionDefList([range_sum_func, double_func, inc_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.complex", "items": [{"__complex__": [3.0, 5.0]}, {"__complex__": [3.0, 1.0]}, {"__complex__": [5.0, 5.0]}, {"__complex__": [3.0, 4.0]}, {"__complex__": [2.0, 1.0]}]}"""
        ),
        function_defs=FunctionDefList([complex_abs_func, f_exp_func, f_ceil_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.list", "items": [[], [3, -2, 0], []]}"""
        ),
        function_defs=FunctionDefList([list_median_func, f_ceil_func, abs_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.tuple", "items": [{"__tuple__": [5, -5, 2]}, {"__tuple__": [-2, 2]}, {"__tuple__": [0, 2, 4]}, {"__tuple__": []}, {"__tuple__": [-2, -5, 3]}]}"""
        ),
        function_defs=FunctionDefList(
            [tuple_reverse_func, tuple_is_empty_func, bool_to_int_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{"a": -4, "b": -2}, {"a": 1}, {"a": 2, "b": -2, "c": -2}, {"a": -1, "b": -5}, {"a": 3}]}"""
        ),
        function_defs=FunctionDefList(
            [dict_is_empty_func, bool_identity_func, bool_to_str_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.int", "items": [-4, 7, 1, 0]}"""
        ),
        function_defs=FunctionDefList([inc_func, int_bit_length_func, dec_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.complex", "items": [{"__complex__": [-1.0, 4.0]}, {"__complex__": [4.0, -4.0]}, {"__complex__": [-5.0, -1.0]}, {"__complex__": [-3.0, 4.0]}, {"__complex__": [-1.0, 2.0]}]}"""
        ),
        function_defs=FunctionDefList(
            [complex_imag_func, float_to_str_func, str_hash_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytes", "items": [{"__bytes__": [66]}, {"__bytes__": [219, 202, 90, 41]}, {"__bytes__": [23, 228, 123]}, {"__bytes__": [215, 253, 154, 116]}, {"__bytes__": []}]}"""
        ),
        function_defs=FunctionDefList([bytes_to_hex_func, upper_func, title_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{"a": 0, "b": -4, "c": 4}, {"a": 4, "b": 2}, {"a": -2, "b": -5}, {"a": -4, "b": -3, "c": 4}, {}]}"""
        ),
        function_defs=FunctionDefList([dict_values_func, list_length_func, inc_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytes", "items": [{"__bytes__": [9, 22, 25]}, {"__bytes__": [76]}, {"__bytes__": [217, 124]}]}"""
        ),
        function_defs=FunctionDefList(
            [bytes_upper_func, bytes_is_ascii_func, bool_identity_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.int", "items": [2, 0, -9]}"""
        ),
        function_defs=FunctionDefList(
            [int_to_bool_func, bool_not_func, bool_identity_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.range", "items": [{"__range__": [1, 5, 1]}, {"__range__": [2, 4, 1]}, {"__range__": [2, 6, 1]}, {"__range__": [1, 1, 1]}, {"__range__": [0, 0, 1]}]}"""
        ),
        function_defs=FunctionDefList(
            [range_sum_func, is_positive_func, bool_not_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytes", "items": [{"__bytes__": [75]}, {"__bytes__": [238, 42]}, {"__bytes__": [3, 134, 58, 123]}]}"""
        ),
        function_defs=FunctionDefList(
            [bytes_is_empty_func, bool_not_func, bool_not_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.complex", "items": [{"__complex__": [-5.0, -2.0]}, {"__complex__": [3.0, -3.0]}, {"__complex__": [-5.0, 2.0]}]}"""
        ),
        function_defs=FunctionDefList(
            [complex_conjugate_func, complex_real_func, f_trunc_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytearray", "items": [{"__bytearray__": [126]}, {"__bytearray__": [26, 118, 104, 47]}, {"__bytearray__": [59, 199, 6, 103]}, {"__bytearray__": []}, {"__bytearray__": [110, 203, 78, 75]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                bytearray_to_bytes_func,
                bytes_reverse_func,
                bytes_is_ascii_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.float", "items": [8.065987867462741, 7.561419933373401, -2.559649594856128]}"""
        ),
        function_defs=FunctionDefList([f_square_func, f_exp_func, f_reciprocal_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.list", "items": [[-5, -4, 5], [-4, 1], []]}"""
        ),
        function_defs=FunctionDefList([list_sum_func, inc_func, sign_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{"a": -5}, {"a": 0}, {"a": -1}, {"a": -4}]}"""
        ),
        function_defs=FunctionDefList(
            [dict_is_empty_func, bool_to_str_func, repeat_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytes", "items": [{"__bytes__": [76, 81, 68, 81]}, {"__bytes__": [92]}, {"__bytes__": [156, 248, 37, 55]}]}"""
        ),
        function_defs=FunctionDefList(
            [bytes_reverse_func, bytes_is_ascii_func, bool_not_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.list", "items": [[], [-2, -3], [], [-4, -4]]}"""
        ),
        function_defs=FunctionDefList(
            [list_unique_func, list_sorted_func, list_unique_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.set", "items": [{"__set__": []}, {"__set__": [3, 0, -5]}, {"__set__": []}, {"__set__": [1, -2]}, {"__set__": []}]}"""
        ),
        function_defs=FunctionDefList(
            [set_to_list_func, list_reverse_func, list_is_empty_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.str", "items": ["1h4", "ABvDi0", "2E", "m"]}"""
        ),
        function_defs=FunctionDefList(
            [strip_func, startswith_a_func, bool_to_float_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bool", "items": [true, true]}"""
        ),
        function_defs=FunctionDefList(
            [bool_to_str_func, str_hash_func, int_to_bool_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.list", "items": [[-5], [4, 3, 4], [-3, -2], [0, -2]]}"""
        ),
        function_defs=FunctionDefList([list_tail_func, list_sum_func, is_even_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.int", "items": [6, -9, -8, 9]}"""
        ),
        function_defs=FunctionDefList(
            [neg_func, int_bit_length_func, int_to_float_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytes", "items": [{"__bytes__": [169]}, {"__bytes__": [236, 131, 72]}, {"__bytes__": [179, 151, 60]}, {"__bytes__": [243, 229, 162, 31]}, {"__bytes__": [157, 217, 96]}]}"""
        ),
        function_defs=FunctionDefList(
            [bytes_length_func, is_negative_func, bool_to_float_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.int", "items": [0, -10, -10, -6, 0]}"""
        ),
        function_defs=FunctionDefList(
            [int_bit_length_func, is_negative_func, bool_to_int_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.set", "items": [{"__set__": [5]}, {"__set__": [2]}, {"__set__": []}, {"__set__": [3, -2]}, {"__set__": [5]}]}"""
        ),
        function_defs=FunctionDefList(
            [set_hash_func, int_to_bool_func, bool_to_str_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bool", "items": [false, false]}"""
        ),
        function_defs=FunctionDefList(
            [bool_to_str_func, str_to_list_func, list_length_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{"a": 4}, {"a": 3}, {"a": -1, "b": 3}]}"""
        ),
        function_defs=FunctionDefList(
            [dict_freeze_func, tuple_reverse_func, tuple_reverse_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.complex", "items": [{"__complex__": [4.0, 4.0]}, {"__complex__": [-3.0, 3.0]}, {"__complex__": [2.0, -5.0]}, {"__complex__": [-1.0, 4.0]}]}"""
        ),
        function_defs=FunctionDefList([complex_phase_func, f_trunc_func, half_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.str", "items": ["", "pz6", "g42"]}"""
        ),
        function_defs=FunctionDefList(
            [capitalize_func, str_is_palindrome_func, bool_not_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytearray", "items": [{"__bytearray__": [229, 226, 188, 205]}, {"__bytearray__": []}, {"__bytearray__": [198, 134, 190]}, {"__bytearray__": [32, 153]}, {"__bytearray__": [127, 145, 28]}]}"""
        ),
        function_defs=FunctionDefList(
            [bytearray_length_func, int_clip_0_100_func, int_to_bool_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.int", "items": [-9, 4, -9, 7, -7]}"""
        ),
        function_defs=FunctionDefList(
            [neg_func, int_is_power_of_two_func, bool_to_str_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.str", "items": ["1PzkT", "6YL", "QPk", "COTDs", "h"]}"""
        ),
        function_defs=FunctionDefList([last_char_func, swapcase_func, duplicate_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.range", "items": [{"__range__": [2, 7, 1]}, {"__range__": [2, 3, 1]}, {"__range__": [3, 8, 1]}]}"""
        ),
        function_defs=FunctionDefList(
            [range_length_func, int_to_float_func, f_fraction_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.set", "items": [{"__set__": [1, 3, -1]}, {"__set__": [2, -4, -3]}, {"__set__": [2, -3]}, {"__set__": [2, 5]}, {"__set__": [-5, -3, -2]}]}"""
        ),
        function_defs=FunctionDefList([set_hash_func, int_to_str_func, is_space_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.str", "items": ["nGgGx", "80bkX", "09o5EE"]}"""
        ),
        function_defs=FunctionDefList(
            [strip_func, str_is_palindrome_func, bool_not_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.tuple", "items": [{"__tuple__": []}, {"__tuple__": [-5]}, {"__tuple__": [-3, -5]}, {"__tuple__": [-3, -5, 5]}]}"""
        ),
        function_defs=FunctionDefList(
            [tuple_to_index_dict_func, dict_length_func, is_even_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bool", "items": [true, true]}"""
        ),
        function_defs=FunctionDefList(
            [bool_identity_func, bool_not_func, bool_identity_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.str", "items": ["", "nWAxW", "f", "McEp", "LH5o7w"]}"""
        ),
        function_defs=FunctionDefList([str_to_list_func, list_sum_func, dec_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{"a": 0}, {"a": 0}, {"a": 4, "b": 4}, {"a": 1, "b": 0, "c": 1}, {"a": 1, "b": -3, "c": 4}]}"""
        ),
        function_defs=FunctionDefList(
            [dict_values_func, list_sorted_func, list_unique_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{}, {"a": 3, "b": -1, "c": 5}, {"a": -3, "b": -2}, {"a": -5, "b": 2, "c": 4}, {}]}"""
        ),
        function_defs=FunctionDefList(
            [dict_keys_func, list_max_func, int_clip_0_100_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bool", "items": [false, true]}"""
        ),
        function_defs=FunctionDefList(
            [bool_identity_func, bool_not_func, bool_to_int_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.str", "items": ["9982K", "J8l9p", "", "b", ""]}"""
        ),
        function_defs=FunctionDefList(
            [str_reverse_words_func, is_lower_func, bool_to_float_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.range", "items": [{"__range__": [0, 5, 1]}, {"__range__": [1, 3, 1]}, {"__range__": [2, 5, 1]}, {"__range__": [1, 1, 1]}, {"__range__": [2, 4, 1]}]}"""
        ),
        function_defs=FunctionDefList(
            [range_length_func, int_to_float_func, f_abs_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{"a": -2, "b": 4, "c": 5}, {"a": 0, "b": 5, "c": 2}, {"a": 3, "b": -1}, {"a": -4, "b": 1, "c": -3}, {"a": 3}]}"""
        ),
        function_defs=FunctionDefList(
            [dict_freeze_func, tuple_count_none_func, abs_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.complex", "items": [{"__complex__": [2.0, -5.0]}, {"__complex__": [-3.0, -2.0]}, {"__complex__": [2.0, 2.0]}, {"__complex__": [2.0, -2.0]}, {"__complex__": [0.0, -1.0]}]}"""
        ),
        function_defs=FunctionDefList(
            [complex_real_func, f_reciprocal_func, f_sin_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{"a": -1}, {"a": -1}, {}, {"a": 0}]}"""
        ),
        function_defs=FunctionDefList(
            [dict_is_empty_func, bool_not_func, bool_to_int_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.set", "items": [{"__set__": [1, 2]}, {"__set__": [2, 4]}, {"__set__": [-4, -1]}]}"""
        ),
        function_defs=FunctionDefList(
            [set_to_list_func, list_is_empty_func, bool_to_str_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.list", "items": [[-1, -4, 3], [5, 2], [-3, 3, -1]]}"""
        ),
        function_defs=FunctionDefList([list_tail_func, list_max_func, half_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.int", "items": [3, 4, -5, 0]}"""
        ),
        function_defs=FunctionDefList(
            [identity_int_func, int_to_bool_func, bool_to_str_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytearray", "items": [{"__bytearray__": []}, {"__bytearray__": [121, 208, 205]}, {"__bytearray__": [42, 72, 37, 80]}, {"__bytearray__": [208, 115, 31]}, {"__bytearray__": [184, 131]}]}"""
        ),
        function_defs=FunctionDefList(
            [bytearray_length_func, int_to_float_func, float_to_str_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.complex", "items": [{"__complex__": [2.0, -2.0]}, {"__complex__": [0.0, 5.0]}, {"__complex__": [4.0, -1.0]}, {"__complex__": [4.0, 5.0]}]}"""
        ),
        function_defs=FunctionDefList([complex_real_func, f_exp_func, f_abs_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.complex", "items": [{"__complex__": [4.0, 5.0]}, {"__complex__": [2.0, 5.0]}, {"__complex__": [3.0, -4.0]}]}"""
        ),
        function_defs=FunctionDefList(
            [complex_imag_func, f_mod1_func, f_is_integer_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.str", "items": ["gWfk", "u", "9QIwA8", "sLEw", "8JWddf"]}"""
        ),
        function_defs=FunctionDefList(
            [str_count_vowels_func, int_clip_0_100_func, mod2_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.tuple", "items": [{"__tuple__": []}, {"__tuple__": [4, 1, 5]}, {"__tuple__": []}, {"__tuple__": []}, {"__tuple__": []}]}"""
        ),
        function_defs=FunctionDefList(
            [tuple_to_list_func, list_max_func, is_even_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.list", "items": [[-4], [-1], [-3, 2, -5]]}"""
        ),
        function_defs=FunctionDefList(
            [list_length_func, is_even_func, bool_to_str_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.range", "items": [{"__range__": [0, 4, 1]}, {"__range__": [1, 4, 1]}, {"__range__": [2, 3, 1]}, {"__range__": [3, 3, 1]}]}"""
        ),
        function_defs=FunctionDefList([range_max_func, mod2_func, is_positive_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bool", "items": [false, false]}"""
        ),
        function_defs=FunctionDefList(
            [bool_to_float_func, f_abs_sqrt_func, f_sin_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.set", "items": [{"__set__": []}, {"__set__": [0, -2]}, {"__set__": []}]}"""
        ),
        function_defs=FunctionDefList([set_hash_func, inc_func, int_to_float_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytes", "items": [{"__bytes__": [179, 62]}, {"__bytes__": [141]}, {"__bytes__": [222, 90, 110, 227]}]}"""
        ),
        function_defs=FunctionDefList(
            [bytes_is_ascii_func, bool_to_int_func, dec_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{"a": 3, "b": 3}, {"a": -5}, {}]}"""
        ),
        function_defs=FunctionDefList(
            [dict_freeze_func, tuple_to_index_dict_func, dict_keyset_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.list", "items": [[1, -1, -2], [], [4, 5, 4], [], []]}"""
        ),
        function_defs=FunctionDefList([list_min_func, neg_func, int_bit_length_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.complex", "items": [{"__complex__": [-1.0, -3.0]}, {"__complex__": [-5.0, 2.0]}, {"__complex__": [4.0, -3.0]}, {"__complex__": [3.0, -3.0]}, {"__complex__": [2.0, -3.0]}]}"""
        ),
        function_defs=FunctionDefList(
            [complex_conjugate_func, complex_abs_func, f_mod1_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.set", "items": [{"__set__": [-5]}, {"__set__": [1, -2]}, {"__set__": []}]}"""
        ),
        function_defs=FunctionDefList(
            [set_is_empty_func, bool_to_int_func, identity_int_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.list", "items": [[-5, -2], [], [1, -5], [5], []]}"""
        ),
        function_defs=FunctionDefList(
            [list_max_func, identity_int_func, int_is_power_of_two_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.float", "items": [7.066908733248937, -1.7483754068470976, 7.093248921724559]}"""
        ),
        function_defs=FunctionDefList([f_log10_func, f_frac_percent_func, abs_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.int", "items": [4, -9, 10, -7, -2]}"""
        ),
        function_defs=FunctionDefList([int_popcount_func, sign_func, dec_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.list", "items": [[], [-5, 0], [2, 4, -4]]}"""
        ),
        function_defs=FunctionDefList(
            [list_length_func, int_to_bool_func, bool_to_float_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.range", "items": [{"__range__": [1, 2, 1]}, {"__range__": [3, 7, 1]}, {"__range__": [1, 3, 1]}]}"""
        ),
        function_defs=FunctionDefList(
            [range_length_func, int_clip_0_100_func, mod2_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.complex", "items": [{"__complex__": [2.0, 1.0]}, {"__complex__": [2.0, -1.0]}, {"__complex__": [-2.0, -2.0]}, {"__complex__": [-3.0, -3.0]}, {"__complex__": [-3.0, -2.0]}]}"""
        ),
        function_defs=FunctionDefList([complex_phase_func, f_floor_func, dec_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytearray", "items": [{"__bytearray__": [142, 110, 242]}, {"__bytearray__": [44, 240, 149]}, {"__bytearray__": [103]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                bytearray_reverse_func,
                bytearray_reverse_func,
                bytearray_to_bytes_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.list", "items": [[], [3, 2, 3], [-3, 2], [4], []]}"""
        ),
        function_defs=FunctionDefList([list_unique_func, list_min_func, mod2_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytearray", "items": [{"__bytearray__": [132, 240]}, {"__bytearray__": [195, 16, 61, 11]}, {"__bytearray__": [4]}, {"__bytearray__": [53, 115, 3, 223]}]}"""
        ),
        function_defs=FunctionDefList(
            [bytearray_length_func, dec_func, int_popcount_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.set", "items": [{"__set__": [4]}, {"__set__": [-3, -1]}, {"__set__": []}, {"__set__": []}, {"__set__": [2]}]}"""
        ),
        function_defs=FunctionDefList(
            [set_is_empty_func, bool_to_str_func, lower_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.float", "items": [6.166861534238208, 1.3557610544532288, 1.8216487100916172]}"""
        ),
        function_defs=FunctionDefList([f_fraction_func, f_exp_func, f_square_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.str", "items": ["ZX", "j", "At", "dgEYf", "OD1qi"]}"""
        ),
        function_defs=FunctionDefList([is_numeric_func, bool_to_str_func, length_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytes", "items": [{"__bytes__": []}, {"__bytes__": [126, 93, 123, 181]}, {"__bytes__": [222, 150, 3, 111]}]}"""
        ),
        function_defs=FunctionDefList(
            [bytes_length_func, identity_int_func, sign_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytes", "items": [{"__bytes__": [83, 8, 243, 250]}, {"__bytes__": []}, {"__bytes__": []}]}"""
        ),
        function_defs=FunctionDefList(
            [bytes_is_empty_func, bool_to_float_func, f_abs_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.tuple", "items": [{"__tuple__": [0]}, {"__tuple__": [2]}, {"__tuple__": [-1]}, {"__tuple__": []}, {"__tuple__": [-5, -3]}]}"""
        ),
        function_defs=FunctionDefList(
            [tuple_reverse_func, tuple_to_list_func, list_tail_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.set", "items": [{"__set__": [3, -5]}, {"__set__": [0, -3, 5]}, {"__set__": [0, 2, -3]}]}"""
        ),
        function_defs=FunctionDefList(
            [set_is_empty_func, bool_not_func, bool_not_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.complex", "items": [{"__complex__": [0.0, -4.0]}, {"__complex__": [1.0, -2.0]}, {"__complex__": [-2.0, 0.0]}, {"__complex__": [0.0, 0.0]}, {"__complex__": [3.0, -1.0]}]}"""
        ),
        function_defs=FunctionDefList(
            [complex_imag_func, f_mod1_func, float_to_str_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytearray", "items": [{"__bytearray__": [120]}, {"__bytearray__": [186]}, {"__bytearray__": [114]}, {"__bytearray__": [202, 26]}, {"__bytearray__": []}]}"""
        ),
        function_defs=FunctionDefList(
            [bytearray_length_func, int_to_str_func, contains_space_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.set", "items": [{"__set__": []}, {"__set__": []}, {"__set__": []}]}"""
        ),
        function_defs=FunctionDefList([set_hash_func, square_func, square_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.complex", "items": [{"__complex__": [2.0, 1.0]}, {"__complex__": [3.0, 3.0]}, {"__complex__": [4.0, 3.0]}]}"""
        ),
        function_defs=FunctionDefList(
            [complex_conjugate_func, complex_real_func, f_mod1_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bool", "items": [true, true]}"""
        ),
        function_defs=FunctionDefList(
            [bool_to_float_func, f_sin_func, float_to_str_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytes", "items": [{"__bytes__": [105, 150, 63, 152]}, {"__bytes__": [156]}, {"__bytes__": [174, 225]}, {"__bytes__": [19, 119, 170]}]}"""
        ),
        function_defs=FunctionDefList(
            [bytes_to_hex_func, is_title_func, bool_to_int_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.tuple", "items": [{"__tuple__": [1, 2]}, {"__tuple__": []}, {"__tuple__": [-3, 4, -3]}, {"__tuple__": [-1, -2]}]}"""
        ),
        function_defs=FunctionDefList(
            [tuple_length_func, is_positive_func, bool_to_int_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.range", "items": [{"__range__": [2, 5, 1]}, {"__range__": [1, 6, 1]}, {"__range__": [1, 5, 1]}, {"__range__": [3, 6, 1]}, {"__range__": [1, 1, 1]}]}"""
        ),
        function_defs=FunctionDefList(
            [range_sum_func, sign_func, int_to_float_func, f_abs_sqrt_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.complex", "items": [{"__complex__": [5.0, 4.0]}, {"__complex__": [3.0, -1.0]}, {"__complex__": [2.0, 4.0]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                complex_conjugate_func,
                complex_imag_func,
                f_reciprocal_func,
                f_square_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.str", "items": ["eoi", "7", "0s0M"]}"""
        ),
        function_defs=FunctionDefList(
            [
                endswith_z_func,
                bool_to_int_func,
                is_odd_func,
                bool_identity_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytes", "items": [{"__bytes__": [241, 228, 133, 176]}, {"__bytes__": [200, 68, 30]}, {"__bytes__": []}, {"__bytes__": []}, {"__bytes__": [71, 165, 168]}]}"""
        ),
        function_defs=FunctionDefList(
            [bytes_length_func, double_func, sign_func, sign_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.range", "items": [{"__range__": [0, 5, 1]}, {"__range__": [2, 3, 1]}, {"__range__": [2, 7, 1]}, {"__range__": [0, 0, 1]}, {"__range__": [1, 1, 1]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                range_max_func,
                int_popcount_func,
                double_func,
                int_bit_length_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.range", "items": [{"__range__": [2, 3, 1]}, {"__range__": [1, 4, 1]}, {"__range__": [1, 1, 1]}, {"__range__": [0, 4, 1]}, {"__range__": [0, 0, 1]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                range_length_func,
                abs_func,
                int_bit_length_func,
                int_to_str_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.int", "items": [-1, -5, -9, -6]}"""
        ),
        function_defs=FunctionDefList(
            [abs_func, abs_func, int_clip_0_100_func, abs_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.complex", "items": [{"__complex__": [5.0, 4.0]}, {"__complex__": [2.0, -1.0]}, {"__complex__": [-3.0, 0.0]}, {"__complex__": [3.0, -4.0]}, {"__complex__": [-1.0, 2.0]}]}"""
        ),
        function_defs=FunctionDefList(
            [complex_phase_func, f_ceil_func, abs_func, half_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.complex", "items": [{"__complex__": [0.0, -4.0]}, {"__complex__": [4.0, 5.0]}, {"__complex__": [-1.0, 1.0]}, {"__complex__": [5.0, 0.0]}]}"""
        ),
        function_defs=FunctionDefList(
            [complex_abs_func, f_mod1_func, f_frac_percent_func, half_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bool", "items": [false, false]}"""
        ),
        function_defs=FunctionDefList(
            [
                bool_to_float_func,
                f_log10_func,
                f_is_integer_func,
                bool_not_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytes", "items": [{"__bytes__": [241, 151, 40]}, {"__bytes__": [110, 133, 140, 75]}, {"__bytes__": [26]}, {"__bytes__": [142, 38, 0, 35]}, {"__bytes__": [180, 193, 94, 25]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                bytes_is_ascii_func,
                bool_to_int_func,
                int_to_str_func,
                is_lower_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.float", "items": [-4.249919946825218, 0.42498308211953173, 6.746778794407351]}"""
        ),
        function_defs=FunctionDefList(
            [
                f_round_func,
                is_positive_func,
                bool_identity_func,
                bool_not_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.complex", "items": [{"__complex__": [5.0, 2.0]}, {"__complex__": [-1.0, -4.0]}, {"__complex__": [1.0, -2.0]}, {"__complex__": [2.0, 3.0]}]}"""
        ),
        function_defs=FunctionDefList(
            [complex_imag_func, f_fraction_func, f_sin_func, f_abs_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.set", "items": [{"__set__": [-5]}, {"__set__": [3, 4, -4]}, {"__set__": []}, {"__set__": []}, {"__set__": [3, -3]}]}"""
        ),
        function_defs=FunctionDefList(
            [set_hash_func, half_func, int_to_bool_func, bool_to_float_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.range", "items": [{"__range__": [0, 5, 1]}, {"__range__": [2, 3, 1]}, {"__range__": [0, 3, 1]}, {"__range__": [1, 4, 1]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                range_sum_func,
                int_bit_length_func,
                int_clip_0_100_func,
                mod2_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.str", "items": ["MK", "j", "cvfXJ"]}"""
        ),
        function_defs=FunctionDefList([lower_func, length_func, dec_func, neg_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{}, {}, {}]}"""
        ),
        function_defs=FunctionDefList(
            [
                dict_freeze_func,
                tuple_is_empty_func,
                bool_to_float_func,
                f_abs_sqrt_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.complex", "items": [{"__complex__": [-2.0, -3.0]}, {"__complex__": [4.0, -5.0]}, {"__complex__": [-2.0, -3.0]}]}"""
        ),
        function_defs=FunctionDefList(
            [complex_phase_func, f_abs_sqrt_func, f_abs_func, f_trunc_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.complex", "items": [{"__complex__": [3.0, 5.0]}, {"__complex__": [-2.0, -5.0]}, {"__complex__": [3.0, 5.0]}, {"__complex__": [-1.0, 2.0]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                complex_abs_func,
                f_reciprocal_func,
                f_floor_func,
                int_popcount_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytearray", "items": [{"__bytearray__": [253]}, {"__bytearray__": []}, {"__bytearray__": [141, 67]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                bytearray_reverse_func,
                bytearray_to_bytes_func,
                bytes_length_func,
                mod2_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.list", "items": [[-2, -5, -3], [-4, 3], [], [4, 5]]}"""
        ),
        function_defs=FunctionDefList(
            [
                list_max_func,
                int_bit_length_func,
                int_bit_length_func,
                is_even_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.complex", "items": [{"__complex__": [-3.0, 3.0]}, {"__complex__": [5.0, 3.0]}, {"__complex__": [3.0, -4.0]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                complex_phase_func,
                f_ceil_func,
                int_to_bool_func,
                bool_not_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.range", "items": [{"__range__": [1, 3, 1]}, {"__range__": [2, 3, 1]}, {"__range__": [2, 5, 1]}, {"__range__": [2, 2, 1]}]}"""
        ),
        function_defs=FunctionDefList(
            [range_list_func, list_sorted_func, list_sum_func, sign_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.range", "items": [{"__range__": [3, 8, 1]}, {"__range__": [0, 4, 1]}, {"__range__": [1, 2, 1]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                range_length_func,
                half_func,
                int_to_bool_func,
                bool_identity_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.float", "items": [4.480826610688936, 5.950878662700047, 7.336265612244354, -1.1847260569522042, -4.548195602906393]}"""
        ),
        function_defs=FunctionDefList(
            [f_floor_func, int_popcount_func, is_odd_func, bool_to_int_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.set", "items": [{"__set__": []}, {"__set__": [0, 3]}, {"__set__": [3, -1]}]}"""
        ),
        function_defs=FunctionDefList(
            [set_size_func, is_odd_func, bool_to_int_func, sign_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.set", "items": [{"__set__": [2]}, {"__set__": [-5]}, {"__set__": []}, {"__set__": [1, -4]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                set_hash_func,
                int_to_float_func,
                f_floor_func,
                int_bit_length_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.complex", "items": [{"__complex__": [-1.0, -5.0]}, {"__complex__": [-3.0, 5.0]}, {"__complex__": [-5.0, 2.0]}, {"__complex__": [2.0, 3.0]}, {"__complex__": [-1.0, -2.0]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                complex_abs_func,
                f_frac_percent_func,
                identity_int_func,
                is_positive_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bool", "items": [false, false]}"""
        ),
        function_defs=FunctionDefList(
            [
                bool_identity_func,
                bool_to_float_func,
                f_abs_sqrt_func,
                f_sin_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bool", "items": [true, true]}"""
        ),
        function_defs=FunctionDefList(
            [
                bool_to_float_func,
                f_fraction_func,
                float_to_str_func,
                strip_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.set", "items": [{"__set__": [4]}, {"__set__": [0, 2, 4]}, {"__set__": [3, -2]}]}"""
        ),
        function_defs=FunctionDefList(
            [set_size_func, dec_func, int_to_str_func, is_title_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.list", "items": [[1], [-2, 3, 3], [1, 0]]}"""
        ),
        function_defs=FunctionDefList(
            [
                list_reverse_func,
                list_sum_func,
                int_to_float_func,
                f_frac_percent_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.tuple", "items": [{"__tuple__": [3, 3, 0]}, {"__tuple__": [-4, -1, -4]}, {"__tuple__": [5]}, {"__tuple__": []}, {"__tuple__": [-1]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                tuple_length_func,
                int_bit_length_func,
                int_to_str_func,
                title_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.tuple", "items": [{"__tuple__": [3, 2]}, {"__tuple__": []}, {"__tuple__": [-4, 3, -4]}, {"__tuple__": [1, 1, 3]}, {"__tuple__": [-3, 2, -4]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                tuple_reverse_func,
                tuple_length_func,
                identity_int_func,
                int_bit_length_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.str", "items": ["", "02x", "odwA7f", "nQbUfv", "M9Xp"]}"""
        ),
        function_defs=FunctionDefList(
            [length_func, is_even_func, bool_to_int_func, is_positive_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.str", "items": ["", "IhQe6", "Y", "dMVZ", "JUbg"]}"""
        ),
        function_defs=FunctionDefList(
            [swapcase_func, length_func, neg_func, int_bit_length_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{"a": 5}, {"a": 2, "b": -2}, {"a": 1, "b": 2, "c": 3}, {"a": -4}]}"""
        ),
        function_defs=FunctionDefList(
            [dict_items_func, list_sum_func, sign_func, is_odd_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.int", "items": [9, 2, 8]}"""
        ),
        function_defs=FunctionDefList(
            [
                is_positive_func,
                bool_to_str_func,
                strip_func,
                startswith_a_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.int", "items": [-2, 4, -1, -2]}"""
        ),
        function_defs=FunctionDefList(
            [sign_func, is_negative_func, bool_to_float_func, f_round_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.int", "items": [-8, 2, 5]}"""
        ),
        function_defs=FunctionDefList(
            [neg_func, neg_func, square_func, int_to_float_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.complex", "items": [{"__complex__": [-5.0, 5.0]}, {"__complex__": [-4.0, 4.0]}, {"__complex__": [3.0, 5.0]}, {"__complex__": [-2.0, 0.0]}, {"__complex__": [2.0, 1.0]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                complex_conjugate_func,
                complex_imag_func,
                float_to_str_func,
                first_char_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.list", "items": [[-3, -2], [1, -5, 1], [-3, 2], []]}"""
        ),
        function_defs=FunctionDefList(
            [list_sorted_func, list_max_func, identity_int_func, double_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.complex", "items": [{"__complex__": [5.0, 4.0]}, {"__complex__": [4.0, -5.0]}, {"__complex__": [-1.0, 0.0]}]}"""
        ),
        function_defs=FunctionDefList(
            [complex_phase_func, f_ceil_func, dec_func, half_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.tuple", "items": [{"__tuple__": []}, {"__tuple__": [5, 1]}, {"__tuple__": []}]}"""
        ),
        function_defs=FunctionDefList(
            [
                tuple_to_index_dict_func,
                dict_keyset_func,
                set_size_func,
                int_to_float_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.range", "items": [{"__range__": [2, 5, 1]}, {"__range__": [0, 3, 1]}, {"__range__": [0, 0, 1]}, {"__range__": [3, 4, 1]}, {"__range__": [2, 7, 1]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                range_sum_func,
                int_is_power_of_two_func,
                bool_identity_func,
                bool_not_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.float", "items": [-2.5593508645030782, -2.82076655971196, 2.7457639379664815, -4.507542348759359]}"""
        ),
        function_defs=FunctionDefList(
            [f_fraction_func, f_mod1_func, f_fraction_func, f_abs_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytes", "items": [{"__bytes__": []}, {"__bytes__": []}, {"__bytes__": [109, 30, 158, 239]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                bytes_upper_func,
                bytes_is_ascii_func,
                bool_not_func,
                bool_to_str_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.tuple", "items": [{"__tuple__": [5]}, {"__tuple__": [4, -4]}, {"__tuple__": [4]}, {"__tuple__": [3]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                tuple_reverse_func,
                tuple_to_list_func,
                list_tail_func,
                list_tail_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.int", "items": [-8, 9, 1, 0]}"""
        ),
        function_defs=FunctionDefList(
            [
                int_is_power_of_two_func,
                bool_to_str_func,
                lower_func,
                count_a_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.list", "items": [[-5, 2, 2], [], [-5], [-3]]}"""
        ),
        function_defs=FunctionDefList([list_sum_func, square_func, abs_func, dec_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.float", "items": [1.3976968766130842, 6.893297339784091, 1.4818655943745558]}"""
        ),
        function_defs=FunctionDefList(
            [
                f_trunc_func,
                is_negative_func,
                bool_to_int_func,
                int_to_str_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{"a": 4, "b": -4, "c": 3}, {"a": 0, "b": -3}, {"a": -4, "b": 4, "c": 2}]}"""
        ),
        function_defs=FunctionDefList(
            [dict_keyset_func, set_hash_func, half_func, is_even_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.list", "items": [[3], [5], [4, 3, 2]]}"""
        ),
        function_defs=FunctionDefList(
            [
                list_is_empty_func,
                bool_to_str_func,
                last_char_func,
                contains_space_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bool", "items": [true, false]}"""
        ),
        function_defs=FunctionDefList(
            [bool_not_func, bool_identity_func, bool_to_str_func, upper_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.range", "items": [{"__range__": [1, 3, 1]}, {"__range__": [1, 3, 1]}, {"__range__": [3, 3, 1]}, {"__range__": [2, 7, 1]}, {"__range__": [3, 8, 1]}]}"""
        ),
        function_defs=FunctionDefList(
            [range_list_func, list_max_func, half_func, identity_int_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.complex", "items": [{"__complex__": [-4.0, 1.0]}, {"__complex__": [-2.0, 1.0]}, {"__complex__": [1.0, 4.0]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                complex_abs_func,
                float_to_str_func,
                str_remove_digits_func,
                startswith_a_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.set", "items": [{"__set__": [0]}, {"__set__": [1, 5]}, {"__set__": [-5, -2]}, {"__set__": [-4]}, {"__set__": [-3, -1, -2]}]}"""
        ),
        function_defs=FunctionDefList(
            [set_is_empty_func, bool_to_str_func, str_hash_func, is_odd_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.range", "items": [{"__range__": [2, 7, 1]}, {"__range__": [1, 5, 1]}, {"__range__": [3, 8, 1]}, {"__range__": [3, 4, 1]}, {"__range__": [3, 4, 1]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                range_list_func,
                list_max_func,
                int_is_power_of_two_func,
                bool_to_float_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.range", "items": [{"__range__": [3, 6, 1]}, {"__range__": [2, 5, 1]}, {"__range__": [3, 5, 1]}, {"__range__": [2, 7, 1]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                range_length_func,
                int_popcount_func,
                int_clip_0_100_func,
                identity_int_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{"a": 5, "b": -5, "c": -2}, {"a": 4, "b": -3, "c": 0}, {"a": 5}, {"a": 1, "b": 1}]}"""
        ),
        function_defs=FunctionDefList(
            [
                dict_values_func,
                list_min_func,
                int_to_float_func,
                f_abs_sqrt_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{"a": -5, "b": 3}, {"a": 1, "b": -2, "c": 5}, {"a": 5}]}"""
        ),
        function_defs=FunctionDefList(
            [
                dict_length_func,
                is_negative_func,
                bool_to_float_func,
                f_mod1_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.tuple", "items": [{"__tuple__": [5]}, {"__tuple__": [-1, -4]}, {"__tuple__": [5, -4, 1]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                tuple_to_index_dict_func,
                dict_keys_func,
                list_unique_func,
                list_median_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bool", "items": [true, false]}"""
        ),
        function_defs=FunctionDefList(
            [
                bool_not_func,
                bool_to_str_func,
                is_alpha_func,
                bool_to_int_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.set", "items": [{"__set__": [-4, -1]}, {"__set__": [3, 2, -5]}, {"__set__": [-4, -2]}, {"__set__": []}, {"__set__": []}]}"""
        ),
        function_defs=FunctionDefList([set_size_func, square_func, abs_func, neg_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.set", "items": [{"__set__": [1, 3, -1]}, {"__set__": [-5, 5, -2]}, {"__set__": []}, {"__set__": []}]}"""
        ),
        function_defs=FunctionDefList(
            [set_size_func, sign_func, int_popcount_func, abs_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.set", "items": [{"__set__": []}, {"__set__": [0, 3, -5]}, {"__set__": [2, 4, -3]}, {"__set__": [-5]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                set_to_list_func,
                list_reverse_func,
                list_reverse_func,
                list_sorted_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.int", "items": [-9, -6, 0]}"""
        ),
        function_defs=FunctionDefList(
            [int_to_float_func, f_ceil_func, dec_func, sign_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.int", "items": [0, -9, -3, -7, 7]}"""
        ),
        function_defs=FunctionDefList(
            [
                neg_func,
                identity_int_func,
                int_is_power_of_two_func,
                bool_to_float_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.complex", "items": [{"__complex__": [1.0, 3.0]}, {"__complex__": [-4.0, 5.0]}, {"__complex__": [4.0, 5.0]}, {"__complex__": [1.0, -4.0]}, {"__complex__": [3.0, -5.0]}]}"""
        ),
        function_defs=FunctionDefList(
            [complex_phase_func, f_ceil_func, double_func, identity_int_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.complex", "items": [{"__complex__": [3.0, -1.0]}, {"__complex__": [5.0, 1.0]}, {"__complex__": [3.0, -3.0]}, {"__complex__": [-3.0, -4.0]}, {"__complex__": [0.0, -1.0]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                complex_conjugate_func,
                complex_real_func,
                f_floor_func,
                abs_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytes", "items": [{"__bytes__": []}, {"__bytes__": [127]}, {"__bytes__": [95, 27, 135]}, {"__bytes__": []}, {"__bytes__": [21]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                bytes_to_hex_func,
                is_title_func,
                bool_to_float_func,
                f_floor_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{}, {"a": 1, "b": 1, "c": -5}, {"a": -3}, {"a": 4, "b": -2, "c": 4}, {"a": -1, "b": 3, "c": 3}]}"""
        ),
        function_defs=FunctionDefList(
            [dict_keys_func, list_max_func, double_func, double_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.range", "items": [{"__range__": [0, 1, 1]}, {"__range__": [2, 7, 1]}, {"__range__": [2, 6, 1]}, {"__range__": [3, 8, 1]}, {"__range__": [3, 8, 1]}]}"""
        ),
        function_defs=FunctionDefList(
            [range_sum_func, neg_func, int_to_bool_func, bool_identity_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{}, {"a": 0}, {}]}"""
        ),
        function_defs=FunctionDefList(
            [
                dict_has_duplicate_values_func,
                bool_to_int_func,
                is_odd_func,
                bool_to_float_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.float", "items": [-2.976861726952313, -8.968577314599743, -4.255292431794775, 3.7038504694170467, -6.588558813578594]}"""
        ),
        function_defs=FunctionDefList(
            [f_fraction_func, f_floor_func, is_odd_func, bool_identity_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.int", "items": [10, -6, 4]}"""
        ),
        function_defs=FunctionDefList(
            [sign_func, inc_func, int_to_bool_func, bool_not_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.range", "items": [{"__range__": [1, 1, 1]}, {"__range__": [1, 5, 1]}, {"__range__": [3, 4, 1]}, {"__range__": [2, 7, 1]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                range_length_func,
                int_to_bool_func,
                bool_identity_func,
                bool_not_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{"a": -1}, {"a": 4, "b": 2}, {"a": 4}]}"""
        ),
        function_defs=FunctionDefList(
            [dict_keys_func, list_tail_func, list_median_func, f_floor_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.complex", "items": [{"__complex__": [4.0, 3.0]}, {"__complex__": [4.0, 5.0]}, {"__complex__": [-3.0, 5.0]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                complex_phase_func,
                f_is_integer_func,
                bool_to_int_func,
                inc_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.complex", "items": [{"__complex__": [-3.0, 5.0]}, {"__complex__": [5.0, 1.0]}, {"__complex__": [3.0, 5.0]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                complex_abs_func,
                f_fraction_func,
                f_frac_percent_func,
                dec_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.complex", "items": [{"__complex__": [0.0, 2.0]}, {"__complex__": [1.0, 2.0]}, {"__complex__": [5.0, 2.0]}]}"""
        ),
        function_defs=FunctionDefList(
            [complex_phase_func, f_square_func, f_log10_func, f_round_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytes", "items": [{"__bytes__": []}, {"__bytes__": [92, 19]}, {"__bytes__": [82]}, {"__bytes__": []}]}"""
        ),
        function_defs=FunctionDefList(
            [
                bytes_is_ascii_func,
                bool_not_func,
                bool_not_func,
                bool_not_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.int", "items": [-7, -3, 6, 9, -3]}"""
        ),
        function_defs=FunctionDefList(
            [dec_func, sign_func, is_odd_func, bool_to_float_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.tuple", "items": [{"__tuple__": [0, -4, -3]}, {"__tuple__": [-2, -5, -4]}, {"__tuple__": [0, -4]}, {"__tuple__": [-3, -5, 3]}, {"__tuple__": [-5, -3, 2]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                tuple_to_index_dict_func,
                dict_freeze_func,
                tuple_to_list_func,
                list_length_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.tuple", "items": [{"__tuple__": [-2, 1]}, {"__tuple__": [0, 4, 1]}, {"__tuple__": [2, -5]}, {"__tuple__": [0, -4, -5]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                tuple_length_func,
                int_to_str_func,
                is_alpha_func,
                bool_to_int_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.int", "items": [-10, -3, 5]}"""
        ),
        function_defs=FunctionDefList(
            [is_positive_func, bool_to_int_func, neg_func, int_to_float_func]
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
                bool_to_int_func,
                int_clip_0_100_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.int", "items": [-2, -6, -10, -9, -2]}"""
        ),
        function_defs=FunctionDefList(
            [
                abs_func,
                int_to_float_func,
                f_is_integer_func,
                bool_to_str_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.str", "items": ["", "o", "CqFyW", ""]}"""
        ),
        function_defs=FunctionDefList(
            [lower_func, str_hash_func, int_popcount_func, is_even_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.tuple", "items": [{"__tuple__": []}, {"__tuple__": []}, {"__tuple__": [2, 5, -1]}, {"__tuple__": [0]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                tuple_reverse_func,
                tuple_count_none_func,
                int_clip_0_100_func,
                mod2_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.list", "items": [[5, 3, 5], [-5], [4, 2, -3], [-2, 5], [-3, -2, 0]]}"""
        ),
        function_defs=FunctionDefList(
            [list_max_func, half_func, square_func, double_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.list", "items": [[4, -3, 2], [4], [1, -4]]}"""
        ),
        function_defs=FunctionDefList(
            [list_tail_func, list_min_func, sign_func, is_odd_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytes", "items": [{"__bytes__": [71, 87, 177, 39]}, {"__bytes__": [14]}, {"__bytes__": [108]}, {"__bytes__": [30, 26]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                bytes_is_ascii_func,
                bool_identity_func,
                bool_to_float_func,
                float_to_str_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.float", "items": [-6.537706829163605, 3.3324150748408847, -4.1081481243128675, -8.49630690258185, 9.011081331719492]}"""
        ),
        function_defs=FunctionDefList(
            [f_fraction_func, f_abs_func, f_ceil_func, int_popcount_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytes", "items": [{"__bytes__": [97, 41]}, {"__bytes__": [224, 78]}, {"__bytes__": [83]}, {"__bytes__": [217]}, {"__bytes__": [90, 43, 173]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                bytes_to_hex_func,
                str_reverse_words_func,
                repeat_func,
                is_space_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytearray", "items": [{"__bytearray__": [244]}, {"__bytearray__": [17, 178, 206]}, {"__bytearray__": [114, 144]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                bytearray_reverse_func,
                bytearray_reverse_func,
                bytearray_reverse_func,
                bytearray_reverse_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytes", "items": [{"__bytes__": [38, 82, 148, 158]}, {"__bytes__": [172]}, {"__bytes__": [225, 251, 77]}, {"__bytes__": [3, 191, 95]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                bytes_is_ascii_func,
                bool_identity_func,
                bool_identity_func,
                bool_to_float_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.float", "items": [7.206437457164494, -5.858752039257267, -3.2975995922652235, 4.584330296219566]}"""
        ),
        function_defs=FunctionDefList(
            [
                f_trunc_func,
                int_to_str_func,
                str_is_palindrome_func,
                bool_to_int_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{"a": -1, "b": -4, "c": 4}, {}, {"a": -2, "b": 5, "c": -5}, {"a": -4, "b": -5, "c": 0}]}"""
        ),
        function_defs=FunctionDefList(
            [dict_values_func, list_unique_func, list_max_func, dec_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytearray", "items": [{"__bytearray__": [146, 239]}, {"__bytearray__": [91, 219, 8, 4]}, {"__bytearray__": [191]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                bytearray_length_func,
                int_is_power_of_two_func,
                bool_not_func,
                bool_identity_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.tuple", "items": [{"__tuple__": [2]}, {"__tuple__": []}, {"__tuple__": [5, -1, -3]}, {"__tuple__": [2]}, {"__tuple__": []}]}"""
        ),
        function_defs=FunctionDefList(
            [
                tuple_to_list_func,
                list_length_func,
                is_negative_func,
                bool_identity_func,
                bool_to_int_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.float", "items": [3.550417670201277, -9.918545851648934, 3.013287002551193, 5.479863874065334]}"""
        ),
        function_defs=FunctionDefList(
            [
                f_floor_func,
                square_func,
                half_func,
                int_clip_0_100_func,
                mod2_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytes", "items": [{"__bytes__": []}, {"__bytes__": [195, 76, 103, 243]}, {"__bytes__": [4]}, {"__bytes__": [24, 84, 101]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                bytes_is_empty_func,
                bool_not_func,
                bool_to_int_func,
                is_odd_func,
                bool_to_float_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{"a": -5}, {}, {}, {"a": 5, "b": 1, "c": 4}, {}]}"""
        ),
        function_defs=FunctionDefList(
            [
                dict_freeze_func,
                tuple_length_func,
                int_clip_0_100_func,
                inc_func,
                is_even_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.float", "items": [5.040887842363107, -0.2458681421145652, -7.372807278207871]}"""
        ),
        function_defs=FunctionDefList(
            [f_trunc_func, neg_func, int_popcount_func, dec_func, neg_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.int", "items": [7, -1, 8]}"""
        ),
        function_defs=FunctionDefList(
            [half_func, neg_func, half_func, int_to_bool_func, bool_not_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytes", "items": [{"__bytes__": [235, 89, 47]}, {"__bytes__": [79, 252]}, {"__bytes__": [67, 185, 146]}, {"__bytes__": [228, 163]}, {"__bytes__": []}]}"""
        ),
        function_defs=FunctionDefList(
            [
                bytes_is_empty_func,
                bool_not_func,
                bool_identity_func,
                bool_to_int_func,
                int_is_power_of_two_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytearray", "items": [{"__bytearray__": [169, 56, 169, 146]}, {"__bytearray__": [26, 63]}, {"__bytearray__": [97, 75]}, {"__bytearray__": [97, 2]}, {"__bytearray__": [197, 18]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                bytearray_length_func,
                square_func,
                dec_func,
                int_to_bool_func,
                bool_to_float_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.list", "items": [[-4], [-2, 0], [-2, 3], []]}"""
        ),
        function_defs=FunctionDefList(
            [
                list_sum_func,
                inc_func,
                dec_func,
                is_even_func,
                bool_to_int_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytes", "items": [{"__bytes__": [135, 164, 208]}, {"__bytes__": [163, 191]}, {"__bytes__": []}]}"""
        ),
        function_defs=FunctionDefList(
            [
                bytes_length_func,
                is_negative_func,
                bool_to_int_func,
                abs_func,
                is_positive_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytearray", "items": [{"__bytearray__": [243, 6]}, {"__bytearray__": [116, 110]}, {"__bytearray__": []}, {"__bytearray__": []}, {"__bytearray__": [249, 89, 96]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                bytearray_reverse_func,
                bytearray_length_func,
                mod2_func,
                half_func,
                abs_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{}, {"a": -5, "b": 0}, {}, {"a": 3, "b": 1, "c": -4}]}"""
        ),
        function_defs=FunctionDefList(
            [
                dict_values_func,
                list_is_empty_func,
                bool_to_int_func,
                dec_func,
                int_is_power_of_two_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.complex", "items": [{"__complex__": [-5.0, 5.0]}, {"__complex__": [3.0, 2.0]}, {"__complex__": [4.0, -4.0]}, {"__complex__": [-5.0, -3.0]}, {"__complex__": [5.0, 4.0]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                complex_imag_func,
                f_sin_func,
                f_square_func,
                f_round_func,
                int_bit_length_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.list", "items": [[0, -1], [-5, 4], [-4, 3, 4]]}"""
        ),
        function_defs=FunctionDefList(
            [
                list_tail_func,
                list_reverse_func,
                list_is_empty_func,
                bool_identity_func,
                bool_to_float_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytearray", "items": [{"__bytearray__": [52]}, {"__bytearray__": [169, 48]}, {"__bytearray__": [42]}, {"__bytearray__": [248, 91, 173]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                bytearray_to_bytes_func,
                bytes_length_func,
                inc_func,
                square_func,
                mod2_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.complex", "items": [{"__complex__": [2.0, 1.0]}, {"__complex__": [-4.0, 2.0]}, {"__complex__": [0.0, 0.0]}, {"__complex__": [-3.0, -5.0]}, {"__complex__": [3.0, 4.0]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                complex_abs_func,
                f_trunc_func,
                int_to_float_func,
                f_round_func,
                is_positive_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.list", "items": [[], [-1], [0, 2, 5], [-3, -3, 5]]}"""
        ),
        function_defs=FunctionDefList(
            [
                list_unique_func,
                list_reverse_func,
                list_reverse_func,
                list_length_func,
                sign_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.int", "items": [-9, -2, 2, 3, -5]}"""
        ),
        function_defs=FunctionDefList(
            [
                is_positive_func,
                bool_to_str_func,
                is_space_func,
                bool_to_float_func,
                f_mod1_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.tuple", "items": [{"__tuple__": [-5]}, {"__tuple__": []}, {"__tuple__": [0, -5, 1]}, {"__tuple__": [4, -3, 0]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                tuple_is_empty_func,
                bool_to_str_func,
                strip_func,
                lower_func,
                str_count_vowels_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.float", "items": [5.022216550005812, -1.3586514470074658, 9.289220460608828]}"""
        ),
        function_defs=FunctionDefList(
            [
                f_frac_percent_func,
                is_negative_func,
                bool_to_str_func,
                str_remove_digits_func,
                strip_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.tuple", "items": [{"__tuple__": [-3]}, {"__tuple__": [-2, -2]}, {"__tuple__": [4, 0, 0]}, {"__tuple__": []}, {"__tuple__": []}]}"""
        ),
        function_defs=FunctionDefList(
            [tuple_length_func, neg_func, mod2_func, abs_func, double_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.list", "items": [[], [-3, 0], [4, 0], [-3, 2, -2], []]}"""
        ),
        function_defs=FunctionDefList(
            [
                list_is_empty_func,
                bool_identity_func,
                bool_to_int_func,
                int_is_power_of_two_func,
                bool_to_int_func,
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
                bool_identity_func,
                bool_to_int_func,
                inc_func,
                double_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.str", "items": ["p", "", "C3", "mu"]}"""
        ),
        function_defs=FunctionDefList(
            [
                capitalize_func,
                contains_space_func,
                bool_to_str_func,
                is_numeric_func,
                bool_to_str_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.complex", "items": [{"__complex__": [3.0, 1.0]}, {"__complex__": [-1.0, -5.0]}, {"__complex__": [-3.0, -1.0]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                complex_abs_func,
                f_mod1_func,
                f_square_func,
                f_exp_func,
                f_frac_percent_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.tuple", "items": [{"__tuple__": [-5]}, {"__tuple__": [1, 0, 5]}, {"__tuple__": []}, {"__tuple__": [-3, 1]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                tuple_count_none_func,
                int_bit_length_func,
                double_func,
                abs_func,
                identity_int_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.range", "items": [{"__range__": [2, 6, 1]}, {"__range__": [1, 4, 1]}, {"__range__": [0, 5, 1]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                range_max_func,
                is_positive_func,
                bool_identity_func,
                bool_to_int_func,
                int_to_bool_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.range", "items": [{"__range__": [1, 4, 1]}, {"__range__": [3, 7, 1]}, {"__range__": [2, 5, 1]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                range_list_func,
                list_is_empty_func,
                bool_to_float_func,
                f_log10_func,
                f_floor_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{"a": 0}, {"a": -4}, {}, {"a": -4}, {}]}"""
        ),
        function_defs=FunctionDefList(
            [
                dict_freeze_func,
                tuple_reverse_func,
                tuple_reverse_func,
                tuple_to_index_dict_func,
                dict_values_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.tuple", "items": [{"__tuple__": [4, -4, 1]}, {"__tuple__": [-1, 1, -3]}, {"__tuple__": [2, 1]}, {"__tuple__": [-2, -2]}, {"__tuple__": [-3, 3, 1]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                tuple_to_list_func,
                list_max_func,
                double_func,
                double_func,
                half_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.int", "items": [-6, -10, 10, 2, -5]}"""
        ),
        function_defs=FunctionDefList(
            [
                half_func,
                identity_int_func,
                inc_func,
                int_is_power_of_two_func,
                bool_to_int_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.str", "items": ["J", "h", "pKIL", "XCM"]}"""
        ),
        function_defs=FunctionDefList(
            [
                is_digit_func,
                bool_to_str_func,
                is_space_func,
                bool_not_func,
                bool_identity_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.float", "items": [-0.40863418702284093, -4.409141582508893, 3.3851663561035537, 7.086319516603098, -3.9723225418723658]}"""
        ),
        function_defs=FunctionDefList(
            [
                f_sin_func,
                f_exp_func,
                f_is_integer_func,
                bool_to_float_func,
                f_sin_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.str", "items": ["N", "HP", "NNNozP"]}"""
        ),
        function_defs=FunctionDefList(
            [
                capitalize_func,
                last_char_func,
                title_func,
                str_remove_digits_func,
                first_char_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.float", "items": [5.592659970884586, 2.600434256416488, 5.396081025373615]}"""
        ),
        function_defs=FunctionDefList(
            [
                f_trunc_func,
                mod2_func,
                half_func,
                int_popcount_func,
                is_even_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.float", "items": [7.8376534268857725, 9.428918412390573, -1.0718203021128154]}"""
        ),
        function_defs=FunctionDefList(
            [
                float_to_str_func,
                is_alpha_func,
                bool_to_float_func,
                f_trunc_func,
                is_positive_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytes", "items": [{"__bytes__": [2, 12]}, {"__bytes__": [113, 226, 18]}, {"__bytes__": [82]}, {"__bytes__": [137, 90, 160, 116]}, {"__bytes__": [119, 204, 253]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                bytes_reverse_func,
                bytes_to_hex_func,
                duplicate_func,
                str_hash_func,
                int_clip_0_100_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.float", "items": [-6.8435772848455585, 7.4586694134557945, 0.15804446834493824]}"""
        ),
        function_defs=FunctionDefList(
            [
                f_abs_sqrt_func,
                float_to_str_func,
                is_digit_func,
                bool_to_float_func,
                f_is_integer_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.int", "items": [3, -2, -5, 1, 2]}"""
        ),
        function_defs=FunctionDefList(
            [
                mod2_func,
                is_odd_func,
                bool_to_int_func,
                int_to_str_func,
                title_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytearray", "items": [{"__bytearray__": [57, 122, 192]}, {"__bytearray__": [138, 174, 194, 88]}, {"__bytearray__": [199, 38, 200, 9]}, {"__bytearray__": [246]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                bytearray_length_func,
                neg_func,
                int_to_bool_func,
                bool_identity_func,
                bool_to_float_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{}, {"a": -3, "b": 3, "c": -4}, {}]}"""
        ),
        function_defs=FunctionDefList(
            [
                dict_freeze_func,
                tuple_length_func,
                identity_int_func,
                int_to_str_func,
                capitalize_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytearray", "items": [{"__bytearray__": [223, 239]}, {"__bytearray__": [218, 210, 245, 59]}, {"__bytearray__": [199, 211, 32]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                bytearray_reverse_func,
                bytearray_to_bytes_func,
                bytes_reverse_func,
                bytes_length_func,
                square_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.tuple", "items": [{"__tuple__": [1]}, {"__tuple__": [4, -5, 4]}, {"__tuple__": []}]}"""
        ),
        function_defs=FunctionDefList(
            [
                tuple_to_list_func,
                list_is_empty_func,
                bool_to_float_func,
                f_fraction_func,
                f_log10_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.list", "items": [[-2, 4, 3], [], [-5], [4, 3, 3], []]}"""
        ),
        function_defs=FunctionDefList(
            [
                list_sorted_func,
                list_max_func,
                mod2_func,
                sign_func,
                int_bit_length_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.range", "items": [{"__range__": [3, 7, 1]}, {"__range__": [1, 1, 1]}, {"__range__": [1, 4, 1]}, {"__range__": [2, 5, 1]}, {"__range__": [1, 3, 1]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                range_list_func,
                list_length_func,
                identity_int_func,
                mod2_func,
                square_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.int", "items": [-6, 5, -2, 1, 1]}"""
        ),
        function_defs=FunctionDefList(
            [
                is_negative_func,
                bool_to_str_func,
                count_a_func,
                int_to_bool_func,
                bool_to_float_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytearray", "items": [{"__bytearray__": [176, 168, 7]}, {"__bytearray__": []}, {"__bytearray__": [207, 179, 254, 220]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                bytearray_to_bytes_func,
                bytes_is_empty_func,
                bool_to_int_func,
                int_is_power_of_two_func,
                bool_to_float_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{"a": -1, "b": 3, "c": -5}, {"a": 4}, {}, {"a": -3, "b": -3, "c": 1}]}"""
        ),
        function_defs=FunctionDefList(
            [
                dict_freeze_func,
                tuple_reverse_func,
                tuple_length_func,
                double_func,
                inc_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.set", "items": [{"__set__": []}, {"__set__": [-4, 4]}, {"__set__": [-5, -4, 5]}]}"""
        ),
        function_defs=FunctionDefList(
            [set_size_func, mod2_func, inc_func, neg_func, square_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bool", "items": [false, false]}"""
        ),
        function_defs=FunctionDefList(
            [
                bool_to_float_func,
                f_square_func,
                f_floor_func,
                is_negative_func,
                bool_identity_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.complex", "items": [{"__complex__": [5.0, 1.0]}, {"__complex__": [-4.0, 2.0]}, {"__complex__": [4.0, -3.0]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                complex_abs_func,
                f_frac_percent_func,
                int_popcount_func,
                is_positive_func,
                bool_to_float_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.float", "items": [-0.9804108714057023, -5.887069627070927, 7.571242140356457, 5.3788343383076445]}"""
        ),
        function_defs=FunctionDefList(
            [
                f_exp_func,
                f_exp_func,
                f_square_func,
                f_fraction_func,
                f_abs_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytearray", "items": [{"__bytearray__": [62]}, {"__bytearray__": [169]}, {"__bytearray__": [46, 101]}, {"__bytearray__": [4, 1, 0]}, {"__bytearray__": [123, 186, 234, 54]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                bytearray_length_func,
                is_odd_func,
                bool_not_func,
                bool_not_func,
                bool_identity_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bool", "items": [true, true]}"""
        ),
        function_defs=FunctionDefList(
            [
                bool_identity_func,
                bool_identity_func,
                bool_identity_func,
                bool_not_func,
                bool_to_int_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.int", "items": [-7, -9, 7]}"""
        ),
        function_defs=FunctionDefList(
            [
                double_func,
                dec_func,
                abs_func,
                int_to_bool_func,
                bool_not_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.str", "items": ["", "", "vYI9Q", "b4"]}"""
        ),
        function_defs=FunctionDefList(
            [
                length_func,
                square_func,
                is_even_func,
                bool_not_func,
                bool_to_float_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytes", "items": [{"__bytes__": [113]}, {"__bytes__": [247]}, {"__bytes__": [57]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                bytes_length_func,
                half_func,
                int_popcount_func,
                int_clip_0_100_func,
                is_odd_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.str", "items": ["mRJjE", "", "Q90V", "YvhtiN", "vn"]}"""
        ),
        function_defs=FunctionDefList(
            [
                startswith_a_func,
                bool_to_int_func,
                dec_func,
                abs_func,
                square_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.tuple", "items": [{"__tuple__": [-3, 4, -3]}, {"__tuple__": []}, {"__tuple__": []}, {"__tuple__": []}, {"__tuple__": []}]}"""
        ),
        function_defs=FunctionDefList(
            [
                tuple_is_empty_func,
                bool_to_str_func,
                is_lower_func,
                bool_identity_func,
                bool_identity_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.range", "items": [{"__range__": [0, 2, 1]}, {"__range__": [1, 4, 1]}, {"__range__": [0, 3, 1]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                range_length_func,
                int_clip_0_100_func,
                half_func,
                abs_func,
                mod2_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.int", "items": [-3, 7, -2]}"""
        ),
        function_defs=FunctionDefList(
            [
                int_to_str_func,
                is_lower_func,
                bool_identity_func,
                bool_not_func,
                bool_not_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bool", "items": [false, true]}"""
        ),
        function_defs=FunctionDefList(
            [
                bool_identity_func,
                bool_to_float_func,
                f_sin_func,
                f_reciprocal_func,
                f_abs_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.set", "items": [{"__set__": []}, {"__set__": [3, -5, -1]}, {"__set__": [0, -4]}, {"__set__": [-5, 5, -1]}, {"__set__": [5]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                set_size_func,
                mod2_func,
                is_even_func,
                bool_to_int_func,
                int_to_bool_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytes", "items": [{"__bytes__": [187]}, {"__bytes__": [142, 89]}, {"__bytes__": [163, 51]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                bytes_is_empty_func,
                bool_to_float_func,
                f_trunc_func,
                abs_func,
                int_to_float_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytes", "items": [{"__bytes__": [53, 88, 44, 90]}, {"__bytes__": []}, {"__bytes__": []}, {"__bytes__": [14, 106, 234, 108]}, {"__bytes__": [160, 26, 81, 65]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                bytes_is_empty_func,
                bool_not_func,
                bool_to_str_func,
                count_a_func,
                is_negative_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.complex", "items": [{"__complex__": [-2.0, -4.0]}, {"__complex__": [4.0, -3.0]}, {"__complex__": [0.0, 2.0]}, {"__complex__": [0.0, 4.0]}]}"""
        ),
        function_defs=FunctionDefList(
            [complex_real_func, f_exp_func, f_ceil_func, mod2_func, dec_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytes", "items": [{"__bytes__": [164]}, {"__bytes__": [164]}, {"__bytes__": [148, 233, 140]}, {"__bytes__": []}, {"__bytes__": [203]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                bytes_upper_func,
                bytes_to_hex_func,
                lower_func,
                count_a_func,
                dec_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.int", "items": [-8, -5, -3]}"""
        ),
        function_defs=FunctionDefList(
            [mod2_func, half_func, half_func, int_to_float_func, f_exp_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytearray", "items": [{"__bytearray__": []}, {"__bytearray__": []}, {"__bytearray__": [253, 92, 242, 29]}, {"__bytearray__": [10, 138, 135]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                bytearray_reverse_func,
                bytearray_reverse_func,
                bytearray_to_bytes_func,
                bytes_is_ascii_func,
                bool_to_int_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytearray", "items": [{"__bytearray__": [85, 43, 20, 150]}, {"__bytearray__": [208, 165, 171, 238]}, {"__bytearray__": []}, {"__bytearray__": [255, 193, 133]}, {"__bytearray__": [120]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                bytearray_reverse_func,
                bytearray_to_bytes_func,
                bytes_upper_func,
                bytes_is_ascii_func,
                bool_to_str_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.int", "items": [5, 7, 8, 4]}"""
        ),
        function_defs=FunctionDefList(
            [
                int_is_power_of_two_func,
                bool_to_float_func,
                f_square_func,
                f_is_integer_func,
                bool_to_int_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.list", "items": [[-4], [-1, -5], [3, 4, -3], [-5, 3], [2, -4]]}"""
        ),
        function_defs=FunctionDefList(
            [
                list_tail_func,
                list_median_func,
                f_trunc_func,
                int_clip_0_100_func,
                int_bit_length_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.str", "items": ["hJq", "Fo4Dg", "rgH"]}"""
        ),
        function_defs=FunctionDefList(
            [
                is_title_func,
                bool_to_str_func,
                is_title_func,
                bool_to_float_func,
                f_is_integer_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bool", "items": [false, true]}"""
        ),
        function_defs=FunctionDefList(
            [
                bool_to_int_func,
                int_is_power_of_two_func,
                bool_to_float_func,
                f_fraction_func,
                f_ceil_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.float", "items": [7.69204149060953, -7.855411594558312, 9.123151048264269]}"""
        ),
        function_defs=FunctionDefList(
            [
                float_to_str_func,
                startswith_a_func,
                bool_to_int_func,
                int_popcount_func,
                double_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.list", "items": [[], [-3, 5, 1], [], [-4]]}"""
        ),
        function_defs=FunctionDefList(
            [
                list_reverse_func,
                list_length_func,
                int_is_power_of_two_func,
                bool_not_func,
                bool_identity_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.list", "items": [[2, -4], [3, 0], [-4]]}"""
        ),
        function_defs=FunctionDefList(
            [
                list_sorted_func,
                list_unique_func,
                list_min_func,
                square_func,
                dec_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.str", "items": ["fi14d", "5", "KQ"]}"""
        ),
        function_defs=FunctionDefList(
            [
                is_title_func,
                bool_identity_func,
                bool_to_int_func,
                is_positive_func,
                bool_not_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.str", "items": ["3sP", "qNVF", "zN"]}"""
        ),
        function_defs=FunctionDefList(
            [
                capitalize_func,
                startswith_a_func,
                bool_to_float_func,
                f_trunc_func,
                int_to_str_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.set", "items": [{"__set__": []}, {"__set__": [-4, -3]}, {"__set__": []}, {"__set__": [-1]}, {"__set__": [2, -3, -1]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                set_size_func,
                half_func,
                int_popcount_func,
                neg_func,
                int_is_power_of_two_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.float", "items": [-4.276889091169787, -5.386837141530512, 0.9704290694686151]}"""
        ),
        function_defs=FunctionDefList(
            [
                f_log10_func,
                f_fraction_func,
                f_reciprocal_func,
                f_fraction_func,
                f_square_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.str", "items": ["F", "qu", "g"]}"""
        ),
        function_defs=FunctionDefList(
            [
                swapcase_func,
                str_count_vowels_func,
                sign_func,
                sign_func,
                int_is_power_of_two_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.list", "items": [[-2, 1], [], [-5, 2, 4]]}"""
        ),
        function_defs=FunctionDefList(
            [
                list_median_func,
                f_mod1_func,
                f_trunc_func,
                half_func,
                half_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.range", "items": [{"__range__": [0, 5, 1]}, {"__range__": [0, 3, 1]}, {"__range__": [2, 7, 1]}, {"__range__": [3, 5, 1]}, {"__range__": [2, 7, 1]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                range_max_func,
                sign_func,
                is_even_func,
                bool_to_float_func,
                f_floor_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.set", "items": [{"__set__": [0, -4]}, {"__set__": [-3, -2, -1]}, {"__set__": [3]}, {"__set__": [-5]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                set_size_func,
                neg_func,
                is_positive_func,
                bool_to_str_func,
                str_to_list_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytes", "items": [{"__bytes__": []}, {"__bytes__": [252, 62, 240, 11]}, {"__bytes__": []}, {"__bytes__": [180]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                bytes_is_ascii_func,
                bool_identity_func,
                bool_identity_func,
                bool_not_func,
                bool_to_str_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{"a": 1, "b": 3}, {}, {"a": -1}, {}, {"a": -5, "b": -3}]}"""
        ),
        function_defs=FunctionDefList(
            [
                dict_values_func,
                list_sorted_func,
                list_sorted_func,
                list_reverse_func,
                list_median_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.list", "items": [[], [], [], [2, -1, -4], [-5, 5, -3]]}"""
        ),
        function_defs=FunctionDefList(
            [
                list_sorted_func,
                list_unique_func,
                list_median_func,
                f_abs_func,
                f_square_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.str", "items": ["tgRF2", "ck", "rXr", "tBK3", ""]}"""
        ),
        function_defs=FunctionDefList(
            [
                is_digit_func,
                bool_to_int_func,
                int_bit_length_func,
                is_odd_func,
                bool_to_int_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.range", "items": [{"__range__": [2, 5, 1]}, {"__range__": [2, 4, 1]}, {"__range__": [3, 6, 1]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                range_length_func,
                int_to_str_func,
                title_func,
                endswith_z_func,
                bool_to_float_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bool", "items": [true, false]}"""
        ),
        function_defs=FunctionDefList(
            [bool_to_int_func, half_func, inc_func, neg_func, abs_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.float", "items": [-7.581225825036877, -2.027245760647089, -0.2519992657308219, -0.5523130908342804, 7.097419502755709]}"""
        ),
        function_defs=FunctionDefList(
            [
                f_square_func,
                f_square_func,
                f_square_func,
                f_square_func,
                f_round_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytes", "items": [{"__bytes__": [85, 200, 117]}, {"__bytes__": [162, 230, 193, 133]}, {"__bytes__": [52]}, {"__bytes__": [150, 167]}, {"__bytes__": []}]}"""
        ),
        function_defs=FunctionDefList(
            [
                bytes_is_ascii_func,
                bool_to_str_func,
                str_reverse_words_func,
                upper_func,
                str_is_palindrome_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.float", "items": [-9.10757589785269, 0.5028593337911502, -6.812296325150102, 9.933373148801877, -1.3045211864144495]}"""
        ),
        function_defs=FunctionDefList(
            [
                f_round_func,
                square_func,
                int_bit_length_func,
                abs_func,
                int_is_power_of_two_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.str", "items": ["SnKSv", "O299WR", "o", "SB8E", "EuNY"]}"""
        ),
        function_defs=FunctionDefList(
            [
                strip_func,
                is_alpha_func,
                bool_identity_func,
                bool_to_int_func,
                is_positive_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.int", "items": [-5, 8, -6, -9]}"""
        ),
        function_defs=FunctionDefList(
            [
                mod2_func,
                abs_func,
                double_func,
                is_even_func,
                bool_to_int_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.list", "items": [[-3, 3], [2, -2], [], [0, -1, -5], []]}"""
        ),
        function_defs=FunctionDefList(
            [
                list_length_func,
                is_odd_func,
                bool_to_str_func,
                str_is_palindrome_func,
                bool_identity_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.list", "items": [[-5, 2], [], [3, 0], [2, 1, -1], [0]]}"""
        ),
        function_defs=FunctionDefList(
            [list_sum_func, square_func, inc_func, square_func, is_odd_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.list", "items": [[-5, 2], [], [3, -2, 3], [0], []]}"""
        ),
        function_defs=FunctionDefList(
            [
                list_tail_func,
                list_max_func,
                inc_func,
                int_to_float_func,
                f_floor_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.list", "items": [[1, -1, -3], [], []]}"""
        ),
        function_defs=FunctionDefList(
            [
                list_max_func,
                inc_func,
                neg_func,
                int_is_power_of_two_func,
                bool_to_int_func,
            ]
        ),
    ),
]

eval_trajectory_specs = TrajectorySpecList(_specs)
