"""Reproducible generation and evaluation for basis-library experiments.

The corpus is intentionally independent from solver output: generation executes a
random-walk witness once, freezes the resulting input/output pair in a deterministic
gzip JSONL file, and evaluation never re-executes that witness.  This makes paired
comparisons between immutable basis snapshots meaningful.

Typical use::

    python -m experiments.basis_library_pilot \
      --mode run --basis-set-id default \
      --discovery-size 250000 --validation-size 25000 --test-size 25000 \
      --output-dir artifacts/basis-library-pilot

Set ``PYTHONHASHSEED`` before launch when explicitly generating with the
checkpoint-era ``checkpoint-rl-6k-with-lp`` basis.

W&B logging is opt-in and hard-limited to the user's personal entity.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
import os
import platform
import random
import re
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import TYPE_CHECKING, Any

from wandering_light.basis_dataset import (
    BasisTaskRecord,
    write_basis_task_records,
)
from wandering_light.basis_set import (
    BasisSet,
    load_basis_set,
    require_reproducible_basis_runtime,
)
from wandering_light.executor import Executor
from wandering_light.function_usage import FunctionUsageTracker
from wandering_light.trajectory import TrajectorySpec
from wandering_light.typed_list import TypedList

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from wandering_light.function_def import FunctionDef, FunctionDefSet


PIPELINE_SCHEMA_VERSION = 1
EXECUTION_ENVIRONMENT_SCHEMA_VERSION = 1
GENERATOR_NAME = "balanced-stratified-random-walk-v2"
VALUE_GENERATOR_NAME = "builtin-behavior-strata-v2"
PERSONAL_WANDB_ENTITY = "abhishekraok-na"
DEFAULT_WANDB_PROJECT = "wandering-light-basis"
DEFAULT_BASIS_SET = "default"
DEFAULT_LOCAL_CHECKPOINT = Path(
    "checkpoints/saved/rl/induction_opt_125m_sft_434k_rl_6k_with_lp"
)
DEFAULT_HF_CHECKPOINT = "abhishekraok/induction-basicfns-opt125m-sft434k-rl-6k-with-lp"
DEFAULT_HF_REVISION = "0c4ea07bfa618321b8dc5ce956ce5a86560d99a7"
DEFAULT_CHECKPOINT_MODEL_BYTES = 500_979_600
DEFAULT_CHECKPOINT_GLOBAL_STEP = 5_760
DEFAULT_CHECKPOINT_WANDB_URL = (
    "https://wandb.ai/abhishekraok-na/wandering-light-rl_induction/runs/dp8ylg8y"
)
DEFAULT_SPLIT_SIZES = {
    "discovery": 250_000,
    "validation": 25_000,
    "test": 25_000,
}
DEFAULT_SPLIT_SEEDS = {
    "discovery": 1_729,
    "validation": 2_718,
    "test": 3_141,
}
NONDETERMINISTIC_PYHASH_NAMES = frozenset({"str_hash", "set_hash"})
_SPLIT_ORDER = ("discovery", "validation", "test")
_COMMIT_SHA_RE = re.compile(r"^[0-9a-fA-F]{40}$")
_LOCAL_TREE_EXCLUDED_PARTS = frozenset({".cache", "__pycache__", "wandb", "runs"})
_LOCAL_TREE_EXCLUDED_FILES = frozenset(
    {
        ".DS_Store",
        "optimizer.pt",
        "scheduler.pt",
        "scaler.pt",
        "rng_state.pth",
    }
)
_LOCAL_TREE_EXCLUDED_SUFFIXES = (".lock", ".partial", ".tmp")
VALUE_STRATA_DESCRIPTION = {
    "builtins.int": "negative, zero, parity, powers of two, clip boundaries, wide random",
    "builtins.float": "negative, zero, fractional, integer, log/exp boundaries, random",
    "builtins.str": (
        "empty, whitespace, case, title, numeric, alphanumeric, palindrome, "
        "a-prefix, z-suffix, multiword, punctuation, unicode, randomized templates"
    ),
    "builtins.bool": "both truth values",
    "builtins.list": "empty, singleton, duplicates, signed and fractional numeric values",
    "builtins.tuple": "empty, singleton, repeated and multiple None values",
    "builtins.set": "empty, singleton, signed and varied-size integer sets",
    "builtins.dict": "empty, unique values, duplicate values, varied key/value counts",
    "builtins.bytes": "empty, ASCII case/space/digits, non-ASCII and random binary",
    "builtins.bytearray": "empty, ASCII, non-ASCII and random binary",
    "builtins.complex": "zero, axes, four quadrants and random components",
    "builtins.range": "empty, singleton, negative endpoints, positive/negative steps",
}
SUPPORTED_RANDOM_TYPES = [
    int,
    float,
    str,
    bool,
    list,
    tuple,
    set,
    dict,
    bytes,
    bytearray,
    complex,
    range,
]


@dataclass(frozen=True)
class ModelSpec:
    """Exact checkpoint identity used by an evaluation."""

    requested: str
    resolved: str
    revision: str | None
    canonical_hf_repo: str | None
    canonical_hf_revision: str | None
    local_files: Mapping[str, Mapping[str, Any]]
    local_tree_digest: str | None
    wandb_run_url: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested": self.requested,
            "resolved": self.resolved,
            "revision": self.revision,
            "canonical_hf_repo": self.canonical_hf_repo,
            "canonical_hf_revision": self.canonical_hf_revision,
            "local_files": dict(self.local_files),
            "local_tree_digest": self.local_tree_digest,
            "wandb_run_url": self.wandb_run_url,
        }


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256_bytes(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        while chunk := input_file.read(1024 * 1024):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _optional_cuda_driver_version(torch: Any) -> str | None:
    """Read the CUDA driver version when the installed torch build exposes it."""
    getters = [
        getattr(torch.cuda, "driver_version", None),
        getattr(getattr(torch, "_C", None), "_cuda_getDriverVersion", None),
    ]
    for getter in getters:
        if not callable(getter):
            continue
        try:
            value = getter()
        except Exception:
            continue
        if value is not None:
            return str(value)
    return None


def _cpu_model_name() -> str:
    name = platform.processor().strip()
    if name:
        return name
    cpuinfo = Path("/proc/cpuinfo")
    try:
        for line in cpuinfo.read_text(encoding="utf-8").splitlines():
            key, separator, value = line.partition(":")
            if (
                separator
                and key.strip().lower() in {"model name", "hardware"}
                and value.strip()
            ):
                return value.strip()
    except OSError:
        pass
    return platform.machine()


def _physical_memory_bytes() -> int | None:
    try:
        return int(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, TypeError, ValueError):
        return None


def capture_execution_environment(
    *,
    requested_device: str,
    requested_batch_size: int,
    budget: int,
    observed_batch_sizes: Sequence[int],
) -> dict[str, Any]:
    """Return content-addressed hardware and effective batching evidence.

    This is captured by the evaluation process itself.  It lets downstream
    analysis distinguish an identical requested ``device=auto`` flag from two
    runs that actually landed on different hardware.
    """
    import torch

    selected_device = requested_device
    if selected_device == "auto":
        selected_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(selected_device)
    accelerator: dict[str, Any]
    if device.type == "cuda":
        index = device.index
        if index is None:
            index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        uuid = getattr(properties, "uuid", None)
        accelerator = {
            "kind": "cuda",
            "index": index,
            "name": str(properties.name),
            "uuid": str(uuid) if uuid is not None else None,
            "total_memory_bytes": int(properties.total_memory),
            "compute_capability": [int(properties.major), int(properties.minor)],
            "driver_version": _optional_cuda_driver_version(torch),
            "cuda_runtime_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
        }
        resolved_device = f"cuda:{index}"
    else:
        accelerator = {
            "kind": "cpu",
            "index": None,
            "name": _cpu_model_name(),
            "uuid": None,
            "total_memory_bytes": _physical_memory_bytes(),
            "compute_capability": None,
            "driver_version": None,
            "cuda_runtime_version": None,
            "cudnn_version": None,
        }
        resolved_device = str(device)

    hardware = {
        "host": {
            "node": platform.node(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_model": _cpu_model_name(),
            "logical_cpu_count": os.cpu_count(),
            "platform": platform.platform(),
        },
        "accelerator": accelerator,
        "software": {
            "python_version": platform.python_version(),
            "torch_version": str(torch.__version__),
        },
    }
    batch_histogram = Counter(observed_batch_sizes)
    inference_batch_histogram: Counter[int] = Counter()
    for task_batch_size in observed_batch_sizes:
        remaining = task_batch_size * budget
        while remaining:
            inference_batch_size = min(remaining, requested_batch_size)
            inference_batch_histogram[inference_batch_size] += 1
            remaining -= inference_batch_size
    batch_protocol = {
        "requested_batch_size": requested_batch_size,
        "solver_inference_batch_size": requested_batch_size,
        "candidates_per_task": budget,
        "observed_batch_count": len(observed_batch_sizes),
        "observed_batch_size_histogram": {
            str(size): count for size, count in sorted(batch_histogram.items())
        },
        "effective_inference_batch_count": sum(inference_batch_histogram.values()),
        "effective_inference_batch_size_histogram": {
            str(size): count
            for size, count in sorted(inference_batch_histogram.items())
        },
    }
    return {
        "schema_version": EXECUTION_ENVIRONMENT_SCHEMA_VERSION,
        "requested_device": requested_device,
        "resolved_device": resolved_device,
        "hardware_fingerprint": _sha256_bytes(
            _canonical_json(hardware).encode("utf-8")
        ),
        "hardware": hardware,
        "batch_protocol": batch_protocol,
    }


def _write_json(value: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def _write_jsonl_gzip(records: Iterable[Mapping[str, Any]], path: Path) -> Path:
    """Write JSONL with canonical rows and a deterministic gzip header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with (
        path.open("wb") as raw_file,
        gzip.GzipFile(filename="", fileobj=raw_file, mode="wb", mtime=0) as compressed,
        io.TextIOWrapper(compressed, encoding="utf-8") as text_file,
    ):
        for record in records:
            text_file.write(_canonical_json(dict(record)) + "\n")
    return path


def validate_personal_wandb_entity(entity: str) -> str:
    """Reject accidental logging to a work organization."""
    normalized = entity.strip().lower()
    if normalized != PERSONAL_WANDB_ENTITY:
        raise ValueError(
            "This pilot may only write to the personal W&B entity "
            f"{PERSONAL_WANDB_ENTITY!r}; got {entity!r}"
        )
    return PERSONAL_WANDB_ENTITY


def _type_name(value_type: type[Any]) -> str:
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _function_identity(function: FunctionDef) -> tuple[str, str]:
    metadata = function.metadata or {}
    stable_id = metadata.get("basis_function_id")
    if not isinstance(stable_id, str) or not stable_id:
        raise ValueError(f"Basis function {function.name!r} has no stable ID")
    return stable_id, function.name


def _item_key(item: Any, item_type: type[Any]) -> str:
    # Reuse TypedList's structural canonicalizer: it normalizes dictionary/set
    # ordering and IEEE NaN without demanding strict-JSON numeric values.
    return repr(TypedList([item], item_type=item_type).canonical_key())


def _has_multiple_output_values(output: TypedList) -> bool:
    return len({_item_key(item, output.item_type) for item in output.items}) >= 2


def _random_ascii_word(rng: random.Random, minimum: int = 1, maximum: int = 9) -> str:
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    return "".join(rng.choices(alphabet, k=rng.randint(minimum, maximum)))


def _random_bytes(rng: random.Random, minimum: int = 1, maximum: int = 8) -> bytes:
    return bytes(rng.getrandbits(8) for _ in range(rng.randint(minimum, maximum)))


def _rich_value_pool(input_type: type[Any], rng: random.Random) -> list[Any]:
    """Return deterministic randomized strata aimed at every current basis branch."""
    if input_type is int:
        boundary_values = [
            -1024,
            -101,
            -100,
            -32,
            -3,
            -2,
            -1,
            0,
            1,
            2,
            3,
            4,
            7,
            8,
            16,
            31,
            32,
            63,
            64,
            99,
            100,
            101,
            255,
            256,
            1024,
        ]
        return [
            *boundary_values,
            *(rng.randint(-10_000, 10_000) for _ in range(12)),
        ]
    if input_type is float:
        boundaries = [
            -100.0,
            -10.5,
            -2.0,
            -1.5,
            -1.0,
            -0.5,
            -0.01,
            -0.0,
            0.0,
            0.01,
            0.5,
            0.99,
            1.0,
            1.5,
            2.0,
            9.99,
            10.0,
            20.0,
            100.0,
        ]
        random_values = [rng.uniform(-20.0, 20.0) for _ in range(10)]
        random_values.extend(
            float(rng.randint(-20, 20)) + rng.choice((0.0, 0.25, 0.5, 0.75))
            for _ in range(8)
        )
        return [*boundaries, *random_values]
    if input_type is str:
        word = _random_ascii_word(rng)
        second_word = _random_ascii_word(rng)
        palindrome_half = _random_ascii_word(rng, 2, 6)
        digits = "".join(rng.choices("0123456789", k=rng.randint(1, 8)))
        return [
            "",
            " ",
            "   ",
            "\t\n",
            "a",
            f"a{word}",
            "z",
            f"{word}z",
            "abc",
            "ABC",
            "AbC",
            "Title Case",
            "two words",
            f"{word} {second_word}",
            f"  {word}  ",
            digits,
            f"{word}{digits}",
            f"a {digits} z",
            "level",
            "RaceCar",
            palindrome_half + palindrome_half[::-1],
            f"{word}!?,.-_",
            "café",
            "東京",
            "ßeta Ωmega",
            word,
            word.upper(),
            word.title(),
        ]
    if input_type is bool:
        return [False, True]
    if input_type is list:
        random_values = [rng.randint(-20, 20) for _ in range(rng.randint(1, 7))]
        duplicate = rng.randint(-9, 9)
        return [
            [],
            [0],
            [1],
            [-1],
            [1, 1],
            [duplicate, duplicate, rng.randint(-9, 9)],
            [-3, 0, 4],
            [3, 1, 2],
            [1.5, -2.0, 0.25],
            random_values,
            list(reversed(random_values)),
        ]
    if input_type is tuple:
        random_values = tuple(rng.randint(-20, 20) for _ in range(rng.randint(1, 7)))
        return [
            (),
            (None,),
            (None, None),
            (0,),
            (1, 1),
            (None, 0),
            (0, None, 1),
            (-3, 0, 4),
            random_values,
            tuple(reversed(random_values)),
        ]
    if input_type is set:
        return [
            set(),
            {0},
            {-1},
            {1, 2},
            {-3, 0, 4},
            set(range(rng.randint(1, 7))),
            {rng.randint(-100, 100) for _ in range(rng.randint(1, 7))},
        ]
    if input_type is dict:
        repeated = rng.randint(-20, 20)
        return [
            {},
            {"a": 0},
            {"a": repeated, "b": repeated},
            {"a": 1, "b": 2},
            {"negative": -1, "zero": 0, "positive": 1},
            {f"k{index}": rng.randint(-20, 20) for index in range(rng.randint(1, 7))},
            {
                f"d{index}": repeated if index < 2 else rng.randint(-20, 20)
                for index in range(rng.randint(2, 7))
            },
        ]
    if input_type is bytes:
        return [
            b"",
            b"ascii",
            b"ASCII",
            b"a z 123",
            b"\x00",
            b"\x7f",
            b"\x80",
            b"\xff\x00A",
            "café".encode(),
            _random_bytes(rng),
            _random_bytes(rng),
        ]
    if input_type is bytearray:
        return [
            bytearray(value)
            for value in (
                b"",
                b"ascii",
                b"ASCII",
                b"\x00",
                b"\x80",
                b"\xff\x00A",
                _random_bytes(rng),
                _random_bytes(rng),
            )
        ]
    if input_type is complex:
        return [
            0j,
            1 + 0j,
            -1 + 0j,
            0 + 1j,
            0 - 1j,
            1 + 1j,
            -1 + 1j,
            -1 - 1j,
            1 - 1j,
            complex(rng.uniform(-20, 20), rng.uniform(-20, 20)),
            complex(rng.uniform(-20, 20), rng.uniform(-20, 20)),
        ]
    if input_type is range:
        random_start = rng.randint(-20, 20)
        positive_step = rng.choice((1, 2, 3, 5))
        negative_step = -rng.choice((1, 2, 3, 5))
        return [
            range(0),
            range(1, 1),
            range(1),
            range(-3, 4),
            range(-10, 11, 2),
            range(10, -1, -1),
            range(5, -10, -3),
            range(
                random_start,
                random_start + positive_step * rng.randint(1, 8),
                positive_step,
            ),
            range(
                random_start,
                random_start + negative_step * rng.randint(1, 8),
                negative_step,
            ),
        ]
    raise ValueError(f"No rich value strata for {_type_name(input_type)}")


def _random_input(input_type: type[Any], rng: random.Random) -> TypedList:
    # Boolean tasks have a small domain. Variable-length lists preserve global
    # input/output deduplication even for large, type-balanced corpora.
    list_length = rng.randint(3, 16) if input_type is bool else rng.randint(3, 8)
    items: list[Any] = []
    while len(items) < list_length:
        pool = _rich_value_pool(input_type, rng)
        rng.shuffle(pool)
        items.extend(pool)
    items = items[:list_length]
    if input_type is bool and set(items) != {False, True}:
        items[0:2] = [False, True]
        rng.shuffle(items)
    return TypedList(items, item_type=input_type)


def _cell_targets(total: int, seed: int) -> dict[tuple[type[Any], int], int]:
    if total < 0:
        raise ValueError("split sizes must be non-negative")
    # 12 input types and 5 lengths are coprime. This ordering visits all 60
    # cross-product cells once while keeping both marginals balanced for a
    # partial cycle as well as for the large, divisible default sizes.
    cells = [
        (SUPPORTED_RANDOM_TYPES[index % len(SUPPORTED_RANDOM_TYPES)], index % 5 + 1)
        for index in range(len(SUPPORTED_RANDOM_TYPES) * 5)
    ]
    base, remainder = divmod(total, len(cells))
    offset = seed % len(cells)
    remainder_cells = {
        cells[(offset + index) % len(cells)] for index in range(remainder)
    }
    return {cell: base + int(cell in remainder_cells) for cell in cells}


def _requires_fixed_python_hash_seed(basis: BasisSet) -> bool:
    return any(
        function.name in NONDETERMINISTIC_PYHASH_NAMES and "hash(" in function.code
        for function in basis.functions
    )


def _require_reproducible_runtime(basis: BasisSet) -> None:
    if _requires_fixed_python_hash_seed(basis):
        require_reproducible_basis_runtime(basis)


def _generate_split(
    *,
    split: str,
    size: int,
    seed: int,
    basis: BasisSet,
    available_functions: FunctionDefSet,
    seen_task_ids: set[str],
    max_attempts_per_record: int,
    progress_every: int,
) -> tuple[list[BasisTaskRecord], dict[str, Any]]:
    rng = random.Random(seed)
    executor = Executor(available_functions)
    targets = _cell_targets(size, seed)
    records: list[BasisTaskRecord] = []
    rejection_counts: Counter[str] = Counter()
    witness_occurrences: Counter[str] = Counter()
    witness_task_coverage: Counter[str] = Counter()
    attempts = 0
    next_progress = progress_every

    for (input_type, witness_length), target in targets.items():
        accepted_for_cell = 0
        cell_attempts = 0
        cell_attempt_limit = max_attempts_per_record * max(1, target)
        while accepted_for_cell < target:
            if cell_attempts >= cell_attempt_limit:
                raise RuntimeError(
                    "Could not fill balanced corpus cell "
                    f"split={split!r}, input_type={_type_name(input_type)!r}, "
                    f"witness_length={witness_length}, accepted={accepted_for_cell}, "
                    f"target={target}, attempts={cell_attempts}. Increase "
                    "--max-attempts-per-record or reduce the split size."
                )
            attempts += 1
            cell_attempts += 1
            input_value = _random_input(input_type, rng)
            spec = TrajectorySpec.create_random_walk(
                input_list=input_value,
                path_length=witness_length,
                available_functions=available_functions.functions,
                rng=rng,
            )
            if len(spec.function_defs) != witness_length:
                rejection_counts["incomplete_walk"] += 1
                continue

            execution = executor.execute_trajectory(spec)
            if not execution.success or execution.trajectory is None:
                rejection_counts["execution_failure"] += 1
                continue
            output_value = execution.trajectory.output
            if input_value == output_value:
                rejection_counts["identity_output"] += 1
                continue
            if not _has_multiple_output_values(output_value):
                rejection_counts["constant_output_example"] += 1
                continue

            witness_ids: list[str] = []
            witness_names: list[str] = []
            for function in spec.function_defs:
                stable_id, name = _function_identity(function)
                witness_ids.append(stable_id)
                witness_names.append(name)
            record = BasisTaskRecord.create(
                split=split,
                input_value=input_value,
                output_value=output_value,
                witness_function_ids=witness_ids,
                witness_function_names=witness_names,
                basis_set_id=basis.basis_set_id,
                basis_set_digest=basis.digest,
                generator=GENERATOR_NAME,
                seed=seed,
                source_index=len(records),
                metadata={
                    "input_type": _type_name(input_type),
                    "requested_witness_length": witness_length,
                },
            )
            if record.task_id in seen_task_ids:
                rejection_counts["duplicate_task"] += 1
                continue
            seen_task_ids.add(record.task_id)
            records.append(record)
            witness_occurrences.update(witness_ids)
            witness_task_coverage.update(set(witness_ids))
            accepted_for_cell += 1
            if progress_every and len(records) >= next_progress:
                print(
                    f"[generate:{split}] accepted={len(records)}/{size} "
                    f"attempts={attempts}",
                    flush=True,
                )
                while next_progress <= len(records):
                    next_progress += progress_every

    if progress_every and records and len(records) % progress_every:
        print(
            f"[generate:{split}] accepted={len(records)}/{size} attempts={attempts}",
            flush=True,
        )

    # Keep batches heterogeneous without consuming the generation RNG stream.
    random.Random(seed ^ 0x5EED5EED).shuffle(records)
    by_type = Counter(record.metadata["input_type"] for record in records)
    by_length = Counter(record.witness_length for record in records)
    witness_exposure = [
        {
            "function_id": function.function_id,
            "function_name": function.name,
            "total_occurrences": witness_occurrences[function.function_id],
            "task_coverage": witness_task_coverage[function.function_id],
        }
        for function in basis.functions
    ]
    return records, {
        "size": len(records),
        "seed": seed,
        "attempts": attempts,
        "rejections": dict(sorted(rejection_counts.items())),
        "by_input_type": dict(sorted(by_type.items())),
        "by_witness_length": {
            str(key): value for key, value in sorted(by_length.items())
        },
        "witness_function_coverage": {
            "functions_exposed": sum(
                row["total_occurrences"] > 0 for row in witness_exposure
            ),
            "functions_available": len(witness_exposure),
            "total_witness_steps": sum(witness_occurrences.values()),
            "functions": witness_exposure,
        },
    }


def generate_corpus(
    *,
    basis_set_id: str,
    split_sizes: Mapping[str, int],
    split_seeds: Mapping[str, int],
    output_dir: str | Path,
    max_attempts_per_record: int = 1_000,
    progress_every: int = 10_000,
    overwrite: bool = False,
) -> tuple[dict[str, Any], Path]:
    """Generate a deterministic, globally deduplicated three-split corpus."""
    if max_attempts_per_record <= 0:
        raise ValueError("max_attempts_per_record must be positive")
    if progress_every < 0:
        raise ValueError("progress_every must be non-negative")
    if set(split_sizes) != set(_SPLIT_ORDER):
        raise ValueError(f"split_sizes must have exactly {_SPLIT_ORDER!r}")
    if set(split_seeds) != set(_SPLIT_ORDER):
        raise ValueError(f"split_seeds must have exactly {_SPLIT_ORDER!r}")
    if len(set(split_seeds.values())) != len(split_seeds):
        raise ValueError("split seeds must be distinct")

    basis = load_basis_set(basis_set_id)
    _require_reproducible_runtime(basis)
    available_functions = basis.as_function_set()
    corpus_dir = Path(output_dir)
    planned_paths = [corpus_dir / f"{split}.jsonl.gz" for split in _SPLIT_ORDER]
    planned_paths.append(corpus_dir / "manifest.json")
    existing = [path for path in planned_paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Corpus outputs already exist; pass --overwrite to replace them: "
            + ", ".join(str(path) for path in existing)
        )

    seen_task_ids: set[str] = set()
    split_metadata: dict[str, Any] = {}
    for split in _SPLIT_ORDER:
        records, metadata = _generate_split(
            split=split,
            size=split_sizes[split],
            seed=split_seeds[split],
            basis=basis,
            available_functions=available_functions,
            seen_task_ids=seen_task_ids,
            max_attempts_per_record=max_attempts_per_record,
            progress_every=progress_every,
        )
        split_path = corpus_dir / f"{split}.jsonl.gz"
        write_basis_task_records(records, split_path)
        metadata.update(
            {
                "path": split_path.name,
                "sha256": _sha256_file(split_path),
            }
        )
        split_metadata[split] = metadata

    manifest: dict[str, Any] = {
        "schema_version": PIPELINE_SCHEMA_VERSION,
        "generator": GENERATOR_NAME,
        "value_generator": {
            "name": VALUE_GENERATOR_NAME,
            "strata": VALUE_STRATA_DESCRIPTION,
        },
        "basis_set_id": basis.basis_set_id,
        "basis_set_digest": basis.digest,
        "global_task_count": len(seen_task_ids),
        "global_dedupe_key": "sha256(canonical_input,canonical_output)",
        "filters": {
            "complete_witness": True,
            "valid_execution": True,
            "reject_identity_output": True,
            "reject_constant_output_example": True,
            "constant_output_definition": "fewer than two distinct output items",
        },
        "balance": {
            "dimensions": ["input_type", "requested_witness_length"],
            "witness_lengths": [1, 2, 3, 4, 5],
            "input_types": [_type_name(t) for t in SUPPORTED_RANDOM_TYPES],
            "maximum_cell_count_difference": 1,
        },
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "splits": split_metadata,
    }
    manifest["manifest_digest"] = _sha256_bytes(_canonical_json(manifest).encode())
    manifest_path = _write_json(manifest, corpus_dir / "manifest.json")
    return manifest, manifest_path


def load_corpus(
    corpus_dir: str | Path,
    *,
    splits: Sequence[str] = _SPLIT_ORDER,
) -> tuple[dict[str, Any], list[BasisTaskRecord]]:
    """Load a corpus with digest, provenance, split, and global-dedupe checks."""
    root = Path(corpus_dir)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stored_digest = manifest.get("manifest_digest")
    digest_payload = dict(manifest)
    digest_payload.pop("manifest_digest", None)
    expected_digest = _sha256_bytes(_canonical_json(digest_payload).encode())
    if stored_digest != expected_digest:
        raise ValueError(
            f"Corpus manifest digest mismatch: {stored_digest!r} != {expected_digest!r}"
        )

    basis = load_basis_set(manifest["basis_set_id"])
    if basis.digest != manifest["basis_set_digest"]:
        raise ValueError("Corpus manifest references a different basis-set digest")

    split_paths: dict[str, Path] = {}
    for split, split_metadata in manifest["splits"].items():
        path_value = split_metadata.get("path")
        if (
            not isinstance(path_value, str)
            or not path_value
            or PurePosixPath(path_value).name != path_value
            or PureWindowsPath(path_value).name != path_value
        ):
            raise ValueError(
                f"Corpus split path for {split!r} must be a safe basename; "
                f"got {path_value!r}"
            )
        split_paths[split] = root / path_value

    unknown_splits = set(splits) - set(manifest["splits"])
    if unknown_splits:
        raise ValueError(f"Unknown corpus splits: {sorted(unknown_splits)}")
    function_names_by_id = {
        function.function_id: function.name for function in basis.functions
    }
    records: list[BasisTaskRecord] = []
    seen_task_ids: set[str] = set()
    records_by_split: dict[str, list[BasisTaskRecord]] = {}
    for split in splits:
        split_metadata = manifest["splits"][split]
        path = split_paths[split]
        if _sha256_file(path) != split_metadata["sha256"]:
            raise ValueError(f"Corpus file digest mismatch: {path}")
        split_records = _load_split_records(
            path,
            expected_basis_set_id=basis.basis_set_id,
            expected_basis_set_digest=basis.digest,
        )
        if len(split_records) != split_metadata["size"]:
            raise ValueError(f"Corpus record count mismatch: {path}")
        for record in split_records:
            if record.split != split:
                raise ValueError(
                    f"Record {record.task_id} is stored in {split!r} but labels "
                    f"itself {record.split!r}"
                )
            if record.task_id in seen_task_ids:
                raise ValueError(
                    f"Duplicate task across corpus splits: {record.task_id}"
                )
            seen_task_ids.add(record.task_id)
            input_type = _type_name(record.input_value.item_type)
            if record.metadata.get("input_type") != input_type:
                raise ValueError(
                    f"Record {record.task_id} input_type metadata mismatch: "
                    f"{record.metadata.get('input_type')!r} != {input_type!r}"
                )
            requested_length = record.metadata.get("requested_witness_length")
            if (
                not isinstance(requested_length, int)
                or isinstance(requested_length, bool)
                or requested_length != record.witness_length
            ):
                raise ValueError(
                    f"Record {record.task_id} witness length mismatch: metadata "
                    f"declares {requested_length!r}, record has "
                    f"{record.witness_length}"
                )
            for index, (function_id, function_name) in enumerate(
                zip(
                    record.witness_function_ids,
                    record.witness_function_names,
                    strict=True,
                )
            ):
                expected_name = function_names_by_id.get(function_id)
                if expected_name is None:
                    raise ValueError(
                        f"Record {record.task_id} witness step {index} references "
                        f"unknown basis-function ID {function_id!r}"
                    )
                if function_name != expected_name:
                    raise ValueError(
                        f"Record {record.task_id} witness step {index} name mismatch "
                        f"for {function_id!r}: {function_name!r} != "
                        f"{expected_name!r}"
                    )
        _validate_split_statistics(split, split_records, split_metadata, basis)
        records_by_split[split] = split_records
        records.extend(split_records)

    all_splits_loaded = len(splits) == len(manifest["splits"]) and set(splits) == set(
        manifest["splits"]
    )
    if all_splits_loaded:
        if manifest.get("global_task_count") != len(seen_task_ids):
            raise ValueError(
                "Corpus global record count mismatch: manifest declares "
                f"{manifest.get('global_task_count')!r}, loaded "
                f"{len(seen_task_ids)}"
            )
        _validate_balance_claims(manifest, records_by_split)
    return manifest, records


def _load_split_records(
    path: Path,
    *,
    expected_basis_set_id: str,
    expected_basis_set_digest: str,
) -> list[BasisTaskRecord]:
    """Decode a split while checking its redundant on-wire witness length."""
    records: list[BasisTaskRecord] = []
    with gzip.open(path, "rt", encoding="utf-8") as text_file:
        for line_number, line in enumerate(text_file, start=1):
            try:
                data = json.loads(line)
                serialized_witness_length = data.get("witness_length")
                record = BasisTaskRecord.from_dict(data)
                if serialized_witness_length != record.witness_length:
                    raise ValueError(
                        "serialized witness_length mismatch: "
                        f"{serialized_witness_length!r} != {record.witness_length}"
                    )
            except Exception as error:
                raise ValueError(
                    f"invalid record at line {line_number}: {error}"
                ) from error
            if record.basis_set_id != expected_basis_set_id:
                raise ValueError(
                    "basis set ID mismatch at line "
                    f"{line_number}: expected {expected_basis_set_id}, "
                    f"got {record.basis_set_id}"
                )
            if record.basis_set_digest != expected_basis_set_digest:
                raise ValueError(
                    "basis set digest mismatch at line "
                    f"{line_number}: expected {expected_basis_set_digest}, "
                    f"got {record.basis_set_digest}"
                )
            records.append(record)
    return records


def _validate_split_statistics(
    split: str,
    records: Sequence[BasisTaskRecord],
    split_metadata: Mapping[str, Any],
    basis: BasisSet,
) -> None:
    """Recompute every result-derived split statistic in the manifest."""
    by_type = Counter(record.metadata["input_type"] for record in records)
    expected_by_type = dict(sorted(by_type.items()))
    if split_metadata.get("by_input_type") != expected_by_type:
        raise ValueError(f"Corpus by-input-type count mismatch for split {split!r}")

    by_length = Counter(record.witness_length for record in records)
    expected_by_length = {str(key): value for key, value in sorted(by_length.items())}
    if split_metadata.get("by_witness_length") != expected_by_length:
        raise ValueError(f"Corpus witness-length count mismatch for split {split!r}")

    witness_occurrences: Counter[str] = Counter()
    witness_task_coverage: Counter[str] = Counter()
    for record in records:
        witness_occurrences.update(record.witness_function_ids)
        witness_task_coverage.update(set(record.witness_function_ids))
    functions = [
        {
            "function_id": function.function_id,
            "function_name": function.name,
            "total_occurrences": witness_occurrences[function.function_id],
            "task_coverage": witness_task_coverage[function.function_id],
        }
        for function in basis.functions
    ]
    expected_coverage = {
        "functions_exposed": sum(row["total_occurrences"] > 0 for row in functions),
        "functions_available": len(functions),
        "total_witness_steps": sum(witness_occurrences.values()),
        "functions": functions,
    }
    if split_metadata.get("witness_function_coverage") != expected_coverage:
        raise ValueError(
            f"Corpus witness-function coverage mismatch for split {split!r}"
        )


def _validate_balance_claims(
    manifest: Mapping[str, Any],
    records_by_split: Mapping[str, Sequence[BasisTaskRecord]],
) -> None:
    """Validate the generator's per-split cross-product balance claim."""
    balance = manifest.get("balance")
    if not isinstance(balance, dict):
        raise ValueError("Corpus manifest has no valid balance claim")
    if balance.get("dimensions") != ["input_type", "requested_witness_length"]:
        raise ValueError("Corpus manifest has unsupported balance dimensions")

    input_types = balance.get("input_types")
    witness_lengths = balance.get("witness_lengths")
    maximum_difference = balance.get("maximum_cell_count_difference")
    if (
        not isinstance(input_types, list)
        or not input_types
        or any(not isinstance(item, str) or not item for item in input_types)
        or len(set(input_types)) != len(input_types)
        or not isinstance(witness_lengths, list)
        or not witness_lengths
        or any(
            not isinstance(item, int) or isinstance(item, bool) or item <= 0
            for item in witness_lengths
        )
        or len(set(witness_lengths)) != len(witness_lengths)
        or not isinstance(maximum_difference, int)
        or isinstance(maximum_difference, bool)
        or maximum_difference < 0
    ):
        raise ValueError("Corpus manifest has an invalid balance claim")

    cells = [
        (input_type, length) for input_type in input_types for length in witness_lengths
    ]
    for split, records in records_by_split.items():
        counts = Counter(
            (record.metadata["input_type"], record.witness_length) for record in records
        )
        unexpected_cells = set(counts) - set(cells)
        if unexpected_cells:
            raise ValueError(
                f"Corpus split {split!r} contains cells outside its balance claim: "
                f"{sorted(unexpected_cells)}"
            )
        cell_counts = [counts[cell] for cell in cells]
        if max(cell_counts) - min(cell_counts) > maximum_difference:
            raise ValueError(
                f"Corpus balance mismatch for split {split!r}: cell counts differ "
                f"by {max(cell_counts) - min(cell_counts)}, exceeding "
                f"{maximum_difference}"
            )


def _is_local_tree_file(root: Path, candidate: Path) -> bool:
    if not candidate.is_file():
        return False
    relative = candidate.relative_to(root)
    if any(part in _LOCAL_TREE_EXCLUDED_PARTS for part in relative.parts):
        return False
    if relative.name in _LOCAL_TREE_EXCLUDED_FILES:
        return False
    return not relative.name.endswith(_LOCAL_TREE_EXCLUDED_SUFFIXES)


def _local_checkpoint_tree(
    path: Path,
) -> tuple[dict[str, dict[str, Any]], str]:
    """Hash every durable regular file under a local inference checkpoint."""
    identities: dict[str, dict[str, Any]] = {}
    for candidate in sorted(path.rglob("*"), key=lambda item: item.as_posix()):
        if not _is_local_tree_file(path, candidate):
            continue
        relative = candidate.relative_to(path).as_posix()
        identities[relative] = {
            "bytes": candidate.stat().st_size,
            "sha256": _sha256_file(candidate),
        }
    tree_payload = [
        {"path": relative, **identity}
        for relative, identity in sorted(identities.items())
    ]
    return identities, _sha256_bytes(_canonical_json(tree_payload).encode())


def _read_wandb_run_url(path: Path) -> str | None:
    if not path.is_file():
        return None
    content = path.read_text(encoding="utf-8").strip()
    for line in content.splitlines():
        if line.startswith("URL="):
            return line.removeprefix("URL=").strip()
    return content if content.startswith(("https://", "http://")) else None


def _verify_default_local_checkpoint(path: Path, wandb_url: str | None) -> None:
    """Prove the known local path has the advertised personal checkpoint lineage."""
    errors: list[str] = []
    weights_path = path / "model.safetensors"
    actual_bytes = weights_path.stat().st_size if weights_path.is_file() else None
    if actual_bytes != DEFAULT_CHECKPOINT_MODEL_BYTES:
        errors.append(
            "model.safetensors bytes "
            f"{actual_bytes!r} != {DEFAULT_CHECKPOINT_MODEL_BYTES}"
        )

    trainer_path = path / "trainer_state.json"
    try:
        trainer_state = json.loads(trainer_path.read_text(encoding="utf-8"))
        global_step = trainer_state.get("global_step")
    except (OSError, ValueError, AttributeError) as error:
        global_step = None
        errors.append(f"invalid trainer_state.json ({error})")
    if global_step != DEFAULT_CHECKPOINT_GLOBAL_STEP:
        errors.append(
            f"trainer global_step {global_step!r} != {DEFAULT_CHECKPOINT_GLOBAL_STEP}"
        )
    if wandb_url != DEFAULT_CHECKPOINT_WANDB_URL:
        errors.append(
            f"wandb_run.url {wandb_url!r} != {DEFAULT_CHECKPOINT_WANDB_URL!r}"
        )
    if errors:
        raise ValueError(
            "Default local checkpoint failed canonical lineage verification: "
            + "; ".join(errors)
        )


def resolve_model_spec(model: str, revision: str | None) -> ModelSpec:
    """Resolve the default local checkpoint or an exact HF repo revision."""
    requested = model
    model_path = Path(model)
    if model == str(DEFAULT_LOCAL_CHECKPOINT) and not model_path.is_dir():
        model = DEFAULT_HF_CHECKPOINT
        model_path = Path(model)

    local_files: dict[str, dict[str, Any]] = {}
    local_tree_digest: str | None = None
    wandb_url: str | None = None
    canonical_hf_repo: str | None = None
    canonical_hf_revision: str | None = None
    resolved_revision = revision
    if model_path.is_dir():
        resolved = str(model_path.resolve())
        resolved_revision = None
        wandb_url = _read_wandb_run_url(model_path / "wandb_run.url")
        if model_path.resolve() == DEFAULT_LOCAL_CHECKPOINT.resolve():
            _verify_default_local_checkpoint(model_path, wandb_url)
            canonical_hf_repo = DEFAULT_HF_CHECKPOINT
            canonical_hf_revision = DEFAULT_HF_REVISION
        local_files, local_tree_digest = _local_checkpoint_tree(model_path)
    else:
        if "/" not in model:
            raise FileNotFoundError(
                f"Model path does not exist and is not an HF repo ID: {model!r}"
            )
        if not isinstance(revision, str) or not _COMMIT_SHA_RE.fullmatch(revision):
            raise ValueError(
                "HF checkpoints require --model-revision to be an immutable "
                "40-hex commit SHA"
            )
        revision = revision.lower()
        resolved_revision = revision
        resolved = model
        canonical_hf_repo = model
        canonical_hf_revision = revision

    return ModelSpec(
        requested=requested,
        resolved=resolved,
        revision=resolved_revision,
        canonical_hf_repo=canonical_hf_repo,
        canonical_hf_revision=canonical_hf_revision,
        local_files=local_files,
        local_tree_digest=local_tree_digest,
        wandb_run_url=wandb_url,
    )


def _create_checkpoint_solver(
    *,
    model_spec: ModelSpec,
    budget: int,
    batch_size: int,
    deterministic_decoding: bool,
    temperature: float,
    max_new_tokens: int,
    seed: int,
    device: str,
    tracker: FunctionUsageTracker,
):
    """Load one exact checkpoint and return the standard verified solver."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from wandering_light.solver import (
        TrainedLLMTokenGenerator,
        create_token_solver,
        remove_thinking,
    )

    if budget <= 0 or batch_size <= 0 or max_new_tokens <= 0:
        raise ValueError("budget, batch_size, and max_new_tokens must be positive")
    if not deterministic_decoding and temperature <= 0:
        raise ValueError("temperature must be positive for sampling")

    class FrozenCheckpointTokenGenerator(TrainedLLMTokenGenerator):
        """Batched generator without unbounded prompt/completion retention."""

        def __init__(self) -> None:
            self.model_or_path = model_spec.resolved
            self.temperature = temperature
            self.inference_batch_size = batch_size
            self.llm_io_history: list[tuple[str, str]] = []
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_spec.resolved,
                revision=model_spec.revision,
            )
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.tokenizer.padding_side = "left"
            self.model = AutoModelForCausalLM.from_pretrained(
                model_spec.resolved,
                revision=model_spec.revision,
                torch_dtype="auto",
                low_cpu_mem_usage=True,
            )
            selected_device = device
            if selected_device == "auto":
                selected_device = "cuda" if torch.cuda.is_available() else "cpu"
            self.device = torch.device(selected_device)
            self.model.to(self.device)
            self.model.eval()
            self.pipeline = None

        def generate(self, prompt: str) -> str:
            return self.generate_batch([prompt])[0]

        def generate_batch(self, prompts: list[str]) -> list[str]:
            responses: list[str] = []
            for start in range(0, len(prompts), self.inference_batch_size):
                batch_prompts = prompts[start : start + self.inference_batch_size]
                encoded = self.tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=False,
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                generation_options: dict[str, Any] = {
                    "max_new_tokens": max_new_tokens,
                    "do_sample": not deterministic_decoding,
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "eos_token_id": self.tokenizer.eos_token_id,
                }
                if not deterministic_decoding:
                    generation_options["temperature"] = temperature
                with torch.inference_mode():
                    output_ids = self.model.generate(
                        **encoded,
                        **generation_options,
                    )
                prompt_width = encoded["input_ids"].shape[1]
                for sequence in output_ids:
                    completion = self.tokenizer.decode(
                        sequence[prompt_width:], skip_special_tokens=True
                    ).strip()
                    responses.append(remove_thinking(completion))
            return responses

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(deterministic_decoding, warn_only=True)
    return create_token_solver(
        FrozenCheckpointTokenGenerator(),
        budget=budget,
        usage_tracker=tracker,
        track_function_usage=True,
    )


def _dimension_summary(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    grouped: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    return {value: _outcome_summary(group) for value, group in sorted(grouped.items())}


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    fraction = index - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _outcome_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    successes = [row for row in rows if row["success"]]
    solution_lengths = [int(row["solution_length"]) for row in successes]
    return {
        "tasks": len(rows),
        "successes": len(successes),
        "failures": len(rows) - len(successes),
        "solve_rate": len(successes) / len(rows) if rows else 0.0,
        "mean_solution_length_success": (
            statistics.fmean(solution_lengths) if solution_lengths else None
        ),
    }


def _compact_error(error: str | None, limit: int = 240) -> str | None:
    if error is None or len(error) <= limit:
        return error
    return error[: limit - 1] + "…"


def _subsequence_summary(
    solved_sequences: Sequence[Sequence[tuple[str, str]]],
    *,
    top_n: int,
) -> list[dict[str, Any]]:
    occurrences: Counter[tuple[str, ...]] = Counter()
    task_coverage: Counter[tuple[str, ...]] = Counter()
    names_by_ids: dict[tuple[str, ...], tuple[str, ...]] = {}
    lengths: dict[tuple[str, ...], int] = {}
    for sequence in solved_sequences:
        seen_in_task: set[tuple[str, ...]] = set()
        for length in range(2, 5):
            for start in range(len(sequence) - length + 1):
                window = sequence[start : start + length]
                ids = tuple(item[0] for item in window)
                names_by_ids[ids] = tuple(item[1] for item in window)
                lengths[ids] = length
                occurrences[ids] += 1
                seen_in_task.add(ids)
        task_coverage.update(seen_in_task)
    ordered = sorted(
        occurrences,
        key=lambda ids: (-occurrences[ids], -task_coverage[ids], ids),
    )[:top_n]
    return [
        {
            "function_ids": list(ids),
            "function_names": list(names_by_ids[ids]),
            "length": lengths[ids],
            "total_occurrences": occurrences[ids],
            "task_coverage": task_coverage[ids],
        }
        for ids in ordered
    ]


def aggregate_results(
    *,
    rows: Sequence[Mapping[str, Any]],
    basis: BasisSet,
    tracker: FunctionUsageTracker,
    solved_sequences: Sequence[Sequence[tuple[str, str]]],
    batch_latencies_seconds: Sequence[float],
    batch_sizes: Sequence[int],
    top_subsequences: int,
) -> dict[str, Any]:
    """Build all solver-independent aggregate statistics."""
    if len(batch_latencies_seconds) != len(batch_sizes):
        raise ValueError("batch latency and size arrays must align")
    usage_snapshot = tracker.snapshot()
    usage_rows = []
    for function in basis.functions:
        stats = usage_snapshot.get(function.function_id)
        usage_rows.append(
            {
                "function_id": function.function_id,
                "function_name": function.name,
                "total_occurrences": stats.invocation_count if stats else 0,
                "task_coverage": stats.solution_count if stats else 0,
                "successful_task_coverage_rate": (
                    stats.solution_count / tracker.successful_solve_count
                    if stats and tracker.successful_solve_count
                    else 0.0
                ),
                "last_used_successful_solve": stats.last_used_solve if stats else None,
            }
        )
    usage_rows.sort(
        key=lambda row: (
            -row["total_occurrences"],
            -row["task_coverage"],
            row["function_id"],
        )
    )
    per_task_latencies = [
        1_000 * latency / size
        for latency, size in zip(batch_latencies_seconds, batch_sizes, strict=True)
        if size
    ]
    total_seconds = sum(batch_latencies_seconds)
    return {
        "overall": _outcome_summary(rows),
        "by_split": _dimension_summary(rows, "split"),
        "by_input_type": _dimension_summary(rows, "input_type"),
        "by_witness_length": _dimension_summary(rows, "witness_length"),
        "solution_length_histogram": {
            str(key): value
            for key, value in sorted(
                Counter(
                    int(row["solution_length"]) for row in rows if row["success"]
                ).items()
            )
        },
        "latency": {
            "wall_seconds": total_seconds,
            "tasks_per_second": len(rows) / total_seconds if total_seconds else None,
            "batch_count": len(batch_sizes),
            "mean_ms_per_task_by_batch": (
                statistics.fmean(per_task_latencies) if per_task_latencies else None
            ),
            "p50_ms_per_task_by_batch": _percentile(per_task_latencies, 0.50),
            "p95_ms_per_task_by_batch": _percentile(per_task_latencies, 0.95),
        },
        "function_usage": {
            "primary_metric": "total_occurrences",
            "successful_solve_count": tracker.successful_solve_count,
            "functions": usage_rows,
        },
        "frequent_contiguous_subsequences": _subsequence_summary(
            solved_sequences, top_n=top_subsequences
        ),
    }


def evaluate_corpus(
    *,
    corpus_dir: str | Path,
    output_dir: str | Path,
    splits: Sequence[str],
    evaluation_basis_set_id: str | None = None,
    model: str,
    model_revision: str | None,
    batch_size: int,
    budget: int,
    deterministic_decoding: bool,
    temperature: float,
    max_new_tokens: int,
    seed: int,
    device: str,
    top_subsequences: int = 100,
    progress_every: int = 10_000,
    overwrite: bool = False,
) -> tuple[dict[str, Any], list[Path]]:
    """Evaluate frozen tasks and record only verified successful usage."""
    manifest, records = load_corpus(corpus_dir, splits=splits)
    if progress_every < 0:
        raise ValueError("progress_every must be non-negative")
    source_basis = load_basis_set(manifest["basis_set_id"])
    basis = load_basis_set(evaluation_basis_set_id or source_basis.basis_set_id)
    _require_reproducible_runtime(basis)
    available_functions = basis.as_function_set()
    tracker = FunctionUsageTracker(basis.basis_set_id, basis.digest)
    model_spec = resolve_model_spec(model, model_revision)

    result_dir = Path(output_dir)
    output_paths = [
        result_dir / "results.jsonl.gz",
        result_dir / "aggregate.json",
        result_dir / FunctionUsageTracker.FILE_NAME,
    ]
    existing = [path for path in output_paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Evaluation outputs already exist; pass --overwrite to replace them: "
            + ", ".join(str(path) for path in existing)
        )

    solver = _create_checkpoint_solver(
        model_spec=model_spec,
        budget=budget,
        batch_size=batch_size,
        deterministic_decoding=deterministic_decoding,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        seed=seed,
        device=device,
        tracker=tracker,
    )
    rows: list[dict[str, Any]] = []
    solved_sequences: list[list[tuple[str, str]]] = []
    batch_latencies: list[float] = []
    batch_sizes: list[int] = []
    next_progress = progress_every
    for batch_index, start in enumerate(range(0, len(records), batch_size)):
        batch = records[start : start + batch_size]
        problems = [(record.input_value, record.output_value) for record in batch]
        started = time.perf_counter()
        results = solver.solve_batch(problems, available_functions)
        elapsed = time.perf_counter() - started
        if len(results) != len(batch):
            raise RuntimeError(
                f"Solver returned {len(results)} results for a batch of {len(batch)}"
            )
        batch_latencies.append(elapsed)
        batch_sizes.append(len(batch))
        for record, result in zip(batch, results, strict=True):
            solution: list[tuple[str, str]] = []
            if result.success:
                if result.trajectory is None:
                    raise RuntimeError("Successful solver result has no trajectory")
                solution = [
                    _function_identity(function)
                    for function in result.trajectory.function_defs
                ]
                solved_sequences.append(solution)
            rows.append(
                {
                    "task_id": record.task_id,
                    "split": record.split,
                    "input_type": record.metadata["input_type"],
                    "witness_length": record.witness_length,
                    "success": result.success,
                    "solution_length": len(solution) if result.success else None,
                    "solution_function_ids": [item[0] for item in solution],
                    "solution_function_names": [item[1] for item in solution],
                    "batch_index": batch_index,
                    "error": (
                        None if result.success else _compact_error(result.error_msg)
                    ),
                }
            )
        processed = start + len(batch)
        if progress_every and processed >= next_progress:
            print(
                f"[evaluate] processed={processed}/{len(records)} "
                f"successes={tracker.successful_solve_count}",
                flush=True,
            )
            while next_progress <= processed:
                next_progress += progress_every

    if progress_every and records and len(records) % progress_every:
        print(
            f"[evaluate] processed={len(records)}/{len(records)} "
            f"successes={tracker.successful_solve_count}",
            flush=True,
        )

    aggregate = {
        "schema_version": PIPELINE_SCHEMA_VERSION,
        "task_source_basis_set_id": source_basis.basis_set_id,
        "task_source_basis_set_digest": source_basis.digest,
        "evaluation_basis_set_id": basis.basis_set_id,
        "evaluation_basis_set_digest": basis.digest,
        # Compatibility fields identify the basis that actually executed.
        "basis_set_id": basis.basis_set_id,
        "basis_set_digest": basis.digest,
        "corpus_manifest_digest": manifest["manifest_digest"],
        "evaluated_splits": list(splits),
        "model": model_spec.to_dict(),
        "decoding": {
            "deterministic": deterministic_decoding,
            "temperature": None if deterministic_decoding else temperature,
            "max_new_tokens": max_new_tokens,
            "seed": seed,
            "budget": budget,
            "requested_batch_size": batch_size,
            "device": device,
        },
        "execution_environment": capture_execution_environment(
            requested_device=device,
            requested_batch_size=batch_size,
            budget=budget,
            observed_batch_sizes=batch_sizes,
        ),
        **aggregate_results(
            rows=rows,
            basis=basis,
            tracker=tracker,
            solved_sequences=solved_sequences,
            batch_latencies_seconds=batch_latencies,
            batch_sizes=batch_sizes,
            top_subsequences=top_subsequences,
        ),
    }
    _write_jsonl_gzip(rows, output_paths[0])
    _write_json(aggregate, output_paths[1])
    tracker.save(output_paths[2])
    return aggregate, output_paths


def log_artifacts_to_wandb(
    *,
    aggregate: Mapping[str, Any] | None,
    paths: Sequence[Path],
    entity: str,
    project: str,
    run_name: str | None,
) -> str:
    """Upload compact pilot artifacts to the explicitly personal W&B entity."""
    import wandb

    entity = validate_personal_wandb_entity(entity)
    if not project.strip():
        raise ValueError("W&B project must not be empty")
    run = wandb.init(
        entity=entity,
        project=project,
        name=run_name,
        job_type="basis-library-pilot",
        config={
            "schema_version": PIPELINE_SCHEMA_VERSION,
            "task_source_basis_set_id": (
                aggregate.get("task_source_basis_set_id") if aggregate else None
            ),
            "task_source_basis_set_digest": (
                aggregate.get("task_source_basis_set_digest") if aggregate else None
            ),
            "evaluation_basis_set_id": (
                aggregate.get("evaluation_basis_set_id") if aggregate else None
            ),
            "evaluation_basis_set_digest": (
                aggregate.get("evaluation_basis_set_digest") if aggregate else None
            ),
            "corpus_manifest_digest": (
                aggregate.get("corpus_manifest_digest") if aggregate else None
            ),
        },
    )
    if run.entity != PERSONAL_WANDB_ENTITY:
        run.finish(exit_code=1)
        raise RuntimeError(f"W&B initialized unexpected entity {run.entity!r}")
    if aggregate is not None:
        overall = aggregate["overall"]
        latency = aggregate["latency"]
        run.log(
            {
                "eval/tasks": overall["tasks"],
                "eval/successes": overall["successes"],
                "eval/solve_rate": overall["solve_rate"],
                "eval/mean_solution_length": overall["mean_solution_length_success"],
                "eval/wall_seconds": latency["wall_seconds"],
                "eval/tasks_per_second": latency["tasks_per_second"],
            }
        )
    artifact = wandb.Artifact(
        name=f"basis-library-pilot-{run.id}",
        type="basis-library-evaluation",
        metadata={"personal_entity": PERSONAL_WANDB_ENTITY},
    )
    for path in paths:
        artifact.add_file(str(path), name=path.name)
    run.log_artifact(artifact)
    url = run.url
    run.finish()
    return url


def _parse_splits(value: str) -> tuple[str, ...]:
    splits = tuple(item.strip() for item in value.split(",") if item.strip())
    if not splits:
        raise argparse.ArgumentTypeError("at least one split is required")
    if len(set(splits)) != len(splits):
        raise argparse.ArgumentTypeError("evaluation splits must be unique")
    unknown = set(splits) - set(_SPLIT_ORDER)
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown splits: {sorted(unknown)}")
    return splits


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("generate", "evaluate", "run"), default="run"
    )
    parser.add_argument("--basis-set-id", default=DEFAULT_BASIS_SET)
    parser.add_argument(
        "--evaluation-basis-set-id",
        help="basis to execute; defaults to the frozen corpus source basis",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/basis-library-pilot")
    )
    parser.add_argument("--corpus-dir", type=Path)
    parser.add_argument(
        "--discovery-size", type=int, default=DEFAULT_SPLIT_SIZES["discovery"]
    )
    parser.add_argument(
        "--validation-size", type=int, default=DEFAULT_SPLIT_SIZES["validation"]
    )
    parser.add_argument("--test-size", type=int, default=DEFAULT_SPLIT_SIZES["test"])
    parser.add_argument(
        "--discovery-seed", type=int, default=DEFAULT_SPLIT_SEEDS["discovery"]
    )
    parser.add_argument(
        "--validation-seed", type=int, default=DEFAULT_SPLIT_SEEDS["validation"]
    )
    parser.add_argument("--test-seed", type=int, default=DEFAULT_SPLIT_SEEDS["test"])
    parser.add_argument("--evaluation-seed", type=int, default=4_242)
    parser.add_argument("--max-attempts-per-record", type=int, default=1_000)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10_000,
        help="report accepted/processed tasks at this interval; 0 disables",
    )
    parser.add_argument("--eval-splits", type=_parse_splits, default=_SPLIT_ORDER)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--budget", type=int, default=1)
    parser.add_argument("--model", default=str(DEFAULT_LOCAL_CHECKPOINT))
    parser.add_argument("--model-revision", default=DEFAULT_HF_REVISION)
    parser.add_argument(
        "--deterministic-decoding",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--top-subsequences", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-entity", default=PERSONAL_WANDB_ENTITY)
    parser.add_argument("--wandb-project", default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--wandb-run-name")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_personal_wandb_entity(args.wandb_entity)
    corpus_dir = args.corpus_dir or args.output_dir / "corpus"
    artifact_paths: list[Path] = []
    aggregate: dict[str, Any] | None = None

    if args.mode in {"generate", "run"}:
        _, manifest_path = generate_corpus(
            basis_set_id=args.basis_set_id,
            split_sizes={
                "discovery": args.discovery_size,
                "validation": args.validation_size,
                "test": args.test_size,
            },
            split_seeds={
                "discovery": args.discovery_seed,
                "validation": args.validation_seed,
                "test": args.test_seed,
            },
            output_dir=corpus_dir,
            max_attempts_per_record=args.max_attempts_per_record,
            progress_every=args.progress_every,
            overwrite=args.overwrite,
        )
        artifact_paths.append(manifest_path)

    if args.mode in {"evaluate", "run"}:
        aggregate, evaluation_paths = evaluate_corpus(
            corpus_dir=corpus_dir,
            output_dir=args.output_dir / "evaluation",
            splits=args.eval_splits,
            evaluation_basis_set_id=args.evaluation_basis_set_id,
            model=args.model,
            model_revision=args.model_revision,
            batch_size=args.batch_size,
            budget=args.budget,
            deterministic_decoding=args.deterministic_decoding,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            seed=args.evaluation_seed,
            device=args.device,
            top_subsequences=args.top_subsequences,
            progress_every=args.progress_every,
            overwrite=args.overwrite,
        )
        artifact_paths.extend(evaluation_paths)

    if args.wandb:
        # Make every W&B artifact independently recoverable: evaluation outputs
        # alone are not enough to reconstruct the frozen task pairs. Revalidate
        # the complete corpus before attaching its manifest and all split files.
        corpus_manifest, _ = load_corpus(corpus_dir)
        artifact_paths.extend(
            [corpus_dir / "manifest.json"]
            + [
                corpus_dir / corpus_manifest["splits"][split]["path"]
                for split in _SPLIT_ORDER
            ]
        )
        artifact_paths = list(dict.fromkeys(path.resolve() for path in artifact_paths))
        run_url = log_artifacts_to_wandb(
            aggregate=aggregate,
            paths=artifact_paths,
            entity=args.wandb_entity,
            project=args.wandb_project,
            run_name=args.wandb_run_name,
        )
        print(f"W&B run: {run_url}")
    print(f"Pilot outputs: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
