"""Successful-solution usage statistics for basis functions.

The executor's ``FunctionDef.usage_count`` is intentionally a raw execution
counter: search procedures can increment it for candidates that never become a
solution.  This module records a narrower signal, only after a solver has
verified that an entire trajectory produces the requested output.
"""

import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Self

from wandering_light.function_def import FunctionDef


@dataclass
class FunctionUsageStats:
    """Usage of one function across successfully verified trajectories."""

    solution_count: int = 0
    invocation_count: int = 0
    last_used_solve: int | None = None


class FunctionUsageTracker:
    """Accumulate and persist basis-function use from verified solutions.

    ``invocation_count`` is the total occurrence count, including repeated
    applications within one trajectory.  ``solution_count`` counts a function
    at most once per trajectory.  ``successful_solve_count`` is the denominator
    for usage rates and includes successful identity (empty-path) solutions.

    Entries use the stable identifier emitted by a basis registry when one is
    present in ``FunctionDef.metadata``.  Name lookup remains supported for
    older callers and legacy function definitions.
    """

    FILE_NAME = "function_usage.json"
    SERIALIZATION_VERSION = 2
    LEGACY_SERIALIZATION_VERSION = 1

    def __init__(
        self,
        basis_set_id: str | None = None,
        basis_digest: str | None = None,
    ) -> None:
        self.basis_set_id = self._validate_provenance_value(
            "basis_set_id", basis_set_id
        )
        self.basis_digest = self._validate_provenance_value(
            "basis_digest", basis_digest
        )
        self._require_complete_provenance(
            self.basis_set_id, self.basis_digest, context="tracker"
        )
        self.successful_solve_count = 0
        self._stats: dict[str, FunctionUsageStats] = {}
        self._function_names: dict[str, str] = {}

    @staticmethod
    def function_identifier(function: FunctionDef) -> str:
        """Return the best available stable identifier for ``function``."""
        metadata = function.metadata or {}
        for key in ("basis_function_id", "basis_function_fingerprint"):
            value = metadata.get(key)
            if value is None:
                continue
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"Function metadata {key!r} must be a non-empty string"
                )
            return value

        if not function.name:
            raise ValueError(
                "Function name cannot be empty when no stable ID is present"
            )
        return function.name

    @classmethod
    def basis_provenance(
        cls, function_defs: Iterable[FunctionDef]
    ) -> tuple[str | None, str | None]:
        """Infer consistent basis provenance from function metadata.

        The registry calls its manifest hash ``basis_set_digest`` while this
        artifact calls the same value ``basis_digest``. Both metadata spellings
        are accepted without importing the registry implementation.
        """
        provenances: list[tuple[str | None, str | None]] = []
        for function in function_defs:
            metadata = function.metadata or {}
            basis_set_id = metadata.get("basis_set_id")
            set_digest = metadata.get("basis_set_digest")
            artifact_digest = metadata.get("basis_digest")
            if (
                set_digest is not None
                and artifact_digest is not None
                and set_digest != artifact_digest
            ):
                raise ValueError(
                    f"Function {function.name!r} has conflicting basis digests"
                )
            basis_digest = set_digest if set_digest is not None else artifact_digest
            validated_id = cls._validate_provenance_value("basis_set_id", basis_set_id)
            validated_digest = cls._validate_provenance_value(
                "basis_digest", basis_digest
            )
            cls._require_complete_provenance(
                validated_id,
                validated_digest,
                context=f"function {function.name!r}",
            )
            provenances.append((validated_id, validated_digest))

        if not provenances:
            return None, None
        unique = set(provenances)
        if len(unique) != 1:
            raise ValueError("Functions contain mixed or incomplete basis provenance")
        return unique.pop()

    def bind_basis(self, basis_set_id: str | None, basis_digest: str | None) -> None:
        """Bind or verify run provenance without recording a successful solve."""
        self._bind_or_check_provenance(basis_set_id, basis_digest)

    def record_solution(
        self,
        function_defs: Iterable[FunctionDef],
        *,
        basis_set_id: str | None = None,
        basis_digest: str | None = None,
    ) -> None:
        """Record the functions in one successfully verified trajectory.

        Optional provenance can bind an empty tracker or check an already bound
        tracker.  This lets callers integrate with a basis registry without
        making this low-level module import that registry.
        """
        functions = list(function_defs)
        if basis_set_id is None and basis_digest is None:
            basis_set_id, basis_digest = self.basis_provenance(functions)
        self._bind_or_check_provenance(basis_set_id, basis_digest)

        identifiers = [self.function_identifier(function) for function in functions]
        names_by_identifier: dict[str, str] = {}
        for identifier, function in zip(identifiers, functions, strict=True):
            previous_name = names_by_identifier.setdefault(identifier, function.name)
            if previous_name != function.name:
                raise ValueError(
                    "Stable function identifier maps to multiple names in one "
                    f"solution: {identifier!r}"
                )
            self._check_function_name(identifier, function.name)

        self.successful_solve_count += 1
        invocation_counts = Counter(identifiers)
        for identifier, invocation_count in invocation_counts.items():
            self._function_names.setdefault(identifier, names_by_identifier[identifier])
            stats = self._stats.setdefault(identifier, FunctionUsageStats())
            stats.solution_count += 1
            stats.invocation_count += invocation_count
            stats.last_used_solve = self.successful_solve_count

    def get(self, function: FunctionDef | str) -> FunctionUsageStats:
        """Return a copy of one function's statistics, or zeroes if unused.

        Strings are interpreted as stable identifiers first, then as legacy
        function names.  An ambiguous name is rejected instead of silently
        combining unrelated stable functions.
        """
        identifier = (
            function
            if isinstance(function, str)
            else self.function_identifier(function)
        )
        stats = self._stats.get(identifier)
        if stats is None and isinstance(function, str):
            matching_identifiers = [
                stable_id
                for stable_id, function_name in self._function_names.items()
                if function_name == function
            ]
            if len(matching_identifiers) > 1:
                raise ValueError(f"Ambiguous function name: {function!r}")
            if matching_identifiers:
                stats = self._stats[matching_identifiers[0]]

        return FunctionUsageStats(**asdict(stats or FunctionUsageStats()))

    def usage_rate(self, function: FunctionDef | str) -> float:
        """Return the fraction of successful solves that used the function."""
        if self.successful_solve_count == 0:
            return 0.0
        return self.get(function).solution_count / self.successful_solve_count

    def unused_functions(
        self, available_functions: Iterable[FunctionDef]
    ) -> list[FunctionDef]:
        """Return available functions that have never appeared in a solution."""
        return [
            function
            for function in available_functions
            if self.get(function).solution_count == 0
        ]

    def least_used_functions(
        self,
        available_functions: Iterable[FunctionDef],
        limit: int | None = None,
    ) -> list[FunctionDef]:
        """Rank functions from least to most used for pruning analysis.

        Total occurrences are the primary measure. Ties are resolved by
        successful-solution coverage, last-use sequence, name, then stable
        identifier so results are deterministic.
        """
        functions = list(available_functions)
        functions.sort(key=self._least_used_sort_key)
        if limit is None:
            return functions
        if limit < 0:
            raise ValueError("limit must be non-negative")
        return functions[:limit]

    def snapshot(self) -> dict[str, FunctionUsageStats]:
        """Return a copy of all statistics, keyed by stable identifier."""
        return {identifier: self.get(identifier) for identifier in sorted(self._stats)}

    def merge(self, other: "FunctionUsageTracker") -> Self:
        """Append another tracker's observations, rejecting mixed bases.

        The receiver's solves are considered earlier than ``other``'s solves,
        so last-use sequence numbers from ``other`` are offset accordingly.
        """
        if not isinstance(other, FunctionUsageTracker):
            raise TypeError("Can only merge another FunctionUsageTracker")
        self._check_merge_provenance(other)

        offset = self.successful_solve_count
        for identifier, other_stats in other._stats.items():
            other_name = other._function_names.get(identifier, identifier)
            self._check_function_name(identifier, other_name)
            self._function_names.setdefault(identifier, other_name)

            stats = self._stats.setdefault(identifier, FunctionUsageStats())
            stats.solution_count += other_stats.solution_count
            stats.invocation_count += other_stats.invocation_count
            if other_stats.last_used_solve is not None:
                stats.last_used_solve = offset + other_stats.last_used_solve

        self.successful_solve_count += other.successful_solve_count
        return self

    def to_dict(self) -> dict[str, Any]:
        """Return the version-2 JSON representation of this tracker."""
        return {
            "version": self.SERIALIZATION_VERSION,
            "basis_set_id": self.basis_set_id,
            "basis_digest": self.basis_digest,
            "successful_solve_count": self.successful_solve_count,
            "functions": {
                identifier: {
                    "function_name": self._function_names.get(identifier, identifier),
                    **asdict(stats),
                }
                for identifier, stats in sorted(self._stats.items())
            },
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        basis_set_id: str | None = None,
        basis_digest: str | None = None,
    ) -> "FunctionUsageTracker":
        """Restore v1 or v2 data and optionally require exact provenance."""
        version = data.get("version")
        if isinstance(version, bool) or version not in {
            cls.LEGACY_SERIALIZATION_VERSION,
            cls.SERIALIZATION_VERSION,
        }:
            raise ValueError(f"Unsupported function usage version: {version}")

        stored_basis_set_id = data.get("basis_set_id") if version == 2 else None
        stored_basis_digest = data.get("basis_digest") if version == 2 else None
        tracker = cls(stored_basis_set_id, stored_basis_digest)
        tracker._require_expected_provenance(basis_set_id, basis_digest)

        tracker.successful_solve_count = cls._non_negative_int(
            "successful_solve_count", data.get("successful_solve_count", 0)
        )
        functions = data.get("functions", {})
        if not isinstance(functions, dict):
            raise ValueError("functions must be a JSON object")

        for identifier, raw_stats in functions.items():
            if not isinstance(identifier, str) or not identifier:
                raise ValueError("Function usage identifiers must be non-empty strings")
            if not isinstance(raw_stats, dict):
                raise ValueError(f"Invalid statistics for function {identifier!r}")

            function_name = (
                raw_stats.get("function_name", identifier)
                if version == cls.SERIALIZATION_VERSION
                else identifier
            )
            if not isinstance(function_name, str) or not function_name:
                raise ValueError(f"Invalid function name for {identifier!r}")

            stats = FunctionUsageStats(
                solution_count=cls._non_negative_int(
                    "solution_count", raw_stats.get("solution_count", 0)
                ),
                invocation_count=cls._non_negative_int(
                    "invocation_count", raw_stats.get("invocation_count", 0)
                ),
                last_used_solve=cls._optional_non_negative_int(
                    "last_used_solve", raw_stats.get("last_used_solve")
                ),
            )
            cls._validate_stats(identifier, stats, tracker.successful_solve_count)
            tracker._stats[identifier] = stats
            tracker._function_names[identifier] = function_name

        return tracker

    def save(self, path: str | Path) -> Path:
        """Save the tracker as JSON and return the resulting path."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8"
        )
        return output_path

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        basis_set_id: str | None = None,
        basis_digest: str | None = None,
    ) -> "FunctionUsageTracker":
        """Load usage JSON and optionally require exact basis provenance."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Function usage file must contain a JSON object")
        return cls.from_dict(data, basis_set_id=basis_set_id, basis_digest=basis_digest)

    def _least_used_sort_key(
        self, function: FunctionDef
    ) -> tuple[int, int, int, str, str]:
        stats = self.get(function)
        last_used_solve = (
            stats.last_used_solve if stats.last_used_solve is not None else -1
        )
        return (
            stats.invocation_count,
            stats.solution_count,
            last_used_solve,
            function.name,
            self.function_identifier(function),
        )

    def _bind_or_check_provenance(
        self,
        basis_set_id: str | None,
        basis_digest: str | None,
    ) -> None:
        requested = (
            self._validate_provenance_value("basis_set_id", basis_set_id),
            self._validate_provenance_value("basis_digest", basis_digest),
        )
        self._require_complete_provenance(*requested, context="requested")
        if requested == (None, None):
            return

        current = (self.basis_set_id, self.basis_digest)
        if current == (None, None) and self.successful_solve_count == 0:
            self.basis_set_id, self.basis_digest = requested
            return
        if current != requested:
            raise ValueError(
                "Incompatible basis provenance: "
                f"tracker={current!r}, requested={requested!r}"
            )

    def _require_expected_provenance(
        self,
        basis_set_id: str | None,
        basis_digest: str | None,
    ) -> None:
        expected = (
            self._validate_provenance_value("basis_set_id", basis_set_id),
            self._validate_provenance_value("basis_digest", basis_digest),
        )
        self._require_complete_provenance(*expected, context="expected")
        stored = (self.basis_set_id, self.basis_digest)
        for label, expected_value, stored_value in zip(
            ("basis_set_id", "basis_digest"), expected, stored, strict=True
        ):
            if expected_value is not None and expected_value != stored_value:
                raise ValueError(
                    f"Incompatible {label}: expected {expected_value!r}, "
                    f"found {stored_value!r}"
                )

    def _check_merge_provenance(self, other: "FunctionUsageTracker") -> None:
        current = (self.basis_set_id, self.basis_digest)
        incoming = (other.basis_set_id, other.basis_digest)
        if current == incoming:
            return
        if (
            current == (None, None)
            and self.successful_solve_count == 0
            and not self._stats
        ):
            self.basis_set_id, self.basis_digest = incoming
            return
        if (
            incoming == (None, None)
            and other.successful_solve_count == 0
            and not other._stats
        ):
            return
        raise ValueError(
            "Cannot merge function usage from incompatible basis provenance: "
            f"{current!r} != {incoming!r}"
        )

    def _check_function_name(self, identifier: str, function_name: str) -> None:
        existing_name = self._function_names.get(identifier)
        if existing_name is not None and existing_name != function_name:
            raise ValueError(
                f"Stable function identifier {identifier!r} changed name from "
                f"{existing_name!r} to {function_name!r}"
            )

    @staticmethod
    def _validate_provenance_value(label: str, value: str | None) -> str | None:
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"{label} must be a non-empty string or None")
        return value

    @staticmethod
    def _require_complete_provenance(
        basis_set_id: str | None,
        basis_digest: str | None,
        *,
        context: str,
    ) -> None:
        if (basis_set_id is None) != (basis_digest is None):
            raise ValueError(
                f"{context} basis_set_id and basis_digest must be provided together"
            )

    @staticmethod
    def _non_negative_int(label: str, value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{label} must be a non-negative integer")
        return value

    @classmethod
    def _optional_non_negative_int(cls, label: str, value: Any) -> int | None:
        if value is None:
            return None
        return cls._non_negative_int(label, value)

    @staticmethod
    def _validate_stats(
        identifier: str,
        stats: FunctionUsageStats,
        successful_solve_count: int,
    ) -> None:
        if stats.solution_count > successful_solve_count:
            raise ValueError(
                f"solution_count for {identifier!r} exceeds successful solves"
            )
        if stats.invocation_count < stats.solution_count:
            raise ValueError(
                f"invocation_count for {identifier!r} is below solution_count"
            )
        if stats.solution_count == 0 and stats.invocation_count != 0:
            raise ValueError(f"unused function {identifier!r} cannot have invocations")
        if stats.last_used_solve is not None and not (
            1 <= stats.last_used_solve <= successful_solve_count
        ):
            raise ValueError(
                f"last_used_solve for {identifier!r} is outside the solve sequence"
            )
        if stats.solution_count == 0 and stats.last_used_solve is not None:
            raise ValueError(
                f"unused function {identifier!r} cannot have last_used_solve"
            )
