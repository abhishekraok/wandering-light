"""Disk-backed catalog for interactive corpus exploration.

Corpus files are gzip JSONL, so selecting one record by repeatedly scanning the
compressed stream is slow while materialising every ``BasisTaskRecord`` is
surprisingly memory hungry.  This module streams each source once into a small
SQLite projection.  Streamlit can then compare, filter, and page through large
corpora without retaining the corpus as Python objects.

Both released corpus schemas are normalized:

* basis-bound ``BasisTaskRecord`` rows (schema version 2), and
* the older shortest-path rows with ``relabeled_functions`` (schema version 1).

The original row is retained as JSON for the one selected task.  Function
associations are normalized into a separate table so filters can distinguish a
witness function from an optimal first or last action.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import sqlite3
import zlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path, PureWindowsPath
from typing import TYPE_CHECKING, Any, Literal
from uuid import uuid4

from wandering_light.basis_dataset import SCHEMA_VERSION as BASIS_TASK_SCHEMA_VERSION
from wandering_light.basis_dataset import (
    BasisTaskRecord,
    task_id_for,
    typed_list_from_builtin_str,
)
from wandering_light.basis_set import BasisSetError, load_basis_set

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

INDEX_SCHEMA_VERSION = 3
FUNCTION_ROLES = ("witness", "optimal_first", "optimal_last")
_QUERYABLE_COLUMNS = frozenset(
    {"split", "distance", "input_type", "output_type", "certification", "root_index"}
)
_INSERT_BATCH_SIZE = 2_000


class CorpusIndexError(ValueError):
    """Raised when a corpus source cannot be safely normalized or indexed."""


@dataclass(frozen=True, slots=True)
class CorpusFile:
    path: Path
    split_hint: str | None = None
    expected_sha256: str | None = None
    expected_records: int | None = None


@dataclass(frozen=True, slots=True)
class CorpusSource:
    """One logical corpus, backed by one or more gzip JSONL files."""

    path: Path
    name: str
    files: tuple[CorpusFile, ...]
    manifest_path: Path | None = None
    missing_files: tuple[Path, ...] = ()
    basis_set_id: str | None = None
    basis_set_digest: str | None = None
    expected_records: int | None = None
    hub_repo_id: str | None = None
    hub_revision: str | None = None

    @property
    def ready(self) -> bool:
        return not self.missing_files and bool(self.files)


@dataclass(frozen=True, slots=True)
class IndexProgress:
    source_name: str
    file_index: int
    file_count: int
    file_path: Path
    line_number: int
    records_indexed: int


@dataclass(frozen=True, slots=True)
class CorpusFilters:
    splits: tuple[str, ...] = ()
    min_distance: int | None = None
    max_distance: int | None = None
    input_types: tuple[str, ...] = ()
    output_types: tuple[str, ...] = ()
    root_indices: tuple[int, ...] = ()
    function_keys: tuple[str, ...] = ()
    function_match: Literal["any", "all"] = "any"
    function_roles: tuple[str, ...] = ("witness",)

    def __post_init__(self) -> None:
        if self.function_match not in ("any", "all"):
            raise ValueError("function_match must be 'any' or 'all'")
        if (
            self.min_distance is not None
            and self.max_distance is not None
            and self.min_distance > self.max_distance
        ):
            raise ValueError("min_distance must not exceed max_distance")
        unknown_roles = set(self.function_roles) - set(FUNCTION_ROLES)
        if unknown_roles:
            raise ValueError(f"unknown function roles: {sorted(unknown_roles)}")


@dataclass(frozen=True, slots=True)
class CorpusStats:
    records: int
    certified_records: int
    min_distance: int | None
    max_distance: int | None
    mean_distance: float | None
    roots: int


@dataclass(frozen=True, slots=True)
class FunctionCount:
    function_key: str
    function_name: str
    role: str
    records: int


@dataclass(frozen=True, slots=True)
class RecordSummary:
    row_id: int
    task_id: str
    split: str
    distance: int | None
    input_type: str
    output_type: str
    certified: bool | None
    witness_function_names: tuple[str, ...]
    input_preview: str
    output_preview: str
    root_index: int | None


@dataclass(frozen=True, slots=True)
class RecordDetail:
    row_id: int
    task_id: str
    schema_kind: str
    split: str
    input: str
    output: str | None
    input_type: str
    output_type: str
    distance: int | None
    certified: bool | None
    witness_function_names: tuple[str, ...]
    witness_function_ids: tuple[str, ...]
    basis_set_id: str | None
    basis_set_digest: str | None
    generator: str | None
    source_index: int | None
    root_index: int | None
    certification: str | None
    metadata: dict[str, Any]
    raw: dict[str, Any]
    functions_by_role: dict[str, tuple[str, ...]]


def _bare_filename(value: Any) -> str:
    if not isinstance(value, str) or value in ("", "."):
        raise CorpusIndexError(f"unsafe corpus filename: {value!r}")
    if Path(value).name != value or PureWindowsPath(value).name != value:
        raise CorpusIndexError(f"unsafe corpus filename: {value!r}")
    return value


def _strip_jsonl_suffix(path: Path) -> str:
    suffix = ".jsonl.gz"
    return path.name[: -len(suffix)] if path.name.endswith(suffix) else path.stem


def corpus_source(path: str | Path) -> CorpusSource:
    """Resolve a standalone gzip file or a manifest-backed corpus directory."""
    candidate = Path(path).expanduser().resolve()
    if candidate.name == "manifest.json":
        candidate = candidate.parent

    if candidate.is_file():
        if not candidate.name.endswith(".jsonl.gz"):
            raise CorpusIndexError(f"not a corpus JSONL gzip file: {candidate}")
        if (candidate.parent / "manifest.json").is_file():
            # A sibling manifest is authoritative even when the caller points
            # directly at one split; retain its digest/provenance checks.
            return corpus_source(candidate.parent)
        return CorpusSource(
            path=candidate,
            name=_strip_jsonl_suffix(candidate),
            files=(CorpusFile(candidate),),
        )

    if not candidate.is_dir():
        raise FileNotFoundError(f"corpus source does not exist: {candidate}")

    manifest_path = candidate / "manifest.json"
    if not manifest_path.exists():
        files = tuple(CorpusFile(item) for item in sorted(candidate.glob("*.jsonl.gz")))
        if not files:
            raise FileNotFoundError(f"no corpus files found under {candidate}")
        return CorpusSource(path=candidate, name=candidate.name, files=files)

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CorpusIndexError(f"could not read {manifest_path}: {error}") from error
    if not isinstance(manifest, dict) or not isinstance(manifest.get("splits"), dict):
        raise CorpusIndexError(f"{manifest_path} has no object-valued 'splits'")
    _validate_manifest_digest(manifest, manifest_path)

    files: list[CorpusFile] = []
    missing: list[Path] = []
    for split, metadata in manifest["splits"].items():
        if not isinstance(split, str) or not isinstance(metadata, dict):
            raise CorpusIndexError(f"invalid split metadata in {manifest_path}")
        filename = _bare_filename(metadata.get("path"))
        expected_sha256 = _optional_sha256(metadata.get("sha256"))
        expected_records = _optional_nonnegative_int(metadata.get("size"))
        split_path = candidate / filename
        if split_path.exists():
            resolved_split = split_path.resolve()
            if not resolved_split.is_relative_to(candidate):
                raise CorpusIndexError(
                    f"corpus split escapes its manifest directory: {split_path}"
                )
            files.append(
                CorpusFile(
                    resolved_split,
                    split,
                    expected_sha256,
                    expected_records,
                )
            )
        else:
            missing.append(split_path)

    hub = manifest.get("hub") if isinstance(manifest.get("hub"), dict) else {}
    expected_records = _optional_nonnegative_int(manifest.get("global_task_count"))
    split_counts = [item.expected_records for item in files]
    if (
        expected_records is not None
        and len(files) == len(manifest["splits"])
        and all(count is not None for count in split_counts)
        and sum(split_counts) != expected_records
    ):
        raise CorpusIndexError(
            f"{manifest_path} global_task_count does not equal its split sizes"
        )
    return CorpusSource(
        path=candidate,
        name=candidate.name,
        files=tuple(files),
        manifest_path=manifest_path,
        missing_files=tuple(missing),
        basis_set_id=_optional_str(manifest.get("basis_set_id")),
        basis_set_digest=_optional_str(manifest.get("basis_set_digest")),
        expected_records=expected_records,
        hub_repo_id=_optional_str(hub.get("repo_id")),
        hub_revision=_optional_str(hub.get("revision")),
    )


def discover_corpus_sources(
    roots: Sequence[str | Path],
    *,
    on_error: Callable[[Path, Exception], None] | None = None,
) -> tuple[CorpusSource, ...]:
    """Find manifest corpora and standalone gzip JSONL files under ``roots``."""
    discovered: list[CorpusSource] = []
    owned_files: set[Path] = set()
    seen_sources: set[Path] = set()
    manifest_directories: set[Path] = set()

    for raw_root in roots:
        root = Path(raw_root).expanduser().resolve()
        if not root.exists():
            continue
        manifests = [root] if (root / "manifest.json").is_file() else []
        manifests.extend(path.parent for path in sorted(root.rglob("manifest.json")))
        for directory in manifests:
            resolved = directory.resolve()
            manifest_directories.add(resolved)
            if resolved in seen_sources:
                continue
            try:
                source = corpus_source(resolved)
            except (CorpusIndexError, FileNotFoundError) as error:
                if on_error is not None:
                    on_error(resolved, error)
                continue
            discovered.append(source)
            seen_sources.add(resolved)
            owned_files.update(item.path.resolve() for item in source.files)
            owned_files.update(item.resolve() for item in source.missing_files)

    for raw_root in roots:
        root = Path(raw_root).expanduser().resolve()
        if not root.exists():
            continue
        paths = [root] if root.is_file() else sorted(root.rglob("*.jsonl.gz"))
        for path in paths:
            resolved = path.resolve()
            lexical = path.absolute()
            if any(
                lexical.is_relative_to(directory) for directory in manifest_directories
            ):
                # A manifest is authoritative even when it is invalid. Never
                # downgrade its split files to unverified standalone sources.
                continue
            if resolved in owned_files or resolved in seen_sources:
                continue
            try:
                source = corpus_source(resolved)
            except (CorpusIndexError, FileNotFoundError) as error:
                if on_error is not None:
                    on_error(resolved, error)
                continue
            discovered.append(source)
            seen_sources.add(resolved)

    return tuple(sorted(discovered, key=lambda source: (source.name, str(source.path))))


def source_signature(source: CorpusSource) -> str:
    """Return a cheap invalidation signature for the files behind ``source``."""
    payload: list[dict[str, Any]] = [
        {"source": str(source.path.resolve()), "schema": INDEX_SCHEMA_VERSION}
    ]
    paths = [item.path for item in source.files]
    if source.manifest_path is not None:
        paths.append(source.manifest_path)
    for path in sorted(paths):
        stat = path.stat()
        corpus_file = next(
            (item for item in source.files if item.path.resolve() == path.resolve()),
            None,
        )
        payload.append(
            {
                "path": str(path.resolve()),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "expected_sha256": (
                    corpus_file.expected_sha256 if corpus_file is not None else None
                ),
                "expected_records": (
                    corpus_file.expected_records if corpus_file is not None else None
                ),
            }
        )
        if source.manifest_path is not None and path == source.manifest_path:
            payload[-1]["content_sha256"] = _sha256_file(path)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def default_index_cache_dir() -> Path:
    configured = os.environ.get("XDG_CACHE_HOME")
    root = Path(configured).expanduser() if configured else Path.home() / ".cache"
    return root / "wandering-light" / "corpus-index"


def ensure_corpus_index(
    source: CorpusSource,
    *,
    cache_dir: str | Path | None = None,
    progress: Callable[[IndexProgress], None] | None = None,
) -> CorpusIndex:
    """Open a valid cached index, or stream-build it atomically when stale."""
    if not source.ready:
        missing = ", ".join(str(path) for path in source.missing_files) or "no files"
        raise FileNotFoundError(f"corpus {source.name!r} is incomplete: {missing}")
    signature = source_signature(source)
    cache_root = (
        Path(cache_dir).expanduser()
        if cache_dir is not None
        else default_index_cache_dir()
    )
    cache_root.mkdir(parents=True, exist_ok=True)
    source_key = hashlib.sha256(str(source.path.resolve()).encode()).hexdigest()[:24]
    database_path = cache_root / f"{source_key}.sqlite3"
    index = CorpusIndex(database_path, source)
    if index.is_valid(signature):
        return index

    temporary = database_path.with_name(
        f".{database_path.name}.{os.getpid()}.{uuid4().hex}.tmp"
    )
    try:
        _build_index(source, temporary, signature=signature, progress=progress)
        os.replace(temporary, database_path)
    finally:
        # Also clean up cancellation/KeyboardInterrupt paths; a partial index
        # for a large corpus can otherwise leave hundreds of megabytes behind.
        temporary.unlink(missing_ok=True)
    return CorpusIndex(database_path, source)


class CorpusIndex:
    """Read-only query facade over a derived SQLite corpus projection."""

    def __init__(self, database_path: str | Path, source: CorpusSource):
        self.database_path = Path(database_path)
        self.source = source

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def is_valid(self, signature: str | None = None) -> bool:
        if not self.database_path.is_file():
            return False
        try:
            with self._connect() as connection:
                rows = dict(connection.execute("SELECT key, value FROM index_meta"))
            return rows.get("index_schema_version") == str(
                INDEX_SCHEMA_VERSION
            ) and rows.get("source_signature") == (
                signature if signature is not None else source_signature(self.source)
            )
        except sqlite3.Error:
            return False

    def stats(self, filters: CorpusFilters | None = None) -> CorpusStats:
        filters = filters or CorpusFilters()
        where, parameters = self._where(filters)
        query = f"""
            SELECT
                COUNT(*) AS records,
                COALESCE(SUM(certified = 1), 0) AS certified_records,
                MIN(distance) AS min_distance,
                MAX(distance) AS max_distance,
                AVG(distance) AS mean_distance,
                COUNT(DISTINCT root_index) AS roots
            FROM records
            {where}
        """
        with self._connect() as connection:
            row = connection.execute(query, parameters).fetchone()
        return CorpusStats(
            records=row["records"],
            certified_records=row["certified_records"],
            min_distance=row["min_distance"],
            max_distance=row["max_distance"],
            mean_distance=row["mean_distance"],
            roots=row["roots"],
        )

    def values(self, column: str) -> tuple[Any, ...]:
        if column not in _QUERYABLE_COLUMNS:
            raise ValueError(f"unsupported corpus column: {column}")
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT DISTINCT {column} AS value FROM records "
                f"WHERE {column} IS NOT NULL ORDER BY {column}"
            )
        return tuple(row["value"] for row in rows)

    def counts(
        self, column: str, filters: CorpusFilters | None = None
    ) -> tuple[tuple[Any, int], ...]:
        filters = filters or CorpusFilters()
        if column not in _QUERYABLE_COLUMNS:
            raise ValueError(f"unsupported corpus column: {column}")
        where, parameters = self._where(filters)
        null_clause = "WHERE" if not where else "AND"
        query = f"""
            SELECT {column} AS value, COUNT(*) AS records
            FROM records
            {where}
            {null_clause} {column} IS NOT NULL
            GROUP BY {column}
            ORDER BY {column}
        """
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return tuple((row["value"], row["records"]) for row in rows)

    def function_counts(
        self,
        *,
        roles: Sequence[str] = ("witness",),
        limit: int | None = None,
    ) -> tuple[FunctionCount, ...]:
        role_values = tuple(dict.fromkeys(roles))
        unknown = set(role_values) - set(FUNCTION_ROLES)
        if unknown:
            raise ValueError(f"unknown function roles: {sorted(unknown)}")
        placeholders = ",".join("?" for _ in role_values)
        query = f"""
            SELECT functions.function_key, functions.function_name,
                   record_functions.role,
                   COUNT(DISTINCT record_functions.record_id) AS records
            FROM record_functions
            JOIN functions ON functions.id = record_functions.function_row_id
            WHERE record_functions.role IN ({placeholders})
            GROUP BY functions.id, record_functions.role
            ORDER BY records DESC, functions.function_name,
                     functions.function_key, record_functions.role
        """
        parameters: list[Any] = list(role_values)
        if limit is not None:
            query += " LIMIT ?"
            parameters.append(limit)
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return tuple(
            FunctionCount(
                function_key=row["function_key"],
                function_name=row["function_name"],
                role=row["role"],
                records=row["records"],
            )
            for row in rows
        )

    def function_totals(
        self,
        *,
        roles: Sequence[str] = ("witness",),
        limit: int | None = None,
    ) -> tuple[FunctionCount, ...]:
        """Count distinct associated tasks across roles without double-counting."""
        role_values = tuple(dict.fromkeys(roles))
        unknown = set(role_values) - set(FUNCTION_ROLES)
        if unknown:
            raise ValueError(f"unknown function roles: {sorted(unknown)}")
        placeholders = ",".join("?" for _ in role_values)
        query = f"""
            SELECT functions.function_key, functions.function_name,
                   COUNT(DISTINCT record_functions.record_id) AS records
            FROM record_functions
            JOIN functions ON functions.id = record_functions.function_row_id
            WHERE record_functions.role IN ({placeholders})
            GROUP BY functions.id
            ORDER BY records DESC, functions.function_name, functions.function_key
        """
        parameters: list[Any] = list(role_values)
        if limit is not None:
            query += " LIMIT ?"
            parameters.append(limit)
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return tuple(
            FunctionCount(
                function_key=row["function_key"],
                function_name=row["function_name"],
                role="associated",
                records=row["records"],
            )
            for row in rows
        )

    def count_records(self, filters: CorpusFilters | None = None) -> int:
        filters = filters or CorpusFilters()
        where, parameters = self._where(filters)
        with self._connect() as connection:
            row = connection.execute(
                f"SELECT COUNT(*) AS records FROM records {where}", parameters
            ).fetchone()
        return row["records"]

    def records(
        self,
        filters: CorpusFilters | None = None,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[RecordSummary, ...]:
        filters = filters or CorpusFilters()
        if limit <= 0 or limit > 1_000:
            raise ValueError("limit must be between 1 and 1000")
        if offset < 0:
            raise ValueError("offset must be non-negative")
        where, parameters = self._where(filters)
        query = f"""
            SELECT id, task_id, split, distance, input_type, output_type,
                   certified, witness_names_json, input_preview, output_preview,
                   root_index
            FROM records
            {where}
            ORDER BY COALESCE(distance, -1) DESC, split, id
            LIMIT ? OFFSET ?
        """
        with self._connect() as connection:
            rows = connection.execute(query, [*parameters, limit, offset]).fetchall()
        return tuple(
            RecordSummary(
                row_id=row["id"],
                task_id=row["task_id"],
                split=row["split"],
                distance=row["distance"],
                input_type=row["input_type"],
                output_type=row["output_type"],
                certified=(
                    None if row["certified"] is None else bool(row["certified"])
                ),
                witness_function_names=tuple(json.loads(row["witness_names_json"])),
                input_preview=row["input_preview"],
                output_preview=row["output_preview"],
                root_index=row["root_index"],
            )
            for row in rows
        )

    def find_task(
        self,
        task_id_prefix: str,
        *,
        filters: CorpusFilters | None = None,
    ) -> tuple[RecordSummary, ...]:
        prefix = task_id_prefix.strip()
        if not prefix:
            return ()
        escaped = prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        where, parameters = self._where(filters or CorpusFilters())
        filter_clause = where.removeprefix("WHERE ")
        if filter_clause:
            filter_clause = f"AND ({filter_clause})"
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT id FROM records
                WHERE task_id LIKE ? ESCAPE '\\'
                {filter_clause}
                ORDER BY task_id
                LIMIT 25
                """,
                [f"{escaped}%", *parameters],
            ).fetchall()
        return tuple(self._summary_by_id(row["id"]) for row in rows)

    def record_matches(self, row_id: int, filters: CorpusFilters) -> bool:
        """Return whether one row still belongs to the current filtered result."""
        where, parameters = self._where(filters)
        filter_clause = where.removeprefix("WHERE ")
        clauses = ["id = ?"]
        if filter_clause:
            clauses.append(f"({filter_clause})")
        with self._connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM records WHERE " + " AND ".join(clauses) + " LIMIT 1",
                [row_id, *parameters],
            ).fetchone()
        return row is not None

    def get_record(self, row_id: int) -> RecordDetail:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM records WHERE id = ?", (row_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown corpus row id {row_id}")
            function_rows = connection.execute(
                """
                SELECT record_functions.role, functions.function_key
                FROM record_functions
                JOIN functions ON functions.id = record_functions.function_row_id
                WHERE record_functions.record_id = ?
                ORDER BY record_functions.role, record_functions.ordinal,
                         functions.function_key
                """,
                (row_id,),
            ).fetchall()
        raw = json.loads(zlib.decompress(row["raw_json"]).decode("utf-8"))
        metadata = raw.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        is_basis_task = row["schema_kind"] == "basis-task-v2"
        by_role: dict[str, list[str]] = {role: [] for role in FUNCTION_ROLES}
        for function_row in function_rows:
            by_role[function_row["role"]].append(function_row["function_key"])
        return RecordDetail(
            row_id=row["id"],
            task_id=row["task_id"],
            schema_kind=row["schema_kind"],
            split=row["split"],
            input=raw["input"],
            output=raw.get("output"),
            input_type=row["input_type"],
            output_type=row["output_type"],
            distance=row["distance"],
            certified=(None if row["certified"] is None else bool(row["certified"])),
            witness_function_names=tuple(json.loads(row["witness_names_json"])),
            witness_function_ids=(
                tuple(raw["witness_function_ids"]) if is_basis_task else ()
            ),
            basis_set_id=raw["basis_set_id"] if is_basis_task else None,
            basis_set_digest=raw["basis_set_digest"] if is_basis_task else None,
            generator=(raw["generator"] if is_basis_task else "legacy-shortest-path"),
            source_index=raw.get("source_index"),
            root_index=row["root_index"],
            certification=row["certification"],
            metadata=metadata,
            raw=raw,
            functions_by_role={key: tuple(value) for key, value in by_role.items()},
        )

    def _summary_by_id(self, row_id: int) -> RecordSummary:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, task_id, split, distance, input_type, output_type,
                       certified, witness_names_json, input_preview, output_preview,
                       root_index
                FROM records WHERE id = ?
                """,
                (row_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown corpus row id {row_id}")
        return RecordSummary(
            row_id=row["id"],
            task_id=row["task_id"],
            split=row["split"],
            distance=row["distance"],
            input_type=row["input_type"],
            output_type=row["output_type"],
            certified=(None if row["certified"] is None else bool(row["certified"])),
            witness_function_names=tuple(json.loads(row["witness_names_json"])),
            input_preview=row["input_preview"],
            output_preview=row["output_preview"],
            root_index=row["root_index"],
        )

    @staticmethod
    def _where(filters: CorpusFilters) -> tuple[str, list[Any]]:
        clauses: list[str] = []
        parameters: list[Any] = []

        def add_membership(column: str, values: Sequence[Any]) -> None:
            if not values:
                return
            placeholders = ",".join("?" for _ in values)
            clauses.append(f"{column} IN ({placeholders})")
            parameters.extend(values)

        add_membership("split", filters.splits)
        add_membership("input_type", filters.input_types)
        add_membership("output_type", filters.output_types)
        add_membership("root_index", filters.root_indices)
        if filters.min_distance is not None:
            clauses.append("distance >= ?")
            parameters.append(filters.min_distance)
        if filters.max_distance is not None:
            clauses.append("distance <= ?")
            parameters.append(filters.max_distance)

        function_keys = tuple(dict.fromkeys(filters.function_keys))
        if function_keys:
            roles = filters.function_roles or FUNCTION_ROLES
            function_placeholders = ",".join("?" for _ in function_keys)
            role_placeholders = ",".join("?" for _ in roles)
            subquery = f"""
                id IN (
                    SELECT record_functions.record_id
                    FROM record_functions
                    JOIN functions
                      ON functions.id = record_functions.function_row_id
                    WHERE functions.function_key IN ({function_placeholders})
                      AND record_functions.role IN ({role_placeholders})
                    GROUP BY record_functions.record_id
            """
            parameters.extend(function_keys)
            parameters.extend(roles)
            if filters.function_match == "all":
                subquery += " HAVING COUNT(DISTINCT functions.function_key) = ?"
                parameters.append(len(function_keys))
            subquery += ")"
            clauses.append(subquery)

        if not clauses:
            return "", parameters
        return "WHERE " + " AND ".join(f"({clause})" for clause in clauses), parameters


def _build_index(
    source: CorpusSource,
    database_path: Path,
    *,
    signature: str,
    progress: Callable[[IndexProgress], None] | None,
) -> None:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path)
    try:
        connection.executescript(
            """
            PRAGMA journal_mode = OFF;
            PRAGMA synchronous = OFF;
            PRAGMA temp_store = FILE;
            PRAGMA foreign_keys = ON;

            CREATE TABLE index_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE records (
                id INTEGER PRIMARY KEY,
                file_path TEXT NOT NULL,
                line_number INTEGER NOT NULL,
                task_id TEXT NOT NULL,
                schema_kind TEXT NOT NULL,
                split TEXT NOT NULL,
                input_type TEXT NOT NULL,
                output_type TEXT NOT NULL,
                distance INTEGER,
                certified INTEGER,
                witness_names_json TEXT NOT NULL,
                root_index INTEGER,
                certification TEXT,
                raw_json BLOB NOT NULL,
                input_preview TEXT NOT NULL,
                output_preview TEXT NOT NULL,
                UNIQUE(file_path, line_number)
            );

            CREATE TABLE functions (
                id INTEGER PRIMARY KEY,
                function_key TEXT NOT NULL UNIQUE,
                function_name TEXT NOT NULL,
                basis_function_id TEXT
            );

            CREATE TABLE record_functions (
                record_id INTEGER NOT NULL REFERENCES records(id),
                function_row_id INTEGER NOT NULL REFERENCES functions(id),
                role TEXT NOT NULL,
                ordinal INTEGER NOT NULL,
                PRIMARY KEY(record_id, function_row_id, role, ordinal)
            ) WITHOUT ROWID;
            """
        )
        id_to_name = _basis_function_names(source)
        records_batch: list[tuple[Any, ...]] = []
        function_rows_batch: list[tuple[Any, ...]] = []
        functions_batch: list[tuple[Any, ...]] = []
        functions_by_key: dict[str, tuple[int, str, str | None]] = {}
        record_id = 0
        next_function_row_id = 0

        def intern_function(
            function_key: str, function_name: str, basis_function_id: str | None
        ) -> int:
            nonlocal next_function_row_id
            existing = functions_by_key.get(function_key)
            if existing is not None:
                row_id, existing_name, existing_basis_id = existing
                if (
                    existing_name != function_name
                    or existing_basis_id != basis_function_id
                ):
                    raise CorpusIndexError(
                        f"function key {function_key!r} has conflicting metadata"
                    )
                return row_id
            next_function_row_id += 1
            functions_by_key[function_key] = (
                next_function_row_id,
                function_name,
                basis_function_id,
            )
            function_rows_batch.append(
                (
                    next_function_row_id,
                    function_key,
                    function_name,
                    basis_function_id,
                )
            )
            return next_function_row_id

        def flush() -> None:
            if function_rows_batch:
                connection.executemany(
                    """
                    INSERT INTO functions(
                        id, function_key, function_name, basis_function_id
                    ) VALUES (?, ?, ?, ?)
                    """,
                    function_rows_batch,
                )
                function_rows_batch.clear()
            if records_batch:
                connection.executemany(
                    """
                    INSERT INTO records(
                        id, file_path, line_number, task_id, schema_kind, split,
                        input_type, output_type, distance, certified,
                        witness_names_json, root_index, certification, raw_json,
                        input_preview, output_preview
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    records_batch,
                )
                records_batch.clear()
            if functions_batch:
                connection.executemany(
                    """
                    INSERT INTO record_functions(
                        record_id, function_row_id, role, ordinal
                    ) VALUES (?, ?, ?, ?)
                    """,
                    functions_batch,
                )
                functions_batch.clear()

        for file_index, corpus_file in enumerate(source.files, start=1):
            if corpus_file.expected_sha256 is not None:
                actual_sha256 = _sha256_file(corpus_file.path)
                if actual_sha256 != corpus_file.expected_sha256:
                    raise CorpusIndexError(
                        f"digest mismatch for {corpus_file.path}: manifest says "
                        f"{corpus_file.expected_sha256}, file is {actual_sha256}"
                    )
            try:
                records_before = record_id
                with gzip.open(corpus_file.path, "rt", encoding="utf-8") as handle:
                    record_id = _index_file(
                        handle,
                        corpus_file=corpus_file,
                        file_index=file_index,
                        file_count=len(source.files),
                        source=source,
                        id_to_name=id_to_name,
                        records_batch=records_batch,
                        functions_batch=functions_batch,
                        intern_function=intern_function,
                        first_record_id=record_id,
                        flush=flush,
                        progress=progress,
                    )
            except OSError as error:
                raise CorpusIndexError(
                    f"could not read {corpus_file.path}: {error}"
                ) from error
            indexed_records = record_id - records_before
            if (
                corpus_file.expected_records is not None
                and indexed_records != corpus_file.expected_records
            ):
                raise CorpusIndexError(
                    f"record count mismatch for {corpus_file.path}: manifest says "
                    f"{corpus_file.expected_records}, file has {indexed_records}"
                )

        if source.expected_records is not None and record_id != source.expected_records:
            raise CorpusIndexError(
                f"record count mismatch for {source.name}: manifest says "
                f"{source.expected_records}, files have {record_id}"
            )

        flush()
        connection.executescript(
            """
            CREATE INDEX records_split_distance_idx ON records(split, distance);
            CREATE INDEX records_types_idx ON records(input_type, output_type);
            CREATE INDEX records_root_idx ON records(root_index);
            CREATE INDEX records_task_id_idx ON records(task_id);
            CREATE INDEX functions_name_idx
                ON functions(function_name, function_key);
            CREATE INDEX record_functions_function_role_idx
                ON record_functions(function_row_id, role, record_id);
            """
        )
        connection.executemany(
            "INSERT INTO index_meta(key, value) VALUES (?, ?)",
            (
                ("index_schema_version", str(INDEX_SCHEMA_VERSION)),
                ("source_signature", signature),
                ("source_name", source.name),
                ("records", str(record_id)),
            ),
        )
        connection.commit()
    finally:
        connection.close()


def _index_file(
    handle,
    *,
    corpus_file: CorpusFile,
    file_index: int,
    file_count: int,
    source: CorpusSource,
    id_to_name: dict[str, str],
    records_batch: list[tuple[Any, ...]],
    functions_batch: list[tuple[Any, ...]],
    intern_function: Callable[[str, str, str | None], int],
    first_record_id: int,
    flush: Callable[[], None],
    progress: Callable[[IndexProgress], None] | None,
) -> int:
    record_id = first_record_id
    line_number = 0
    for line_number, line in enumerate(handle, start=1):
        try:
            raw = json.loads(line)
            normalized = _normalize_record(
                raw,
                split_hint=corpus_file.split_hint,
                source=source,
                id_to_name=id_to_name,
                fallback_id=f"{file_index}:{line_number}",
            )
        except Exception as error:
            raise CorpusIndexError(
                f"invalid corpus record at {corpus_file.path}:{line_number}: {error}"
            ) from error
        record_id += 1
        records_batch.append(
            (
                record_id,
                str(corpus_file.path),
                line_number,
                normalized["task_id"],
                normalized["schema_kind"],
                normalized["split"],
                normalized["input_type"],
                normalized["output_type"],
                normalized["distance"],
                normalized["certified"],
                _json(normalized["witness_names"]),
                normalized["root_index"],
                normalized["certification"],
                sqlite3.Binary(zlib.compress(_json(raw).encode("utf-8"), level=1)),
                _serialized_preview(normalized["input"]),
                _serialized_preview(normalized["output"]),
            )
        )
        functions_batch.extend(
            (
                record_id,
                intern_function(function_key, name, function_id),
                role,
                ordinal,
            )
            for role, ordinal, function_key, name, function_id in normalized[
                "functions"
            ]
        )
        if len(records_batch) >= _INSERT_BATCH_SIZE:
            flush()
        if progress is not None and line_number % _INSERT_BATCH_SIZE == 0:
            progress(
                IndexProgress(
                    source_name=source.name,
                    file_index=file_index,
                    file_count=file_count,
                    file_path=corpus_file.path,
                    line_number=line_number,
                    records_indexed=record_id,
                )
            )
    if progress is not None:
        progress(
            IndexProgress(
                source_name=source.name,
                file_index=file_index,
                file_count=file_count,
                file_path=corpus_file.path,
                line_number=line_number,
                records_indexed=record_id,
            )
        )
    return record_id


def _normalize_record(
    raw: Any,
    *,
    split_hint: str | None,
    source: CorpusSource,
    id_to_name: dict[str, str],
    fallback_id: str,
) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise CorpusIndexError("record must be a JSON object")
    input_value = raw.get("input")
    output_value = raw.get("output")
    if not isinstance(input_value, str):
        raise CorpusIndexError("record input must be a serialized TypedList string")
    if output_value is not None and not isinstance(output_value, str):
        raise CorpusIndexError(
            "record output must be null or a serialized TypedList string"
        )

    metadata = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
    schema_version = raw.get("schema_version")
    basis_markers = {"basis_set_id", "witness_function_ids", "witness_function_names"}
    if schema_version == BASIS_TASK_SCHEMA_VERSION:
        is_basis_task = True
    elif schema_version in (None, 1) and not basis_markers.intersection(raw):
        is_basis_task = False
    elif schema_version in (None, 1):
        raise CorpusIndexError("record mixes legacy schema with basis-task fields")
    else:
        raise CorpusIndexError(f"unsupported corpus schema version: {schema_version!r}")
    if is_basis_task:
        validated = BasisTaskRecord.from_dict(raw)
        schema_kind = "basis-task-v2"
        witness_names = list(validated.witness_function_names)
        witness_ids = list(validated.witness_function_ids)
        distance = _optional_int(metadata.get("certified_distance"))
        certified: bool | None = True if distance is not None else None
        basis_set_id = validated.basis_set_id
        basis_set_digest = validated.basis_set_digest
        if source.basis_set_id is not None and basis_set_id != source.basis_set_id:
            raise CorpusIndexError(
                f"record basis {basis_set_id!r} does not match manifest "
                f"{source.basis_set_id!r}"
            )
        if (
            source.basis_set_digest is not None
            and basis_set_digest != source.basis_set_digest
        ):
            raise CorpusIndexError("record basis digest does not match manifest")
        generator = validated.generator
        task_id = validated.task_id
        if not id_to_name:
            id_to_name = _registered_basis_function_names(
                basis_set_id, basis_set_digest
            )
    else:
        typed_list_from_builtin_str(input_value)
        if output_value is not None:
            typed_list_from_builtin_str(output_value)
        schema_kind = "shortest-path-v1"
        witness_names = _string_list(
            raw.get("relabeled_functions", raw.get("original_functions", [])),
            "relabeled functions",
        )
        witness_ids = []
        distance = _optional_int(raw.get("relabeled_length", raw.get("upper_bound")))
        certified = _optional_bool(raw.get("certified"))
        basis_set_id = None
        basis_set_digest = None
        generator = "legacy-shortest-path"
        task_id = _legacy_task_id(input_value, output_value, fallback_id)

    input_type = _optional_str(metadata.get("input_type")) or _serialized_type(
        input_value
    )
    output_type = _optional_str(metadata.get("output_type")) or _serialized_type(
        output_value
    )
    split = _optional_str(raw.get("split")) or split_hint or "unknown"
    if split_hint is not None and split != split_hint:
        raise CorpusIndexError(
            f"record split {split!r} does not match manifest split {split_hint!r}"
        )

    functions: list[tuple[str, int, str, str, str | None]] = []
    functions.extend(
        _function_associations(
            "witness", witness_names, witness_ids, id_to_name=id_to_name
        )
    )
    functions.extend(
        _function_associations(
            "optimal_first",
            _string_list(
                metadata.get("optimal_first_action_names", []),
                "optimal first names",
            ),
            _string_list(
                metadata.get("optimal_first_action_ids", []), "optimal first IDs"
            ),
            id_to_name=id_to_name,
        )
    )
    functions.extend(
        _function_associations(
            "optimal_last",
            _string_list(
                metadata.get("optimal_last_action_names", []), "optimal last names"
            ),
            _string_list(
                metadata.get("optimal_last_action_ids", []), "optimal last IDs"
            ),
            id_to_name=id_to_name,
        )
    )

    return {
        "task_id": task_id,
        "schema_kind": schema_kind,
        "split": split,
        "input": input_value,
        "output": output_value,
        "input_type": input_type,
        "output_type": output_type,
        "distance": distance,
        "certified": None if certified is None else int(certified),
        "witness_names": witness_names,
        "witness_ids": witness_ids,
        "basis_set_id": basis_set_id,
        "basis_set_digest": basis_set_digest,
        "generator": generator,
        "source_index": _optional_int(raw.get("source_index")),
        "root_index": _optional_int(metadata.get("root_index")),
        "certification": _optional_str(metadata.get("certification"))
        or _optional_str(raw.get("method")),
        "metadata": metadata,
        "functions": functions,
    }


def _function_associations(
    role: str,
    names: Sequence[str],
    function_ids: Sequence[str],
    *,
    id_to_name: dict[str, str],
) -> list[tuple[str, int, str, str, str | None]]:
    associations: list[tuple[str, int, str, str, str | None]] = []
    seen: set[tuple[str, str]] = set()

    def append(*, ordinal: int, name: str, function_id: str | None = None) -> None:
        function_key = function_id or f"name:{name}"
        dedupe_key = (function_key, role)
        if dedupe_key in seen:
            return
        seen.add(dedupe_key)
        associations.append((role, ordinal, function_key, name, function_id))

    represented_names: set[str] = set()
    for ordinal, function_id in enumerate(function_ids):
        mapped_name = id_to_name.get(function_id) or _name_from_function_id(function_id)
        positional_name = names[ordinal] if ordinal < len(names) else None
        if (
            role == "witness"
            and mapped_name is not None
            and positional_name is not None
            and mapped_name != positional_name
        ):
            raise CorpusIndexError(
                f"witness function ID {function_id!r} resolves to {mapped_name!r}, "
                f"not {positional_name!r}"
            )
        name = mapped_name or positional_name or function_id
        append(ordinal=ordinal, name=name, function_id=function_id)
        represented_names.add(name)

    # Optimal-action IDs and names are independently sorted in deep_corpus_v1,
    # so they must not be zipped positionally. Add any name not recovered from
    # the registered basis as a name-only association.
    for ordinal, name in enumerate(names):
        if name not in represented_names:
            append(ordinal=ordinal, name=name)
    return associations


def _name_from_function_id(function_id: str) -> str | None:
    parts = function_id.split(":")
    if len(parts) == 3 and parts[0] == "bf" and parts[1]:
        return parts[1]
    return None


def _basis_function_names(source: CorpusSource) -> dict[str, str]:
    if source.basis_set_id is None:
        return {}
    return _registered_basis_function_names(
        source.basis_set_id, source.basis_set_digest
    )


@lru_cache(maxsize=16)
def _registered_basis_function_names(
    basis_set_id: str, basis_set_digest: str | None
) -> dict[str, str]:
    try:
        basis = load_basis_set(basis_set_id)
    except BasisSetError as error:
        raise CorpusIndexError(
            f"corpus basis {basis_set_id!r} is not available: {error}"
        ) from error
    if basis_set_digest is not None and basis.digest != basis_set_digest:
        raise CorpusIndexError(
            f"corpus basis digest {basis_set_digest} does not match "
            f"installed {basis.digest}"
        )
    return {function.function_id: function.name for function in basis.functions}


def _legacy_task_id(input_value: str, output_value: str | None, fallback: str) -> str:
    if output_value is None:
        return f"legacy:{fallback}"
    return task_id_for(input_value, output_value)


def _serialized_type(value: str | None) -> str:
    if value is None:
        return "unknown"
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as error:
        raise CorpusIndexError(f"invalid serialized TypedList JSON: {error}") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("type"), str):
        raise CorpusIndexError("serialized TypedList has no string 'type'")
    return payload["type"]


def _serialized_preview(value: str | None, limit: int = 140) -> str:
    if value is None:
        return "(none)"
    try:
        payload = json.loads(value)
        item_type = str(payload.get("type", "unknown")).removeprefix("builtins.")
        items = json.dumps(
            payload.get("items"), ensure_ascii=False, separators=(",", ":")
        )
        preview = f"{item_type}: {items}"
    except (AttributeError, json.JSONDecodeError):
        preview = value
    if len(preview) <= limit:
        return preview
    return preview[: limit - 1] + "…"


def _string_list(value: Any, context: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise CorpusIndexError(f"{context} must be an array of strings")
    return value


def _optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _optional_sha256(value: Any) -> str | None:
    if value is None:
        return None
    if (
        not isinstance(value, str)
        or len(value) != 71
        or not value.startswith("sha256:")
    ):
        raise CorpusIndexError(f"invalid sha256 digest: {value!r}")
    try:
        int(value.removeprefix("sha256:"), 16)
    except ValueError as error:
        raise CorpusIndexError(f"invalid sha256 digest: {value!r}") from error
    return value


def _validate_manifest_digest(manifest: dict[str, Any], path: Path) -> None:
    """Validate the corpus body while treating hosting metadata as external.

    PR #37 defines ``hub`` as a location annotation added after generation, so
    neither it nor ``manifest_digest`` belongs to the digested corpus body.
    """
    stored = _optional_sha256(manifest.get("manifest_digest"))
    if stored is None:
        return
    payload = dict(manifest)
    payload.pop("manifest_digest", None)
    payload.pop("hub", None)
    try:
        canonical = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise CorpusIndexError(
            f"invalid manifest JSON tree in {path}: {error}"
        ) from error
    actual = f"sha256:{hashlib.sha256(canonical).hexdigest()}"
    if actual != stored:
        raise CorpusIndexError(
            f"manifest digest mismatch for {path}: says {stored}, computed {actual}"
        )


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise CorpusIndexError(f"expected integer, got {value!r}")
    return value


def _optional_nonnegative_int(value: Any) -> int | None:
    parsed = _optional_int(value)
    if parsed is not None and parsed < 0:
        raise CorpusIndexError(f"expected non-negative integer, got {value!r}")
    return parsed


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise CorpusIndexError(f"expected boolean, got {value!r}")
    return value


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def iter_raw_records(source: CorpusSource) -> Iterator[dict[str, Any]]:
    """Stream original rows, primarily for audits and one-off migrations."""
    for corpus_file in source.files:
        with gzip.open(corpus_file.path, "rt", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as error:
                    raise CorpusIndexError(
                        f"invalid JSON at {corpus_file.path}:{line_number}: {error}"
                    ) from error
                if not isinstance(raw, dict):
                    raise CorpusIndexError(
                        f"record at {corpus_file.path}:{line_number} is not an object"
                    )
                yield raw
