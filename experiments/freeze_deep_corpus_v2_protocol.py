"""Freeze the immutable evaluation protocol for issue #54.

This command is intentionally a gate, not a convenience wrapper.  It verifies
the complete pinned corpus and shared tokenizer, rejects source changes, builds
the balanced validation/test cohorts, and writes all three lock artifacts in a
single directory rename.  The held-out test cohort is selected here, before any
checkpoint selection; opening it for scoring is controlled separately by the
test-opening ledger in :mod:`wandering_light.evals.deep_corpus_protocol`.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from experiments import generate_deep_corpus as deep
from experiments.prepare_deep_corpus_v2 import (
    BASIS_SET_DIGEST,
    BASIS_SET_ID,
    PROFILE_NAME,
)
from experiments.prepare_deep_corpus_v2 import (
    OUTPUT_DIR as DEFAULT_CORPUS_DIR,
)
from experiments.prepare_deep_corpus_v2 import (
    SEED as ISSUE_54_SEED,
)
from wandering_light.evals.deep_corpus_protocol import (
    COHORT_SIZE,
    SAMPLE_PREFIXES,
    build_bfs_settings,
    build_evaluator_source_identity,
    build_frozen_cohorts,
    build_generation_config,
    build_protocol_lock,
    build_tokenizer_identity,
    canonical_copy,
    file_identity,
    sha256_json,
    validate_frozen_cohorts,
    validate_protocol_lock,
)
from wandering_light.training.deep_corpus_tokenizer import (
    DEFAULT_REFERENCE_CORPUS_DIR,
    EXPECTED_CORPUS_REPO_ID,
    FIXED_MAX_LENGTH,
    TOKENIZER_FILENAMES,
    load_verified_v2_discovery,
    validate_frozen_tokenizer,
)
from wandering_light.training.deep_corpus_tokenizer import (
    MANIFEST_FILENAME as TOKENIZER_MANIFEST_FILENAME,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_SELECTION_SEED = 20_260_826
BFS_EXECUTION_SAFETY_BUDGET = 20_000_000
GENERATION_TEMPERATURE = 0.8
GENERATION_TOP_P = 0.95
GENERATION_MAX_NEW_TOKENS = 64
GENERATION_BATCH_SIZE = 32

PROTOCOL_LOCK_FILENAME = "protocol_lock.json"
VALIDATION_COHORT_FILENAME = "validation_cohort.json"
TEST_COHORT_FILENAME = "test_cohort.json"
ARTIFACT_FILENAMES = (
    PROTOCOL_LOCK_FILENAME,
    VALIDATION_COHORT_FILENAME,
    TEST_COHORT_FILENAME,
)

# These are semantic roots, not the complete set.  ``evaluator_source_paths``
# follows every repository-local import transitively, so Executor, Trajectory,
# TypedList, the BFS solver, prompt rendering, basis loading, corpus verification,
# and their local dependencies are all locked as well.
EVALUATOR_SOURCE_ROOTS = (
    "experiments/freeze_deep_corpus_v2_protocol.py",
    "experiments/evaluate_deep_corpus_v2_scratch.py",
    "experiments/create_deep_corpus_v2_scratch_report.py",
    "experiments/publish_deep_corpus_v2_scratch_report.py",
    "experiments/train_deep_corpus_v2_scratch.py",
    "experiments/evaluate_deep_corpus_sft.py",
    "experiments/train_deep_corpus_sft.py",
    "wandering_light/evals/deep_corpus_protocol.py",
    "wandering_light/evals/run_evaluation.py",
)
EVALUATOR_STATIC_FILES = (
    "wandering_light/basis_sets/index.json",
    "wandering_light/basis_sets/wl-core-v1.json",
)
REQUIRED_EVALUATOR_FILES = frozenset(
    {
        *EVALUATOR_SOURCE_ROOTS,
        *EVALUATOR_STATIC_FILES,
        "wandering_light/basis_dataset.py",
        "wandering_light/basis_set.py",
        "wandering_light/executor.py",
        "wandering_light/function_def.py",
        "wandering_light/llm_utils.py",
        "wandering_light/solver.py",
        "wandering_light/trajectory.py",
        "wandering_light/training/deep_corpus_scratch.py",
        "wandering_light/typed_list.py",
    }
)

_GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_HF_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_LOCAL_MODULE_PREFIXES = ("experiments", "wandering_light")


def _canonical_pretty_json(value: Any) -> str:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def _safe_relative_file(root: Path, relative: Any, *, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{label} must be a non-empty relative file path")
    candidate = Path(relative)
    if candidate.is_absolute() or candidate.name != relative:
        raise ValueError(f"{label} is not a safe payload filename: {relative!r}")
    return root / candidate


def corpus_payload_paths(corpus_dir: str | Path) -> tuple[Path, ...]:
    """Return only split payloads declared by the source manifest.

    These large files may be fetched into an otherwise clean checkout and are
    the sole corpus paths exempted from the dirty-tree gate.  The manifest is
    deliberately not exempt: changing it must stop the freeze.
    """

    root = Path(corpus_dir).resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Deep-corpus-v2 manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    splits = manifest.get("splits")
    if not isinstance(splits, Mapping) or set(splits) != {
        "discovery",
        "validation",
        "test",
    }:
        raise ValueError("V2 corpus manifest must declare exactly three splits")
    payloads: list[Path] = []
    for split in ("discovery", "validation", "test"):
        metadata = splits[split]
        if not isinstance(metadata, Mapping):
            raise ValueError(f"V2 corpus {split} metadata must be a mapping")
        payloads.append(
            _safe_relative_file(root, metadata.get("path"), label=f"{split} path")
        )
    return tuple(payloads)


def _git_status_paths(repo_root: Path) -> tuple[Path, ...]:
    result = subprocess.run(
        [
            "git",
            "-C",
            str(repo_root),
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
        ],
        check=True,
        stdout=subprocess.PIPE,
    )
    chunks = result.stdout.split(b"\0")
    paths: list[Path] = []
    index = 0
    while index < len(chunks):
        entry = chunks[index]
        index += 1
        if not entry:
            continue
        if len(entry) < 4 or entry[2:3] != b" ":
            raise RuntimeError("Could not parse git porcelain status")
        status = entry[:2].decode("ascii", errors="strict")
        paths.append(Path(os.fsdecode(entry[3:])))
        if "R" in status or "C" in status:
            if index >= len(chunks) or not chunks[index]:
                raise RuntimeError("Git rename status omitted its source path")
            paths.append(Path(os.fsdecode(chunks[index])))
            index += 1
    return tuple(paths)


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def assert_clean_source_tree(
    repo_root: str | Path, *, ignored_paths: Iterable[str | Path] = ()
) -> str:
    """Require a clean checkout, except for explicitly named data destinations."""

    root = Path(repo_root).resolve()
    top_level = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()
    if Path(top_level).resolve() != root:
        raise ValueError(f"repo_root is not the Git top level: {root}")

    ignored = tuple(Path(path).resolve() for path in ignored_paths)
    dirty = []
    for relative in _git_status_paths(root):
        absolute = (root / relative).resolve()
        if any(_is_within(absolute, allowed) for allowed in ignored):
            continue
        dirty.append(relative.as_posix())
    if dirty:
        rendered = ", ".join(sorted(set(dirty)))
        raise RuntimeError(
            "Refusing to freeze evaluator identities from a dirty source tree: "
            f"{rendered}"
        )

    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()
    if not _GIT_COMMIT_RE.fullmatch(commit):
        raise RuntimeError(f"Git HEAD is not a full commit identity: {commit!r}")
    return commit


def _is_local_module(module: str) -> bool:
    return any(
        module == prefix or module.startswith(f"{prefix}.")
        for prefix in _LOCAL_MODULE_PREFIXES
    )


def _module_candidates(module: str, repo_root: Path) -> tuple[Path, ...]:
    if not _is_local_module(module):
        return ()
    parts = module.split(".")
    base = repo_root.joinpath(*parts)
    candidates: list[Path] = []
    module_file = base.with_suffix(".py")
    package_file = base / "__init__.py"
    if module_file.is_file():
        candidates.append(module_file)
    if package_file.is_file():
        candidates.append(package_file)
    for length in range(1, len(parts)):
        package_init = repo_root.joinpath(*parts[:length], "__init__.py")
        if package_init.is_file():
            candidates.append(package_init)
    return tuple(candidates)


def _module_name(path: Path, repo_root: Path) -> str:
    relative = path.relative_to(repo_root)
    parts = list(relative.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _absolute_import_from(node: ast.ImportFrom, current_package: str) -> str:
    if node.level == 0:
        return node.module or ""
    package_parts = current_package.split(".") if current_package else []
    if not package_parts:
        return ""
    ascend = node.level - 1
    if ascend > len(package_parts):
        return ""
    prefix = package_parts[: len(package_parts) - ascend]
    if node.module:
        prefix.extend(node.module.split("."))
    return ".".join(prefix)


def _local_import_modules(path: Path, repo_root: Path) -> set[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError) as error:
        raise ValueError(f"Cannot inspect evaluator source imports: {path}") from error
    current_module = _module_name(path, repo_root)
    current_package = (
        current_module
        if path.name == "__init__.py"
        else current_module.rpartition(".")[0]
    )
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = _absolute_import_from(node, current_package)
            if base:
                modules.add(base)
                for alias in node.names:
                    if alias.name != "*":
                        modules.add(f"{base}.{alias.name}")
    return {module for module in modules if _is_local_module(module)}


def evaluator_source_paths(
    repo_root: str | Path,
    *,
    source_roots: Sequence[str | Path] = EVALUATOR_SOURCE_ROOTS,
    static_files: Sequence[str | Path] = EVALUATOR_STATIC_FILES,
    required_files: Iterable[str | Path] | None = None,
    extra_roots: Iterable[str | Path] = (),
) -> tuple[Path, ...]:
    """Resolve the complete repository-local evaluator dependency closure."""

    root = Path(repo_root).resolve()
    relative_roots = [*source_roots, *extra_roots]
    initial = [root / Path(path) for path in relative_roots]
    initial.extend(root / Path(path) for path in static_files)
    for path in initial:
        if not path.is_file() or not _is_within(path.resolve(), root):
            raise FileNotFoundError(f"Evaluator dependency root is missing: {path}")

    pending: deque[Path] = deque(initial)
    discovered: set[Path] = set()
    while pending:
        path = pending.popleft().resolve()
        if path in discovered:
            continue
        discovered.add(path)
        if path.suffix != ".py":
            continue
        for module in _local_import_modules(path, root):
            for dependency in _module_candidates(module, root):
                resolved = dependency.resolve()
                if resolved not in discovered:
                    pending.append(resolved)

    relative_discovered = {path.relative_to(root).as_posix() for path in discovered}
    if required_files is None:
        if (
            tuple(Path(path).as_posix() for path in source_roots)
            == EVALUATOR_SOURCE_ROOTS
            and tuple(Path(path).as_posix() for path in static_files)
            == EVALUATOR_STATIC_FILES
        ):
            expected_files = REQUIRED_EVALUATOR_FILES
        else:
            expected_files = frozenset(
                Path(path).as_posix() for path in (*source_roots, *static_files)
            )
    else:
        expected_files = frozenset(Path(path).as_posix() for path in required_files)
    missing = sorted(expected_files - relative_discovered)
    if missing:
        raise RuntimeError(
            "Evaluator dependency closure omitted required semantic sources: "
            + ", ".join(missing)
        )
    return tuple(sorted(discovered, key=lambda path: path.relative_to(root).as_posix()))


def _validate_issue_54_manifest(
    manifest: Mapping[str, Any], *, expected_git_commit: str | None = None
) -> None:
    target = manifest.get("target_contract")
    hub = manifest.get("hub")
    if manifest.get("basis_set_id") != BASIS_SET_ID:
        raise ValueError("V2 corpus uses the wrong basis-set ID")
    if manifest.get("basis_set_digest") != BASIS_SET_DIGEST:
        raise ValueError("V2 corpus uses the wrong frozen basis-set digest")
    if manifest.get("seed") != ISSUE_54_SEED:
        raise ValueError("V2 corpus was not generated with issue #54's seed")
    if not isinstance(target, Mapping) or target.get("profile") != PROFILE_NAME:
        raise ValueError("Corpus manifest is not the frozen issue #54 V2 profile")
    generator_source = target.get("generator_source")
    if (
        not isinstance(generator_source, Mapping)
        or not _GIT_COMMIT_RE.fullmatch(str(generator_source.get("git_commit", "")))
        or generator_source.get("tree_state") != "clean"
        or generator_source.get("runner") != "experiments/prepare_deep_corpus_v2.py"
        or generator_source.get("generator") != "experiments/generate_deep_corpus.py"
    ):
        raise ValueError("V2 corpus lacks clean generator-source provenance")
    if expected_git_commit is not None and not _GIT_COMMIT_RE.fullmatch(
        expected_git_commit
    ):
        raise ValueError("Expected protocol source commit is malformed")
    if (
        not isinstance(hub, Mapping)
        or hub.get("repo_id") != EXPECTED_CORPUS_REPO_ID
        or hub.get("repo_type") != "dataset"
        or not _HF_COMMIT_RE.fullmatch(str(hub.get("revision", "")))
    ):
        raise ValueError("V2 corpus is not pinned to its immutable Hub revision")


def _git_commit_is_ancestor(repo_root: Path, ancestor: str, descendant: str) -> bool:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=repo_root,
        check=False,
        capture_output=True,
    )
    if result.returncode not in {0, 1}:
        raise RuntimeError("Could not verify corpus-generator Git ancestry")
    return result.returncode == 0


def _git_file_identity(repo_root: Path, commit: str, relative: str) -> dict[str, Any]:
    try:
        payload = subprocess.run(
            ["git", "show", f"{commit}:{relative}"],
            cwd=repo_root,
            check=True,
            capture_output=True,
        ).stdout
    except subprocess.CalledProcessError as error:
        raise ValueError(
            f"Corpus generator commit does not contain {relative}"
        ) from error
    return {
        "path": relative,
        "bytes": len(payload),
        "sha256": f"sha256:{hashlib.sha256(payload).hexdigest()}",
    }


def _generator_source_binding(
    manifest: Mapping[str, Any],
    *,
    repo_root: Path,
    git_commit: str,
    evaluator_sources: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind the corpus generator claim to live clean files and the source lock."""

    _validate_issue_54_manifest(manifest, expected_git_commit=git_commit)
    claimed = manifest["target_contract"]["generator_source"]
    generator_git_commit = claimed["git_commit"]
    if not _git_commit_is_ancestor(repo_root, generator_git_commit, git_commit):
        raise ValueError(
            "Corpus generator commit is not an ancestor of the protocol source commit"
        )
    frozen_by_path = {
        row.get("path"): row
        for row in evaluator_sources.get("files", [])
        if isinstance(row, Mapping)
    }
    files = []
    for role in ("runner", "generator"):
        relative = claimed[role]
        path = repo_root / relative
        if not path.is_file():
            raise ValueError(f"Corpus {role} source file is missing: {relative}")
        live = file_identity(path, root=repo_root)
        committed = _git_file_identity(repo_root, generator_git_commit, relative)
        if committed != live:
            raise ValueError(
                f"Corpus {role} source changed after the recorded generator commit"
            )
        if frozen_by_path.get(relative) != live:
            raise ValueError(
                f"Corpus {role} source is not covered by the frozen source identity"
            )
        files.append({"role": role, **live})
    body = {
        "git_commit": generator_git_commit,
        "protocol_git_commit": git_commit,
        "generator_is_protocol_ancestor": True,
        "tree_state": "clean",
        "files": files,
        "evaluator_source_sha256": evaluator_sources["source_sha256"],
    }
    return {**body, "generator_source_sha256": sha256_json(body)}


def _validate_tokenizer_source(
    tokenizer_manifest: Mapping[str, Any],
    *,
    corpus_manifest: Mapping[str, Any],
    verified_source_identity: Mapping[str, Any],
    basis_identity: Mapping[str, Any],
) -> None:
    if canonical_copy(tokenizer_manifest.get("source")) != canonical_copy(
        verified_source_identity
    ):
        raise ValueError(
            "Frozen tokenizer source identity does not match the currently verified "
            "V2 corpus manifest and discovery payload"
        )
    if canonical_copy(tokenizer_manifest.get("basis_set")) != canonical_copy(
        basis_identity
    ):
        raise ValueError("Frozen tokenizer and verified corpus use different bases")
    if tokenizer_manifest.get("source", {}).get(
        "manifest_digest"
    ) != corpus_manifest.get("manifest_digest"):
        raise ValueError("Frozen tokenizer was trained from another corpus manifest")
    if tokenizer_manifest.get("tokenizer", {}).get("max_length") != FIXED_MAX_LENGTH:
        raise ValueError(
            f"Frozen tokenizer maximum length must remain {FIXED_MAX_LENGTH}"
        )


def _corpus_identity(
    corpus_dir: Path,
    manifest: Mapping[str, Any],
    records: Sequence[Any],
    verified_source_identity: Mapping[str, Any],
    generator_source_binding: Mapping[str, Any],
) -> dict[str, Any]:
    split_files = corpus_payload_paths(corpus_dir)
    identities = [
        file_identity(corpus_dir / "manifest.json", root=corpus_dir),
        *(file_identity(path, root=corpus_dir) for path in split_files),
    ]
    task_ids = [str(record.task_id) for record in records]
    root_digests = [str(record.metadata["root_digest"]) for record in records]
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("Verified V2 corpus contains duplicate task IDs")
    return {
        "profile": PROFILE_NAME,
        "manifest": canonical_copy(dict(manifest)),
        "files": sorted(identities, key=lambda row: row["path"]),
        "records": len(records),
        "ordered_task_ids_sha256": sha256_json(task_ids),
        "task_id_set_sha256": sha256_json(sorted(task_ids)),
        "root_digest_set_sha256": sha256_json(sorted(set(root_digests))),
        "full_verification": canonical_copy(
            dict(verified_source_identity.get("verification", {}))
        ),
        "generator_source": canonical_copy(dict(generator_source_binding)),
    }


def _tokenizer_identity(
    tokenizer_dir: Path, tokenizer_manifest: Mapping[str, Any]
) -> dict[str, Any]:
    paths = [
        tokenizer_dir / TOKENIZER_MANIFEST_FILENAME,
        *(tokenizer_dir / name for name in TOKENIZER_FILENAMES),
    ]
    tokenizer = tokenizer_manifest["tokenizer"]
    return build_tokenizer_identity(
        paths,
        root=tokenizer_dir,
        metadata={
            "artifact_type": tokenizer_manifest["artifact_type"],
            "artifact_digest": tokenizer_manifest["artifact_digest"],
            "source_manifest_digest": tokenizer_manifest["source"]["manifest_digest"],
            "basis_set": tokenizer_manifest["basis_set"],
            "max_length": tokenizer["max_length"],
            "base_vocab_size": tokenizer["base_vocab_size"],
            "total_vocab_size": tokenizer["total_vocab_size"],
        },
    )


def build_protocol_artifacts(
    *,
    repo_root: Path,
    git_commit: str,
    corpus_dir: Path,
    corpus_manifest: Mapping[str, Any],
    records: Sequence[Any],
    verified_source_identity: Mapping[str, Any],
    tokenizer_dir: Path,
    tokenizer_manifest: Mapping[str, Any],
    source_paths: Sequence[Path],
    source_roots: Sequence[str] = EVALUATOR_SOURCE_ROOTS,
) -> dict[str, Any]:
    """Build and fully revalidate the three in-memory protocol artifacts."""

    _validate_issue_54_manifest(corpus_manifest, expected_git_commit=git_commit)
    cohorts = build_frozen_cohorts(records, selection_seed=PROTOCOL_SELECTION_SEED)
    validate_frozen_cohorts(cohorts, records=records)
    generation = build_generation_config(
        sampling_seed=PROTOCOL_SELECTION_SEED,
        temperature=GENERATION_TEMPERATURE,
        top_p=GENERATION_TOP_P,
        max_new_tokens=GENERATION_MAX_NEW_TOKENS,
        batch_size=GENERATION_BATCH_SIZE,
        max_sequence_length=FIXED_MAX_LENGTH,
    )
    bfs = build_bfs_settings(
        budget=BFS_EXECUTION_SAFETY_BUDGET,
        max_depth=8,
        extra={
            "budget_role": "per-task execution safety ceiling",
            "matched_cost_unit": "basis-function executions",
        },
    )
    tokenizer = _tokenizer_identity(tokenizer_dir, tokenizer_manifest)
    evaluator_sources = build_evaluator_source_identity(
        source_paths,
        root=repo_root,
        metadata={
            "git_commit": git_commit,
            "dependency_resolution": "transitive repository-local AST import closure",
            "roots": list(source_roots),
            "basis_set_id": BASIS_SET_ID,
            "basis_set_digest": BASIS_SET_DIGEST,
        },
    )
    generator_source = _generator_source_binding(
        corpus_manifest,
        repo_root=repo_root,
        git_commit=git_commit,
        evaluator_sources=evaluator_sources,
    )
    protocol_lock = build_protocol_lock(
        corpus_identity=_corpus_identity(
            corpus_dir,
            corpus_manifest,
            records,
            verified_source_identity,
            generator_source,
        ),
        cohorts=cohorts,
        generation=generation,
        bfs=bfs,
        tokenizer=tokenizer,
        evaluator_sources=evaluator_sources,
    )
    validate_protocol_lock(
        protocol_lock,
        records=records,
        tokenizer_root=tokenizer_dir,
        evaluator_root=repo_root,
    )
    return {
        PROTOCOL_LOCK_FILENAME: protocol_lock,
        VALIDATION_COHORT_FILENAME: cohorts["validation"],
        TEST_COHORT_FILENAME: cohorts["test"],
    }


def _load_artifact_directory(output_dir: Path) -> dict[str, Any]:
    return {
        name: json.loads((output_dir / name).read_text(encoding="utf-8"))
        for name in ARTIFACT_FILENAMES
    }


def write_protocol_artifacts(
    output_dir: str | Path,
    artifacts: Mapping[str, Any],
    *,
    repo_root: str | Path,
    tokenizer_dir: str | Path,
    records: Sequence[Any],
) -> Path:
    """Atomically publish all protocol artifacts into one fresh directory."""

    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(
            f"Protocol output must be a fresh directory: {destination}"
        )
    if set(artifacts) != set(ARTIFACT_FILENAMES):
        raise ValueError("Protocol artifact set is incomplete or has extra files")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.building-", dir=destination.parent
        )
    )
    try:
        for name in ARTIFACT_FILENAMES:
            path = temporary / name
            with path.open("x", encoding="utf-8") as output_file:
                output_file.write(_canonical_pretty_json(artifacts[name]))
                output_file.flush()
                os.fsync(output_file.fileno())

        loaded = _load_artifact_directory(temporary)
        if loaded != canonical_copy(dict(artifacts)):
            raise RuntimeError("Protocol artifacts changed during serialization")
        protocol_lock = loaded[PROTOCOL_LOCK_FILENAME]
        if (
            loaded[VALIDATION_COHORT_FILENAME] != protocol_lock["cohorts"]["validation"]
            or loaded[TEST_COHORT_FILENAME] != protocol_lock["cohorts"]["test"]
        ):
            raise RuntimeError("Explicit cohort artifacts differ from protocol lock")
        validate_protocol_lock(
            protocol_lock,
            records=records,
            tokenizer_root=tokenizer_dir,
            evaluator_root=repo_root,
        )
        if destination.exists():
            raise FileExistsError(
                f"Protocol output appeared while it was being built: {destination}"
            )
        temporary.rename(destination)
        return destination
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def freeze_protocol(
    *,
    corpus_dir: str | Path,
    tokenizer_dir: str | Path,
    output_dir: str | Path,
    repo_root: str | Path = REPO_ROOT,
    dry_run: bool = False,
    reference_corpus_dirs: Sequence[str | Path] = (DEFAULT_REFERENCE_CORPUS_DIR,),
    evaluator_source_roots: Sequence[str | Path] = EVALUATOR_SOURCE_ROOTS,
    evaluator_static_files: Sequence[str | Path] = EVALUATOR_STATIC_FILES,
    extra_evaluator_sources: Sequence[str | Path] = (),
) -> dict[str, Any]:
    """Verify inputs, build the lock, and optionally publish it atomically."""

    root = Path(repo_root).resolve()
    corpus = Path(corpus_dir).resolve()
    tokenizer = Path(tokenizer_dir).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Protocol output must be a fresh path: {output}")

    payload_paths = corpus_payload_paths(corpus)
    reference_payload_paths = tuple(
        payload
        for reference_dir in reference_corpus_dirs
        for payload in corpus_payload_paths(reference_dir)
    )
    commit = assert_clean_source_tree(
        root,
        ignored_paths=(output, *payload_paths, *reference_payload_paths),
    )

    # This loader checks the immutable Hub revision, verifies all three splits,
    # re-executes every witness, loads the exact basis, and reconstructs the
    # discovery-only tokenizer source identity.
    if not reference_corpus_dirs:
        raise ValueError("At least one frozen reference corpus is required")
    verified = load_verified_v2_discovery(
        corpus,
        reference_corpus_dirs=reference_corpus_dirs,
    )
    _validate_issue_54_manifest(verified.manifest)
    corpus_manifest, records = deep.load_corpus(corpus)
    if canonical_copy(corpus_manifest) != canonical_copy(verified.manifest):
        raise ValueError("Corpus manifest changed after complete verification")

    tokenizer_manifest = validate_frozen_tokenizer(
        tokenizer,
        expected_source_manifest_digest=verified.manifest["manifest_digest"],
    )
    _validate_tokenizer_source(
        tokenizer_manifest,
        corpus_manifest=verified.manifest,
        verified_source_identity=verified.source_identity,
        basis_identity=verified.basis.identity_dict(),
    )

    source_roots = (
        *(Path(path).as_posix() for path in evaluator_source_roots),
        *(Path(path).as_posix() for path in extra_evaluator_sources),
    )
    source_paths = evaluator_source_paths(
        root,
        source_roots=evaluator_source_roots,
        static_files=evaluator_static_files,
        extra_roots=extra_evaluator_sources,
    )
    artifacts = build_protocol_artifacts(
        repo_root=root,
        git_commit=commit,
        corpus_dir=corpus,
        corpus_manifest=corpus_manifest,
        records=records,
        verified_source_identity=verified.source_identity,
        tokenizer_dir=tokenizer,
        tokenizer_manifest=tokenizer_manifest,
        source_paths=source_paths,
        source_roots=source_roots,
    )
    lock = artifacts[PROTOCOL_LOCK_FILENAME]
    status = "dry-run"
    if not dry_run:
        write_protocol_artifacts(
            output,
            artifacts,
            repo_root=root,
            tokenizer_dir=tokenizer,
            records=records,
        )
        status = "written"
    return {
        "status": status,
        "output_dir": str(output),
        "protocol_lock_sha256": lock["protocol_lock_sha256"],
        "cohorts_sha256": lock["cohorts"]["cohorts_sha256"],
        "validation_cohort_sha256": lock["cohorts"]["validation"]["cohort_sha256"],
        "test_cohort_sha256": lock["cohorts"]["test"]["cohort_sha256"],
        "validation_tasks": lock["cohorts"]["validation"]["size"],
        "test_tasks": lock["cohorts"]["test"]["size"],
        "sample_prefixes": list(SAMPLE_PREFIXES),
        "generation_seed": lock["generation"]["sampling_seed"],
        "bfs_execution_safety_budget": lock["bfs"]["budget"],
        "evaluator_source_files": len(lock["evaluator_sources"]["files"]),
        "git_commit": commit,
        "wrote_files": list(ARTIFACT_FILENAMES) if not dry_run else [],
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--reference-corpus-dir",
        action="append",
        default=[],
        type=Path,
        help="Frozen corpus whose root/task identities V2 excludes (repeatable)",
    )
    parser.add_argument(
        "--evaluator-source",
        action="append",
        default=[],
        type=Path,
        help=(
            "Additional repository-relative evaluator root to include in the "
            "transitive source lock (repeatable)"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    result = freeze_protocol(
        corpus_dir=args.corpus_dir,
        tokenizer_dir=args.tokenizer_dir,
        output_dir=args.output_dir,
        repo_root=args.repo_root,
        dry_run=args.dry_run,
        reference_corpus_dirs=tuple(
            args.reference_corpus_dir or (DEFAULT_REFERENCE_CORPUS_DIR,)
        ),
        extra_evaluator_sources=tuple(args.evaluator_source),
    )
    if result["validation_tasks"] != COHORT_SIZE or result["test_tasks"] != (
        COHORT_SIZE
    ):
        raise AssertionError("Protocol freeze produced an invalid cohort size")
    print(_canonical_pretty_json(result), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
