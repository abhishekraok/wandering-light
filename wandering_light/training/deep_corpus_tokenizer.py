"""Deterministic shared tokenizer for the deep-corpus-v2 scratch study.

The tokenizer is deliberately independent of every model architecture.  It is
trained once from canonical prompt/completion text derived from the verified
``discovery`` split, then frozen as a content-addressed artifact.  Validation and
test records are verified as part of the corpus, but their text is never passed
to the tokenizer trainer.

The low-level tokenizer is byte-level BPE with byte fallback enabled and the
complete byte alphabet seeded into the vocabulary.  Basis-function names are
registered *after* BPE training as normal ``AddedToken`` instances: each name is
one token, yet decoding with ``skip_special_tokens=True`` retains it.  The four
special tokens have a fixed order and therefore fixed IDs.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from wandering_light.basis_set import load_basis_set

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerFast

    from wandering_light.basis_dataset import BasisTaskRecord
    from wandering_light.basis_set import BasisSet
    from wandering_light.function_def import FunctionDefSet


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "wandering-light-deep-corpus-v2-shared-tokenizer"
EXPECTED_BASIS_SET_ID = "wl-core-v1"
EXPECTED_CORPUS_REPO_ID = "abhishekraok/wandering-light-deep-corpus-v2"
DEFAULT_CORPUS_DIR = Path("wandering_light/training/data/deep_corpus_v2")
DEFAULT_REFERENCE_CORPUS_DIR = Path("wandering_light/training/data/deep_corpus_v1")
DEFAULT_VOCAB_SIZE = 4096
DEFAULT_MIN_FREQUENCY = 2
FIXED_MAX_LENGTH = 2048
DISCOVERY_SPLIT = "discovery"
CORPUS_SPLITS = (DISCOVERY_SPLIT, "validation", "test")
REQUIRED_DISTANCES = tuple(range(1, 9))

# Order is part of the protocol. BpeTrainer assigns these IDs before the byte
# alphabet and learned vocabulary.
FIXED_SPECIAL_TOKENS: tuple[tuple[str, str], ...] = (
    ("pad_token", "<|pad|>"),
    ("bos_token", "<|bos|>"),
    ("eos_token", "<|eos|>"),
    ("unk_token", "<|unk|>"),
)

TOKENIZER_FILENAMES = (
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
MANIFEST_FILENAME = "tokenizer_manifest.json"

_HF_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


@dataclass(frozen=True)
class SharedTokenizerConfig:
    """Frozen tokenizer-training parameters."""

    vocab_size: int = DEFAULT_VOCAB_SIZE
    min_frequency: int = DEFAULT_MIN_FREQUENCY
    max_length: int = FIXED_MAX_LENGTH


@dataclass(frozen=True)
class VerifiedDiscoveryCorpus:
    """Verified V2 discovery rows and their immutable source identity."""

    manifest: Mapping[str, Any]
    basis: BasisSet
    records: tuple[BasisTaskRecord, ...]
    rows: tuple[Mapping[str, Any], ...]
    source_identity: Mapping[str, Any]


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


def _canonical_digest(value: Any) -> str:
    return _sha256_bytes(_canonical_json(value).encode("utf-8"))


def _line_digest(rows: Iterable[Any]) -> str:
    """Hash canonical JSON rows with unambiguous line framing, in order."""

    digest = hashlib.sha256()
    for row in rows:
        digest.update(_canonical_json(row).encode("utf-8"))
        digest.update(b"\n")
    return f"sha256:{digest.hexdigest()}"


def _validate_config(config: SharedTokenizerConfig) -> None:
    if config.max_length != FIXED_MAX_LENGTH:
        raise ValueError(
            f"The shared tokenizer maximum length is frozen at {FIXED_MAX_LENGTH}; "
            f"got {config.max_length}"
        )
    # Four special tokens plus the complete byte alphabet require at least 260
    # base vocabulary entries. Basis names are normal added tokens and may extend
    # the final vocabulary beyond the BPE trainer's target.
    if config.vocab_size < 260:
        raise ValueError("vocab_size must leave room for specials and all 256 bytes")
    if config.min_frequency <= 0:
        raise ValueError("min_frequency must be positive")


def basis_function_names(basis: BasisSet) -> tuple[str, ...]:
    """Return the stable, deterministic normal-token registration order."""

    if basis.basis_set_id != EXPECTED_BASIS_SET_ID:
        raise ValueError(
            f"Shared tokenizer requires {EXPECTED_BASIS_SET_ID!r}; "
            f"got {basis.basis_set_id!r}"
        )
    names = tuple(sorted(function.name for function in basis.functions))
    if len(names) != len(set(names)):
        raise ValueError("Basis contains duplicate function names")
    special_contents = {content for _, content in FIXED_SPECIAL_TOKENS}
    overlap = sorted(set(names) & special_contents)
    if overlap:
        raise ValueError(f"Basis names overlap fixed special tokens: {overlap}")
    return names


def canonical_corpus_rows(
    records: Sequence[BasisTaskRecord],
    available_functions: FunctionDefSet,
    *,
    allowed_splits: Sequence[str] = CORPUS_SPLITS,
    row_builder: Callable[[BasisTaskRecord, FunctionDefSet], Mapping[str, Any]]
    | None = None,
) -> tuple[Mapping[str, Any], ...]:
    """Build canonical PR #53 prompt/completion rows without changing split.

    ``row_builder`` is injectable for small pure tests. Production uses PR #53's
    public ``training_row_for_record`` helper, which validates and re-executes the
    recorded witness while constructing the exact training prompt.
    """

    if row_builder is None:
        from experiments.train_deep_corpus_sft import training_row_for_record

        row_builder = training_row_for_record

    allowed = tuple(allowed_splits)
    if not allowed or not set(allowed).issubset(CORPUS_SPLITS):
        raise ValueError(f"Invalid canonical-row split allowlist: {allowed!r}")

    rows: list[Mapping[str, Any]] = []
    for record in records:
        if record.split not in allowed:
            raise ValueError(
                f"Canonical row construction accepts only {allowed!r}; task "
                f"{record.task_id} is labeled {record.split!r}"
            )
        row = dict(row_builder(record, available_functions))
        if row.get("task_id") != record.task_id or row.get("split") != record.split:
            raise ValueError("Canonical row builder changed task or split identity")
        rows.append(row)
    _validate_rows(
        rows,
        allowed_splits=allowed,
        require_all_distances=False,
    )
    return tuple(rows)


def canonical_discovery_rows(
    records: Sequence[BasisTaskRecord],
    available_functions: FunctionDefSet,
    *,
    row_builder: Callable[[BasisTaskRecord, FunctionDefSet], Mapping[str, Any]]
    | None = None,
) -> tuple[Mapping[str, Any], ...]:
    """Build canonical rows under the discovery-only tokenizer-training gate."""

    return canonical_corpus_rows(
        records,
        available_functions,
        allowed_splits=(DISCOVERY_SPLIT,),
        row_builder=row_builder,
    )


def _validate_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    allowed_splits: Sequence[str] = CORPUS_SPLITS,
    require_all_distances: bool = True,
) -> None:
    if not rows:
        raise ValueError("Tokenization requires at least one corpus row")
    allowed = tuple(allowed_splits)
    if not allowed or not set(allowed).issubset(CORPUS_SPLITS):
        raise ValueError(f"Invalid tokenization split allowlist: {allowed!r}")
    task_ids: set[str] = set()
    observed_distances: set[int] = set()
    for index, row in enumerate(rows):
        task_id = row.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError(f"Tokenizer row {index} has no task ID")
        if task_id in task_ids:
            raise ValueError(f"Duplicate tokenizer task ID {task_id}")
        task_ids.add(task_id)
        if row.get("split") not in allowed:
            raise ValueError(
                f"Tokenization accepts only {allowed!r}; task {task_id} is "
                f"labeled {row.get('split')!r}"
            )
        distance = row.get("certified_distance")
        if type(distance) is not int or distance not in REQUIRED_DISTANCES:
            raise ValueError(
                f"Task {task_id} has invalid certified distance {distance!r}"
            )
        observed_distances.add(distance)
        for field in ("prompt", "completion"):
            if not isinstance(row.get(field), str):
                raise ValueError(f"Task {task_id} has no string {field}")
        if not row["completion"]:
            raise ValueError(f"Task {task_id} has an empty completion")
    missing = sorted(set(REQUIRED_DISTANCES) - observed_distances)
    if require_all_distances and missing:
        raise ValueError(
            f"Deep-corpus-v2 rows must cover distances 1 through 8; missing {missing}"
        )


def _validate_discovery_training_rows(
    rows: Sequence[Mapping[str, Any]],
) -> None:
    _validate_rows(
        rows,
        allowed_splits=(DISCOVERY_SPLIT,),
        require_all_distances=True,
    )


def _validate_pinned_v2_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("basis_set_id") != EXPECTED_BASIS_SET_ID:
        raise ValueError("V2 corpus manifest does not identify wl-core-v1")
    if not _SHA256_RE.fullmatch(str(manifest.get("basis_set_digest", ""))):
        raise ValueError("V2 corpus manifest has no valid basis digest")
    if not _SHA256_RE.fullmatch(str(manifest.get("manifest_digest", ""))):
        raise ValueError("V2 corpus manifest has no valid content digest")
    splits = manifest.get("splits")
    if not isinstance(splits, Mapping) or set(splits) != {
        "discovery",
        "validation",
        "test",
    }:
        raise ValueError("V2 corpus must have discovery, validation, and test splits")
    hub = manifest.get("hub")
    if not isinstance(hub, Mapping) or hub.get("repo_id") != EXPECTED_CORPUS_REPO_ID:
        raise ValueError(
            "Tokenizer training requires the corpus published at the exact V2 Hub repo"
        )
    if hub.get("repo_type") != "dataset" or not _HF_COMMIT_RE.fullmatch(
        str(hub.get("revision", ""))
    ):
        raise ValueError(
            "Tokenizer training requires a pinned immutable V2 Hub revision"
        )


def _source_identity(
    *,
    corpus_dir: Path,
    manifest: Mapping[str, Any],
    records: Sequence[BasisTaskRecord],
    verification: Mapping[str, Any],
) -> dict[str, Any]:
    manifest_path = corpus_dir / "manifest.json"
    discovery_metadata = manifest["splits"][DISCOVERY_SPLIT]
    discovery_path = corpus_dir / discovery_metadata["path"]
    task_ids = [record.task_id for record in records]
    if discovery_metadata.get("size") != len(records):
        raise ValueError("Discovery record count differs from the corpus manifest")
    if verification.get("ok") is not True:
        raise ValueError("Deep-corpus-v2 verification did not pass")
    if verification.get("records") != manifest.get("global_task_count"):
        raise ValueError("Corpus verifier and manifest disagree on global task count")
    if verification.get("witness_failures") or verification.get(
        "roots_leaking_across_splits"
    ):
        raise ValueError("Corpus verification found witness failures or root leakage")
    return {
        "corpus": "deep-corpus-v2",
        "manifest_digest": manifest["manifest_digest"],
        "manifest_file_sha256": _sha256_file(manifest_path),
        "basis_set_id": manifest["basis_set_id"],
        "basis_set_digest": manifest["basis_set_digest"],
        "generator": manifest.get("generator"),
        "generator_seed": manifest.get("seed"),
        "hub": dict(manifest["hub"]),
        "verification": {
            "records": verification["records"],
            "distance_histogram": dict(verification["distance_histogram"]),
            "all_witnesses_reexecuted": not verification["witness_failures"],
            "root_split_leaks": list(verification["roots_leaking_across_splits"]),
            "target_contract_failures": list(
                verification.get("target_contract_failures", [])
            ),
            "reference_root_digest_overlaps": list(
                verification.get("reference_root_digest_overlaps", [])
            ),
            "reference_root_key_overlaps": list(
                verification.get("reference_root_key_overlaps", [])
            ),
            "reference_task_id_overlaps": list(
                verification.get("reference_task_id_overlaps", [])
            ),
            "ok": verification["ok"],
        },
        "discovery": {
            "split": DISCOVERY_SPLIT,
            "records": len(records),
            "roots": discovery_metadata.get("roots"),
            "by_certified_distance": dict(discovery_metadata["by_certified_distance"]),
            "split_file": {
                "name": discovery_path.name,
                "bytes": discovery_path.stat().st_size,
                "sha256": _sha256_file(discovery_path),
            },
            "ordered_task_ids_digest": _canonical_digest(task_ids),
            "task_id_set_digest": _canonical_digest(sorted(task_ids)),
        },
    }


def load_verified_v2_discovery(
    corpus_dir: str | Path = DEFAULT_CORPUS_DIR,
    *,
    reference_corpus_dirs: Sequence[str | Path] = (),
) -> VerifiedDiscoveryCorpus:
    """Verify all V2 splits, then expose canonical discovery text only."""

    from experiments import generate_deep_corpus as deep

    root = Path(corpus_dir)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Deep-corpus-v2 manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _validate_pinned_v2_manifest(manifest)

    # The complete verifier reloads every split and re-executes every witness.
    # Only after that succeeds do we load records for one explicitly named split.
    reference_identities = tuple(
        deep.collect_corpus_identities(reference_dir)
        for reference_dir in reference_corpus_dirs
    )
    verification = deep.verify_corpus(
        root,
        recertify=0,
        seed=manifest["seed"],
        reference_identities=reference_identities,
    )
    loaded_manifest, records = deep.load_corpus(root, splits=(DISCOVERY_SPLIT,))
    if loaded_manifest != manifest:
        raise ValueError("Corpus manifest changed while tokenizer inputs were prepared")

    basis = load_basis_set(manifest["basis_set_id"])
    if basis.digest != manifest["basis_set_digest"]:
        raise ValueError("Loaded basis digest differs from the V2 corpus manifest")
    rows = canonical_discovery_rows(records, basis.as_function_set())
    source = _source_identity(
        corpus_dir=root,
        manifest=manifest,
        records=records,
        verification=verification,
    )
    return VerifiedDiscoveryCorpus(
        manifest=manifest,
        basis=basis,
        records=tuple(records),
        rows=rows,
        source_identity=source,
    )


def _training_text_rows(rows: Sequence[Mapping[str, Any]]) -> Iterable[list[str]]:
    for row in rows:
        yield [row["task_id"], "prompt", row["prompt"]]
        yield [row["task_id"], "completion", row["completion"]]


def train_shared_tokenizer(
    rows: Sequence[Mapping[str, Any]],
    basis: BasisSet,
    *,
    config: SharedTokenizerConfig | None = None,
) -> PreTrainedTokenizerFast:
    """Train the deterministic byte-level tokenizer entirely in memory."""

    if config is None:
        config = SharedTokenizerConfig()
    _validate_config(config)
    _validate_discovery_training_rows(rows)
    names = basis_function_names(basis)

    from tokenizers import AddedToken, Tokenizer, decoders, pre_tokenizers
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.processors import TemplateProcessing
    from tokenizers.trainers import BpeTrainer
    from transformers import PreTrainedTokenizerFast

    special_added_tokens = [
        AddedToken(
            content,
            single_word=False,
            lstrip=False,
            rstrip=False,
            normalized=False,
            special=True,
        )
        for _, content in FIXED_SPECIAL_TOKENS
    ]
    backend = Tokenizer(
        BPE(
            unk_token=dict(FIXED_SPECIAL_TOKENS)["unk_token"],
            byte_fallback=True,
            fuse_unk=False,
        )
    )
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False,
        trim_offsets=True,
        use_regex=True,
    )
    backend.decoder = decoders.ByteLevel()
    trainer = BpeTrainer(
        vocab_size=config.vocab_size,
        min_frequency=config.min_frequency,
        show_progress=False,
        special_tokens=special_added_tokens,
        initial_alphabet=ByteLevel.alphabet(),
    )

    def text_iterator() -> Iterable[str]:
        for row in rows:
            yield row["prompt"]
            yield row["completion"]

    backend.train_from_iterator(
        text_iterator(),
        trainer=trainer,
        length=2 * len(rows),
    )
    # Register basis names after training. Passing these through BpeTrainer's
    # ``special_tokens`` parameter would mark them special even when AddedToken
    # says otherwise; add_tokens preserves special=False and existing token IDs.
    backend.add_tokens(
        [
            AddedToken(
                name,
                single_word=False,
                lstrip=False,
                rstrip=False,
                normalized=False,
                special=False,
            )
            for name in names
        ]
    )
    special_map = dict(FIXED_SPECIAL_TOKENS)
    bos_token = special_map["bos_token"]
    bos_token_id = backend.token_to_id(bos_token)
    if bos_token_id is None:
        raise ValueError("BOS token disappeared during tokenizer training")
    # TRL tokenizes both the prompt and prompt+completion with
    # add_special_tokens=True. This fixed post-processor makes that public path
    # prepend exactly one BOS while leaving EOS under the dataset's control.
    backend.post_processor = TemplateProcessing(
        single=f"{bos_token} $A",
        pair=f"{bos_token} $A $B",
        special_tokens=[(bos_token, bos_token_id)],
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        model_max_length=config.max_length,
        clean_up_tokenization_spaces=False,
        **special_map,
    )
    validate_tokenizer_runtime(tokenizer, basis, expected_max_length=config.max_length)
    return tokenizer


def encode_bytes(tokenizer: PreTrainedTokenizerFast, value: bytes) -> list[int]:
    """Encode arbitrary bytes through a reversible Latin-1 text bridge."""

    return tokenizer.encode(value.decode("latin-1"), add_special_tokens=False)


def decode_bytes(tokenizer: PreTrainedTokenizerFast, token_ids: Sequence[int]) -> bytes:
    """Reverse :func:`encode_bytes` without dropping any registered token."""

    text = tokenizer.decode(
        list(token_ids),
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    return text.encode("latin-1")


def validate_tokenizer_runtime(
    tokenizer: PreTrainedTokenizerFast,
    basis: BasisSet,
    *,
    expected_max_length: int = FIXED_MAX_LENGTH,
) -> None:
    """Validate fixed IDs, byte fallback, roundtrip, and atomic normal names."""

    if tokenizer.model_max_length != expected_max_length:
        raise ValueError(
            f"Tokenizer maximum length drifted: {tokenizer.model_max_length}"
        )
    for expected_id, (role, content) in enumerate(FIXED_SPECIAL_TOKENS):
        if getattr(tokenizer, role) != content:
            raise ValueError(f"Tokenizer {role} drifted from {content!r}")
        if tokenizer.convert_tokens_to_ids(content) != expected_id:
            raise ValueError(f"Tokenizer {role} does not have fixed ID {expected_id}")

    backend_payload = json.loads(tokenizer.backend_tokenizer.to_str())
    if backend_payload.get("model", {}).get("byte_fallback") is not True:
        raise ValueError("Tokenizer BPE model does not enable byte fallback")
    if backend_payload.get("pre_tokenizer", {}).get("type") != "ByteLevel":
        raise ValueError("Tokenizer does not use the byte-level pre-tokenizer")
    if backend_payload.get("decoder", {}).get("type") != "ByteLevel":
        raise ValueError("Tokenizer does not use the byte-level decoder")
    if backend_payload.get("post_processor", {}).get("type") != "TemplateProcessing":
        raise ValueError("Tokenizer does not use the fixed BOS template processor")

    bos_id = tokenizer.bos_token_id
    if bos_id is None:
        raise ValueError("Tokenizer has no BOS token ID")
    for text in ("", "post-processor probe"):
        plain_ids = tokenizer.encode(text, add_special_tokens=False)
        special_ids = tokenizer.encode(text, add_special_tokens=True)
        if special_ids != [bos_id, *plain_ids]:
            raise ValueError("Tokenizer must prepend exactly one BOS token")

    added = tokenizer.backend_tokenizer.get_added_tokens_decoder()
    all_special = set(tokenizer.all_special_tokens)
    for name in basis_function_names(basis):
        token_id = tokenizer.convert_tokens_to_ids(name)
        if tokenizer.encode(name, add_special_tokens=False) != [token_id]:
            raise ValueError(f"Basis function {name!r} is not atomic")
        registered = added.get(token_id)
        if registered is None or registered.content != name or registered.special:
            raise ValueError(f"Basis function {name!r} is not a normal added token")
        if name in all_special:
            raise ValueError(f"Basis function {name!r} is incorrectly special")
        decoded = tokenizer.decode(
            [token_id],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if decoded != name:
            raise ValueError(f"Basis function {name!r} is skipped during decoding")

    for text in ("", "plain ASCII", "café 東京 ßeta Ωmega", "\x00\uffff"):
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(
            token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if decoded != text:
            raise ValueError(f"Tokenizer does not round-trip text probe {text!r}")
    all_bytes = bytes(range(256))
    if decode_bytes(tokenizer, encode_bytes(tokenizer, all_bytes)) != all_bytes:
        raise ValueError("Tokenizer does not round-trip all byte values")


def encode_training_row(
    tokenizer: PreTrainedTokenizerFast,
    row: Mapping[str, Any],
) -> dict[str, Any]:
    """Mirror TRL's prompt/completion tokenization and completion mask exactly."""

    prompt_ids = tokenizer.encode(row["prompt"], add_special_tokens=False)
    completion_ids = tokenizer.encode(row["completion"], add_special_tokens=False)
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    if bos_id is None or eos_id is None:
        raise ValueError("Shared tokenizer requires BOS and EOS IDs")
    eos_token = tokenizer.eos_token
    if not isinstance(eos_token, str) or not eos_token:
        raise ValueError("Shared tokenizer requires an EOS token string")
    if row["completion"].endswith(eos_token):
        raise ValueError("Canonical completion rows must not contain the EOS suffix")
    completion_with_eos = [*completion_ids, eos_id]
    combined = [bos_id, *prompt_ids, *completion_with_eos]
    prompt_with_bos = tokenizer.encode(row["prompt"], add_special_tokens=True)
    trl_combined = tokenizer.encode(
        row["prompt"] + row["completion"] + eos_token,
        add_special_tokens=True,
    )
    if prompt_with_bos != [bos_id, *prompt_ids]:
        raise ValueError("Public prompt tokenization differs from the BOS contract")
    if trl_combined != combined:
        raise ValueError(
            "TRL prompt+completion tokenization differs from separately encoded IDs"
        )
    if trl_combined[: len(prompt_with_bos)] != prompt_with_bos:
        raise ValueError("TRL prompt IDs are not a prefix of prompt+completion IDs")
    completion_mask = [0] * len(prompt_with_bos) + [1] * (
        len(trl_combined) - len(prompt_with_bos)
    )
    return {
        "task_id": row["task_id"],
        "split": row["split"],
        "certified_distance": row["certified_distance"],
        "prompt_ids": prompt_ids,
        "prompt_with_bos_ids": prompt_with_bos,
        "completion_ids": completion_ids,
        "completion_with_eos_ids": completion_with_eos,
        "combined_ids": combined,
        "completion_mask": completion_mask,
    }


def _histogram_quantile(
    histogram: Counter[int], numerator: int, denominator: int
) -> int:
    count = sum(histogram.values())
    if not count:
        raise ValueError("Cannot compute a quantile of an empty histogram")
    rank = max(1, math.ceil(count * numerator / denominator))
    cumulative = 0
    for value, frequency in sorted(histogram.items()):
        cumulative += frequency
        if cumulative >= rank:
            return value
    raise AssertionError("Histogram rank exceeded its count")


def _length_statistics(histogram: Counter[int]) -> dict[str, Any]:
    count = sum(histogram.values())
    total = sum(length * frequency for length, frequency in histogram.items())
    if not count:
        return {
            "count": 0,
            "total": 0,
            "min": None,
            "max": None,
            "mean": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "histogram": {},
        }
    return {
        "count": count,
        "total": total,
        "min": min(histogram),
        "max": max(histogram),
        "mean": total / count,
        "p50": _histogram_quantile(histogram, 50, 100),
        "p90": _histogram_quantile(histogram, 90, 100),
        "p95": _histogram_quantile(histogram, 95, 100),
        "p99": _histogram_quantile(histogram, 99, 100),
        "histogram": {
            str(length): frequency for length, frequency in sorted(histogram.items())
        },
    }


def _statistics_payload(
    counters: Mapping[str, Mapping[int, Mapping[str, Counter[int]]]],
    scope: str,
) -> dict[str, Any]:
    def summarize(distance: int) -> dict[str, Any]:
        return {
            field: _length_statistics(counters[scope][distance][field])
            for field in ("prompt", "completion", "completion_with_eos", "combined")
        }

    aggregate: dict[str, Counter[int]] = {
        field: Counter()
        for field in ("prompt", "completion", "completion_with_eos", "combined")
    }
    for distance in REQUIRED_DISTANCES:
        for field, histogram in aggregate.items():
            histogram.update(counters[scope][distance][field])
    return {
        "definitions": {
            "prompt": "prompt IDs without BOS",
            "completion": "completion IDs without EOS",
            "completion_with_eos": "completion IDs followed by exactly one EOS",
            "combined": "BOS + prompt + completion + EOS",
        },
        "overall": {
            field: _length_statistics(histogram)
            for field, histogram in aggregate.items()
        },
        "by_certified_distance": {
            str(distance): summarize(distance) for distance in REQUIRED_DISTANCES
        },
    }


def build_tokenization_audit(
    tokenizer: PreTrainedTokenizerFast,
    rows: Sequence[Mapping[str, Any]],
    *,
    max_length: int = FIXED_MAX_LENGTH,
) -> dict[str, Any]:
    """Freeze selection, full length distributions, and ordered token-ID hashes."""

    if max_length != FIXED_MAX_LENGTH:
        raise ValueError(f"Tokenization max_length must remain {FIXED_MAX_LENGTH}")
    _validate_rows(rows, require_all_distances=False)
    scopes = ("selected", "retained")
    fields = ("prompt", "completion", "completion_with_eos", "combined")
    counters: defaultdict[str, defaultdict[int, dict[str, Counter[int]]]]
    counters = defaultdict(
        lambda: defaultdict(lambda: {field: Counter() for field in fields})
    )
    digesters = {
        scope: {field: hashlib.sha256() for field in fields} for scope in scopes
    }
    retained: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []

    def update_digest(
        scope: str, field: str, task_id: str, token_ids: list[int]
    ) -> None:
        digesters[scope][field].update(
            _canonical_json([task_id, token_ids]).encode("utf-8")
        )
        digesters[scope][field].update(b"\n")

    for row in rows:
        encoded = encode_training_row(tokenizer, row)
        distance = encoded["certified_distance"]
        lengths = {
            "prompt": len(encoded["prompt_ids"]),
            "completion": len(encoded["completion_ids"]),
            "completion_with_eos": len(encoded["completion_with_eos_ids"]),
            "combined": len(encoded["combined_ids"]),
        }
        token_lists = {
            "prompt": encoded["prompt_ids"],
            "completion": encoded["completion_ids"],
            "completion_with_eos": encoded["completion_with_eos_ids"],
            "combined": encoded["combined_ids"],
        }
        for field in fields:
            counters["selected"][distance][field][lengths[field]] += 1
            update_digest("selected", field, encoded["task_id"], token_lists[field])
        selection_row = {
            "task_id": encoded["task_id"],
            "split": encoded["split"],
            "certified_distance": distance,
            "prompt_tokens": lengths["prompt"],
            "completion_tokens": lengths["completion"],
            "completion_tokens_with_eos": lengths["completion_with_eos"],
            "combined_tokens": lengths["combined"],
        }
        if lengths["combined"] <= max_length:
            retained.append(selection_row)
            for field in fields:
                counters["retained"][distance][field][lengths[field]] += 1
                update_digest("retained", field, encoded["task_id"], token_lists[field])
        else:
            excluded.append(
                {
                    **selection_row,
                    "reason": "combined_tokens_exceed_fixed_max_length",
                    "max_length": max_length,
                }
            )

    if not retained:
        raise ValueError("Fixed maximum length retained no corpus examples")
    token_id_digests = {
        scope: {
            field: f"sha256:{digesters[scope][field].hexdigest()}" for field in fields
        }
        for scope in scopes
    }
    selected_ids = [str(row["task_id"]) for row in rows]
    retained_ids = [row["task_id"] for row in retained]
    excluded_ids = [row["task_id"] for row in excluded]
    return {
        "policy": (
            "BOS + prompt + completion + EOS; retain iff the complete sequence "
            f"has at most {max_length} tokens"
        ),
        "max_length": max_length,
        "selected_records": len(rows),
        "retained_records": len(retained),
        "excluded_records": len(excluded),
        "selected_task_ids_digest": _canonical_digest(selected_ids),
        "retained_task_ids_digest": _canonical_digest(retained_ids),
        "excluded_task_ids_digest": _canonical_digest(excluded_ids),
        "ordered_token_id_digest_algorithm": (
            "sha256 of canonical JSON [task_id, token_ids] plus LF in source order"
        ),
        "token_id_digests": token_id_digests,
        "retained": retained,
        "excluded": excluded,
        "statistics": {scope: _statistics_payload(counters, scope) for scope in scopes},
    }


def _tokenizer_identity(
    tokenizer: PreTrainedTokenizerFast,
    basis: BasisSet,
    artifact_dir: Path,
    config: SharedTokenizerConfig,
) -> dict[str, Any]:
    import tokenizers
    import transformers

    file_hashes = {
        name: {
            "bytes": (artifact_dir / name).stat().st_size,
            "sha256": _sha256_file(artifact_dir / name),
        }
        for name in TOKENIZER_FILENAMES
    }
    tokenizer_json = json.loads(
        (artifact_dir / "tokenizer.json").read_text(encoding="utf-8")
    )
    return {
        "implementation": {
            "tokenizers_version": tokenizers.__version__,
            "transformers_version": transformers.__version__,
            "model": "BPE",
            "pre_tokenizer": "ByteLevel",
            "decoder": "ByteLevel",
            "post_processor": "TemplateProcessing: BOS + sequence",
            "byte_fallback": True,
            "byte_bridge": "latin-1 bijection for arbitrary raw bytes",
            "normalizer": None,
            "target_bpe_vocab_size": config.vocab_size,
            "min_frequency": config.min_frequency,
        },
        "max_length": config.max_length,
        "base_vocab_size": tokenizer.vocab_size,
        "total_vocab_size": len(tokenizer),
        "canonical_tokenizer_json_digest": _canonical_digest(tokenizer_json),
        "files": file_hashes,
        "special_tokens": [
            {
                "role": role,
                "content": content,
                "id": tokenizer.convert_tokens_to_ids(content),
            }
            for role, content in FIXED_SPECIAL_TOKENS
        ],
        "basis_function_tokens": [
            {
                "name": name,
                "id": tokenizer.convert_tokens_to_ids(name),
                "special": False,
            }
            for name in basis_function_names(basis)
        ],
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _compact_frozen_selection(audit: Mapping[str, Any]) -> dict[str, Any]:
    """Keep all exclusions and content hashes without duplicating every row.

    The retained rows are deterministically reconstructed and re-audited by each
    consumer. Storing hundreds of thousands of per-row summaries in git would
    add no identity beyond the ordered task/token digests already frozen here.
    """

    selection = dict(audit)
    retained = selection.pop("retained", None)
    if not isinstance(retained, list) or len(retained) != selection.get(
        "retained_records"
    ):
        raise ValueError("Tokenization audit has an invalid retained-row list")
    selection["retained_rows"] = {
        "storage": "omitted-from-artifact; deterministically reconstructed",
        "ordering": "source corpus order after fixed maximum-length filter",
        "identity": "retained_task_ids_digest and retained token_id_digests",
    }
    return selection


def freeze_shared_tokenizer(
    output_dir: str | Path,
    *,
    rows: Sequence[Mapping[str, Any]],
    basis: BasisSet,
    source_identity: Mapping[str, Any],
    config: SharedTokenizerConfig | None = None,
) -> dict[str, Any]:
    """Train, atomically save, hash, reload, and validate one frozen artifact."""

    if config is None:
        config = SharedTokenizerConfig()
    _validate_config(config)
    _validate_discovery_training_rows(rows)
    _validate_source_identity(source_identity, rows, basis)
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(
            f"Tokenizer artifact already exists; choose a fresh path: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.building-", dir=destination.parent
        )
    )
    try:
        tokenizer = train_shared_tokenizer(rows, basis, config=config)
        saved = tokenizer.save_pretrained(temporary)
        saved_names = {Path(path).name for path in saved}
        if saved_names != set(TOKENIZER_FILENAMES):
            raise RuntimeError(
                "Unexpected tokenizer artifact files: "
                f"expected={sorted(TOKENIZER_FILENAMES)}, got={sorted(saved_names)}"
            )
        selection = _compact_frozen_selection(
            build_tokenization_audit(
                tokenizer,
                rows,
                max_length=config.max_length,
            )
        )
        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": ARTIFACT_TYPE,
            "basis_set": basis.identity_dict(),
            "source": dict(source_identity),
            "training_text": {
                "split": DISCOVERY_SPLIT,
                "discovery_only": True,
                "records": len(rows),
                "texts": 2 * len(rows),
                "ordered_prompt_completion_digest": _line_digest(
                    _training_text_rows(rows)
                ),
            },
            "tokenizer": _tokenizer_identity(
                tokenizer,
                basis,
                temporary,
                config,
            ),
            "selection": selection,
        }
        manifest["artifact_digest"] = _canonical_digest(manifest)
        _write_json(temporary / MANIFEST_FILENAME, manifest)
        validate_frozen_tokenizer(temporary)
        temporary.rename(destination)
        return validate_frozen_tokenizer(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _validate_source_identity(
    source: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    basis: BasisSet,
) -> None:
    if source.get("corpus") != "deep-corpus-v2":
        raise ValueError("Tokenizer source must identify deep-corpus-v2")
    if (
        source.get("basis_set_id") != basis.basis_set_id
        or source.get("basis_set_digest") != basis.digest
    ):
        raise ValueError("Tokenizer source and loaded basis identity differ")
    if not _SHA256_RE.fullmatch(str(source.get("manifest_digest", ""))):
        raise ValueError("Tokenizer source has no corpus manifest digest")
    hub = source.get("hub")
    if not isinstance(hub, Mapping) or hub.get("repo_id") != EXPECTED_CORPUS_REPO_ID:
        raise ValueError("Tokenizer source has the wrong V2 Hub repository")
    if not _HF_COMMIT_RE.fullmatch(str(hub.get("revision", ""))):
        raise ValueError("Tokenizer source has no immutable Hub revision")
    discovery = source.get("discovery")
    ordered_ids = [str(row["task_id"]) for row in rows]
    if (
        not isinstance(discovery, Mapping)
        or discovery.get("split") != DISCOVERY_SPLIT
        or discovery.get("records") != len(rows)
        or discovery.get("ordered_task_ids_digest") != _canonical_digest(ordered_ids)
    ):
        raise ValueError("Tokenizer source does not match ordered discovery rows")


def validate_frozen_tokenizer(
    artifact_dir: str | Path,
    *,
    expected_source_manifest_digest: str | None = None,
) -> dict[str, Any]:
    """Reload and validate all hashes and tokenizer protocol invariants."""

    from transformers import PreTrainedTokenizerFast

    root = Path(artifact_dir)
    manifest_path = root / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Frozen tokenizer manifest is missing: {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Frozen tokenizer has an unsupported schema version")
    if manifest.get("artifact_type") != ARTIFACT_TYPE:
        raise ValueError("Artifact is not the deep-corpus-v2 shared tokenizer")
    stored_digest = manifest.get("artifact_digest")
    digest_payload = dict(manifest)
    digest_payload.pop("artifact_digest", None)
    expected_digest = _canonical_digest(digest_payload)
    if stored_digest != expected_digest:
        raise ValueError(
            f"Tokenizer artifact digest mismatch: {stored_digest!r} != "
            f"{expected_digest!r}"
        )
    if (
        expected_source_manifest_digest is not None
        and manifest.get("source", {}).get("manifest_digest")
        != expected_source_manifest_digest
    ):
        raise ValueError("Frozen tokenizer was trained from another corpus manifest")

    files = manifest.get("tokenizer", {}).get("files")
    if not isinstance(files, Mapping) or set(files) != set(TOKENIZER_FILENAMES):
        raise ValueError("Frozen tokenizer file identity is incomplete")
    for name in TOKENIZER_FILENAMES:
        path = root / name
        identity = files[name]
        if not path.is_file():
            raise FileNotFoundError(f"Frozen tokenizer file is missing: {path}")
        if path.stat().st_size != identity.get("bytes") or _sha256_file(
            path
        ) != identity.get("sha256"):
            raise ValueError(f"Frozen tokenizer file digest mismatch: {name}")
    tokenizer_payload = json.loads((root / "tokenizer.json").read_text("utf-8"))
    if _canonical_digest(tokenizer_payload) != manifest["tokenizer"].get(
        "canonical_tokenizer_json_digest"
    ):
        raise ValueError("Frozen tokenizer canonical content digest mismatch")

    basis_identity = manifest.get("basis_set", {})
    basis = load_basis_set(basis_identity.get("basis_set_id", ""))
    if basis.identity_dict() != basis_identity:
        raise ValueError("Frozen tokenizer basis identity no longer matches")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(root, local_files_only=True)
    validate_tokenizer_runtime(
        tokenizer,
        basis,
        expected_max_length=manifest["tokenizer"]["max_length"],
    )
    expected_specials = [
        {
            "role": role,
            "content": content,
            "id": tokenizer.convert_tokens_to_ids(content),
        }
        for role, content in FIXED_SPECIAL_TOKENS
    ]
    if manifest["tokenizer"].get("special_tokens") != expected_specials:
        raise ValueError("Frozen tokenizer special-token identity drifted")
    expected_names = [
        {
            "name": name,
            "id": tokenizer.convert_tokens_to_ids(name),
            "special": False,
        }
        for name in basis_function_names(basis)
    ]
    if manifest["tokenizer"].get("basis_function_tokens") != expected_names:
        raise ValueError("Frozen tokenizer basis-token identity drifted")
    if manifest["tokenizer"].get("total_vocab_size") != len(tokenizer):
        raise ValueError("Frozen tokenizer vocabulary size drifted")
    selection = manifest.get("selection")
    if not isinstance(selection, Mapping):
        raise ValueError("Frozen tokenizer has no selection audit")
    selected_records = selection.get("selected_records")
    retained_records = selection.get("retained_records")
    excluded_records = selection.get("excluded_records")
    excluded = selection.get("excluded")
    if (
        type(selected_records) is not int
        or type(retained_records) is not int
        or type(excluded_records) is not int
        or not isinstance(excluded, list)
        or selected_records != retained_records + excluded_records
        or excluded_records != len(excluded)
        or selection.get("max_length") != FIXED_MAX_LENGTH
        or "retained" in selection
        or selection.get("retained_rows", {}).get("storage")
        != "omitted-from-artifact; deterministically reconstructed"
    ):
        raise ValueError("Frozen tokenizer selection audit is inconsistent")
    return manifest


def dry_run_summary(source: VerifiedDiscoveryCorpus) -> dict[str, Any]:
    """Return a side-effect-free preview before tokenizer training or writes."""

    return {
        "status": "dry-run",
        "artifact_type": ARTIFACT_TYPE,
        "fixed_max_length": FIXED_MAX_LENGTH,
        "fixed_special_tokens": [
            {"role": role, "content": content, "id": index}
            for index, (role, content) in enumerate(FIXED_SPECIAL_TOKENS)
        ],
        "basis_set": source.basis.identity_dict(),
        "basis_function_tokens": len(basis_function_names(source.basis)),
        "source": dict(source.source_identity),
        "preview": [
            {
                "task_id": row["task_id"],
                "certified_distance": row["certified_distance"],
                "prompt": row["prompt"],
                "completion": row["completion"],
            }
            for row in source.rows[:8]
        ],
        "note": (
            "All corpus files, splits, basis identity, and witnesses were verified; "
            "only discovery rows were materialized. No tokenizer was trained or saved."
        ),
    }
