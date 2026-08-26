"""Prepare, verify, and publish the frozen deep-corpus-v2 profile for issue #54."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

from experiments import generate_deep_corpus as deep
from wandering_light import corpus_hub

if TYPE_CHECKING:
    from collections.abc import Sequence


PROFILE_NAME = "deep-corpus-v2-issue-54"
BASIS_SET_ID = "wl-core-v1"
BASIS_SET_DIGEST = (
    "sha256:ffc602fb24db249df7eb4f6b0d5ba38d5a5070b2c68c6c5f94c7f387de682494"
)
SEED = 20_260_826
OUTPUT_DIR = Path("wandering_light/training/data/deep_corpus_v2")
V1_CORPUS_DIR = Path("wandering_light/training/data/deep_corpus_v1")
V1_MANIFEST_DIGEST = (
    "sha256:0efed8ca6c57942c1672ed0a2e6e3abe87b009168f9b26140408458ded8f41af"
)
V1_HUB_REVISION = "fd22029677638702b6b13c572757a71e2f81335b"
HUB_REPO_ID = "abhishekraok/wandering-light-deep-corpus-v2"
REPO_ROOT = Path(__file__).resolve().parents[1]
_GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")

ROOTS = 2_400
SPLIT_COUNTS = {"discovery": 1_880, "validation": 260, "test": 260}
DEEP_ROOTS = 360
DEEP_ROOT_TYPES = (
    "builtins.bool",
    "builtins.list",
    "builtins.tuple",
    "builtins.set",
    "builtins.dict",
    "builtins.bytes",
    "builtins.bytearray",
    "builtins.complex",
    "builtins.range",
)
DISTANCE_TARGETS = {
    "1": 18_800,
    "2": 47_500,
    "3": 48_000,
    "4": 48_000,
    "5": 48_000,
    "6": 48_000,
    "7": 48_000,
    "8": 7_200,
}


def default_resume_dir(output_dir: str | Path) -> Path:
    """Keep checkpoints beside, rather than inside, the atomically finalized corpus."""
    output_path = Path(output_dir)
    return output_path.parent / f".{output_path.name}.resume"


def target_contract(
    reference_manifest_digest: str = V1_MANIFEST_DIGEST,
    *,
    generator_git_commit: str | None = None,
) -> dict[str, Any]:
    """The issue-54 acceptance contract embedded in the immutable manifest."""
    contract: dict[str, Any] = {
        "profile": PROFILE_NAME,
        "root_count": ROOTS,
        "split_root_counts": dict(SPLIT_COUNTS),
        "certified_distance_range": [1, 8],
        "distance_targets": dict(DISTANCE_TARGETS),
        "distance_relative_tolerance": 0.05,
        # A root may hit its resource cap while expanding the next BFS layer.
        # The generator emits only examples at already-completed layers (plus
        # the separately certified one-step frontier), so root-level exhaustion
        # is not required for exact shortest-distance certification.
        "reference_manifest_digests": [reference_manifest_digest],
    }
    if generator_git_commit is not None:
        if not _GIT_COMMIT_RE.fullmatch(generator_git_commit):
            raise ValueError("generator_git_commit must be a full 40-hex commit")
        contract["generator_source"] = {
            "git_commit": generator_git_commit,
            "tree_state": "clean",
            "runner": "experiments/prepare_deep_corpus_v2.py",
            "generator": "experiments/generate_deep_corpus.py",
        }
    return contract


def require_clean_generator_source(repo_root: Path = REPO_ROOT) -> str:
    """Return HEAD only when every generator dependency is committed."""

    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise RuntimeError(
            "Corpus generation requires a clean source tree so its git commit "
            "fully identifies the certification implementation"
        )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not _GIT_COMMIT_RE.fullmatch(commit):
        raise RuntimeError(f"Git returned an invalid full commit: {commit!r}")
    return commit


def frozen_generation_options(
    *,
    output_dir: str | Path = OUTPUT_DIR,
    reference_identities: Sequence[deep.CorpusIdentities] = (),
    workers: int = 1,
    resume_dir: str | Path | None = None,
    generator_git_commit: str | None = None,
    overwrite: bool = False,
    progress: bool = True,
) -> dict[str, Any]:
    """Return the exact generation call for the primary V2 corpus."""
    output_path = Path(output_dir)
    return {
        "basis_set_id": BASIS_SET_ID,
        "output_dir": output_path,
        "roots": ROOTS,
        "max_depth": 6,
        "deep_roots": DEEP_ROOTS,
        "deep_max_depth": 7,
        "min_distance": 1,
        "tasks_per_distance": 20,
        "frontier_sample": 24,
        "max_states": 600_000,
        "max_transitions": 2_500_000,
        "deep_max_states": 1_200_000,
        "deep_max_transitions": 8_000_000,
        "deep_root_types": DEEP_ROOT_TYPES,
        "seed": SEED,
        "split_counts": SPLIT_COUNTS,
        "reference_identities": tuple(reference_identities),
        "target_contract": target_contract(generator_git_commit=generator_git_commit),
        "workers": workers,
        "resume_dir": (
            default_resume_dir(output_path) if resume_dir is None else Path(resume_dir)
        ),
        "allow_constant_outputs": False,
        "verify_witnesses": True,
        "overwrite": overwrite,
        "progress": progress,
    }


def _reference_identities(
    corpus_dir: Path, *, fetch_if_missing: bool
) -> deep.CorpusIdentities:
    manifest_path = corpus_dir / corpus_hub.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("manifest_digest") != V1_MANIFEST_DIGEST:
        raise ValueError(
            "V1 reference manifest digest does not match the frozen profile"
        )
    if manifest.get("basis_set_id") != BASIS_SET_ID:
        raise ValueError("V1 reference uses the wrong basis set")
    if manifest.get("basis_set_digest") != BASIS_SET_DIGEST:
        raise ValueError("V1 reference uses the wrong basis digest")
    if manifest.get("hub", {}).get("revision") != V1_HUB_REVISION:
        raise ValueError("V1 reference Hub revision does not match the frozen profile")
    try:
        corpus_hub.verify_corpus_files(corpus_dir)
    except FileNotFoundError:
        if not fetch_if_missing:
            raise
        corpus_hub.fetch_corpus(manifest_path, destination=corpus_dir)
    return deep.collect_corpus_identities(corpus_dir)


def _verify(
    output_dir: Path,
    reference: deep.CorpusIdentities,
    *,
    recertify: int,
) -> dict[str, Any]:
    result = deep.verify_corpus(
        output_dir,
        recertify=recertify,
        seed=SEED,
        reference_identities=(reference,),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("profile", "run", "verify", "publish"), default="profile"
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--v1-corpus-dir", type=Path, default=V1_CORPUS_DIR)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--resume-dir",
        type=Path,
        default=None,
        help="atomic root-checkpoint directory; defaults beside output-dir",
    )
    parser.add_argument("--recertify", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--fetch-reference", action=argparse.BooleanOptionalAction, default=True
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "profile":
        options = frozen_generation_options(
            output_dir=args.output_dir,
            workers=args.workers,
            resume_dir=args.resume_dir,
            overwrite=args.overwrite,
            progress=not args.quiet,
        )
        serializable = {
            key: value
            for key, value in options.items()
            if key not in {"reference_identities", "progress", "overwrite"}
        }
        for key in ("output_dir", "resume_dir"):
            serializable[key] = str(serializable[key])
        print(json.dumps(serializable, indent=2, sort_keys=True))
        return 0

    reference = _reference_identities(
        args.v1_corpus_dir, fetch_if_missing=args.fetch_reference
    )
    resume_dir = (
        default_resume_dir(args.output_dir)
        if args.resume_dir is None
        else args.resume_dir
    )
    if args.mode == "run":
        generator_git_commit = require_clean_generator_source()
        manifest, path = deep.generate_corpus(
            **frozen_generation_options(
                output_dir=args.output_dir,
                reference_identities=(reference,),
                workers=args.workers,
                resume_dir=resume_dir,
                generator_git_commit=generator_git_commit,
                overwrite=args.overwrite,
                progress=not args.quiet,
            )
        )
        if manifest["basis_set_digest"] != BASIS_SET_DIGEST:
            raise RuntimeError("generated corpus resolved to the wrong basis digest")
        print(f"wrote {path}")
        result = _verify(args.output_dir, reference, recertify=args.recertify)
        if result["ok"]:
            deep.remove_resume_directory(resume_dir)
        return 0 if result["ok"] else 1
    if args.mode == "verify":
        result = _verify(args.output_dir, reference, recertify=args.recertify)
        if result["ok"]:
            deep.remove_resume_directory(resume_dir)
        return 0 if result["ok"] else 1

    result = _verify(args.output_dir, reference, recertify=args.recertify)
    if not result["ok"]:
        return 1
    deep.remove_resume_directory(resume_dir)
    revision = corpus_hub.publish_corpus(
        args.output_dir,
        repo_id=HUB_REPO_ID,
        commit_message=f"Publish {PROFILE_NAME}",
    )
    corpus_hub.verify_published_corpus(
        args.output_dir, repo_id=HUB_REPO_ID, revision=revision
    )
    corpus_hub.record_hub_location(
        args.output_dir, repo_id=HUB_REPO_ID, revision=revision
    )
    # The hub block is outside the corpus digest, but loading once more proves
    # that recording the immutable location did not invalidate the manifest.
    deep.load_corpus(args.output_dir)
    print(f"published {HUB_REPO_ID}@{revision}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
