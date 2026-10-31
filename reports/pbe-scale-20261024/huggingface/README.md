---
license: apache-2.0
language:
  - en
pretty_name: Wandering Light Stack PBE Scale v1
size_categories:
  - 1M<n<10M
tags:
  - synthetic
  - program-synthesis
  - programming-by-example
  - stack-machine
configs:
  - config_name: default
    default: true
    data_files:
      - split: train
        path: data/tasks_train.jsonl.gz
  - config_name: views16
    data_files:
      - split: train
        path: data/tasks_train_16views.jsonl.gz
---

# Wandering Light Stack PBE Scale v1

Synthetic **training data for programming by example (PBE)**: infer a program that manipulates a stack of values from six input/output pairs. This release contains **68,499 retained behaviors**, each with a checked stack witness (a program reproducing its checked outputs) and a separately expressed Python reference, plus **1,095,984 reshuffled task views** of those same behaviors.

| Configuration / file | Rows | Contents |
|---|---:|---|
| `default`: `data/tasks_train.jsonl.gz` | 68,499 | One original task per behavior; six shown pairs and all remaining checked pairs |
| `views16`: `data/tasks_train_16views.jsonl.gz` | 1,095,984 | 16 six-pair views per behavior; fixed hidden bank |
| `metadata/behaviors.jsonl.gz` | 68,499 | Candidate/reference and verification metadata; separate schema from task rows |

Both configurations have only a `train` split. This corpus supplements training; the project's existing [v4 dataset](https://huggingface.co/datasets/abhishekraok/wandering-light-stack-pbe-v4/tree/00c2a9996d32df8d1894b45f4246a66b6df46c25) remains pinned at revision `00c2a9996d32df8d1894b45f4246a66b6df46c25`, with its evaluation unchanged. The `held_out_*` columns contain within-task checks, not an independent evaluation split. The configurations share behaviors and witnesses: do not use one as validation for the other, or randomly split views to measure generalization to new functions.

```python
from datasets import load_dataset

repo = "abhishekraok/wandering-light-stack-pbe-scale-v1"
train = load_dataset(repo, split="train")
views = load_dataset(repo, "views16", split="train", streaming=True)
row = next(iter(views))
print(row["inputs"], row["outputs"], row["program"])
```

The explicit YAML `data_files` keep metadata JSONL out of the task loader.

## How the data was generated

| Candidate source | Candidates | Retained after deduplication and overlap exclusion |
|---|---:|---:|
| Typed compositions | 50,000 | 29,477 |
| Long typed compositions | 30,000 | 22,204 |
| Explicit algorithm compositions | 20,000 | 13,308 |
| Directly Codex-authored | 6,648 | 3,353 |
| Bootstrap | 24 | 10 |
| Simple anchors | 153 | 147 |
| **Total** | **106,825** | **68,499** |

**Typed compositions.** Deterministic, seeded enumeration combines 2–4 stages from **23 declared list transformations**: pointwise operations, filters, deduplication, ordering, running sums, neighboring-value operations, centering and mean-based selection. Starts use one list, a list and scalar threshold/baseline, or two lists combined by aligned arithmetic or concatenation. Readouts return a list or scalar summary, such as count, sum, range or position-weighted sum. Wrappers can compare the result with the original input or neighboring differences, or conditionally reverse the input. Obvious redundant/constant combinations are filtered. The long source uses the same vocabulary with longer witnesses.

**Explicit algorithm compositions.** A second deterministic enumeration combines **39 named sequence algorithms** with input transforms, optional postprocessing and readouts. Examples include running extrema, record sequences, run lengths, local peaks, occurrence counts, ranks and prefix relations. These sources enumerate a finite composition space in seeded order; each row is not a separate model invention or independent algorithmic idea. Numeric-literal sweeps do not supply their volume.

Each component supplies both a stack implementation and a Python expression/body; the generator assembles the chosen composition in both languages. The implementations **share the composition design** while expressing its components separately. Executing them independently checks implementation agreement, not independent algorithmic coverage. For example, filtering above an argument, reversing, retaining negatives and taking the range has an unnecessary reversal, since the final range is permutation invariant.

**Direct authoring.** The [saved campaign configuration](metadata/authoring-config.json) used `gpt-6.1-sol` through `codex exec`, with reasoning effort `low`. Batches received machine semantics, canonical macros, a themed prompt and the candidate schema. They were asked to write meaningful macro programs and pure Python `def f(...)` references from the stated semantics, without interpreting stack code or mechanically translating stack operations. The model supplied both implementations; there was no separate reference author. Direct authoring is distinct from deterministic enumeration. Bootstrap and anchor candidates add simple seed witnesses that can replace inflated witnesses during merging. No stage collected or translated library execution traces; the table above lists the generation methods used.

## Inputs, verification and selection

The intended domain is integers **−10 through 10**, and flat lists of those integers with lengths **0 through 12**, including empties, zeros and negatives. Signatures are `f(xs)`, `f(xs, n)`, `f(xs, ys)`, `f(n)` and `f(a, b)`. Outputs must be integers or flat integer lists, with at most 256 entries and absolute integer values at most 10¹². Acceptance does not prove totality over all lists in this domain or specify behavior outside it.

Validation uses seed `20261024` and probe policy `v3-edge-cartesian`:

| Signature | Shared probes | Candidate-specific probes | Checked pairs | Original shown / held out |
|---|---:|---:|---:|---:|
| One list; list + integer; two lists | 192 | 16 | 208 | 6 / 202 |
| One integer | 21 exhaustive | 0 | 21 | 6 / 15 |
| Two integers | 441 exhaustive | 0 | 441 | 6 / 435 |

Shared banks start with edge cases: empties, short lists, repeated values, alternating signs, extrema and longer sequences. List-plus-integer banks cross short edge lists with scalar values; two-list banks include a Cartesian bank of short lists. Remaining cases use seeded sampling: list lengths uniformly from 0–12, values uniformly from −10–10. Candidate-specific probes use a seed derived from the candidate digest and exclude shared inputs. The shared bank is deliberately nonuniform, with many short inputs.

References pass syntax/capability checks and execute in bounded child processes. Macro-expanded stack witnesses must terminate within the interpreter's fuel limit and leave exactly one value matching the Python output on every checked input. Mismatches, refusals and resource failures are rejected. Retained witnesses passed **14,255,789 reference/stack pairs**. Loading the saved dataset executes neither references nor generated programs.

Deduplication fingerprints the input signature and reference outputs on the **shared bank**. Constants and projections of input arguments are rejected there. Candidate-specific probes verify witnesses but do not define fingerprints. Validation and merging remove **30,615 shared-probe duplicates, 6,967 shared-probe constants and 732 other failures**, leaving **68,511** representatives. Among verified alternatives, merging selects the **shortest known expanded witness**, breaking equal-length ties by candidate digest. This is empirical equivalence on finite probes, not proof of semantic identity or minimality: different functions can collide and be removed, and unseen duplicates can remain.

Original prompts preferentially select candidate-specific pairs with varied first-list lengths and at least two distinct outputs; the remaining checked pairs become `held_out_*`. Every retained original task has at least two distinct shown outputs.

## Finite overlap exclusion

A local audit compared candidate Python references with all **45 frozen v4 evaluation references**, using up to **64 distinct sampler-valid inputs per same-signature comparison**, drawn from 512 attempts. Exact sampled agreement removed **12 candidates**, leaving **68,499**. The **108 inconclusive candidates** each accept one integer, a signature absent from this evaluation; they remain in training. All 45 evaluation references were resolved.

This excludes detected exact matches only. It does not establish strict family disjointness, absence of related compositions, or absence of semantic overlap beyond the finite inputs. The evaluation payload remains unchanged; this release contains only the new training corpus. See [audit evidence](audits/exclusions.json).

## What the 16 views add

Augmentation uses seed `0` and **reuses already checked pairs**, without executing programs or adding functions. Each behavior gets 16 distinct six-input shown sets with varied outputs, drawn from a common pool. Views may share shown pairs. A fixed hidden bank is disjoint from every shown view for that behavior: **5 of 21 pairs** for one-integer tasks, otherwise **32 of 208 or 441 pairs**. The export contains **6,575,904 shown pairs**; it multiplies presentations of 68,499 behaviors rather than creating a million independent functions. Its hidden pairs can have appeared as shown pairs in the original configuration.

## Task schema and metadata

Task rows follow the project's `Task` schema (`schema_version=2`, `basis="mark-28"`).

| Fields | Meaning |
|---|---|
| `behaviour`, `summary` | Candidate identifier; short description (truncated to 200 characters) |
| `inputs`, `outputs` | Six paired **stack-format strings**, not JSON argument arrays |
| `held_out_inputs`, `held_out_outputs` | Other paired stack-format strings, excluded from the shown prompt |
| `program`, `length` | Macro-expanded token list and token count, including quotation syntax |
| `macro_program` | Original macro program string |
| `examples`, `held_out` | Shown and held-out pair counts |
| `fuel` | Original: summed measured fuel on six shown inputs. `views16`: **0 is an unmeasured sentinel**, also tagged `fuel-unmeasured` |
| `tags`, `skeleton` | Declared family/template, optional view tag; program shape with literals abstracted |
| `split`, `task_hash_split` | `split` is always `train`; `task_hash_split` is a legacy hash annotation, not a release split |
| `witness_source`, `alternative_programs` | Generic `"authored"` label and empty alternatives list; `"authored"` does **not** identify directly model-authored rows |

For example, `[ (-5 -3 2) ]` is a stack containing one list, and `[ -6 ]` a stack containing its scalar sum. Each string represents the whole initial or final stack.

`metadata/behaviors.jsonl.gz` preserves accepted validation records excluding embedded tasks and machine-path bookkeeping. Its `candidate` object contains the complete description, Python `reference`, original macro `program`, input kind, family/template and any parameters/private macros. It also preserves candidate digests, empirical signatures, verification counts and provenance IDs. Use these records and [source snapshots](sources/experiments/pbe_scale/generate.py) to inspect generation evidence; `witness_source` alone does not distinguish sources. Optional candidate fields vary between records; read this metadata as JSON rather than imposing one inferred table schema:

```python
import gzip
import json
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo, "metadata/behaviors.jsonl.gz", repo_type="dataset"
)
with gzip.open(path, "rt") as stream:
    metadata = (json.loads(line) for line in stream)
    print(next(metadata)["candidate"]["reference"])
```

## Diversity and length limits

Retained rows comprise 51,681 typed pipelines (75.4%), 13,308 algorithm compositions (19.4%), 3,353 directly Codex-authored rows (4.9%) and 157 anchors/bootstrap rows. Pipelines contain 8,756 ordered stage sequences excluding starts, wrappers and readouts. This is compositional variety within a small vocabulary; neither family labels nor row counts certify independent concepts.

| Expanded witness tokens | Behaviors |
|---|---:|
| 1–49 | 726 |
| 50–99 | 7,390 |
| 100–199 | 35,794 |
| 200–399 | 24,283 |
| 400–845 | 306 |

There are **68,467 distinct expanded witness strings**; different input-signature namespaces can reuse the same witness. Both deterministic generators filter by expanded-token length: retained pipelines span 50–400 tokens and explicit algorithm compositions 25–200; directly authored programs can be longer. Length is an upper bound from an available witness, not certified difficulty. The structural audit finds a removable declared stage in **7,722 witnesses (11.27%)**. Pipeline medians are 193 expanded tokens versus 30 macro tokens despite only 2–4 transformation stages.

On the shared bank used for deduplication, **21,388 rows (31.22%)** have a same-signature neighbor matching at least 95% of outputs. This is not a duplicate-function count or independent novelty test. Input coverage matters: 170 of 192 shared list-plus-integer probes have list length at most two; the same nearest pairs' median agreement falls from 96.88% across the bank to 70.59% on its 17 cases of length at least four. Original prompts are broader: **76.06%** of their shown list-plus-integer pairs have list length at least four. These observations do not establish longer-input or longer-program generalization. Evidence: [structure](audits/diversity/structural-summary.json), [output agreement](audits/diversity/semantic-summary.json), [input sensitivity](audits/diversity/semantic-sensitivity.json), [prompt coverage](audits/diversity/prompt-coverage.json).

![Composition reuse and input sensitivity](audits/diversity/diversity.png)

## Provenance and project

[manifest.json](manifest.json) records release hashes, counts, configuration and provenance; `sources/` preserves release-time source snapshots, including the [authoring driver](sources/experiments/pbe_scale/campaign.py), [anchors](sources/experiments/pbe_scale/anchors.py), [project dependencies](sources/pyproject.toml) and [lockfile](sources/uv.lock). `metadata/provenance/` preserves original origin records. The [append-only provenance correction](audits/provenance-correction.json) explains that some historical commit/dirty fields describe the copied launcher's directory instead of its imported source checkout. Original artifact hashes remain preserved; the sidecar records runtime revision `25eb317` and the unavailable historical dirty-state evidence. Read that correction alongside original provenance fields; release-time snapshots are not a claim that all historical files were clean at that revision.

Source and construction/diversity report: [wandering-light](https://github.com/abhishekraok/wandering-light), [campaign report](https://github.com/abhishekraok/wandering-light/blob/cc4951d6f14703ec79426bbc359a35df60011f6c/reports/pbe-scale-20261024/README.md). License: [Apache-2.0](LICENSE), matching the project repository.
