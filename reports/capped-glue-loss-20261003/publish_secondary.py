"""Publish only already independently verified #178 secondary summaries."""

import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path

OUT = Path(__file__).resolve().parent
TRIAL = (
    Path.home() / "repos/wandering-light/checkpoints/capped-glue-loss-20261003/trial"
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pinned(artifact):
    path = Path(artifact["path"])
    assert digest(path) == artifact["sha256"], path
    return json.loads(path.read_text())


verification = json.loads((OUT / "independent-verification.json").read_text())
assert verification["verified"] and verification["models_deserialized"] == 0
assert digest(TRIAL / "final.json") == verification["final_sha256"]
final = json.loads((TRIAL / "final.json").read_text())
assert final["complete"] and final["exit_code"] == 0
stage = pinned(final["stage_a"])
assert final["stage_a"]["sha256"] == verification["stage_sha256"]
rows = stage["rows"]
metrics = [
    "unmasked_nll",
    "legal_nll",
    "negative_log_legal_mass",
    "best_legal_deficit",
    "legal_rank",
    "legal_size",
]


def grouped_means(labels, side, role=None):
    chosen = [
        x for x in labels if x["side"] == side and (role is None or x["class"] == role)
    ]
    n = sum(x["rows"] for x in chosen)
    return dict(
        rows=n, **{m: math.fsum(x[m]["sum"] for x in chosen) / n for m in metrics}
    )


def exposure(selected):
    examples = [e for r in selected for e in r["encoder_diagnostics"]["examples"]]
    result = {"rows": len(selected), "example_states": len(examples)}
    for name in ["state_pre_cap_width", "target_pre_cap_width"]:
        values = [e[name] for e in examples]
        result[name] = {
            "min": min(values),
            "median": statistics.median(values),
            "max": max(values),
            "histogram": dict(sorted(Counter(values).items())),
        }
    for name in ["state_truncated", "target_truncated"]:
        result[name + "_example_states"] = sum(e[name] for e in examples)
        result[name + "_rows"] = sum(
            r["encoder_diagnostics"]["any_" + name] for r in selected
        )
    result["body_depth_counts"] = dict(
        sorted(Counter(r["depth"] for r in selected if r["side"] == "body").items())
    )
    return result


counts = {}
for side in ["top", "body"]:
    content = [r for r in rows if r["side"] == side and not r["structural"]]
    matched = [r for r in content if r["matched"]]
    counts[side] = {
        "content": len(content),
        "common_support": len(matched),
        "omitted_content": len(content) - len(matched),
        "structural": sum(r["side"] == side and r["structural"] for r in rows),
        "before_opener": sum(r["before_opener"] for r in content),
        "between_closer_and_combinator": sum(
            r["between_closer_and_combinator"] for r in content
        ),
    }
widths = {
    "all": exposure(rows),
    "common_support": exposure([r for r in rows if r["matched"]]),
}
bundle = {
    "schema": "capped-glue-secondary-report-v1",
    "verified_final_sha256": verification["final_sha256"],
    "verified_stage_sha256": verification["stage_sha256"],
    "weighting": "occurrence weighted summaries; behaviour shares retain each behaviour; no new gate",
    "counts": counts,
    "counts_by_label": stage["counts"]["by_label"],
    "encoder_exposure": widths,
    "limits": stage["limits"],
    "alternative_first_refusal_counts": stage["alternative_first_refusal_counts"],
    "training_row_multiplicity": "unavailable_not_reconstructed",
    "checkpoints": [],
}
lines = [
    "## Required secondary reporting",
    "",
    "These are descriptive summaries of the same complete sealed panel. No selection,",
    "inference, threshold, gate or trial has changed. [SECONDARY.json](SECONDARY.json)",
    "retains the exact independently checked all/common joint label summaries",
    "(side, content/structural, span position, exact op and depth), every behaviour's",
    "token/loss/deficit shares, and the assembly-excluded cell/behaviour results.",
    "The publication script is retained beside this report as `publish_secondary.py`.",
    "",
    "### Common support and omitted exposure",
    "",
    "| side | all content | common support | omitted content | structural |",
    "|---|---:|---:|---:|---:|",
]
for side, c in counts.items():
    lines.append(
        f"| {side} | {c['content']} | {c['common_support']} | {c['omitted_content']} | {c['structural']} |"
    )
lines += [
    "",
    "All common-support content is middle of a combinator-delimited span.",
    "The omitted content includes unmatched middle as well as boundary tokens;",
    "structural tokens are excluded from the primary. The full joint breakdown is",
    "in SECONDARY.json. Top-content flags refer to any later top-owned opener or an",
    "earlier depth-one closer followed by a later run in the same span; they can overlap:",
    f"{counts['top']['before_opener']} occurrences before an opener and",
    f"{counts['top']['between_closer_and_combinator']} between a closer and combinator.",
    "",
    "### Legal support and loss decomposition",
    "",
    "The following means weight common-support **occurrences**, unlike the primary",
    "equal-cell/then-equal-behaviour means. Each row satisfies unmasked NLL = legal",
    "NLL + negative log legal probability mass. Rank counts strictly greater legal",
    "logits; ties share rank. Deficit is best legal logit minus witness logit.",
    "",
    "| encoding | seed | side | rows | unmasked NLL | legal NLL | -log legal mass | deficit | rank | legal size |",
    "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
]
assembly_lines = []
share_lines = []
for checkpoint in final["checkpoints"]:
    summary = pinned(checkpoint["summary"])
    assert (
        summary["encoding"] == checkpoint["encoding"]
        and summary["seed"] == checkpoint["seed"]
    )
    assert summary["all_rows"]["rows"] == len(rows)
    assert summary["common_support_rows"]["rows"] == sum(r["matched"] for r in rows)
    # Copy every verified descriptive summary unchanged, excluding only duplicated
    # primary data and the raw-row metric list (whose retained source is hash pinned).
    copied = {
        k: v for k, v in summary.items() if k not in ["primary", "per_row_metrics"]
    }
    copied["source_summary"] = checkpoint["summary"]
    copied["occurrence_weighted_means"] = {}
    for population in ["all_rows", "common_support_rows"]:
        copied["occurrence_weighted_means"][population] = {
            side: grouped_means(summary[population]["by_label"], side)
            for side in ["top", "body"]
        }
    for side in ["top", "body"]:
        m = copied["occurrence_weighted_means"]["common_support_rows"][side]
        assert math.isclose(
            m["unmasked_nll"],
            m["legal_nll"] + m["negative_log_legal_mass"],
            abs_tol=1e-10,
        )
        lines.append(
            f"| {summary['encoding']} | {summary['seed']} | {side} | {m['rows']} | "
            + " | ".join(f"{m[k]:.6f}" for k in metrics)
            + " |"
        )
    shares = summary["behaviour_loss_and_exposure"]
    copied["share_mean_defined_behaviours"] = {
        k: sum(b[k] is not None for b in shares)
        for k in [
            "top_token_share",
            "top_unmasked_loss_share",
            "top_legal_deficit_share",
        ]
    }
    share_lines.append(
        f"| {summary['encoding']} | {summary['seed']} | "
        + " | ".join(
            f"{statistics.fmean(b[k] for b in shares if b[k] is not None):.6f}"
            for k in [
                "top_token_share",
                "top_unmasked_loss_share",
                "top_legal_deficit_share",
            ]
        )
        + f" | {copied['share_mean_defined_behaviours']['top_legal_deficit_share']} |"
    )
    excluded = summary["without_assembly_descriptive"]
    assembly_lines.append(
        f"| {summary['encoding']} | {summary['seed']} | {summary['primary']['delta']:+.6f} | "
        f"{excluded['delta']:+.6f} | {len(excluded['cells'])} | {len(excluded['behaviours'])} |"
    )
    bundle["checkpoints"].append(copied)
lines += [
    "",
    "Legal sets depend on the shown execution state and the bounded interpreter.",
    "Under Mark, emitting the same op name has different semantics from executing it.",
    "These secondaries do not identify a causal quotation/frame effect.",
    "",
    "### Token, loss and deficit shares",
    "",
    "All 64 selected behaviours contribute. This table is the arithmetic mean of each",
    "behaviour's top share across **all tokens**, including structural tokens, without",
    "pooling seeds or encodings. The complete per-behaviour shares remain in SECONDARY.json.",
    "An undefined zero-total metric share is retained as null and excluded from that",
    "metric's mean; token share is always defined.",
    "",
    "| encoding | seed | top token share | top unmasked-loss share | top legal-deficit share | defined deficit behaviours |",
    "|---|---:|---:|---:|---:|---:|",
    *share_lines,
    "",
    "### Typed encoder width and truncation",
    "",
    "Widths count typed `encode_stack` tokens **before** caps; these are shown-example",
    "states at witness prefixes, not independent tasks. A row is flagged if any shown",
    "example truncates. State cap is 160 and target cap is 64 in both encodings.",
    "",
    "| population | rows | example states | state width min/median/max | target width min/median/max | state truncation rows/examples | target truncation rows/examples |",
    "|---|---:|---:|---|---|---|---|",
]
for population, x in widths.items():
    state = x["state_pre_cap_width"]
    target = x["target_pre_cap_width"]
    lines.append(
        f"| {population} | {x['rows']} | {x['example_states']} | "
        f"{state['min']}/{state['median']}/{state['max']} | "
        f"{target['min']}/{target['median']}/{target['max']} | "
        f"{x['state_truncated_rows']}/{x['state_truncated_example_states']} | "
        f"{x['target_truncated_rows']}/{x['target_truncated_example_states']} |"
    )
lines += [
    "",
    f"Body-depth counts are {widths['all']['body_depth_counts']} across all rows and",
    f"{widths['common_support']['body_depth_counts']} in common support. Full width histograms",
    "are retained in SECONDARY.json. Training task-pool membership does not establish",
    "that these rows appeared in the with-replacement training sample; exact multiplicity",
    "was not reconstructed and remains unavailable.",
    "",
    "### Assembly exclusion and resource scope",
    "",
    "`quote`, `cons` and `cat` cells are flagged assembly cells. Removing them changes",
    "the descriptive population below; it supplies no new gate or inference.",
    "",
    "| encoding | seed | primary delta | delta without assembly | remaining cells | remaining behaviours |",
    "|---|---:|---:|---:|---:|---:|",
    *assembly_lines,
    "",
    "There were no duplicate selected capped-program texts. Alternative legality checks",
    f"had {stage['alternative_first_refusal_counts']['ordinary']} ordinary first refusals,",
    f"{stage['alternative_first_refusal_counts']['fuel']} fuel and {stage['alternative_first_refusal_counts']['depth']} depth refusals;",
    "no scored row is resource-affected. Refusal reasons record only the first refusing",
    "shown example/op prefix. Ordinary refusal means illegal under this bounded mask;",
    "resource refusal, if present, would not establish semantic impossibility.",
    "Witness replay succeeded on every shown example. Per-example/op fuel was 100,000,",
    "initial depth 0 and maximum quotation depth 32. This diagnostic supplies neither",
    "comparative solve timing nor a claim about omitted boundaries or held-out tasks.",
    "",
]
(OUT / "SECONDARY.json").write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n")
result = (OUT / "RESULT.md").read_text()
result = result.split("## Required secondary reporting")[0]
result = result.replace(
    "Raw per-step logits and full secondary summaries remain under main\ncheckpoints/capped-glue-loss-20261003/trial; paths/hashes are in measurement.json.\nBefore drawing further descriptive conclusions, inspect the required secondary\ndiagnostics; none changes the registered gate.\n",
    "Raw per-step logits remain under main checkpoints/capped-glue-loss-20261003/trial;\npaths/hashes are in measurement.json. Required secondary summaries are published below.\n",
)
(OUT / "RESULT.md").write_text(result.rstrip() + "\n\n" + "\n".join(lines))
print(
    "Published unchanged verified summaries and descriptive tables:",
    len(bundle["checkpoints"]),
    "checkpoints;",
    (OUT / "SECONDARY.json").stat().st_size,
    "bytes",
)
