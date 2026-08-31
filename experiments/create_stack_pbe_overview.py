"""Render the verified stack-PBE corpus overview as one standalone HTML file."""

from __future__ import annotations

import argparse
import html
from pathlib import Path
from typing import Any

from experiments.stack_pbe_scope import analyze_verified_corpus


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _number(value: int | float) -> str:
    if isinstance(value, float) and value.is_integer():
        return f"{int(value):,}"
    return f"{value:,}"


def _percent(value: float | None, digits: int = 1) -> str:
    return "—" if value is None else f"{value:.{digits}%}"


def _length_chart(analysis: dict[str, Any]) -> str:
    histogram = {
        int(length): count for length, count in analysis["length_histogram"].items()
    }
    min_length = analysis["min_length"]
    max_length = analysis["max_length"]
    median = analysis["median_length"]
    width, height = 960, 290
    left, right, top, bottom = 54, 18, 18, 42
    plot_width = width - left - right
    plot_height = height - top - bottom
    max_count = max(histogram.values())

    def x(length: float) -> float:
        return left + (length - min_length) / (max_length - min_length) * plot_width

    def y(count: float) -> float:
        return top + plot_height - count / max_count * plot_height

    bar_width = max(1.2, plot_width / (max_length - min_length + 1) * 0.82)
    bars = []
    for length, count in sorted(histogram.items()):
        bar_height = plot_height - (y(count) - top)
        bars.append(
            f'<rect class="bar" x="{x(length) - bar_width / 2:.2f}" '
            f'y="{y(count):.2f}" width="{bar_width:.2f}" height="{bar_height:.2f}">'
            f"<title>{length} ops: {count:,} tasks</title></rect>"
        )
    x_ticks = sorted({min_length, 50, 100, 150, 200, max_length})
    x_axis = "".join(
        f'<line class="tick" x1="{x(tick):.2f}" y1="{top + plot_height}" '
        f'x2="{x(tick):.2f}" y2="{top + plot_height + 5}"/>'
        f'<text class="axis" x="{x(tick):.2f}" y="{height - 15}" '
        f'text-anchor="middle">{tick}</text>'
        for tick in x_ticks
    )
    y_ticks = sorted({0, max_count // 2, max_count})
    y_axis = "".join(
        f'<line class="grid" x1="{left}" y1="{y(tick):.2f}" '
        f'x2="{left + plot_width}" y2="{y(tick):.2f}"/>'
        f'<text class="axis" x="{left - 8}" y="{y(tick) + 4:.2f}" '
        f'text-anchor="end">{tick:,}</text>'
        for tick in y_ticks
    )
    return f"""<svg class="chart length-chart" viewBox="0 0 {width} {height}" role="img"
     aria-labelledby="length-title length-desc">
  <title id="length-title">Witness length distribution</title>
  <desc id="length-desc">Task counts for every witness length from {min_length} to
    {max_length} operations. The median is {_number(median)} operations.</desc>
  {y_axis}
  <line class="axis-line" x1="{left}" y1="{top + plot_height}"
        x2="{left + plot_width}" y2="{top + plot_height}"/>
  {"".join(bars)}
  <line class="median" x1="{x(median):.2f}" y1="{top}"
        x2="{x(median):.2f}" y2="{top + plot_height}"/>
  <text class="annotation" x="{x(median) + 5:.2f}" y="{top + 13}">median {_number(median)}</text>
  {x_axis}
  <text class="axis-title" x="{left + plot_width / 2}" y="{height - 1}"
        text-anchor="middle">witness length (operations)</text>
  <text class="axis-title" transform="translate(13 {top + plot_height / 2}) rotate(-90)"
        text-anchor="middle">tasks</text>
</svg>"""


def _tag_chart(analysis: dict[str, Any]) -> str:
    tag_data = analysis["tag_length"]
    bins = tag_data["bins"]
    tags = tag_data["tags"]
    width, height = 960, 330
    left, right, top, bottom = 58, 22, 20, 52
    plot_width = width - left - right
    plot_height = height - top - bottom
    max_length = analysis["max_length"]

    def x(value: float) -> float:
        return left + (value - 1) / (max_length - 1) * plot_width

    def y(value: float) -> float:
        return top + (1 - value) * plot_height

    grids = "".join(
        f'<line class="grid" x1="{left}" y1="{y(value):.2f}" '
        f'x2="{left + plot_width}" y2="{y(value):.2f}"/>'
        f'<text class="axis" x="{left - 8}" y="{y(value) + 4:.2f}" '
        f'text-anchor="end">{value:.0%}</text>'
        for value in (0, 0.25, 0.5, 0.75, 1)
    )
    x_ticks = sorted({1, 50, 100, 150, 200, max_length})
    axes = "".join(
        f'<line class="tick" x1="{x(value):.2f}" y1="{top + plot_height}" '
        f'x2="{x(value):.2f}" y2="{top + plot_height + 5}"/>'
        f'<text class="axis" x="{x(value):.2f}" y="{height - 24}" '
        f'text-anchor="middle">{value}</text>'
        for value in x_ticks
    )
    lines = []
    for index, tag in enumerate(tags, 1):
        points = [
            (x((row["start"] + row["stop"]) / 2), y(row["incidence"][tag]), row)
            for row in bins
        ]
        path = " ".join(f"{px:.2f},{py:.2f}" for px, py, _ in points)
        circles = "".join(
            f'<circle class="series-{index}" cx="{px:.2f}" cy="{py:.2f}" r="3">'
            f"<title>{tag}, {row['label']} ops: {row['counts'][tag]}/"
            f"{row['behaviours']} behaviours ({row['incidence'][tag]:.1%})</title></circle>"
            for px, py, row in points
        )
        lines.append(
            f'<polyline class="tag-line series-{index}" points="{path}"/>{circles}'
        )
    return f"""<div class="legend" aria-label="Program tag legend">
  {"".join(f'<span><i class="swatch series-{i}"></i>{_escape(tag)}</span>' for i, tag in enumerate(tags, 1))}
</div>
<svg class="chart tag-chart" viewBox="0 0 {width} {height}" role="img"
     aria-labelledby="tag-title tag-desc">
  <title id="tag-title">Program-tag incidence by witness length</title>
  <desc id="tag-desc">Share of behaviours carrying each overlapping program tag,
    grouped into ten-operation witness-length bins.</desc>
  {grids}
  <line class="axis-line" x1="{left}" y1="{top + plot_height}"
        x2="{left + plot_width}" y2="{top + plot_height}"/>
  {"".join(lines)}
  {axes}
  <text class="axis-title" x="{left + plot_width / 2}" y="{height - 3}"
        text-anchor="middle">witness length (operations)</text>
  <text class="axis-title" transform="translate(14 {top + plot_height / 2}) rotate(-90)"
        text-anchor="middle">behaviours in bin</text>
</svg>"""


def _scaffold_chart(summary: dict[str, Any], chart_id: str) -> str:
    rows = summary["histogram"]
    width, height = 450, 220
    left, right, top, bottom = 42, 12, 16, 38
    plot_width = width - left - right
    plot_height = height - top - bottom
    max_count = max(row["count"] for row in rows) or 1
    slot = plot_width / len(rows)
    bars = []
    for index, row in enumerate(rows):
        bar_height = row["count"] / max_count * plot_height
        bars.append(
            f'<rect class="bar" x="{left + index * slot + 2:.2f}" '
            f'y="{top + plot_height - bar_height:.2f}" width="{slot - 4:.2f}" '
            f'height="{bar_height:.2f}"><title>{_escape(row["label"])}: '
            f"{row['count']} behaviours</title></rect>"
        )
    return f"""<svg class="chart scaffold-chart" viewBox="0 0 {width} {height}" role="img"
     aria-labelledby="{chart_id}-title {chart_id}-desc">
  <title id="{chart_id}-title">Scaffold sharing distribution</title>
  <desc id="{chart_id}-desc">Eligible behaviours grouped by the share of their
    eight-operation windows found in the comparison set.</desc>
  <line class="grid" x1="{left}" y1="{top}" x2="{left + plot_width}" y2="{top}"/>
  <text class="axis" x="{left - 7}" y="{top + 4}" text-anchor="end">{max_count}</text>
  <text class="axis" x="{left - 7}" y="{top + plot_height + 4}" text-anchor="end">0</text>
  {"".join(bars)}
  <line class="axis-line" x1="{left}" y1="{top + plot_height}"
        x2="{left + plot_width}" y2="{top + plot_height}"/>
  <text class="axis" x="{left}" y="{height - 15}" text-anchor="middle">0%</text>
  <text class="axis" x="{left + plot_width / 2}" y="{height - 15}" text-anchor="middle">50%</text>
  <text class="axis" x="{left + plot_width}" y="{height - 15}" text-anchor="middle">100%</text>
  <text class="axis-title" x="{left + plot_width / 2}" y="{height - 1}"
        text-anchor="middle">shared 8-op windows</text>
</svg>"""


def _shape_rows(analysis: dict[str, Any]) -> str:
    rows = []
    for shape in analysis["input_shapes"]:
        train = shape["splits"].get("train", 0)
        eval_count = shape["splits"].get("eval", 0)
        width = shape["tasks"] / analysis["tasks"] * 100
        rows.append(
            f"""<tr>
  <th scope="row">{_escape(shape["label"])}<div class="mini-track"><span style="width:{width:.3f}%"></span></div></th>
  <td class="num">{shape["tasks"]:,}</td>
  <td class="num">{shape["behaviours"]}</td>
  <td class="num">{train:,}</td>
  <td class="num">{eval_count:,}</td>
</tr>"""
        )
    return "\n".join(rows)


def _behaviour_rows(analysis: dict[str, Any]) -> str:
    rows = []
    for row in analysis["behaviour_rows"]:
        tags = " ".join(row["tags"])
        search = " ".join(
            (
                row["behaviour"],
                row["summary"],
                row["input_shape"],
                tags,
            )
        ).lower()
        rows.append(
            f"""<tr data-search="{_escape(search)}" data-split="{_escape(row["split"])}">
  <th scope="row"><code>{_escape(row["behaviour"])}</code><span class="row-summary">{_escape(row["summary"])}</span></th>
  <td><span class="split split-{_escape(row["split"])}">{_escape(row["split"])}</span></td>
  <td class="num">{row["tasks"]:,}</td>
  <td class="num">{row["length"]}</td>
  <td>{_escape(row["input_shape"])}</td>
  <td>{_escape(tags)}</td>
  <td class="num">{_percent(row["overall_scaffold_share"])}</td>
  <td class="num">{_percent(row["eval_to_train_scaffold_share"])}</td>
</tr>"""
        )
    return "\n".join(rows)


def render_report(analysis: dict[str, Any]) -> str:
    """Render a deterministic standalone document from an analysis dictionary."""
    split = analysis["splits"]
    train = split["train"]
    eval_split = split["eval"]
    groups = analysis["inseparable_groups"]
    group_profile = ", ".join(
        f"{count} {'group' if count == 1 else 'groups'} of size {size}"
        for size, count in sorted(
            groups["size_histogram"].items(), key=lambda item: int(item[0])
        )
    )
    scaffolds = analysis["scaffolds"]
    overall = scaffolds["overall"]
    eval_to_train = scaffolds["eval_to_train"]
    provenance = analysis["provenance"]
    repo = provenance["hub_repo_id"]
    revision = provenance["hub_revision"]
    hub_url = f"https://huggingface.co/datasets/{repo}/tree/{revision}"

    payload_rows = "\n".join(
        f"""<tr>
  <th scope="row">{_escape(row["split"])}</th>
  <td><code>{_escape(row["path"])}</code></td>
  <td class="num">{row["records"]:,}</td>
  <td class="num">{row["bytes"]:,}</td>
  <td><code class="digest">{_escape(row["sha256"])}</code></td>
</tr>"""
        for row in provenance["payloads"]
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light dark">
<title>stack_pbe_v4 corpus overview</title>
<style>
:root {{
  --bg: #f6f3ec; --surface: #fffdf8; --ink: #24231f; --muted: #68645b;
  --line: #d9d2c5; --accent: #1c6b62; --accent-soft: #dcebe6;
  --eval: #a44a37; --eval-soft: #f3ded7; --grid: #ded8cd;
  --series-1: #1c6b62; --series-2: #a44a37; --series-3: #8267a8;
  --series-4: #b57b16; --series-5: #4775a5;
}}
@media (prefers-color-scheme: dark) {{
  :root {{
    --bg: #181917; --surface: #22231f; --ink: #efede6; --muted: #bcb7aa;
    --line: #45463f; --accent: #75c3b6; --accent-soft: #243e39;
    --eval: #ee9b86; --eval-soft: #472f2a; --grid: #3c3d38;
    --series-1: #75c3b6; --series-2: #ee9b86; --series-3: #b9a1d8;
    --series-4: #e1b65d; --series-5: #8eb5dd;
  }}
}}
* {{ box-sizing: border-box; }}
html {{ scroll-behavior: smooth; }}
body {{ margin: 0; background: var(--bg); color: var(--ink); font: 16px/1.55 ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
main {{ width: min(1160px, calc(100% - 36px)); margin: 0 auto; padding: 54px 0 80px; }}
h1, h2, h3 {{ line-height: 1.15; letter-spacing: -0.025em; }}
h1 {{ margin: 0; font-size: clamp(2.25rem, 6vw, 4.8rem); max-width: 900px; }}
h2 {{ margin: 0 0 10px; font-size: clamp(1.6rem, 3vw, 2.25rem); }}
h3 {{ margin: 0; font-size: 1.05rem; }}
p {{ max-width: 78ch; }}
a {{ color: var(--accent); text-underline-offset: 3px; }}
code {{ font: .88em/1.45 ui-monospace, SFMono-Regular, Consolas, monospace; }}
.eyebrow {{ color: var(--accent); text-transform: uppercase; letter-spacing: .12em; font-size: .75rem; font-weight: 700; }}
.lead {{ color: var(--muted); font-size: 1.12rem; max-width: 760px; margin: 18px 0 8px; }}
.source-line {{ color: var(--muted); font-size: .9rem; margin: 0; }}
.stats {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; margin: 34px 0 0; }}
.stat {{ padding: 17px 18px; background: var(--surface); border: 1px solid var(--line); border-radius: 10px; }}
.stat b {{ display: block; font-size: 1.65rem; line-height: 1.1; font-variant-numeric: tabular-nums; }}
.stat span {{ color: var(--muted); font-size: .83rem; }}
section {{ padding-top: 64px; }}
.section-head {{ display: flex; align-items: end; justify-content: space-between; gap: 24px; margin-bottom: 22px; }}
.section-head p {{ color: var(--muted); margin: 0; }}
.chart {{ display: block; width: 100%; height: auto; overflow: visible; }}
.chart .bar {{ fill: var(--accent); }}
.chart .grid {{ stroke: var(--grid); stroke-width: 1; }}
.chart .axis-line, .chart .tick {{ stroke: var(--muted); stroke-width: 1; }}
.chart .axis, .chart .axis-title, .chart .annotation {{ fill: var(--muted); font: 12px ui-sans-serif, system-ui, sans-serif; }}
.chart .axis-title, .chart .annotation {{ fill: var(--ink); }}
.chart .median {{ stroke: var(--eval); stroke-width: 1.5; stroke-dasharray: 5 4; }}
.legend {{ display: flex; flex-wrap: wrap; gap: 8px 20px; margin: -4px 0 8px; font-size: .86rem; color: var(--muted); }}
.legend span {{ display: inline-flex; align-items: center; gap: 7px; }}
.swatch {{ width: 18px; height: 3px; display: inline-block; border-radius: 2px; background: currentColor; }}
circle.series-1, circle.series-2, circle.series-3, circle.series-4, circle.series-5 {{ stroke: var(--bg); stroke-width: 1; }}
.series-1 {{ color: var(--series-1); stroke: var(--series-1); fill: var(--series-1); }}
.series-2 {{ color: var(--series-2); stroke: var(--series-2); fill: var(--series-2); }}
.series-3 {{ color: var(--series-3); stroke: var(--series-3); fill: var(--series-3); }}
.series-4 {{ color: var(--series-4); stroke: var(--series-4); fill: var(--series-4); }}
.series-5 {{ color: var(--series-5); stroke: var(--series-5); fill: var(--series-5); }}
.tag-line {{ fill: none; stroke-width: 2; }}
.table-wrap {{ overflow-x: auto; border-top: 1px solid var(--line); }}
table {{ width: 100%; border-collapse: collapse; font-size: .91rem; }}
th, td {{ text-align: left; vertical-align: top; padding: 11px 12px; border-bottom: 1px solid var(--line); }}
thead th {{ color: var(--muted); font-size: .75rem; text-transform: uppercase; letter-spacing: .06em; white-space: nowrap; }}
tbody th {{ font-weight: 600; }}
.num {{ text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }}
.mini-track {{ height: 3px; width: 100%; max-width: 260px; margin-top: 7px; background: var(--line); }}
.mini-track span {{ display: block; height: 100%; background: var(--accent); }}
.split-grid, .scaffold-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }}
.split-panel, .scaffold-panel {{ padding: 20px; background: var(--surface); border: 1px solid var(--line); border-radius: 10px; }}
.split-panel dl {{ display: grid; grid-template-columns: 1fr auto; gap: 7px 20px; margin: 16px 0 0; }}
.split-panel dt {{ color: var(--muted); }}
.split-panel dd {{ margin: 0; font-variant-numeric: tabular-nums; font-weight: 650; }}
.zero-overlap {{ margin-top: 18px; padding: 13px 16px; background: var(--accent-soft); border-left: 4px solid var(--accent); }}
.zero-overlap b {{ color: var(--accent); }}
.scaffold-panel header {{ display: flex; justify-content: space-between; gap: 12px; align-items: baseline; }}
.scaffold-panel header b {{ color: var(--accent); font-size: 1.45rem; }}
.scaffold-panel p {{ color: var(--muted); font-size: .88rem; margin: 5px 0 0; }}
.scaffold-facts {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-top: 14px; }}
.scaffold-facts div {{ border-top: 1px solid var(--line); padding-top: 8px; }}
.scaffold-facts b {{ display: block; font-variant-numeric: tabular-nums; }}
.scaffold-facts span {{ color: var(--muted); font-size: .75rem; }}
.group-note {{ display: flex; gap: 18px; align-items: baseline; border-top: 1px solid var(--line); border-bottom: 1px solid var(--line); padding: 17px 0; margin: 28px 0 0; }}
.group-note b {{ font-size: 1.35rem; white-space: nowrap; }}
.group-note span {{ color: var(--muted); }}
.filters {{ display: flex; flex-wrap: wrap; gap: 10px; align-items: center; margin: 20px 0 12px; }}
.filters label {{ color: var(--muted); font-size: .84rem; }}
input, select {{ font: inherit; color: var(--ink); background: var(--surface); border: 1px solid var(--line); border-radius: 7px; padding: 9px 11px; }}
input {{ min-width: min(360px, 100%); flex: 1; }}
.result-count {{ color: var(--muted); font-size: .84rem; margin-left: auto; }}
.behaviour-table th:first-child {{ min-width: 245px; }}
.row-summary {{ display: block; color: var(--muted); font-size: .78rem; font-weight: 400; margin-top: 3px; max-width: 40ch; }}
.split {{ display: inline-block; padding: 2px 7px; border-radius: 999px; font-size: .75rem; font-weight: 650; }}
.split-train {{ color: var(--accent); background: var(--accent-soft); }}
.split-eval {{ color: var(--eval); background: var(--eval-soft); }}
.provenance-list {{ display: grid; grid-template-columns: 160px 1fr; gap: 9px 18px; margin: 22px 0; }}
.provenance-list dt {{ color: var(--muted); }}
.provenance-list dd {{ margin: 0; min-width: 0; }}
.digest {{ overflow-wrap: anywhere; }}
.note {{ color: var(--muted); font-size: .86rem; }}
@media (max-width: 820px) {{
  .stats {{ grid-template-columns: repeat(2, 1fr); }}
  .split-grid, .scaffold-grid {{ grid-template-columns: 1fr; }}
  .section-head {{ display: block; }} .section-head p {{ margin-top: 8px; }}
}}
@media (max-width: 520px) {{
  main {{ width: min(100% - 24px, 1160px); padding-top: 34px; }}
  .stats {{ grid-template-columns: 1fr 1fr; }}
  .stat {{ padding: 13px; }} .stat b {{ font-size: 1.35rem; }}
  section {{ padding-top: 48px; }}
  .provenance-list {{ grid-template-columns: 1fr; gap: 2px; }}
  .provenance-list dd {{ margin-bottom: 10px; }}
  .group-note {{ display: block; }}
  .result-count {{ width: 100%; margin-left: 0; }}
}}
@media print {{
  :root {{ --bg: white; --surface: white; --ink: black; --muted: #444; --line: #bbb; }}
  main {{ width: 100%; padding: 0; }} section {{ break-inside: avoid; }}
  .filters {{ display: none; }}
}}
</style>
</head>
<body>
<main>
<header>
  <div class="eyebrow">verified corpus snapshot</div>
  <h1><code>stack_pbe_v4</code><br>corpus overview</h1>
  <p class="lead">A behaviour-first program-synthesis corpus with fresh task instances,
    a skeleton-disjoint train/eval split, and long witnesses built from a 28-op basis.</p>
  <p class="source-line">Pinned to <a href="{_escape(hub_url)}">{_escape(repo)} @ {_escape(revision[:12])}</a>.</p>
  <div class="stats" aria-label="Corpus headline statistics">
    <div class="stat"><b>{analysis["tasks"]:,}</b><span>tasks</span></div>
    <div class="stat"><b>{analysis["behaviours"]}</b><span>behaviours</span></div>
    <div class="stat"><b>{analysis["skeletons"]}</b><span>program skeletons</span></div>
    <div class="stat"><b>{analysis["min_length"]}-{analysis["max_length"]}</b><span>witness length</span></div>
    <div class="stat"><b>{_number(analysis["median_length"])}</b><span>median witness length</span></div>
  </div>
</header>

<section aria-labelledby="length-heading">
  <div class="section-head">
    <div><h2 id="length-heading">Witness length</h2><p>Exact task counts; each behaviour contributes {analysis["tasks"] // analysis["behaviours"]} independently drawn tasks.</p></div>
  </div>
  {_length_chart(analysis)}
</section>

<section aria-labelledby="tag-heading">
  <div class="section-head">
    <div><h2 id="tag-heading">Program structure vs. length</h2><p>Tags overlap: a witness can branch, fold, map, filter, and close over task data.</p></div>
  </div>
  {_tag_chart(analysis)}
</section>

<section aria-labelledby="split-heading">
  <div class="section-head">
    <div><h2 id="split-heading">Skeleton-disjoint split</h2><p>The split unit is the normalized program skeleton, not the task or behaviour.</p></div>
  </div>
  <div class="split-grid">
    <article class="split-panel"><h3>Train</h3><dl>
      <dt>Tasks</dt><dd>{train["tasks"]:,}</dd><dt>Behaviours</dt><dd>{train["behaviours"]}</dd><dt>Skeletons</dt><dd>{train["skeletons"]}</dd>
    </dl></article>
    <article class="split-panel"><h3>Eval</h3><dl>
      <dt>Tasks</dt><dd>{eval_split["tasks"]:,}</dd><dt>Behaviours</dt><dd>{eval_split["behaviours"]}</dd><dt>Skeletons</dt><dd>{eval_split["skeletons"]}</dd>
    </dl></article>
  </div>
  <p class="zero-overlap"><b>{len(analysis["skeleton_overlap"])} skeleton overlap.</b> No normalized program shape occurs in both train and eval. Behaviour overlap is also {len(analysis["behaviour_overlap"])}.</p>
</section>

<section aria-labelledby="shape-heading">
  <div class="section-head">
    <div><h2 id="shape-heading">Input-shape mix</h2><p>Parsed from every shown and held-out stack. All {analysis["tasks"]:,} tasks and {analysis["behaviours"]} behaviours are accounted for.</p></div>
  </div>
  <div class="table-wrap"><table>
    <thead><tr><th>Structural input shape</th><th class="num">Tasks</th><th class="num">Behaviours</th><th class="num">Train tasks</th><th class="num">Eval tasks</th></tr></thead>
    <tbody>{_shape_rows(analysis)}</tbody>
    <tfoot><tr><th>Total</th><td class="num">{analysis["tasks"]:,}</td><td class="num">{analysis["behaviours"]}</td><td class="num">{train["tasks"]:,}</td><td class="num">{eval_split["tasks"]:,}</td></tr></tfoot>
  </table></div>
</section>

<section aria-labelledby="scaffold-heading">
  <div class="section-head">
    <div><h2 id="scaffold-heading">Shared scaffolding</h2><p>Per-behaviour shares of contiguous {scaffolds["window"]}-operation windows. Short witnesses are excluded, not scored as zero.</p></div>
  </div>
  <div class="scaffold-grid">
    <article class="scaffold-panel">
      <header><h3>Another corpus witness</h3><b>{_percent(overall["median"])}</b></header>
      <p>Median share among {overall["eligible"]}/{overall["targets"]} eligible witnesses; {overall["excluded_short"]} have fewer than {scaffolds["window"]} ops.</p>
      {_scaffold_chart(overall, "overall-scaffold")}
      <div class="scaffold-facts"><div><b>{overall["matched_windows"]:,}/{overall["windows"]:,}</b><span>matched windows</span></div><div><b>{_percent(overall["aggregate_share"])}</b><span>aggregate share</span></div><div><b>{_percent(overall["mean"])}</b><span>mean behaviour share</span></div></div>
    </article>
    <article class="scaffold-panel">
      <header><h3>Eval windows found in train</h3><b>{_percent(eval_to_train["median"])}</b></header>
      <p>Median share among {eval_to_train["eligible"]}/{eval_to_train["targets"]} eligible eval witnesses; {eval_to_train["excluded_short"]} have fewer than {scaffolds["window"]} ops.</p>
      {_scaffold_chart(eval_to_train, "eval-scaffold")}
      <div class="scaffold-facts"><div><b>{eval_to_train["matched_windows"]:,}/{eval_to_train["windows"]:,}</b><span>matched windows</span></div><div><b>{_percent(eval_to_train["aggregate_share"])}</b><span>aggregate share</span></div><div><b>{_percent(eval_to_train["mean"])}</b><span>mean behaviour share</span></div></div>
    </article>
  </div>
  <div class="group-note"><b>{groups["groups"]} inseparable groups</b><span>{_escape(group_profile)} across {groups["behaviours"]} behaviours; the largest has {groups["largest"]}. Grouping is derived only from component names in this corpus snapshot.</span></div>
</section>

<section id="behaviours" aria-labelledby="behaviour-heading">
  <div class="section-head">
    <div><h2 id="behaviour-heading">Behaviours</h2><p>One consistency-checked representative per behaviour; task count is computed from corpus rows.</p></div>
  </div>
  <div class="filters">
    <label for="behaviour-filter">Filter</label><input id="behaviour-filter" type="search" placeholder="name, summary, shape, or tag" autocomplete="off">
    <label for="split-filter">Split</label><select id="split-filter"><option value="">all</option><option value="train">train</option><option value="eval">eval</option></select>
    <span id="behaviour-count" class="result-count" aria-live="polite">{analysis["behaviours"]} behaviours</span>
  </div>
  <div class="table-wrap"><table class="behaviour-table">
    <thead><tr><th>Behaviour</th><th>Split</th><th class="num">Tasks</th><th class="num">Ops</th><th>Input shape</th><th>Tags</th><th class="num">Shared</th><th class="num">Eval → train</th></tr></thead>
    <tbody id="behaviour-rows">{_behaviour_rows(analysis)}</tbody>
  </table></div>
</section>

<section aria-labelledby="provenance-heading">
  <div class="section-head"><div><h2 id="provenance-heading">Provenance</h2><p>Both payloads were SHA-256 verified against the tracked manifest before rendering.</p></div></div>
  <dl class="provenance-list">
    <dt>Hub snapshot</dt><dd><a href="{_escape(hub_url)}"><code>{_escape(repo)}@{_escape(revision)}</code></a></dd>
    <dt>Manifest</dt><dd><code>{_escape(provenance["manifest"])}</code> · <code class="digest">{_escape(provenance["manifest_sha256"])}</code></dd>
    <dt>Catalogue digest</dt><dd><code class="digest">{_escape(provenance["catalogue_digest"])}</code></dd>
    <dt>Basis</dt><dd><code>{_escape(", ".join(analysis["bases"]))}</code></dd>
  </dl>
  <div class="table-wrap"><table>
    <thead><tr><th>Split</th><th>Payload</th><th class="num">Records</th><th class="num">Bytes</th><th>SHA-256</th></tr></thead>
    <tbody>{payload_rows}</tbody>
  </table></div>
  <p class="note">All statistics are derived from the pinned corpus rows. No live catalogue state is consulted.</p>
</section>
</main>
<script>
(() => {{
  const query = document.getElementById("behaviour-filter");
  const split = document.getElementById("split-filter");
  const count = document.getElementById("behaviour-count");
  const rows = Array.from(document.querySelectorAll("#behaviour-rows tr"));
  const apply = () => {{
    const term = query.value.trim().toLowerCase();
    const wantedSplit = split.value;
    let visible = 0;
    for (const row of rows) {{
      const show = (!term || row.dataset.search.includes(term)) &&
        (!wantedSplit || row.dataset.split === wantedSplit);
      row.hidden = !show;
      if (show) visible += 1;
    }}
    count.textContent = `${{visible}} behaviour${{visible === 1 ? "" : "s"}}`;
  }};
  query.addEventListener("input", apply);
  split.addEventListener("change", apply);
}})();
</script>
</body>
</html>
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="wandering_light/evals/data/stack_pbe_v4")
    parser.add_argument("--out", default="reports/stack-pbe-v4-20260830/index.html")
    args = parser.parse_args(argv)

    analysis = analyze_verified_corpus(args.corpus)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_report(analysis), encoding="utf-8")
    print(f"wrote {output} ({output.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
