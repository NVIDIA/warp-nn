# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Reports (Markdown, JSON, CSV and HTML) of the benchmark results."""

from __future__ import annotations

from typing import Any, Callable, Iterable

import csv
import dataclasses
import html
import io
import itertools
import json

from ._benchmark import BASELINE, CURRENT, PASSES, REFERENCES, TORCH, Result
from ._specs import CATEGORIES


NOT_DIFFERENTIABLE = "—"
NOT_AVAILABLE = "n/a"
NOT_SIGNIFICANT = "~"
# markers of the speedup cells in the Markdown report (which has no colors), by cell state
MARKDOWN_MARKERS = {"faster": "🟢", "slower": "🔴", "same": "⚪"}


@dataclasses.dataclass(frozen=True)
class Cell:
    """Speedup cell of the report table."""

    text: str
    """Displayed text."""

    state: str
    """One of: ``"faster"``, ``"slower"``, ``"same"`` (not significant), ``"na"`` (not available) and ``"none"``."""

    details: str = ""
    """Details (e.g. absolute timings and confidence interval)."""


def _labels(metadata: dict[str, Any]) -> dict[str, str]:
    return {TORCH: "PyTorch", CURRENT: "warp-nn", BASELINE: metadata.get("baseline_ref") or "baseline"}


def _columns(metadata: dict[str, Any]) -> list[tuple[str, str, str]]:
    # (pass, reference implementation, header) of each speedup column (without the baseline if not benchmarked)
    labels = _labels(metadata)
    references = [ref for ref in REFERENCES if ref != BASELINE or metadata.get("baseline_ref")]
    return [(pass_, ref, f"{pass_.capitalize()} vs {labels[ref]}") for ref in references for pass_ in PASSES]


def _confidence(metadata: dict[str, Any]) -> str:
    return f"{100 * metadata.get('confidence', 0.95):g}%"


def format_time(seconds: float) -> str:
    """Format a duration with a suitable unit.

    :param seconds: The duration (in seconds).

    :return: The formatted duration.
    """
    if round(seconds * 1e6, 1) < 1000.0:
        return f"{seconds * 1e6:.1f} µs"
    if round(seconds * 1e3, 2) < 1000.0:
        return f"{seconds * 1e3:.2f} ms"
    return f"{seconds:.2f} s"


def make_cell(result: Result, pass_: str, reference: str, metadata: dict[str, Any]) -> Cell:
    """Create the speedup cell of the current implementation against a reference implementation.

    :param result: The benchmark result.
    :param pass_: The pass (``"forward"`` or ``"backward"``).
    :param reference: The reference implementation.
    :param metadata: The report metadata.

    :return: The cell.
    """
    if pass_ == "backward" and not result.differentiable:
        return Cell(NOT_DIFFERENTIABLE, "none", "not differentiable")
    comparison = result.comparisons.get(pass_, {}).get(reference)
    if comparison is None:
        errors = [result.errors[key] for key in ("benchmark", pass_, CURRENT, reference) if key in result.errors]
        return Cell(NOT_AVAILABLE, "na", "; ".join(errors))
    text = f"{comparison.speedup:.2f}x"
    if not comparison.significant:
        state, text = "same", f"{text} {NOT_SIGNIFICANT}"
    else:
        state = "faster" if comparison.speedup > 1.0 else "slower"
    labels = _labels(metadata)
    timings = result.timings[pass_]
    details = ", ".join(
        f"{labels[name]}: {format_time(timings[name].median)} (IQR {format_time(timings[name].iqr)})"
        for name in (CURRENT, reference)
    )
    if len(comparison.strata) > 1:
        details += "; per instance set: " + ", ".join(f"{value:.3f}x" for value in comparison.strata)
    details += f"; {_confidence(metadata)} CI [{comparison.ci_low:.3f}, {comparison.ci_high:.3f}]"
    details += f"; n={timings[CURRENT].samples}"
    return Cell(text, state, details)


def _group(results: Iterable[Result]) -> list[tuple[str, list[list[Result]]]]:
    # results grouped by category (in the order of CATEGORIES) and by module (in the order of the results)
    results = list(results)
    groups = []
    for category in CATEGORIES:
        selected = [result for result in results if result.category == category]
        modules = [list(group) for _, group in itertools.groupby(selected, key=lambda result: result.module)]
        if modules:
            groups.append((category, modules))
    return groups


def _metadata_lines(metadata: dict[str, Any]) -> list[tuple[str, str]]:
    current = metadata.get("current_commit", "unknown")
    if metadata.get("current_dirty"):
        current += " (with uncommitted changes)"
    baseline = "not benchmarked"
    if metadata.get("baseline_ref"):
        baseline = f"{metadata['baseline_ref']} ({metadata.get('baseline_commit', 'unknown')})"
    lines = []
    if metadata.get("gpu"):
        lines.append(("GPU", metadata["gpu"]))
    if metadata.get("cpu"):
        lines.append(("CPU", metadata["cpu"]))
    return lines + [
        ("Date", metadata.get("date", "")),
        ("Mode", "CUDA graph" if metadata.get("cuda_graph") else "eager"),
        ("Device", f"{metadata.get('device', '')} ({metadata.get('device_name', '')})"),
        ("warp-nn (current)", current),
        ("warp-nn (baseline)", baseline),
        ("Versions", metadata.get("versions", "")),
        ("TF32 (PyTorch)", "allowed" if metadata.get("allow_tf32") else "disabled"),
        ("Timing", metadata.get("timing", "")),
    ]


def _legend(metadata: dict[str, Any]) -> str:
    min_effect = 100 * metadata.get("min_effect", 0.02)
    return (
        "Speedup of warp-nn (current) = reference time / warp-nn time, where the reference (PyTorch or the "
        f"baseline warp-nn implementation) is 1x: > 1x means that warp-nn is faster. `{NOT_SIGNIFICANT}`: no "
        f"significant difference ({_confidence(metadata)} confidence interval includes 1x, the difference is "
        f"below {min_effect:g}%, or the instance sets disagree). `{NOT_AVAILABLE}`: not "
        f"available (see errors). `{NOT_DIFFERENTIABLE}`: not differentiable."
    )


def _errors(results: Iterable[Result]) -> list[tuple[str, int, str, str]]:
    return [
        (result.module, result.batch_size, key, error) for result in results for key, error in result.errors.items()
    ]


def _markdown_cell(cell: Cell) -> str:
    marker = MARKDOWN_MARKERS.get(cell.state)
    return f"{marker} {cell.text}" if marker else cell.text


def to_markdown(results: list[Result], metadata: dict[str, Any]) -> str:
    """Format the benchmark results as Markdown.

    :param results: The benchmark results.
    :param metadata: The report metadata.

    :return: The Markdown report.
    """
    columns = _columns(metadata)
    lines = ["# warp-nn modules benchmark", ""]
    lines += [f"- **{key}:** {value}" for key, value in _metadata_lines(metadata)]
    markers = (
        f"{MARKDOWN_MARKERS['faster']} significantly faster, {MARKDOWN_MARKERS['slower']} significantly slower, "
        f"{MARKDOWN_MARKERS['same']} no significant difference."
    )
    lines += ["", f"{_legend(metadata)} {markers}", ""]
    for category, modules in _group(results):
        lines += [f"## {category.capitalize()}", ""]
        lines.append("| Module | Batch size | " + " | ".join(header for _, _, header in columns) + " |")
        lines.append("|:---|---:|" + "---:|" * len(columns))
        for group in modules:
            for i, result in enumerate(group):
                cells = [_markdown_cell(make_cell(result, pass_, ref, metadata)) for pass_, ref, _ in columns]
                name = f"**{result.module}**" if i == 0 else ""
                lines.append(f"| {name} | {result.batch_size} | " + " | ".join(cells) + " |")
        lines.append("")
    errors = _errors(results)
    if errors:
        lines += ["## Errors", ""]
        lines += [
            # backticks would end the inline code span
            f"- {module} (batch size {batch_size}), {key}: `{error.replace('`', chr(39))}`"
            for module, batch_size, key, error in errors
        ]
        lines.append("")
    return "\n".join(lines)


def _to_dict(result: Result) -> dict[str, Any]:
    return {
        "category": result.category,
        "module": result.module,
        "batch_size": result.batch_size,
        "differentiable": result.differentiable,
        "timings": {
            pass_: {
                name: {
                    "median_us": 1e6 * summary.median,
                    "q1_us": 1e6 * summary.q1,
                    "q3_us": 1e6 * summary.q3,
                    "samples": summary.samples,
                    "iterations": summary.iterations,
                }
                for name, summary in timings.items()
            }
            for pass_, timings in result.timings.items()
        },
        "comparisons": {
            pass_: {reference: dataclasses.asdict(comparison) for reference, comparison in comparisons.items()}
            for pass_, comparisons in result.comparisons.items()
        },
        "errors": dict(result.errors),
    }


def to_json(results: list[Result], metadata: dict[str, Any]) -> str:
    """Format the benchmark results as JSON (with the raw summary statistics).

    :param results: The benchmark results.
    :param metadata: The report metadata.

    :return: The JSON report.
    """
    return json.dumps({"metadata": metadata, "results": [_to_dict(result) for result in results]}, indent=2) + "\n"


def to_csv(results: list[Result], metadata: dict[str, Any]) -> str:
    """Format the benchmark results as CSV, with one row per module, batch size and pass.

    :param results: The benchmark results.
    :param metadata: The report metadata.

    :return: The CSV report.
    """
    names = (CURRENT, *REFERENCES)
    fields = ["category", "module", "batch_size", "pass"]
    for name in names:
        fields += [f"{name}_median_us", f"{name}_q1_us", f"{name}_q3_us", f"{name}_samples", f"{name}_iterations"]
    for reference in REFERENCES:
        fields += [f"speedup_vs_{reference}", f"ci_low_vs_{reference}", f"ci_high_vs_{reference}"]
        fields += [f"significant_vs_{reference}"]
    fields.append("errors")
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for result in results:
        for pass_ in PASSES:
            if pass_ == "backward" and not result.differentiable:
                continue
            row = {"category": result.category, "module": result.module, "batch_size": result.batch_size}
            row["pass"] = pass_
            for name, summary in result.timings.get(pass_, {}).items():
                row[f"{name}_median_us"] = f"{1e6 * summary.median:.4f}"
                row[f"{name}_q1_us"] = f"{1e6 * summary.q1:.4f}"
                row[f"{name}_q3_us"] = f"{1e6 * summary.q3:.4f}"
                row[f"{name}_samples"] = summary.samples
                row[f"{name}_iterations"] = summary.iterations
            for reference, comparison in result.comparisons.get(pass_, {}).items():
                row[f"speedup_vs_{reference}"] = f"{comparison.speedup:.4f}"
                row[f"ci_low_vs_{reference}"] = f"{comparison.ci_low:.4f}"
                row[f"ci_high_vs_{reference}"] = f"{comparison.ci_high:.4f}"
                row[f"significant_vs_{reference}"] = comparison.significant
            row["errors"] = "; ".join(f"{key}: {error}" for key, error in result.errors.items())
            writer.writerow(row)
    return buffer.getvalue()


_HTML_STYLE = """
body { font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 2em; color: #1f2328; }
h1 { font-size: 1.6em; } h2 { font-size: 1.25em; margin-top: 1.6em; }
table { border-collapse: collapse; margin: 0.5em 0; font-size: 0.92em; }
th, td { border: 1px solid #d0d7de; padding: 4px 10px; }
th { background: #f6f8fa; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
td.module { font-weight: 600; vertical-align: top; }
td.faster { background: #dafbe1; color: #116329; font-weight: 600; }
td.slower { background: #ffebe9; color: #a40e26; font-weight: 600; }
td.same { color: #57606a; }
td.na, td.none { color: #8c959f; text-align: center; }
table.metadata td { border: none; padding: 2px 10px 2px 0; }
p.legend { color: #57606a; max-width: 60em; }
"""


def to_html(results: list[Result], metadata: dict[str, Any]) -> str:
    """Format the benchmark results as a standalone HTML page (e.g. to be exposed as a CI artifact).

    The speedup cells are color-coded (faster, slower or not significant),
    and their tooltips show the absolute timings and the confidence interval.

    :param results: The benchmark results.
    :param metadata: The report metadata.

    :return: The HTML report.
    """

    def escape(value: Any) -> str:
        return html.escape(str(value))

    columns = _columns(metadata)
    parts = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        '<head><meta charset="utf-8"><title>warp-nn modules benchmark</title>',
        f"<style>{_HTML_STYLE}</style></head>",
        "<body>",
        "<h1>warp-nn modules benchmark</h1>",
        '<table class="metadata">',
    ]
    parts += [
        f"<tr><td><b>{escape(key)}</b></td><td>{escape(value)}</td></tr>" for key, value in _metadata_lines(metadata)
    ]
    parts += ["</table>", f'<p class="legend">{escape(_legend(metadata)).replace("`", "")}</p>']
    for category, modules in _group(results):
        parts += [
            f"<h2>{escape(category.capitalize())}</h2>",
            "<table>",
            "<thead><tr><th>Module</th><th>Batch size</th>",
        ]
        parts += [f"<th>{escape(header)}</th>" for _, _, header in columns]
        parts += ["</tr></thead>", "<tbody>"]
        for group in modules:
            for i, result in enumerate(group):
                row = "<tr>"
                if i == 0:
                    row += f'<td class="module" rowspan="{len(group)}">{escape(result.module)}</td>'
                row += f'<td class="num">{result.batch_size}</td>'
                for pass_, ref, _ in columns:
                    cell = make_cell(result, pass_, ref, metadata)
                    row += f'<td class="num {cell.state}" title="{escape(cell.details)}">{escape(cell.text)}</td>'
                parts.append(row + "</tr>")
        parts += ["</tbody>", "</table>"]
    errors = _errors(results)
    if errors:
        parts += ["<h2>Errors</h2>", "<ul>"]
        parts += [
            f"<li>{escape(module)} (batch size {batch_size}), {escape(key)}: <code>{escape(error)}</code></li>"
            for module, batch_size, key, error in errors
        ]
        parts.append("</ul>")
    parts += ["</body>", "</html>", ""]
    return "\n".join(parts)


# report formats: file extension and formatting function, by format name
FORMATS: dict[str, tuple[str, Callable[[list[Result], dict[str, Any]], str]]] = {
    "markdown": (".md", to_markdown),
    "json": (".json", to_json),
    "csv": (".csv", to_csv),
    "html": (".html", to_html),
}
