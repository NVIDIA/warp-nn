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

import pytest

import csv
import dataclasses
import io
import json

from benchmarks import _report
from benchmarks._benchmark import Result
from benchmarks._timing import Comparison, Summary


METADATA = {
    "gpu": "GPU model (sm_89)",
    "cpu": "CPU model, 8 logical cores",
    "date": "2026-10-03T00:00:00+00:00",
    "device": "cuda:0",
    "device_name": "GPU",
    "cuda_graph": False,
    "current_commit": "abc",
    "current_dirty": True,
    "baseline_ref": "develop",
    "baseline_commit": "def",
    "confidence": 0.95,
    "min_effect": 0.02,
}


def _summary(median: float) -> Summary:
    return Summary(median=median, q1=0.9 * median, q3=1.1 * median, samples=30, iterations=10)


def _comparison(speedup: float, significant: bool = True) -> Comparison:
    return Comparison(speedup=speedup, ci_low=0.95 * speedup, ci_high=1.05 * speedup, significant=significant)


def _results() -> list[Result]:
    linear_1 = Result("layers", "Linear", 1, True)
    linear_1.timings = {
        "forward": {"current": _summary(10e-6), "torch": _summary(20e-6), "baseline": _summary(10e-6)},
        "backward": {"current": _summary(40e-6), "torch": _summary(20e-6)},
    }
    linear_1.comparisons = {
        "forward": {"torch": _comparison(2.0), "baseline": _comparison(1.004, significant=False)},
        "backward": {"torch": _comparison(0.5)},
    }
    linear_1.errors = {"baseline": "AttributeError: <missing>"}
    linear_4096 = Result("layers", "Linear", 4096, True)
    linear_4096.timings = {"forward": {"current": _summary(2e-3), "torch": _summary(1e-3)}}
    linear_4096.comparisons = {"forward": {"torch": _comparison(0.5)}}
    bitwise = Result("operators", "BitwiseAnd", 1, False)
    bitwise.timings = {"forward": {"current": _summary(10e-6), "torch": _summary(10e-6)}}
    bitwise.comparisons = {"forward": {"torch": _comparison(1.0, significant=False)}}
    relu = Result("activations", "ReLU", 1, True)
    return [linear_1, linear_4096, bitwise, relu]


def test_format_time():
    assert _report.format_time(12.34e-6) == "12.3 µs"
    assert _report.format_time(1.5e-3) == "1.50 ms"
    assert _report.format_time(2.0) == "2.00 s"
    # rounding up to the next unit
    assert _report.format_time(0.99996e-3) == "1.00 ms"


def test_make_cell_states():
    linear, _, bitwise, relu = _results()
    faster = _report.make_cell(linear, "forward", "torch", METADATA)
    assert (faster.text, faster.state) == ("2.00x", "faster")
    assert "warp-nn: 10.0 µs" in faster.details and "PyTorch: 20.0 µs" in faster.details
    assert "95% CI [1.900, 2.100]" in faster.details and "n=30" in faster.details
    linear.comparisons["forward"]["torch"] = dataclasses.replace(_comparison(2.0), strata=(1.9, 2.1))
    assert "per instance set: 1.900x, 2.100x" in _report.make_cell(linear, "forward", "torch", METADATA).details
    same = _report.make_cell(linear, "forward", "baseline", METADATA)
    assert (same.text, same.state) == ("1.00x ~", "same")
    assert "develop: 10.0 µs" in same.details
    slower = _report.make_cell(linear, "backward", "torch", METADATA)
    assert (slower.text, slower.state) == ("0.50x", "slower")
    not_available = _report.make_cell(linear, "backward", "baseline", METADATA)
    assert (not_available.text, not_available.state) == ("n/a", "na")
    assert "AttributeError" in not_available.details
    not_differentiable = _report.make_cell(bitwise, "backward", "torch", METADATA)
    assert (not_differentiable.text, not_differentiable.state) == ("—", "none")
    assert _report.make_cell(relu, "forward", "torch", METADATA).state == "na"


def test_to_markdown():
    report = _report.to_markdown(_results(), METADATA)
    lines = report.splitlines()
    header = (
        "| Module | Batch size | Forward vs PyTorch | Backward vs PyTorch | Forward vs develop | Backward vs develop |"
    )
    assert header in lines
    # categories in a fixed order, and the module name only in its first row
    assert report.index("## Activations") < report.index("## Layers") < report.index("## Operators")
    # markers of the (significantly) faster, slower and not significantly different cells
    assert "| **Linear** | 1 | 🟢 2.00x | 🔴 0.50x | ⚪ 1.00x ~ | n/a |" in lines
    assert "|  | 4096 | 🔴 0.50x | n/a | n/a | n/a |" in lines
    assert "| **BitwiseAnd** | 1 | ⚪ 1.00x ~ | — | n/a | — |" in lines
    assert "🟢 significantly faster, 🔴 significantly slower, ⚪ no significant difference." in report
    assert "abc (with uncommitted changes)" in report
    # the GPU specifications on top
    assert lines[2] == "- **GPU:** GPU model (sm_89)"
    assert lines[3] == "- **CPU:** CPU model, 8 logical cores"
    assert "develop (def)" in report
    assert "## Errors" in report and "Linear (batch size 1), baseline: `AttributeError: <missing>`" in report


def test_to_markdown_without_device_specifications():
    # e.g. a CPU device, or metadata without the specifications
    metadata = {key: value for key, value in METADATA.items() if key not in ("gpu", "cpu")}
    lines = _report.to_markdown(_results()[:1], {**metadata, "gpu": None}).splitlines()
    assert lines[2].startswith("- **Date:**")
    assert not any(line.startswith(("- **GPU:**", "- **CPU:**")) for line in lines)


def test_to_markdown_without_baseline():
    metadata = {**METADATA, "baseline_ref": None, "baseline_commit": None}
    report = _report.to_markdown(_results()[:1], metadata)
    # no baseline columns
    assert "| Module | Batch size | Forward vs PyTorch | Backward vs PyTorch |" in report
    assert "vs baseline" not in report
    assert "not benchmarked" in report


def test_to_json():
    data = json.loads(_report.to_json(_results(), METADATA))
    assert data["metadata"] == METADATA
    linear = data["results"][0]
    assert (linear["module"], linear["batch_size"], linear["differentiable"]) == ("Linear", 1, True)
    assert linear["timings"]["forward"]["torch"]["median_us"] == pytest.approx(20.0)
    assert linear["comparisons"]["forward"]["baseline"]["significant"] is False
    assert linear["errors"] == {"baseline": "AttributeError: <missing>"}


def test_to_csv():
    rows = list(csv.DictReader(io.StringIO(_report.to_csv(_results(), METADATA))))
    # one row per module, batch size and pass (no backward pass for non-differentiable modules)
    assert [(row["module"], row["batch_size"], row["pass"]) for row in rows] == [
        ("Linear", "1", "forward"),
        ("Linear", "1", "backward"),
        ("Linear", "4096", "forward"),
        ("Linear", "4096", "backward"),
        ("BitwiseAnd", "1", "forward"),
        ("ReLU", "1", "forward"),
        ("ReLU", "1", "backward"),
    ]
    assert float(rows[0]["current_median_us"]) == pytest.approx(10.0)
    assert float(rows[0]["speedup_vs_torch"]) == pytest.approx(2.0)
    assert rows[0]["significant_vs_baseline"] == "False"
    assert rows[1]["baseline_median_us"] == "" and rows[1]["speedup_vs_baseline"] == ""
    assert "AttributeError" in rows[0]["errors"]


def test_to_html():
    report = _report.to_html(_results(), METADATA)
    assert report.startswith("<!DOCTYPE html>")
    assert '<td class="module" rowspan="2">Linear</td>' in report
    assert 'class="num faster"' in report and 'class="num slower"' in report and 'class="num same"' in report
    # escaped content
    assert "&lt;missing&gt;" in report and "<missing>" not in report


def test_formats():
    assert set(_report.FORMATS) == {"markdown", "json", "csv", "html"}
    for extension, formatter in _report.FORMATS.values():
        assert extension.startswith(".")
        assert formatter(_results(), METADATA)
