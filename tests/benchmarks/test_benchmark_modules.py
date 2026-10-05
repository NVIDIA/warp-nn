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

import dataclasses
import json
import sys
import types
import torch

import numpy as np
import warp as wp

import warp_nn.nn as nn
from benchmarks import _benchmark, _timing, benchmark_modules
from benchmarks._benchmark import (
    BASELINE,
    CURRENT,
    StatisticsConfig,
    allocate_padding,
    describe_error,
    make_synchronize,
    run_benchmark,
    warm_up_device,
)
from benchmarks._specs import SPECS
from benchmarks._timing import TimingConfig


DEVICE = "cuda:0" if wp.is_cuda_available() else "cpu"
FAST_TIMING = TimingConfig(samples=2, min_samples=1, min_sample_time=1e-4, warmup_time=0.0)
FAST_ARGS = ["--samples", "2", "--min-samples", "1", "--min-sample-time", "0.0001"]
FAST_ARGS += ["--warmup-time", "0", "--device-warmup", "0", "--device", DEVICE]
_SPECS = {spec.name: spec for spec in SPECS}


def test_describe_error():
    assert describe_error(ValueError("first line\nsecond line")) == "ValueError: first line"
    assert describe_error(KeyError()) == "KeyError"
    assert len(describe_error(RuntimeError("x" * 1000), max_length=50)) == 50


def test_describe_device():
    description = benchmark_modules.describe_device(wp.get_device(DEVICE))
    assert description.startswith(wp.get_device(DEVICE).name)
    if wp.get_device(DEVICE).is_cuda:
        assert f"sm_{wp.get_device(DEVICE).arch}" in description and "CUDA driver" in description
    assert "logical cores" in benchmark_modules.describe_device(wp.get_device("cpu"))
    assert "logical cores" in benchmark_modules.describe_host()


def test_make_synchronize():
    make_synchronize(DEVICE)()
    make_synchronize("cpu")()


def test_warm_up_device():
    warm_up_device(DEVICE, 0.01)
    warm_up_device("cpu", 1.0)  # no-op


def test_run_benchmark():
    result = run_benchmark(
        _SPECS["ReLU"],
        2,
        implementations={CURRENT: nn, BASELINE: nn},
        device=DEVICE,
        cuda_graph=False,
        timing=FAST_TIMING,
        statistics=StatisticsConfig(),
    )
    assert (result.category, result.module, result.batch_size, result.differentiable) == (
        "activations",
        "ReLU",
        2,
        True,
    )
    assert not result.errors
    for pass_ in ("forward", "backward"):
        assert set(result.timings[pass_]) == {"torch", CURRENT, BASELINE}
        assert set(result.comparisons[pass_]) == {"torch", BASELINE}


def test_run_benchmark_unavailable_baseline():
    # e.g. a module that does not exist in the baseline implementation
    result = run_benchmark(
        _SPECS["ReLU"],
        2,
        implementations={CURRENT: nn, BASELINE: types.SimpleNamespace()},
        device=DEVICE,
        cuda_graph=False,
        timing=FAST_TIMING,
        statistics=StatisticsConfig(),
    )
    assert set(result.errors) == {BASELINE}
    assert result.errors[BASELINE].startswith("AttributeError")
    assert set(result.comparisons["forward"]) == {"torch"}


def test_run_benchmark_non_differentiable():
    result = run_benchmark(
        _SPECS["BitwiseAnd"],
        2,
        implementations={CURRENT: nn},
        device=DEVICE,
        cuda_graph=False,
        timing=FAST_TIMING,
        statistics=StatisticsConfig(),
    )
    assert not result.differentiable
    assert set(result.timings) == {"forward"}


def test_select_specs():
    select = benchmark_modules.select_specs
    assert len(select(SPECS, categories=None, patterns=None)) == len(SPECS)
    assert {spec.category for spec in select(SPECS, categories=["operators"], patterns=None)} == {"operators"}
    assert [spec.name for spec in select(SPECS, categories=None, patterns=["conv*", "LINEAR"])] == [
        "Conv1D",
        "Conv2D",
        "Linear",
    ]
    assert not select(SPECS, categories=["activations"], patterns=["Linear"])


def test_parse_args():
    args = benchmark_modules.parse_args([])
    assert args.device == ("cuda:0" if wp.is_cuda_available() else "cpu")
    assert args.batch_sizes == [1, 50, 4096, 8192, 32768]
    assert args.baseline_ref == "develop"
    assert args.format == ["markdown"]


def test_parse_args_tf32():
    assert benchmark_modules.parse_args(["--device", DEVICE]).disable_tf32 is False
    assert benchmark_modules.parse_args(["--disable-tf32", "--device", DEVICE]).disable_tf32 is True


@pytest.mark.parametrize("disable", [False, True])
def test_main_tf32(monkeypatch, tmp_path, disable):
    flags = []
    run_benchmark = benchmark_modules.run_benchmark

    def spy(*args, **kwargs):
        flags.append((torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32))
        return run_benchmark(*args, **kwargs)

    monkeypatch.setattr(benchmark_modules, "run_benchmark", spy)
    previous = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    argv = ["--modules", "Abs", "--batch-sizes", "2", "--no-baseline", *FAST_ARGS, "--format", "json"]
    argv += ["--output-dir", str(tmp_path), *(["--disable-tf32"] if disable else [])]
    assert benchmark_modules.main(argv) == 0
    # the flags are set while benchmarking, reported, and restored afterwards
    assert flags == [(not disable, not disable)]
    assert json.loads((tmp_path / "benchmark_modules.json").read_text())["metadata"]["allow_tf32"] is not disable
    assert (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32) == previous


def test_main_list(capsys):
    assert benchmark_modules.main(["--list", "--modules", "Linear"]) == 0
    assert capsys.readouterr().out.split() == ["layers", "Linear"]


def test_main_no_match():
    assert benchmark_modules.main(["--modules", "NotAModule"]) == 1


def test_main_stdout(capsys):
    argv = ["--modules", "ReLU", "--batch-sizes", "2", "--no-baseline", "--format", "markdown", *FAST_ARGS]
    assert benchmark_modules.main(argv) == 0
    output = capsys.readouterr().out
    assert "| **ReLU** | 2 |" in output
    assert "Forward vs PyTorch" in output and "vs baseline" not in output


def test_main_output_files(tmp_path):
    argv = ["--modules", "Abs", "BitwiseNot", "--batch-sizes", "1", "3", "--no-baseline", *FAST_ARGS]
    argv += ["--format", "markdown", "json", "csv", "html", "--output-dir", str(tmp_path), "--output-name", "report"]
    assert benchmark_modules.main(argv) == 0
    assert sorted(path.name for path in tmp_path.iterdir()) == ["report.csv", "report.html", "report.json", "report.md"]
    data = json.loads((tmp_path / "report.json").read_text())
    assert [(result["module"], result["batch_size"]) for result in data["results"]] == [
        ("Abs", 1),
        ("Abs", 3),
        ("BitwiseNot", 1),
        ("BitwiseNot", 3),
    ]
    assert data["metadata"]["baseline_ref"] is None
    assert data["metadata"]["device"] == str(wp.get_device(DEVICE))


def test_main_missing_baseline(capsys):
    path = list(sys.path)
    argv = ["--modules", "ReLU", "--baseline-ref", "this-ref-does-not-exist", *FAST_ARGS]
    assert benchmark_modules.main(argv) == 1
    assert "--no-baseline" in capsys.readouterr().err
    assert sys.path == path
    assert "warp_nn_baseline" not in sys.modules


def test_main_baseline_twice(tmp_path):
    # the baseline (here, the committed version of the repository) is imported and unloaded on every run
    argv = ["--modules", "Abs", "--batch-sizes", "2", "--baseline-ref", "HEAD", *FAST_ARGS, "--format", "json"]
    for i in range(2):
        assert benchmark_modules.main([*argv, "--output-dir", str(tmp_path), "--output-name", f"run{i}"]) == 0
        data = json.loads((tmp_path / f"run{i}.json").read_text())
        assert data["metadata"]["baseline_ref"] == "HEAD"
        assert set(data["results"][0]["comparisons"]["forward"]) == {"torch", "baseline"}
        assert "warp_nn_baseline" not in sys.modules
        assert not any("warp_nn_baseline_" in path for path in sys.path)


@pytest.mark.parametrize(
    "argv, message",
    [
        (["--min-sample-time", "0"], "must be positive"),
        (["--max-time", "-1"], "must be positive"),
        (["--warmup-time", "-1"], "must be non-negative"),
        (["--device-warmup", "-1"], "must be non-negative"),
        (["--min-effect", "-0.1"], "must be non-negative"),
        (["--device", "not-a-device"], "invalid --device"),
        (["--batch-sizes", "0"], "must be positive"),
        (["--confidence", "1.5"], "must be in"),
        (["--samples", "5", "--min-samples", "10"], "min-samples <= samples"),
        (["--output-name", "a/b"], "must be a file name"),
        (["--instance-sets", "0"], "--instance-sets must be positive"),
        (["--robust", "--instance-sets", "1"], "--robust requires at least 2 instance sets"),
    ],
)
def test_parse_args_invalid_values(capsys, argv, message):
    with pytest.raises(SystemExit):
        benchmark_modules.parse_args([*argv, "--device", DEVICE] if "--device" not in argv else argv)
    assert message in capsys.readouterr().err


def test_parse_args_output_dir_is_file(capsys, tmp_path):
    (tmp_path / "file").write_text("")
    with pytest.raises(SystemExit):
        benchmark_modules.parse_args(["--output-dir", str(tmp_path / "file"), "--device", DEVICE])
    assert "is not a directory" in capsys.readouterr().err


@pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA is not available")
def test_parse_args_cuda_graph_on_cpu(capsys):
    with pytest.raises(SystemExit):
        benchmark_modules.parse_args(["--cuda-graph", "--device", "cpu"])
    assert "requires a CUDA device" in capsys.readouterr().err


def test_main_report_write_failure(capsys, tmp_path):
    # the report cannot be written (a directory exists with its name): it is printed instead
    (tmp_path / "report.md").mkdir()
    argv = ["--modules", "Abs", "--batch-sizes", "2", "--no-baseline", *FAST_ARGS]
    argv += ["--output-dir", str(tmp_path), "--output-name", "report"]
    assert benchmark_modules.main(argv) == 0
    captured = capsys.readouterr()
    assert "Unable to write the report" in captured.err
    assert "| **Abs** | 2 |" in captured.out


def test_main_unexpected_failure(monkeypatch, capsys):
    def fail(*args, **kwargs):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(benchmark_modules, "run_benchmark", fail)
    argv = ["--modules", "Abs", "--batch-sizes", "2", "--no-baseline", *FAST_ARGS]
    assert benchmark_modules.main(argv) == 0
    output = capsys.readouterr().out
    assert "| **Abs** | 2 | n/a | n/a |" in output
    assert "Abs (batch size 2), benchmark: `RuntimeError: unexpected`" in output


def test_run_benchmark_measurement_failure(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("measurement")

    monkeypatch.setattr(_benchmark, "measure", fail)
    result = run_benchmark(
        _SPECS["ReLU"],
        2,
        implementations={CURRENT: nn},
        device=DEVICE,
        cuda_graph=False,
        timing=FAST_TIMING,
        statistics=StatisticsConfig(),
    )
    assert result.errors == {"forward": "RuntimeError: measurement", "backward": "RuntimeError: measurement"}
    assert not result.timings and not result.comparisons


def test_run_benchmark_other_instance_failure():
    # the other instance of the baseline cannot be built (e.g. out of memory): only the first set is benchmarked
    relu = _SPECS["ReLU"]
    baseline = types.SimpleNamespace(ReLU=nn.ReLU)
    builds = []

    def warp_factory(namespace):
        if namespace is baseline:
            builds.append(BASELINE)
            if len(builds) == 2:
                raise MemoryError("out of memory")
        return relu.warp(namespace)

    spec = dataclasses.replace(relu, warp=warp_factory)
    timing = TimingConfig(samples=6, min_samples=1, min_sample_time=1e-4, warmup_time=0.0)
    result = run_benchmark(
        spec,
        2,
        implementations={CURRENT: nn, BASELINE: baseline},
        device=DEVICE,
        cuda_graph=False,
        timing=timing,
        statistics=StatisticsConfig(),
    )
    assert result.errors == {f"{BASELINE} (other instance)": "MemoryError: out of memory"}
    assert set(result.comparisons["forward"]) == {"torch", BASELINE}
    # all the samples from the first instance set
    assert all(summary.samples == 6 for summary in result.timings["forward"].values())


@pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA is not available")
def test_run_benchmark_cuda_graph():
    result = run_benchmark(
        _SPECS["Linear"],
        4,
        implementations={CURRENT: nn, BASELINE: nn},
        device=DEVICE,
        cuda_graph=True,
        timing=FAST_TIMING,
        statistics=StatisticsConfig(),
    )
    assert not result.errors
    assert set(result.comparisons["backward"]) == {"torch", BASELINE}


def test_run_benchmark_swapped_instances():
    # two instances of each implementation are built in swapped orders, and their samples are pooled
    builds = []
    relu = _SPECS["ReLU"]
    baseline = types.SimpleNamespace(ReLU=nn.ReLU)

    def warp_factory(namespace):
        builds.append(BASELINE if namespace is baseline else CURRENT)
        return relu.warp(namespace)

    def torch_factory():
        builds.append("torch")
        return relu.torch()

    spec = dataclasses.replace(relu, warp=warp_factory, torch=torch_factory)
    timing = TimingConfig(samples=6, min_samples=1, min_sample_time=1e-4, warmup_time=0.0)
    result = run_benchmark(
        spec,
        2,
        implementations={CURRENT: nn, BASELINE: baseline},
        device=DEVICE,
        cuda_graph=False,
        timing=timing,
        statistics=StatisticsConfig(),
    )
    assert builds == ["torch", CURRENT, BASELINE, "torch", BASELINE, CURRENT]
    # 3 samples per instance set, rounded up to a complete cycle of the 6 orderings of the 3 implementations
    assert all(summary.samples == 12 for summary in result.timings["forward"].values())


def test_run_benchmark_instance_sets():
    # the instance sets alternate the order of the warp-nn implementations (PyTorch first)
    builds = []
    relu = _SPECS["ReLU"]
    baseline = types.SimpleNamespace(ReLU=nn.ReLU)

    def warp_factory(namespace):
        builds.append(BASELINE if namespace is baseline else CURRENT)
        return relu.warp(namespace)

    def torch_factory():
        builds.append("torch")
        return relu.torch()

    spec = dataclasses.replace(relu, warp=warp_factory, torch=torch_factory)
    timing = TimingConfig(samples=12, min_samples=1, min_sample_time=1e-4, warmup_time=0.0)
    result = run_benchmark(
        spec,
        2,
        implementations={CURRENT: nn, BASELINE: baseline},
        device=DEVICE,
        cuda_graph=False,
        timing=timing,
        statistics=StatisticsConfig(),
        instance_sets=3,
    )
    assert builds == ["torch", CURRENT, BASELINE, "torch", BASELINE, CURRENT, "torch", CURRENT, BASELINE]
    assert len(result.comparisons["forward"][BASELINE].strata) == 3
    # 4 samples per instance set, rounded up to a complete cycle of the 6 orderings of the 3 implementations
    assert all(summary.samples == 18 for summary in result.timings["forward"].values())
    with pytest.raises(ValueError, match="instance sets"):
        run_benchmark(
            spec,
            2,
            implementations={CURRENT: nn},
            device=DEVICE,
            cuda_graph=False,
            timing=timing,
            statistics=StatisticsConfig(),
            instance_sets=0,
        )


def test_allocate_padding():
    rng = np.random.default_rng(0)
    for name, array_type in (("torch", torch.Tensor), (CURRENT, wp.array)):
        sizes = set()
        for _ in range(5):
            padding = allocate_padding(name, device=DEVICE, rng=rng)
            assert isinstance(padding, array_type)
            size = padding.numel() if isinstance(padding, torch.Tensor) else padding.size
            assert 0 < size <= 1 << 22 and size % 256 == 0
            sizes.add(size)
        assert len(sizes) > 1


def test_run_benchmark_randomize_placement(monkeypatch):
    paddings = []

    def record(name, *, device, rng):
        paddings.append(name)
        return allocate_padding(name, device=device, rng=rng)

    monkeypatch.setattr(_benchmark, "allocate_padding", record)
    kwargs = dict(implementations={CURRENT: nn, BASELINE: nn}, device=DEVICE, cuda_graph=False, timing=FAST_TIMING)
    result = run_benchmark(_SPECS["ReLU"], 2, statistics=StatisticsConfig(), instance_sets=3, **kwargs)
    assert not paddings and not result.errors
    result = run_benchmark(
        _SPECS["ReLU"], 2, statistics=StatisticsConfig(), instance_sets=3, randomize_placement=True, **kwargs
    )
    # a padding block before building each instance
    assert paddings == ["torch", CURRENT, BASELINE, "torch", BASELINE, CURRENT, "torch", CURRENT, BASELINE]
    assert not result.errors and len(result.comparisons["forward"][BASELINE].strata) == 3


def test_parse_args_robust():
    args = benchmark_modules.parse_args(["--device", DEVICE])
    assert (args.robust, args.instance_sets) == (False, 2)
    args = benchmark_modules.parse_args(["--robust", "--device", DEVICE])
    assert (args.robust, args.instance_sets) == (True, benchmark_modules.ROBUST_INSTANCE_SETS)
    args = benchmark_modules.parse_args(["--robust", "--instance-sets", "4", "--device", DEVICE])
    assert (args.robust, args.instance_sets) == (True, 4)


def test_main_robust(tmp_path):
    argv = ["--modules", "Abs", "--batch-sizes", "2", "--no-baseline", "--robust", "--instance-sets", "2", *FAST_ARGS]
    argv += ["--format", "json", "--output-dir", str(tmp_path)]
    assert benchmark_modules.main(argv) == 0
    metadata = json.loads((tmp_path / "benchmark_modules.json").read_text())["metadata"]
    assert metadata["robust"] is True
    assert "randomized memory placement" in metadata["timing"]


def test_run_benchmark_unequal_rounds(monkeypatch):
    # the instance sets measured different numbers of rounds (e.g. within the time budget): they are truncated to the
    # same number of rounds, so that they are equally weighted
    calls = []

    def fake_measure(fns, *, synchronize, config):
        calls.append(None)
        rounds = 12 if len(calls) % 2 else 6
        return {name: _timing.Measurement(times=np.full(rounds, 1e-3), iterations=1) for name in fns}

    monkeypatch.setattr(_benchmark, "measure", fake_measure)
    result = run_benchmark(
        _SPECS["ReLU"],
        2,
        implementations={CURRENT: nn, BASELINE: nn},
        device=DEVICE,
        cuda_graph=False,
        timing=FAST_TIMING,
        statistics=StatisticsConfig(),
    )
    for pass_ in ("forward", "backward"):
        assert all(summary.samples == 12 for summary in result.timings[pass_].values())
        assert len(result.comparisons[pass_][BASELINE].strata) == 2


def test_run_benchmark_keeps_leading_instance_sets():
    # the third instance of the baseline cannot be built: the first two instance sets are benchmarked
    relu = _SPECS["ReLU"]
    baseline = types.SimpleNamespace(ReLU=nn.ReLU)
    builds = []

    def warp_factory(namespace):
        if namespace is baseline:
            builds.append(BASELINE)
            if len(builds) == 3:
                raise MemoryError("out of memory")
        return relu.warp(namespace)

    spec = dataclasses.replace(relu, warp=warp_factory)
    result = run_benchmark(
        spec,
        2,
        implementations={CURRENT: nn, BASELINE: baseline},
        device=DEVICE,
        cuda_graph=False,
        timing=FAST_TIMING,
        statistics=StatisticsConfig(),
        instance_sets=4,
    )
    assert result.errors == {f"{BASELINE} (other instance)": "MemoryError: out of memory"}
    assert len(result.comparisons["forward"][BASELINE].strata) == 2


def test_help_instance_sets(capsys):
    with pytest.raises(SystemExit):
        benchmark_modules.parse_args(["--help"])
    help_text = " ".join(capsys.readouterr().out.split())
    # the help of --instance-sets (without the automatic default, since it depends on --robust)
    entry = help_text[help_text.index("--instance-sets INSTANCE_SETS instance sets") : help_text.index("--robust ")]
    assert entry.endswith(f"(default: 2, or {benchmark_modules.ROBUST_INSTANCE_SETS} with --robust) ")
    assert "==SUPPRESS==" not in help_text
