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

"""Benchmark the performance of the warp-nn modules (activations, layers and operators).

The forward and backward passes of the current warp-nn implementation (the working tree) are compared against
their PyTorch counterparts and against a baseline warp-nn implementation (by default, the ``develop`` branch),
for the batch sizes (number of environments) most commonly used in Isaac Lab tasks.

Usage (from the repository root)::

    python -m benchmarks.benchmark_modules [--cuda-graph] [--modules Linear "Conv*"] [--format markdown html] ...

Run with ``--help`` for all the options.
"""

from __future__ import annotations

from typing import Any

import argparse
import contextlib
import datetime
import fnmatch
import gc
import os
import pathlib
import platform
import subprocess
import sys
import tempfile
import time


if __package__ in (None, ""):  # run as a script (python benchmarks/benchmark_modules.py)
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    __package__ = "benchmarks"

import torch

import numpy as np
import warp as wp

import warp_nn
import warp_nn.nn

from ._baseline import import_baseline, unload_baseline
from ._benchmark import BASELINE, CURRENT, Result, StatisticsConfig, describe_error, run_benchmark, warm_up_device
from ._report import FORMATS
from ._specs import CATEGORIES, SPECS, ModuleSpec
from ._timing import TimingConfig


REPOSITORY = pathlib.Path(__file__).resolve().parents[1]
ROBUST_INSTANCE_SETS = 6  # instance sets of the robust mode (all of them must agree for a significant difference)
# single environment, Isaac Lab's play, training and large training defaults, and PPO mini-batch (4096 envs x 16 / 2)
BATCH_SIZES = (1, 50, 4096, 8192, 32768)


def select_specs(
    specs: tuple[ModuleSpec, ...], *, categories: list[str] | None, patterns: list[str] | None
) -> list[ModuleSpec]:
    """Select the module specifications by category and by (case-insensitive, shell-style) name patterns.

    :param specs: The module specifications.
    :param categories: The categories to select, or None to select all of them.
    :param patterns: The name patterns to select, or None to select all the names.

    :return: The selected module specifications.
    """
    return [
        spec
        for spec in specs
        if (categories is None or spec.category in categories)
        and (patterns is None or any(fnmatch.fnmatch(spec.name.lower(), pattern.lower()) for pattern in patterns))
    ]


def _git_output(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=REPOSITORY, check=True, capture_output=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return ""


def describe_device(device: wp.Device) -> str:
    """Describe the specifications of a device (e.g. for a GPU: model, architecture, memory, driver).

    :param device: The device.

    :return: The description.
    """
    if not device.is_cuda:
        return describe_host()
    driver = ".".join(str(value) for value in wp.get_cuda_driver_version())
    toolkit = ".".join(str(value) for value in wp.get_cuda_toolkit_version())
    return (
        f"{device.name} (sm_{device.arch}, {device.sm_count} SMs, {device.total_memory / 1024**3:.0f} GiB), "
        f"CUDA driver {driver}, CUDA toolkit {toolkit} (Warp) / {torch.version.cuda or 'n/a'} (PyTorch)"
    )


def describe_host() -> str:
    """Describe the host CPU (model and number of logical cores).

    :return: The description.
    """
    model = ""
    with contextlib.suppress(OSError):
        for line in pathlib.Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                model = line.split(":", 1)[1].strip()
                break
    return f"{model or platform.processor() or platform.machine()}, {os.cpu_count()} logical cores"


def collect_metadata(args: argparse.Namespace, *, date: str, baseline_commit: str | None) -> dict[str, Any]:
    """Collect the metadata of the benchmark (environment and configuration).

    :param args: The command line arguments.
    :param date: The (start) date of the benchmark.
    :param baseline_commit: The commit hash of the baseline implementation, or None if it is not benchmarked.

    :return: The metadata.
    """
    device = wp.get_device(args.device)
    return {
        "date": date,
        "device": str(device),
        "device_name": device.name,
        "gpu": describe_device(device) if device.is_cuda else None,
        "cpu": describe_host(),
        "cuda_graph": args.cuda_graph,
        "current_commit": _git_output("rev-parse", "HEAD").strip() or "unknown",
        "current_dirty": bool(_git_output("status", "--porcelain", "--", "warp_nn").strip()),
        "baseline_ref": None if baseline_commit is None else args.baseline_ref,
        "baseline_commit": baseline_commit,
        "versions": (
            f"warp-nn {warp_nn.__version__}, Warp {wp.__version__}, PyTorch {torch.__version__}, "
            f"Python {platform.python_version()}"
        ),
        "allow_tf32": not args.disable_tf32,
        "robust": args.robust,
        "batch_sizes": args.batch_sizes,
        "timing": (
            f"{args.samples} interleaved samples (min. {args.min_samples}, within {args.max_time:g} s) of at least "
            f"{1e3 * args.min_sample_time:g} ms each, shared by {args.instance_sets} instance sets (at least a "
            f"complete cycle of orderings each{', randomized memory placement' if args.robust else ''}), "
            f"{1e3 * args.warmup_time:g} ms warmup per implementation, {args.device_warmup:g} s device warmup"
        ),
        "confidence": args.confidence,
        "min_effect": args.min_effect,
        "seed": args.seed,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line arguments.

    :param argv: The command line arguments (without the program name), or None to use ``sys.argv``.

    :return: The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.benchmark_modules",
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_argument_group("selection")
    group.add_argument("--categories", nargs="+", choices=CATEGORIES, help="module categories (default: all)")
    group.add_argument("--modules", nargs="+", metavar="PATTERN", help="module name patterns, e.g. 'Conv*'")
    group.add_argument("--batch-sizes", nargs="+", type=int, default=list(BATCH_SIZES), help="batch sizes")
    group.add_argument("--list", action="store_true", help="list the selected modules and exit")
    group = parser.add_argument_group("implementations")
    group.add_argument("--baseline-ref", default="develop", help="git reference of the baseline implementation")
    group.add_argument("--no-baseline", action="store_true", help="do not benchmark the baseline implementation")
    group.add_argument("--device", default="cuda:0" if wp.is_cuda_available() else "cpu", help="device")
    group.add_argument("--cuda-graph", action="store_true", help="replay captured CUDA graphs (no host overhead)")
    group.add_argument(
        "--disable-tf32",
        action="store_true",
        help="disable TF32 math in PyTorch (cuBLAS and cuDNN matmuls/convolutions)",
    )
    group = parser.add_argument_group("timing and statistics")
    group.add_argument("--samples", type=int, default=30, help="timing samples (rounds) per implementation")
    group.add_argument("--min-samples", type=int, default=10, help="minimum samples when over the time budget")
    group.add_argument("--min-sample-time", type=float, default=20e-3, help="minimum duration (s) of a sample")
    group.add_argument("--max-time", type=float, default=5.0, help="time budget (s) of the rounds of a measurement")
    group.add_argument("--warmup-time", type=float, default=0.05, help="warmup time (s) per implementation")
    group.add_argument("--device-warmup", type=float, default=2.0, help="device warmup time (s) before benchmarking")
    group.add_argument("--confidence", type=float, default=0.95, help="confidence level of the speedups")
    group.add_argument("--min-effect", type=float, default=0.02, help="minimum relative significant difference")
    group.add_argument("--seed", type=int, default=0, help="seed of the random number generators")
    group.add_argument(
        "--instance-sets",
        type=int,
        default=argparse.SUPPRESS,
        help=f"instance sets (alternating build orders) sharing the samples, each with at least a complete cycle of "
        f"orderings (default: 2, or {ROBUST_INSTANCE_SETS} with --robust)",
    )
    group.add_argument(
        "--robust",
        action="store_true",
        help="randomize the memory placement of the instances, and use more instance sets, so that placement effects "
        "average out (slower: for decisions on large-batch, memory-bound modules)",
    )
    group = parser.add_argument_group("output")
    group.add_argument("--format", nargs="+", choices=list(FORMATS), default=["markdown"], help="report formats")
    group.add_argument(
        "--output-dir", type=pathlib.Path, help="directory of the report files (default: print them to stdout)"
    )
    group.add_argument("--output-name", default="benchmark_modules", help="base name of the report files")
    args = parser.parse_args(argv)
    try:
        device = wp.get_device(args.device)
    except Exception as e:
        parser.error(f"invalid --device '{args.device}': {e}")
    if args.cuda_graph and not device.is_cuda:
        parser.error("--cuda-graph requires a CUDA device")
    if any(batch_size < 1 for batch_size in args.batch_sizes):
        parser.error("--batch-sizes must be positive")
    if not 0.0 < args.confidence < 1.0:
        parser.error("--confidence must be in (0, 1)")
    if args.samples < 1 or not 1 <= args.min_samples <= args.samples:
        parser.error("--samples and --min-samples must satisfy 1 <= min-samples <= samples")
    if not hasattr(args, "instance_sets"):
        args.instance_sets = ROBUST_INSTANCE_SETS if args.robust else 2
    if args.instance_sets < 1:
        parser.error("--instance-sets must be positive")
    if args.robust and args.instance_sets < 2:
        parser.error("--robust requires at least 2 instance sets (to average out the randomized placements)")
    if args.min_sample_time <= 0.0 or args.max_time <= 0.0:
        parser.error("--min-sample-time and --max-time must be positive")
    if args.warmup_time < 0.0 or args.device_warmup < 0.0 or args.min_effect < 0.0:
        parser.error("--warmup-time, --device-warmup and --min-effect must be non-negative")
    if args.output_dir is not None and args.output_dir.exists() and not args.output_dir.is_dir():
        parser.error(f"--output-dir '{args.output_dir}' is not a directory")
    if pathlib.Path(args.output_name).name != args.output_name:
        parser.error("--output-name must be a file name (without directories)")
    return args


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> int:
    """Run the benchmarks and write the reports.

    The global settings modified while benchmarking (Warp's log level, PyTorch's TF32 flags) are restored afterwards.

    :param argv: The command line arguments (without the program name), or None to use ``sys.argv``.

    :return: The exit status.
    """
    previous = (wp.config.log_level, torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    # silence Warp's initialization and module loading messages, which would be mixed with the reports on stdout
    wp.config.log_level = wp.LOG_WARNING
    try:
        return _main(argv)
    finally:
        wp.config.log_level, torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous


def _main(argv: list[str] | None) -> int:
    args = parse_args(argv)
    if REPOSITORY not in pathlib.Path(warp_nn.__file__).resolve().parents:
        _log(f"Warning: the benchmarked warp-nn ({warp_nn.__file__}) is not the one of this repository ({REPOSITORY})")
    specs = select_specs(SPECS, categories=args.categories, patterns=args.modules)
    if args.list:
        print("\n".join(f"{spec.category:<12} {spec.name}" for spec in specs))
        return 0
    if not specs:
        _log("No module matches the selection")
        return 1
    # create the output directory before benchmarking, so that an invalid directory does not lose the results
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    # this script allows TF32 math in PyTorch's matrix multiplications and convolutions (PyTorch itself only allows it
    # in cuDNN convolutions by default), or uses full single precision math like warp-nn (--disable-tf32)
    torch.backends.cuda.matmul.allow_tf32 = not args.disable_tf32
    torch.backends.cudnn.allow_tf32 = not args.disable_tf32
    timing = TimingConfig(
        samples=args.samples,
        min_samples=args.min_samples,
        min_sample_time=args.min_sample_time,
        max_time=args.max_time,
        warmup_time=args.warmup_time,
    )
    statistics = StatisticsConfig(confidence=args.confidence, min_effect=args.min_effect)
    with contextlib.ExitStack() as stack:
        implementations = {CURRENT: warp_nn.nn}
        baseline_commit = None
        if not args.no_baseline:
            # the exported package must exist while benchmarking (the kernels' source is read on instantiation)
            directory = pathlib.Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="warp_nn_baseline_")))
            try:
                implementations[BASELINE], baseline_commit = import_baseline(
                    args.baseline_ref, repository=REPOSITORY, destination=directory
                )
            except Exception as e:
                _log(f"Unable to import the baseline '{args.baseline_ref}' (use --no-baseline to skip it): {e}")
                return 1
            # undo the import (module caches and search path) before removing the directory
            stack.callback(unload_baseline, destination=directory)
            _log(f"Baseline: {args.baseline_ref} ({baseline_commit})")
        date = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
        warm_up_device(args.device, args.device_warmup)
        results: list[Result] = []
        total, start = len(specs) * len(args.batch_sizes), time.perf_counter()
        for spec in specs:
            for batch_size in args.batch_sizes:
                elapsed = time.perf_counter() - start
                _log(f"[{len(results) + 1:>4}/{total}] {elapsed:7.1f} s  {spec.category}: {spec.name} ({batch_size})")
                try:
                    result = run_benchmark(
                        spec,
                        batch_size,
                        implementations=implementations,
                        device=args.device,
                        cuda_graph=args.cuda_graph,
                        timing=timing,
                        statistics=statistics,
                        seed=args.seed,
                        instance_sets=args.instance_sets,
                        randomize_placement=args.robust,
                    )
                except Exception as e:  # unexpected failure: report it and keep benchmarking the other modules
                    result = Result(spec.category, spec.name, batch_size, spec.differentiable)
                    result.errors["benchmark"] = describe_error(e)
                for key, error in result.errors.items():
                    _log(f"       {key}: {error}")
                results.append(result)
                # release the memory of the benchmarked modules before the next benchmark
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        metadata = collect_metadata(args, date=date, baseline_commit=baseline_commit)
    for name in args.format:
        extension, formatter = FORMATS[name]
        report = formatter(results, metadata)
        if args.output_dir is None:
            print(report)
            continue
        path = args.output_dir / f"{args.output_name}{extension}"
        try:
            path.write_text(report, encoding="utf-8")
            _log(f"Report written to {path}")
        except OSError as e:  # do not lose the results of a long benchmark
            _log(f"Unable to write the report to {path} ({e}), printing it instead")
            print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
