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

"""Benchmark of a module, for a batch size, against its PyTorch counterpart and a baseline warp-nn implementation."""

from __future__ import annotations

from typing import Callable

import dataclasses
import functools
import math
import time
import zlib
from types import ModuleType
import torch

import numpy as np
import warp as wp

from ._runners import Runner, build_torch_runner, build_warp_runner
from ._specs import ModuleSpec
from ._timing import Comparison, Measurement, Summary, TimingConfig, compare, measure, summarize


PASSES = ("forward", "backward")
CURRENT = "current"  # name of the benchmarked (current) warp-nn implementation
TORCH = "torch"  # name of the PyTorch implementation (reference)
BASELINE = "baseline"  # name of the baseline warp-nn implementation (reference)
REFERENCES = (TORCH, BASELINE)  # implementations the current implementation is compared against
_SLOW_BUILD_TIME = 1.0  # build time (in seconds) above which the device is warmed up again (for as long)
_MAX_PADDING = 1 << 22  # maximum size (in bytes) of the random padding allocated before building an instance
_PADDING_ALIGNMENT = 256  # alignment (in bytes) of the random padding sizes


def allocate_padding(name: str, *, device: str, rng: np.random.Generator) -> torch.Tensor | wp.array:
    """Allocate a randomly sized padding block in the memory pool of an implementation's framework.

    Allocating it before building an instance of the implementation randomizes the memory placement of the instance.
    The performance of some (e.g. memory-bound) kernels depends on the placement of their arrays (e.g. by tens of
    percent), so randomizing it turns its effects into random (rather than systematic) differences between instances.

    :param name: The implementation name (PyTorch or a warp-nn implementation).
    :param device: The device.
    :param rng: The random number generator.

    :return: The padding block, which must be kept alive while the instance is used.
    """
    size = int(rng.integers(1, _MAX_PADDING // _PADDING_ALIGNMENT + 1)) * _PADDING_ALIGNMENT
    if name == TORCH:
        return torch.empty(size, dtype=torch.uint8, device=device)
    return wp.empty(size, dtype=wp.uint8, device=device)


@dataclasses.dataclass(frozen=True)
class StatisticsConfig:
    """Configuration of the comparisons between implementations."""

    confidence: float = 0.95
    """Confidence level of the confidence interval of the speedups."""

    min_effect: float = 0.02
    """Minimum relative difference for a speedup to be significant."""


@dataclasses.dataclass
class Result:
    """Result of the benchmark of a module for a batch size."""

    category: str
    module: str
    batch_size: int
    differentiable: bool
    timings: dict[str, dict[str, Summary]] = dataclasses.field(default_factory=dict)
    """Summary statistics of the timing samples, by pass and implementation."""

    comparisons: dict[str, dict[str, Comparison]] = dataclasses.field(default_factory=dict)
    """Comparison of the current implementation against each reference implementation, by pass and reference."""

    errors: dict[str, str] = dataclasses.field(default_factory=dict)
    """Errors, by implementation (when building it) or by pass (when timing it)."""


def describe_error(error: BaseException, max_length: int = 300) -> str:
    """Describe an exception in a single, length-limited line.

    :param error: The exception.
    :param max_length: The maximum length of the description.

    :return: The description.
    """
    lines = str(error).strip().splitlines()
    description = f"{type(error).__name__}: {lines[0] if lines else ''}".strip().rstrip(":")
    return description if len(description) <= max_length else description[: max_length - 3] + "..."


def make_synchronize(device: str) -> Callable[[], None]:
    """Create a function that waits for the completion of all the work queued on a device (by any framework).

    The same function synchronizes all the implementations (PyTorch and warp-nn), so that they are timed alike.
    On CUDA devices, ``torch.cuda.synchronize`` synchronizes the whole device (``cudaDeviceSynchronize``), not only the
    current PyTorch stream: it waits for all the streams of the (primary) CUDA context, which Warp and PyTorch share,
    including the Warp streams (e.g. Warp's default stream, which is not PyTorch's current stream, or other
    non-blocking streams). It is then equivalent to ``wp.synchronize_device``.

    :param device: The device.

    :return: The synchronization function.
    """
    if wp.get_device(device).is_cuda:
        return functools.partial(torch.cuda.synchronize, device)
    return lambda: None


def warm_up_device(device: str, duration: float) -> None:
    """Keep a device busy for a while, so that its clocks ramp up before benchmarking.

    :param device: The device.
    :param duration: The warmup duration (in seconds).
    """
    if not wp.get_device(device).is_cuda or duration <= 0.0:
        return
    a = torch.rand((2048, 2048), device=device)
    deadline = time.perf_counter() + duration
    while time.perf_counter() < deadline:
        for _ in range(10):
            a = torch.mm(a, a).clamp_(-1.0, 1.0)
        torch.cuda.synchronize(device)


def run_benchmark(
    spec: ModuleSpec,
    batch_size: int,
    *,
    implementations: dict[str, ModuleType],
    device: str,
    cuda_graph: bool,
    timing: TimingConfig,
    statistics: StatisticsConfig,
    seed: int = 0,
    instance_sets: int = 2,
    randomize_placement: bool = False,
) -> Result:
    """Benchmark a module, for a batch size, against its PyTorch counterpart and the other warp-nn implementations.

    :param spec: The module specification.
    :param batch_size: The batch size.
    :param implementations: The ``nn`` namespace of the warp-nn implementations, by implementation name.
        It must contain the current implementation (:py:data:`CURRENT`), and can contain the baseline one
        (:py:data:`BASELINE`).
    :param device: The device.
    :param cuda_graph: Whether to replay captured CUDA graphs (rather than running eagerly).
    :param timing: The timing configuration.
    :param statistics: The statistics configuration.
    :param seed: The seed of the random number generators (inputs and parameters).
    :param instance_sets: The number of instance sets (sets of instances of all the implementations, which share the
        samples), whose build orders alternate. More sets average out the effects of the memory placement of the
        instances, at the cost of more memory and calibration time.
    :param randomize_placement: Whether to randomize the memory placement of the instances (with randomly sized
        padding blocks allocated before building them), so that placement effects average out over the instance sets
        (which should then be several) rather than biasing the comparisons systematically.

    :return: The benchmark result.

    :raises ValueError: If the number of instance sets is not positive.
    """
    if instance_sets < 1:
        raise ValueError(f"The number of instance sets must be positive (got {instance_sets})")
    result = Result(category=spec.category, module=spec.name, batch_size=batch_size, differentiable=spec.differentiable)
    arrays = spec.sample_inputs(batch_size, np.random.default_rng(seed))
    builders = {TORCH: functools.partial(build_torch_runner, spec, arrays, device=device, cuda_graph=cuda_graph)}
    for name, nn in implementations.items():
        builders[name] = functools.partial(build_warp_runner, spec, nn, arrays, device=device, cuda_graph=cuda_graph)
    # build several instances of each implementation (instance sets), in alternating orders, and pool their
    # measurements, so that the effects of the build order (e.g. on the placement of the arrays in memory, which can
    # make an instance a few percent slower than an identical one) average out. Building the runners executes the
    # passes once (warmup execution), which compiles/loads the kernels and allocates the cached arrays, so that none
    # of it is timed
    # PyTorch is built first, and the order of the warp-nn implementations is reversed in every other set, so that
    # they swap their positions and neighbors (mirroring the whole order would keep the middle one in place)
    warp_names = list(implementations)
    orders = [[TORCH, *(warp_names if i % 2 == 0 else warp_names[::-1])] for i in range(instance_sets)]
    instances: list[dict[str, Runner]] = [{} for _ in orders]
    failures: list[dict[str, str]] = [{} for _ in orders]
    start = time.perf_counter()
    # padding blocks (kept alive while benchmarking) and their (reproducible) random number generator
    paddings = []
    padding_rng = np.random.default_rng([seed, batch_size, zlib.crc32(spec.name.encode())])
    for runners, errors, order in zip(instances, failures, orders):
        for name in order:
            # the parameters are initialized from the global NumPy (warp-nn) and PyTorch random number generators
            np.random.seed(seed)
            torch.manual_seed(seed)
            try:
                if randomize_placement:
                    paddings.append(allocate_padding(name, device=device, rng=padding_rng))
                runners[name] = builders[name]()
            except Exception as e:
                errors[name] = describe_error(e)
    # implementations that cannot be built are not benchmarked. If only other instances of an implementation cannot
    # be built (e.g. out of memory), only the leading instance sets that could be completely built are benchmarked
    result.errors.update(failures[0])
    for index, errors in enumerate(failures[1:], start=1):
        if set(errors) - set(failures[0]):
            result.errors.update(
                {f"{name} (other instance)": error for name, error in errors.items() if name not in failures[0]}
            )
            del instances[index:]
            break
    # the device idles (and its clocks drop) during long builds (e.g. compiling kernels), so ramp them up again
    if time.perf_counter() - start > _SLOW_BUILD_TIME:
        warm_up_device(device, _SLOW_BUILD_TIME)
    synchronize = make_synchronize(device)
    # the instance sets share the samples and the time budget (each set measures at least a complete cycle of the
    # orderings of the implementations, so that it is balanced on its own: e.g. 6 rounds for 3 implementations)
    config = dataclasses.replace(
        timing,
        samples=math.ceil(timing.samples / len(instances)),
        min_samples=math.ceil(timing.min_samples / len(instances)),
        max_time=timing.max_time / len(instances),
    )
    for pass_ in PASSES:
        sets = [
            {
                name: fn
                for name, runner in runners.items()
                if name not in result.errors and (fn := getattr(runner, pass_))
            }
            for runners in instances
        ]
        if CURRENT not in sets[0]:
            continue
        try:
            subsets = [measure(fns, synchronize=synchronize, config=config) for fns in sets]
        except Exception as e:
            result.errors[pass_] = describe_error(e)
            continue
        # pool the (paired) samples of the instance sets, with the same number of rounds for each of them (the time
        # budget can result in different numbers), so that they are equally weighted
        rounds = min(len(measurement.times) for measurements in subsets for measurement in measurements.values())
        pooled = {
            name: Measurement(
                times=np.concatenate([measurements[name].times[:rounds] for measurements in subsets]),
                iterations=subsets[0][name].iterations,
            )
            for name in subsets[0]
        }
        result.timings[pass_] = {name: summarize(measurement) for name, measurement in pooled.items()}
        result.comparisons[pass_] = {
            reference: compare(
                pooled[reference].times,
                pooled[CURRENT].times,
                confidence=statistics.confidence,
                min_effect=statistics.min_effect,
                strata=len(subsets),
                seed=seed,
            )
            for reference in REFERENCES
            if reference in pooled
        }
    return result
