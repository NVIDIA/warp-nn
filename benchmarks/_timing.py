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

"""Timing and statistics of the benchmarks.

The timing protocol is designed to produce results that can be used to make optimization decisions:

- Each implementation is warmed up (avoiding anomalous first-call latencies, e.g. lazy initializations or waking up
  worker threads) before calibrating the number of iterations per timing sample.
- Each timing sample runs a callable for a batch of iterations (calibrated so that a sample lasts at least a
  minimum time), synchronizing the device before and after the batch. The per-iteration time is then the end-to-end
  throughput, including the host overhead (e.g. Python, kernel launches) that is not hidden by the device execution.
  An untimed iteration (pre-roll) precedes the batch, so that the sample does not depend on the state (e.g. host
  caches) left by the previous sample of another implementation.
- The samples of the compared implementations are interleaved (one sample of each implementation per round), so that
  slow drifts of the device state (e.g. clocks, temperature) affect all of them alike. The rounds cycle through all
  the orderings of the implementations, so that each implementation is preceded by each other one equally often
  (e.g. the state of the host caches left by the previous implementation does not favor any of them).
- The comparison between two implementations is computed from the per-round (paired) ratios of their samples, which
  cancels the drifts that are slower than a round. Its uncertainty is estimated by bootstrapping the rounds.
- The garbage collector is disabled while timing.
"""

from __future__ import annotations

from typing import Callable, Mapping

import dataclasses
import gc
import itertools
import math
import time

import numpy as np


@dataclasses.dataclass(frozen=True)
class TimingConfig:
    """Configuration of the timing protocol."""

    samples: int = 30
    """Number of timing samples (rounds) per implementation.

    It is rounded up to a multiple of the number of orderings of the implementations (e.g. 6 for 3 implementations).
    """

    min_samples: int = 10
    """Minimum number of timing samples when the time budget does not allow ``samples`` rounds."""

    min_sample_time: float = 20e-3
    """Minimum duration (in seconds) of a timing sample (the iterations are batched until it is reached)."""

    max_time: float = 5.0
    """Time budget (in seconds) for the rounds of a measurement (``min_samples`` rounds are always run)."""

    warmup_time: float = 0.05
    """Warmup time (in seconds) per implementation, before the calibration of the iterations per sample."""

    max_iterations: int = 1 << 20
    """Maximum number of iterations per timing sample."""


@dataclasses.dataclass(frozen=True)
class Measurement:
    """Timing samples of an implementation."""

    times: np.ndarray
    """Per-iteration time (in seconds) of each timing sample."""

    iterations: int
    """Number of iterations per timing sample."""


@dataclasses.dataclass(frozen=True)
class Summary:
    """Summary statistics of the timing samples of an implementation."""

    median: float
    """Median of the per-iteration time (in seconds)."""

    q1: float
    """First quartile of the per-iteration time (in seconds)."""

    q3: float
    """Third quartile of the per-iteration time (in seconds)."""

    samples: int
    """Number of timing samples."""

    iterations: int
    """Number of iterations per timing sample (of the first instance set, when pooling several of them)."""

    @property
    def iqr(self) -> float:
        """Interquartile range of the per-iteration time (in seconds)."""
        return self.q3 - self.q1


@dataclasses.dataclass(frozen=True)
class Comparison:
    """Comparison (speedup) of a candidate implementation against a reference implementation."""

    speedup: float
    """Ratio of the reference time to the candidate time (> 1: the candidate is faster).

    It is the median of the per-round ratios (or, if several strata, the median of the per-stratum medians,
    in log space: the geometric mean of two strata).
    """

    ci_low: float
    """Lower bound of the confidence interval of the speedup (extended to include the per-stratum speedups)."""

    ci_high: float
    """Upper bound of the confidence interval of the speedup (extended to include the per-stratum speedups)."""

    significant: bool
    """Whether the difference is significant.

    The confidence interval excludes 1, and the speedup of every stratum shows a non-negligible effect in the same
    direction (e.g. not only one of the instances of an implementation is slower, because of its memory placement).
    """

    strata: tuple[float, ...] = ()
    """Speedup of each stratum."""


def _run(fn: Callable[[], object], iterations: int, synchronize: Callable[[], None]) -> float:
    # time (in seconds) a batch of iterations, from an idle device to the completion of all the queued work,
    # after an untimed iteration (pre-roll)
    fn()
    synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    synchronize()
    return time.perf_counter() - start


def calibrate(fn: Callable[[], object], *, synchronize: Callable[[], None], config: TimingConfig) -> tuple[int, float]:
    """Find the number of iterations per timing sample so that a sample lasts at least the minimum sample time.

    :param fn: The callable to time.
    :param synchronize: Function that waits for the completion of the work queued on the device.
    :param config: The timing configuration.

    :return: The number of iterations per sample, and the duration (in seconds) of the last calibration sample.
    """
    iterations = 1
    while True:
        # the minimum of two runs, so that a single slow outlier (e.g. a stall) does not stop the calibration early
        elapsed = min(_run(fn, iterations, synchronize), _run(fn, iterations, synchronize))
        if elapsed >= config.min_sample_time or iterations >= config.max_iterations:
            return iterations, elapsed
        # aim slightly above the target, to avoid falling just below it, while at least doubling the iterations
        estimate = math.ceil(1.2 * iterations * config.min_sample_time / max(elapsed, 1e-9))
        iterations = min(config.max_iterations, max(2 * iterations, estimate))


def measure(
    fns: Mapping[str, Callable[[], object]], *, synchronize: Callable[[], None], config: TimingConfig
) -> dict[str, Measurement]:
    """Time several implementations by interleaving their timing samples.

    Every round takes one sample of each implementation, cycling through all the orderings of the implementations.
    All the implementations get the same number of samples (rounds), so that their samples can be paired.

    :param fns: The callables to time, by implementation name.
    :param synchronize: Function that waits for the completion of the work queued on the device.
    :param config: The timing configuration.

    :return: The timing samples, by implementation name.
    """
    names = list(fns)
    gc_enabled = gc.isenabled()
    gc.disable()
    try:
        # warm up (at least one iteration), in growing batches of iterations between synchronizations,
        # so that the device is kept busy (and its clocks ramp up) even for short passes
        for name in names:
            deadline = time.perf_counter() + config.warmup_time
            iterations = 1
            while (
                _run(fns[name], iterations, synchronize) < config.warmup_time / 10
                and iterations < config.max_iterations
            ):
                iterations *= 2
            while time.perf_counter() < deadline:
                _run(fns[name], iterations, synchronize)
        # calibrate the iterations per sample
        iterations, round_time = {}, 0.0
        for name in names:
            iterations[name], elapsed = calibrate(fns[name], synchronize=synchronize, config=config)
            round_time += elapsed * (iterations[name] + 1) / iterations[name]  # including the pre-roll
        # interleaved timing rounds, within the time budget, rounded up to complete cycles of the orderings
        rounds = config.samples
        if rounds * round_time > config.max_time:
            rounds = max(config.min_samples, int(config.max_time / round_time))
        orderings = list(itertools.permutations(names))
        rounds = math.ceil(rounds / len(orderings)) * len(orderings)
        times = {name: np.empty(rounds) for name in names}
        for i in range(rounds):
            for name in orderings[i % len(orderings)]:
                times[name][i] = _run(fns[name], iterations[name], synchronize) / iterations[name]
    finally:
        if gc_enabled:
            gc.enable()
    return {name: Measurement(times=times[name], iterations=iterations[name]) for name in names}


def summarize(measurement: Measurement) -> Summary:
    """Compute the summary statistics of the timing samples of an implementation.

    :param measurement: The timing samples.

    :return: The summary statistics.
    """
    q1, median, q3 = np.quantile(measurement.times, [0.25, 0.5, 0.75])
    return Summary(
        median=float(median),
        q1=float(q1),
        q3=float(q3),
        samples=len(measurement.times),
        iterations=measurement.iterations,
    )


def compare(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    confidence: float = 0.95,
    min_effect: float = 0.02,
    strata: int = 1,
    resamples: int = 2000,
    seed: int = 0,
) -> Comparison:
    """Compare a candidate implementation against a reference implementation from their paired timing samples.

    The samples can be split into strata (equal, contiguous blocks of samples, e.g. measured with different instances
    of the implementations), whose differences are not attributable to the implementations. The speedup is then the
    median of the per-stratum speedups in log space (giving each stratum the same weight, and robust to an outlier
    stratum when there are more than two), and the bootstrap resamples the samples within each stratum.
    The difference is significant only if all the strata agree on it.

    :param reference: Per-iteration time of each timing sample (round) of the reference implementation.
    :param candidate: Per-iteration time of each timing sample (round) of the candidate implementation.
    :param confidence: Confidence level of the (percentile bootstrap) confidence interval of the speedup.
    :param min_effect: Minimum relative difference (e.g. 0.02 for 2%) for the difference to be significant.
    :param strata: Number of strata.
    :param resamples: Number of bootstrap resamples.
    :param seed: Seed of the bootstrap random number generator (for reproducible confidence intervals).

    :return: The comparison.

    :raises ValueError: If the timing samples are not paired (different number of samples), are empty,
        or cannot be split into the strata.
    """
    reference, candidate = np.asarray(reference, dtype=float), np.asarray(candidate, dtype=float)
    if reference.shape != candidate.shape or reference.ndim != 1 or not reference.size:
        raise ValueError(f"Expected paired, non-empty timing samples (got {reference.shape} and {candidate.shape})")
    if strata < 1 or reference.size % strata:
        raise ValueError(f"Unable to split {reference.size} timing samples into {strata} strata")
    ratios = (reference / candidate).reshape(strata, -1)  # (strata, samples per stratum)
    strata_speedups = np.median(ratios, axis=1)
    speedup = float(np.exp(np.median(np.log(strata_speedups))))
    # bootstrap within each stratum: (resamples, strata, samples per stratum)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, ratios.shape[1], size=(resamples, *ratios.shape))
    resampled = ratios[np.arange(strata)[:, None], indices]
    bootstrap = np.exp(np.median(np.log(np.median(resampled, axis=2)), axis=1))
    alpha = (1.0 - confidence) / 2.0
    ci_low, ci_high = (float(value) for value in np.quantile(bootstrap, [alpha, 1.0 - alpha]))
    # the strata can disagree beyond their sampling noise (e.g. instances with different memory placements): the
    # interval then covers the per-stratum speedups, and the difference is significant only if all of them agree
    strata_speedups = tuple(float(value) for value in strata_speedups)
    ci_low, ci_high = min(ci_low, *strata_speedups), max(ci_high, *strata_speedups)
    faster = all(value >= 1.0 + min_effect for value in strata_speedups)
    slower = all(value <= 1.0 / (1.0 + min_effect) for value in strata_speedups)
    significant = (ci_low > 1.0 and faster) or (ci_high < 1.0 and slower)
    return Comparison(speedup=speedup, ci_low=ci_low, ci_high=ci_high, significant=significant, strata=strata_speedups)
