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

import gc
import itertools
import time

import numpy as np

from benchmarks import _timing
from benchmarks._timing import Measurement, TimingConfig, calibrate, compare, measure, summarize


def _noop():
    pass


def _no_sync():
    pass


class _FakeRun:
    """Replacement of ``_timing._run`` that records the timed callables and returns a fixed per-iteration time."""

    def __init__(self, time_per_iteration: float):
        self.time_per_iteration = time_per_iteration
        self.calls = []

    def __call__(self, fn, iterations, synchronize):
        self.calls.append((fn, iterations))
        return self.time_per_iteration * iterations


def test_calibrate_reaches_min_sample_time():
    config = TimingConfig(min_sample_time=5e-3)
    iterations, elapsed = calibrate(lambda: time.sleep(5e-4), synchronize=_no_sync, config=config)
    assert iterations > 1
    assert elapsed >= config.min_sample_time


def test_calibrate_respects_max_iterations():
    iterations, _ = calibrate(_noop, synchronize=_no_sync, config=TimingConfig(min_sample_time=10.0, max_iterations=8))
    assert iterations == 8


def test_calibrate_single_iteration_for_slow_callables():
    iterations, _ = calibrate(lambda: time.sleep(2e-3), synchronize=_no_sync, config=TimingConfig(min_sample_time=1e-3))
    assert iterations == 1


def test_calibrate_ignores_slow_outlier():
    # the second call (the first timed one, after the pre-roll) stalls: it must not end the calibration early
    calls = []

    def fn():
        calls.append(1)
        time.sleep(20e-3 if len(calls) == 2 else 1e-5)

    iterations, _ = calibrate(fn, synchronize=_no_sync, config=TimingConfig(min_sample_time=5e-3))
    assert iterations > 1


def test_run_includes_untimed_pre_roll():
    calls = []
    _timing._run(lambda: calls.append(1), 3, _no_sync)
    assert len(calls) == 4


def test_measure_interleaves_all_orderings(monkeypatch):
    fake = _FakeRun(1e-3)
    monkeypatch.setattr(_timing, "_run", fake)
    fns = {"a": lambda: None, "b": lambda: None, "c": lambda: None}
    names = {fn: name for name, fn in fns.items()}
    # 4 samples are rounded up to a complete cycle of the 6 orderings of 3 implementations
    measurements = measure(fns, synchronize=_no_sync, config=TimingConfig(samples=4, min_samples=1, warmup_time=0.0))
    assert all(len(measurement.times) == 6 for measurement in measurements.values())
    timed = [names[fn] for fn, _ in fake.calls[-18:]]
    rounds = [tuple(timed[i : i + 3]) for i in range(0, 18, 3)]
    assert sorted(rounds) == sorted(itertools.permutations("abc"))


def test_measure_per_iteration_times(monkeypatch):
    monkeypatch.setattr(_timing, "_run", _FakeRun(2e-3))
    config = TimingConfig(samples=6, min_samples=1, min_sample_time=10e-3, warmup_time=0.0)
    measurements = measure({"a": _noop, "b": _noop}, synchronize=_no_sync, config=config)
    for measurement in measurements.values():
        assert measurement.iterations >= 5
        np.testing.assert_allclose(measurement.times, 2e-3)


def test_measure_warms_up_in_batches(monkeypatch):
    fake = _FakeRun(1e-6)
    monkeypatch.setattr(_timing, "_run", fake)
    config = TimingConfig(samples=1, min_samples=1, min_sample_time=1e-3, warmup_time=0.01)
    measure({"a": _noop}, synchronize=_no_sync, config=config)
    # the iterations double until a warmup batch lasts a tenth of the warmup time
    assert max(iterations for _, iterations in fake.calls[:20]) == 1024


def test_measure_time_budget(monkeypatch):
    # every iteration takes 1 s: a round of 3 implementations (with the pre-roll) takes 6 s
    monkeypatch.setattr(_timing, "_run", _FakeRun(1.0))
    config = TimingConfig(samples=30, min_samples=10, min_sample_time=1e-3, max_time=10.0, warmup_time=0.0)
    measurements = measure({"a": _noop, "b": _noop, "c": _noop}, synchronize=_no_sync, config=config)
    # the minimum number of samples, rounded up to a complete cycle of orderings
    assert all(len(measurement.times) == 12 for measurement in measurements.values())


def test_measure_restores_garbage_collector():
    assert gc.isenabled()
    config = TimingConfig(samples=1, min_samples=1, min_sample_time=1e-4, warmup_time=0.0)
    measure({"a": _noop}, synchronize=_no_sync, config=config)
    assert gc.isenabled()

    def fail():
        raise RuntimeError("failure")

    with pytest.raises(RuntimeError, match="failure"):
        measure({"a": fail}, synchronize=_no_sync, config=config)
    assert gc.isenabled()


def test_measure_synchronizes_around_samples():
    events = []
    config = TimingConfig(samples=2, min_samples=1, min_sample_time=1e-4, warmup_time=0.0)
    measure({"a": lambda: events.append("fn")}, synchronize=lambda: events.append("sync"), config=config)
    assert events[-1] == "sync"
    assert events.count("sync") >= 2 * 2


def test_summarize():
    summary = summarize(Measurement(times=np.array([1.0, 2.0, 3.0, 4.0, 5.0]), iterations=7))
    assert summary.median == 3.0
    assert summary.q1 == 2.0
    assert summary.q3 == 4.0
    assert summary.iqr == 2.0
    assert summary.samples == 5
    assert summary.iterations == 7


def test_compare_identical():
    times = np.random.default_rng(0).uniform(1.0, 1.1, size=30)
    comparison = compare(times, times.copy())
    assert comparison.speedup == 1.0
    assert not comparison.significant


def test_compare_faster_candidate():
    rng = np.random.default_rng(0)
    reference = rng.uniform(2.0, 2.2, size=30)
    candidate = reference / 2.0 * rng.uniform(0.98, 1.02, size=30)
    comparison = compare(reference, candidate)
    assert comparison.speedup == pytest.approx(2.0, rel=0.02)
    assert comparison.ci_low <= comparison.speedup <= comparison.ci_high
    assert comparison.ci_low > 1.0
    assert comparison.significant


def test_compare_slower_candidate():
    reference = np.full(30, 1.0)
    candidate = np.full(30, 1.25)
    comparison = compare(reference, candidate)
    assert comparison.speedup == pytest.approx(0.8)
    assert comparison.significant


def test_compare_min_effect():
    # a consistent, but negligible, difference
    reference = np.full(30, 1.01)
    candidate = np.full(30, 1.0)
    assert compare(reference, candidate, min_effect=0.0).significant
    assert not compare(reference, candidate, min_effect=0.02).significant


def test_compare_reproducible():
    rng = np.random.default_rng(1)
    reference, candidate = rng.uniform(1.0, 2.0, size=20), rng.uniform(1.0, 2.0, size=20)
    assert compare(reference, candidate, seed=3) == compare(reference, candidate, seed=3)


def test_compare_paired_samples_required():
    with pytest.raises(ValueError, match="paired"):
        compare(np.ones(3), np.ones(4))
    with pytest.raises(ValueError, match="paired"):
        compare(np.ones(0), np.ones(0))


def test_compare_strata():
    # each stratum (e.g. instance set) has a constant bias, of opposite signs: they cancel out
    reference = np.ones(20)
    candidate = np.r_[np.full(10, 1.02), np.full(10, 1.0 / 1.02)]
    comparison = compare(reference, candidate, strata=2, min_effect=0.0)
    assert comparison.speedup == pytest.approx(1.0)
    assert comparison.strata == pytest.approx((1.0 / 1.02, 1.02))
    # the confidence interval covers the per-stratum speedups
    assert comparison.ci_low == pytest.approx(1.0 / 1.02) and comparison.ci_high == pytest.approx(1.02)
    assert not comparison.significant
    # a single stratum is the median of the ratios
    assert compare(reference, candidate).speedup == pytest.approx(np.median(reference / candidate))


def test_compare_strata_weights():
    # the strata are equally weighted (geometric mean of their speedups)
    reference = np.ones(4)
    candidate = np.array([0.5, 0.5, 2.0, 2.0])
    assert compare(reference, candidate, strata=2).speedup == pytest.approx(1.0)


def test_compare_invalid_strata():
    with pytest.raises(ValueError, match="strata"):
        compare(np.ones(5), np.ones(5), strata=2)
    with pytest.raises(ValueError, match="strata"):
        compare(np.ones(4), np.ones(4), strata=0)


def test_compare_strata_disagreement():
    # one stratum (e.g. an instance with an unfavorable memory placement) is much slower: the pooled interval would
    # exclude 1, but the strata disagree, so the difference is not significant
    rng = np.random.default_rng(0)
    reference = rng.uniform(1.0, 1.001, size=24)
    candidate = np.r_[reference[:12] * 1.5, reference[12:]]
    comparison = compare(reference, candidate, strata=2)
    assert comparison.strata == pytest.approx((1 / 1.5, 1.0))
    assert comparison.ci_low == pytest.approx(1 / 1.5) and comparison.ci_high >= 1.0
    assert not comparison.significant


def test_compare_strata_agreement():
    # all the strata agree on a non-negligible slowdown (with different magnitudes)
    reference = np.ones(24)
    candidate = np.r_[np.full(12, 1.5), np.full(12, 1.2)]
    comparison = compare(reference, candidate, strata=2)
    assert comparison.significant
    assert comparison.ci_high <= 1 / 1.2 + 1e-12
    # one of them below the minimum effect
    candidate = np.r_[np.full(12, 1.5), np.full(12, 1.01)]
    assert not compare(reference, candidate, strata=2, min_effect=0.02).significant


def test_compare_strata_outlier():
    # one of four strata is an outlier (e.g. an instance with an unfavorable memory placement): the median of the
    # per-stratum speedups is robust to it, and the strata disagree, so the difference is not significant
    reference = np.ones(40)
    candidate = np.r_[np.full(10, 1 / 1.11), np.full(10, 1.0), np.full(10, 1 / 1.02), np.full(10, 1.0)]
    comparison = compare(reference, candidate, strata=4)
    assert comparison.speedup == pytest.approx(np.sqrt(1.0 * 1.02))
    assert comparison.strata == pytest.approx((1.11, 1.0, 1.02, 1.0))
    assert comparison.ci_high == pytest.approx(1.11)
    assert not comparison.significant


def test_compare_strata_agreement_faster():
    # all the strata agree on a non-negligible speedup (with different magnitudes)
    reference = np.ones(24)
    candidate = np.r_[np.full(12, 1 / 1.5), np.full(12, 1 / 1.2)]
    comparison = compare(reference, candidate, strata=2)
    assert comparison.speedup == pytest.approx(np.sqrt(1.5 * 1.2))
    assert comparison.significant
    assert comparison.ci_low == pytest.approx(1.2)
