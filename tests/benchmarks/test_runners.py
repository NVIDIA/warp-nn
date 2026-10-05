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
import gc
import weakref
import torch

import numpy as np
import warp as wp

import warp_nn.nn as nn
from benchmarks._runners import build_torch_runner, build_warp_runner, flatten
from benchmarks._specs import CATEGORIES, SPECS, InputSpec


DEVICE = "cuda:0" if wp.is_cuda_available() else "cpu"
_SPECS = {spec.name: spec for spec in SPECS}


def _to_numpy(data) -> np.ndarray:
    return data.numpy() if isinstance(data, wp.array) else data.detach().cpu().numpy()


def _grads(inputs) -> list[np.ndarray]:
    return [_to_numpy(input.grad).copy() for input in inputs]


def test_flatten():
    assert flatten(1) == [1]
    assert flatten((1, (2, 3), [4])) == [1, 2, 3, 4]


def test_input_spec_sample():
    rng = np.random.default_rng(0)
    array = InputSpec((3, 4), low=0.5, high=2.0).sample(5, rng)
    assert array.shape == (5, 3, 4) and array.dtype == np.float32
    assert array.min() >= 0.5 and array.max() <= 2.0
    array = InputSpec((3,), integer=True).sample(2, rng)
    assert array.shape == (2, 3) and array.dtype == np.int32


def test_specs():
    names = [spec.name for spec in SPECS]
    assert len(names) == len(set(names))
    assert all(spec.category in CATEGORIES for spec in SPECS)
    assert {spec.category for spec in SPECS} == set(CATEGORIES)


@pytest.mark.parametrize("spec", SPECS, ids=lambda spec: spec.name)
def test_specs_match_torch(spec):
    # the warp-nn modules and their PyTorch counterparts take the same inputs and produce outputs with the same
    # shapes (and values, for the modules without parameters/randomness), and both passes run
    arrays = spec.sample_inputs(3, np.random.default_rng(0))
    torch_runner = build_torch_runner(spec, arrays, device=DEVICE, cuda_graph=False)
    warp_runner = build_warp_runner(spec, nn, arrays, device=DEVICE, cuda_graph=False)
    torch_outputs = [_to_numpy(output) for output in flatten(torch_runner.forward())]
    warp_outputs = [_to_numpy(output) for output in flatten(warp_runner.forward())]
    assert [output.shape for output in warp_outputs] == [output.shape for output in torch_outputs]
    torch_module = spec.torch()
    stateless = not isinstance(torch_module, torch.nn.Module) or not list(torch_module.parameters())
    if stateless and spec.name != "Dropout":
        for warp_output, torch_output in zip(warp_outputs, torch_outputs):
            np.testing.assert_allclose(warp_output, torch_output, rtol=1e-2, atol=1e-3)
    assert (warp_runner.backward is None) == (torch_runner.backward is None) == (not spec.differentiable)
    if spec.differentiable:
        torch_runner.backward()
        warp_runner.backward()


@pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA is not available")
@pytest.mark.parametrize("name", ["ReLU", "Add", "LSTM", "Sequential (MLP)"])
def test_cuda_graph_runners(name):
    # replaying the captured graphs runs the passes: the backward pass accumulates the gradients of the inputs
    spec = _SPECS[name]
    arrays = spec.sample_inputs(4, np.random.default_rng(0))
    for runner in (
        build_torch_runner(spec, arrays, device=DEVICE, cuda_graph=True),
        build_warp_runner(spec, nn, arrays, device=DEVICE, cuda_graph=True),
    ):
        runner.forward()
        before = _grads(runner.inputs)
        runner.backward()
        torch.cuda.synchronize()
        wp.synchronize()
        after = _grads(runner.inputs)
        assert any(not np.array_equal(a, b) for a, b in zip(before, after))


def test_non_differentiable_runners():
    spec = _SPECS["BitwiseAnd"]
    arrays = spec.sample_inputs(2, np.random.default_rng(0))
    for runner in (
        build_torch_runner(spec, arrays, device=DEVICE, cuda_graph=False),
        build_warp_runner(spec, nn, arrays, device=DEVICE, cuda_graph=False),
    ):
        assert runner.backward is None
        np.testing.assert_array_equal(_to_numpy(runner.forward()), np.bitwise_and(*arrays))


@pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA is not available")
def test_cuda_graph_runners_keep_captured_objects_alive():
    # the memory referenced by the captured graphs (e.g. the modules' parameters and cached arrays) must not be freed
    # (and reused) while the graphs can be replayed
    spec = _SPECS["Linear"]
    modules = []

    def record(factory):
        def create(*args):
            module = factory(*args)
            modules.append(weakref.ref(module))
            return module

        return create

    spec = dataclasses.replace(spec, warp=record(spec.warp), torch=record(spec.torch))
    arrays = spec.sample_inputs(4, np.random.default_rng(0))
    runners = [
        build_torch_runner(spec, arrays, device=DEVICE, cuda_graph=True),
        build_warp_runner(spec, nn, arrays, device=DEVICE, cuda_graph=True),
    ]
    gc.collect()
    assert len(modules) == 2
    assert all(module() is not None for module in modules)
    del runners
    gc.collect()
    assert all(module() is None for module in modules)
