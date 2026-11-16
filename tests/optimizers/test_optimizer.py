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

import numpy as np
import warp as wp

from warp_nn.optimizers import SGD, Adam

from .. import utilities


def _create_parameters(gradients: list[np.ndarray], device: str) -> list[wp.array]:
    parameters = [
        wp.zeros(gradient.shape, dtype=wp.float32, device=device, requires_grad=True) for gradient in gradients
    ]
    _assign_gradients(parameters, gradients)
    return parameters


def _assign_gradients(parameters: list[wp.array], gradients: list[np.ndarray]) -> None:
    for parameter, gradient in zip(parameters, gradients):
        parameter.grad.assign(gradient)


def _clip(gradients: list[np.ndarray], max_norm: float) -> list[np.ndarray]:
    total_norm = np.sqrt(sum(np.sum(gradient**2) for gradient in gradients))
    return [gradient * min(1.0, max_norm / total_norm) for gradient in gradients]


# test-specific parameters
@pytest.mark.parametrize("disable_graph", [True, False])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_clip_by_total_norm(capsys, device, disable_graph):
    if not utilities.is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")

    # gradients whose sizes are not a multiple of the 1D tile size, with a total norm greater than 1
    gradients = [utilities.sample_array(shape=shape) for shape in [(10, 20), (7,)]]
    assert np.sqrt(sum(np.sum(gradient**2) for gradient in gradients)) > 1.0

    # each `max_norm` specializes the kernels: optimizers (created one after the other in the same process)
    # must not share them, and every call (including the first one, when a CUDA graph is captured) must clip
    for max_norm in [0.5, 1.0, 1000.0]:
        parameters = _create_parameters(gradients, device)
        optimizer = SGD(parameters, device=device)
        for i in range(2):
            msg = f"max_norm: {max_norm}, call {i + 1}/2"
            _assign_gradients(parameters, gradients)
            optimizer.clip_by_total_norm(max_norm, disable_graph=disable_graph)
            expected = [wp.array(gradient) for gradient in _clip(gradients, max_norm)]
            utilities.check_arrays(expected, optimizer.gradients, flatten=True, atol=1e-5, msg=msg)


# test-specific parameters
@pytest.mark.parametrize("disable_graph", [True, False])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_clip_by_total_norm_and_step(capsys, device, disable_graph):
    if not utilities.is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")

    gradients = [utilities.sample_array(shape=shape) for shape in [(10, 20), (7,)]]
    parameters = _create_parameters(gradients, device)
    # SGD (lr: 1, no momentum): parameters -= clipped gradients
    optimizer = SGD(parameters, lr=1.0, max_norm=0.5, device=device, disable_graph=disable_graph)
    expected_parameters = [np.zeros_like(gradient) for gradient in gradients]
    # interleave explicit clipping calls (which may capture their own graph) with optimization steps
    # (whose captured graph must follow `max_norm` changes)
    for i, (action, max_norm) in enumerate([("clip", 0.5), ("step", 0.5), ("clip", 1.0), ("step", 1.0), ("step", 1.0)]):
        msg = f"{action} (max_norm: {max_norm}), action {i + 1}"
        _assign_gradients(parameters, gradients)
        if action == "clip":
            optimizer.clip_by_total_norm(max_norm, disable_graph=disable_graph)
        else:
            optimizer.step()
            expected_parameters = [p - g for p, g in zip(expected_parameters, _clip(gradients, max_norm))]
        expected_gradients = [wp.array(gradient) for gradient in _clip(gradients, max_norm)]
        utilities.check_arrays(expected_gradients, optimizer.gradients, flatten=True, atol=1e-5, msg=msg)
        utilities.check_arrays([wp.array(p) for p in expected_parameters], parameters, flatten=True, atol=1e-5, msg=msg)


def test_clip_by_total_norm_external_capture(capsys):
    device = "cuda"
    if not utilities.is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")

    gradients = [utilities.sample_array(shape=shape) for shape in [(10, 20), (7,)]]
    parameters = _create_parameters(gradients, device)
    optimizer = SGD(parameters, device=device)
    # capture (and replay) the clipping graph
    optimizer.clip_by_total_norm(0.5)
    # clipping without graph capture must launch the kernels (and not replay the previous graph),
    # so that it can be captured as part of an external graph
    with wp.ScopedCapture(device=device) as capture:
        optimizer.clip_by_total_norm(0.5, disable_graph=True)
    for i in range(2):
        _assign_gradients(parameters, gradients)
        wp.capture_launch(capture.graph)
        expected = [wp.array(gradient) for gradient in _clip(gradients, 0.5)]
        utilities.check_arrays(expected, optimizer.gradients, flatten=True, atol=1e-5, msg=f"replay {i + 1}/2")


@pytest.mark.parametrize("max_norm", [None, 1.0])
@pytest.mark.parametrize("disable_graph", [True, False])
@pytest.mark.parametrize("optimizer_class", [Adam, SGD])
def test_step_single_block_dim(capsys, optimizer_class, disable_graph, max_norm):
    device = "cuda"
    if not utilities.is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")

    # all the kernels of the optimization step (and of the gradient clipping) must be launched with the same block
    # dimension: Warp loads (compiles) the whole module once per block dimension, and compiling the tiled kernels for
    # other block dimensions (e.g. 1) is extremely slow on CUDA.
    # Note: it relies on Warp internals (`Module.execs`, keyed by (context, block_dim), and `Module.unload`)
    parameters = _create_parameters([utilities.sample_array(shape=(10, 20))], device)
    optimizer = optimizer_class(parameters, max_norm=max_norm, device=device, disable_graph=disable_graph)
    kernels = [optimizer._kernel_increase_timestep, optimizer._kernel_step]
    if max_norm is not None:
        kernels += [optimizer._kernel_sum_squares, optimizer._kernel_clip_by_total_norm]
    modules = list({id(kernel.module): kernel.module for kernel in kernels}.values())
    if not all(isinstance(getattr(module, "execs", None), dict) and hasattr(module, "unload") for module in modules):
        pytest.skip("Warp's `Module.execs` or `Module.unload` is not available")
    # unload the modules, so that only what this step loads is checked (the modules are shared by the whole session)
    for module in modules:
        module.unload()
    optimizer.step()
    context = wp.get_device(device).context
    for module in modules:
        try:
            block_dims = {block_dim for ctx, block_dim in module.execs if ctx == context}
        except (TypeError, ValueError):
            pytest.skip("Unexpected layout of Warp's `Module.execs` keys")
        assert block_dims == {optimizer._config.block_dim}, f"module: {module.name}"
