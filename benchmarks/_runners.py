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

"""Forward and backward passes of the benchmarked implementations (warp-nn and PyTorch).

- Forward pass: inference, i.e. without recording the operations for differentiation
  (no ``wp.Tape`` for warp-nn, ``torch.no_grad()`` for PyTorch), as in the rollouts of reinforcement learning.
- Backward pass: backpropagation of the gradients of all the outputs (seeded with ones) to the inputs and parameters,
  through the operations recorded by a single forward pass (``wp.Tape.backward(grads=...)`` for warp-nn,
  ``torch.autograd.backward(..., retain_graph=True)`` for PyTorch). Gradients are accumulated (not zeroed).
  The warp-nn backward pass includes copying the incoming gradients into the outputs' gradients, as required by
  ``wp.Tape.backward`` (whose adjoint kernels zero them), while PyTorch uses the incoming gradients directly.

Both passes can be run eagerly (including the host overhead) or by replaying a captured CUDA graph.
"""

from __future__ import annotations

from typing import Any, Callable

import dataclasses
import functools
from types import ModuleType
import torch

import numpy as np
import warp as wp

from ._specs import ModuleSpec


_TORCH_WARMUP_ITERATIONS = 3  # warmup iterations, on the capture stream, before capturing a PyTorch CUDA graph


@dataclasses.dataclass
class Runner:
    """Forward and backward passes of an implementation of a module, for a given input."""

    forward: Callable[[], Any]
    """Forward pass."""

    backward: Callable[[], Any] | None
    """Backward pass, or None if the module is not differentiable."""

    inputs: list[Any]
    """Input arrays/tensors."""


def flatten(data: Any) -> list[Any]:
    """Flatten (possibly nested) tuples or lists of arrays/tensors, such as the outputs of a LSTM module.

    :param data: The (possibly nested) data.

    :return: The flattened data.
    """
    if isinstance(data, (tuple, list)):
        return [item for element in data for item in flatten(element)]
    return [data]


class _Replay:
    """Replay of a captured CUDA graph.

    It keeps the captured callable alive, and with it all the objects (e.g. modules, parameters, input/output arrays,
    tapes) whose memory is referenced by the graph, which would otherwise be freed (and reused) after the capture.
    """

    def __init__(self, replay: Callable[[], Any], captured: Callable[[], Any]):
        self._replay = replay
        self._captured = captured

    def __call__(self) -> None:
        self._replay()


def _capture_torch(make: Callable[[], Callable[[], Any]], device: str) -> _Replay:
    # the callable is created (e.g. by running a forward pass for the backward pass) and warmed up on the capture
    # stream, since the backward operations run on the stream of their corresponding forward operations
    stream = torch.cuda.Stream(device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        fn = make()
        for _ in range(_TORCH_WARMUP_ITERATIONS):
            fn()
    torch.cuda.current_stream(device).wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        fn()
    graph.replay()
    return _Replay(graph.replay, fn)


def _capture_warp(fn: Callable[[], Any], device: str) -> _Replay:
    # the passes are already warmed up: do not load all the registered modules (e.g. of other benchmarks)
    with wp.ScopedCapture(device=device, force_module_load=False) as capture:
        fn()
    graph = capture.graph
    # launch the graph right after capturing it, since the capture only records the work
    wp.capture_launch(graph)
    return _Replay(functools.partial(wp.capture_launch, graph), fn)


def build_torch_runner(spec: ModuleSpec, arrays: list[np.ndarray], *, device: str, cuda_graph: bool) -> Runner:
    """Build the forward and backward passes of the PyTorch counterpart of a module.

    :param spec: The module specification.
    :param arrays: The input arrays.
    :param device: The device.
    :param cuda_graph: Whether to replay captured CUDA graphs (rather than running eagerly).

    :return: The runner.
    """
    module = spec.torch()
    if isinstance(module, torch.nn.Module):
        module = module.to(device)
    inputs = [torch.tensor(array, device=device, requires_grad=spec.differentiable) for array in arrays]

    def make_forward():
        def forward():
            with torch.no_grad():
                return spec.call(module, *inputs)

        return forward

    def make_backward():
        outputs = [output for output in flatten(spec.call(module, *inputs)) if output.requires_grad]
        grads = [torch.ones_like(output) for output in outputs]
        return lambda: torch.autograd.backward(outputs, grads, retain_graph=True)

    if cuda_graph:
        forward = _capture_torch(make_forward, device)
        backward = _capture_torch(make_backward, device) if spec.differentiable else None
    else:
        forward = make_forward()
        forward()
        backward = make_backward() if spec.differentiable else None
        if backward is not None:
            backward()
    return Runner(forward=forward, backward=backward, inputs=inputs)


def build_warp_runner(
    spec: ModuleSpec, nn: ModuleType, arrays: list[np.ndarray], *, device: str, cuda_graph: bool
) -> Runner:
    """Build the forward and backward passes of a warp-nn module.

    :param spec: The module specification.
    :param nn: The ``nn`` namespace of the warp-nn implementation (current or baseline).
    :param arrays: The input arrays.
    :param device: The device.
    :param cuda_graph: Whether to replay captured CUDA graphs (rather than running eagerly).

    :return: The runner.
    """
    module = spec.warp(nn).to(device)
    inputs = [wp.array(array, device=device, requires_grad=spec.differentiable) for array in arrays]

    def forward():
        return spec.call(module, *inputs)

    # warmup execution (always): the first calls compile/load the kernels and allocate the cached arrays,
    # which must not be timed nor happen during a CUDA graph capture
    forward()
    backward = None
    if spec.differentiable:
        tape = wp.Tape()
        with tape:
            outputs = [output for output in flatten(spec.call(module, *inputs)) if output.requires_grad]
        # the incoming gradients are copied into the outputs' gradients on every call,
        # since the backward pass may consume (zero) them
        # (the outputs must own gradient arrays, otherwise the incoming gradients would be adopted and then zeroed)
        assert all(output.grad is not None for output in outputs), "The outputs must have gradient arrays"
        grads = {output: wp.ones_like(output, requires_grad=False) for output in outputs}

        def backward():
            tape.backward(grads=grads)

        backward()
    if cuda_graph:
        forward = _capture_warp(forward, device)
        if backward is not None:
            backward = _capture_warp(backward, device)
    return Runner(forward=forward, backward=backward, inputs=inputs)
