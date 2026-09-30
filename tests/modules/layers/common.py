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

from typing import Any

import torch

import numpy as np
import warp as wp

from ... import utilities


@wp.kernel
def _loss_1d(a: wp.array1d[Any], loss: wp.array1d[wp.float32]):
    i = wp.tid()
    wp.atomic_add(loss, 0, loss.dtype(a[i]))


@wp.kernel
def _loss_2d(a: wp.array2d[Any], loss: wp.array1d[wp.float32]):
    i, j = wp.tid()
    wp.atomic_add(loss, 0, loss.dtype(a[i, j]))


@wp.kernel
def _loss_3d(a: wp.array3d[Any], loss: wp.array1d[wp.float32]):
    i, j, k = wp.tid()
    wp.atomic_add(loss, 0, loss.dtype(a[i, j, k]))


@wp.kernel
def _loss_4d(a: wp.array4d[Any], loss: wp.array1d[wp.float32]):
    i, j, k, l = wp.tid()
    wp.atomic_add(loss, 0, loss.dtype(a[i, j, k, l]))


def _flatten(data) -> list:
    # flatten (possibly nested) tuples of arrays, such as the outputs of the LSTM module: (output, (hidden, cell))
    return [item for element in data for item in _flatten(element)] if isinstance(data, tuple) else [data]


def _run_rnn(*, warp_module, torch_module, device, shape, hidden_shapes: list | None):
    # run the forward pass of a recurrent module (cell or multi-layer) and its PyTorch counterpart, where
    # hidden_shapes is the shape of each initial state (hidden state and, for LSTM, cell state), or None to use
    # the default (zeros) initial states. Return the inputs, initial states and (flattened) outputs of both,
    # as well as the tape recording the Warp forward pass
    # move modules to target device
    warp_module.to(device)
    torch_module.to(device)
    # init parameters to same values
    utilities.init_parameters(torch_module.parameters())
    utilities.init_parameters(warp_module.parameters(as_array=True))
    # create inputs
    array = utilities.sample_array(shape)
    torch_input = torch.tensor(array, device=device, requires_grad=True)
    warp_input = wp.array(array, device=device, requires_grad=True)
    torch_hidden, warp_hidden = None, None
    if hidden_shapes is not None:
        arrays = [utilities.sample_array(hidden_shape) for hidden_shape in hidden_shapes]
        torch_hidden = tuple(torch.tensor(array, device=device, requires_grad=True) for array in arrays)
        warp_hidden = tuple(wp.array(array, device=device, requires_grad=True) for array in arrays)
        if len(arrays) == 1:
            torch_hidden, warp_hidden = torch_hidden[0], warp_hidden[0]
    # forward pass
    torch_outputs = _flatten(torch_module(torch_input, torch_hidden))
    tape = wp.Tape()
    with tape:
        warp_outputs = _flatten(warp_module(warp_input, warp_hidden))
    return (torch_input, torch_hidden, torch_outputs), (warp_input, warp_hidden, warp_outputs), tape


def check_forward(*, warp_module, torch_module, device, dtype, shape, rtol: float = 1e-02, atol: float = 1e-03):
    # move modules to target device
    warp_module.to(device)
    torch_module.to(device)
    # init parameters to same values
    utilities.init_parameters(torch_module.parameters())
    utilities.init_parameters(warp_module.parameters(as_array=True))
    # create inputs
    array = utilities.sample_array(shape, dtype=dtype)
    torch_input = torch.tensor(array, device=device)
    warp_input = wp.array(array, device=device)
    # forward pass
    warp_output = warp_module(warp_input)
    torch_output = torch_module(torch_input)
    # check outputs
    utilities.check_arrays(torch_output, warp_output, rtol=rtol, atol=atol)


def check_gradients(
    *,
    warp_module,
    torch_module,
    device,
    dtype,
    shape,
    rtol: float = 1e-02,
    atol: float = 1e-03,
    weighted: bool = False,
):
    # weighted: whether the loss is a weighted sum of the outputs (with random weights), rather than their sum,
    # since the gradient of the sum of the outputs vanishes for some modules (e.g. normalization layers)
    # move modules to target device
    warp_module.to(device)
    torch_module.to(device)
    # init parameters to same values
    utilities.init_parameters(torch_module.parameters())
    utilities.init_parameters(warp_module.parameters(as_array=True))
    # create inputs
    array = utilities.sample_array(shape, dtype=dtype)
    torch_input = torch.tensor(array, device=device, requires_grad=True)
    warp_input = wp.array(array, device=device, requires_grad=True)
    # compute loss
    # - torch
    torch_output = torch_module(torch_input)
    if weighted:
        weights = utilities.sample_array(tuple(torch_output.shape))
        torch_loss = (torch_output * torch.tensor(weights, device=device)).sum()
    else:
        torch_loss = torch_output.sum()
    torch_loss.backward()
    # - warp
    tape = wp.Tape()
    loss = wp.zeros((1,), dtype=wp.float32, requires_grad=True, device=device)
    with tape:
        warp_output = warp_module(warp_input)
        if not weighted:
            wp.launch(
                {1: _loss_1d, 2: _loss_2d, 3: _loss_3d, 4: _loss_4d}[len(warp_output.shape)],
                dim=warp_output.shape,
                inputs=[warp_output],
                outputs=[loss],
                device=device,
            )
    if weighted:
        tape.backward(grads={warp_output: wp.array(weights, device=device)})
    else:
        tape.backward(loss)
    # check gradients
    utilities.check_arrays(torch_input.grad, warp_input.grad, rtol=rtol, atol=atol)
    utilities.check_arrays(
        [parameter.grad for parameter in torch_module.parameters()],
        [parameter.grad for parameter in warp_module.parameters(as_array=True)],
        flatten=True,
        rtol=rtol,
        atol=atol,
    )


def check_forward_rnn(
    *,
    warp_module,
    torch_module,
    device,
    shape,
    hidden_shapes: list | None,
    rtol: float = 1e-02,
    atol: float = 1e-03,
):
    # check the outputs of a recurrent module (cell or multi-layer) against PyTorch (see _run_rnn for the arguments)
    (_, _, torch_outputs), (_, _, warp_outputs), _ = _run_rnn(
        warp_module=warp_module, torch_module=torch_module, device=device, shape=shape, hidden_shapes=hidden_shapes
    )
    utilities.check_arrays(torch_outputs, warp_outputs, rtol=rtol, atol=atol)


def check_gradients_rnn(
    *,
    warp_module,
    torch_module,
    device,
    shape,
    hidden_shapes: list | None,
    rtol: float = 1e-02,
    atol: float = 1e-02,
):
    # check the gradients (of the input, initial states and parameters) of a recurrent module (cell or multi-layer)
    # against PyTorch (see _run_rnn for the arguments), where the loss is a weighted sum of all the outputs
    (torch_input, torch_hidden, torch_outputs), (warp_input, warp_hidden, warp_outputs), tape = _run_rnn(
        warp_module=warp_module, torch_module=torch_module, device=device, shape=shape, hidden_shapes=hidden_shapes
    )
    # backward pass
    weights = [utilities.sample_array(tuple(output.shape)) for output in torch_outputs]
    sum((output * torch.tensor(w, device=device)).sum() for output, w in zip(torch_outputs, weights)).backward()
    tape.backward(grads={output: wp.array(w, device=device) for output, w in zip(warp_outputs, weights)})
    # check gradients (whose magnitude grows with the sizes and number of layers, so the absolute tolerance is
    # scaled by the magnitude of the reference gradients). Backpropagating through saturated gates over several
    # time steps and layers is ill-conditioned in single precision (even PyTorch deviates from a double precision
    # reference by ~1e-2 in some cases), hence the looser tolerance
    torch_grads = [torch_input.grad]
    warp_grads = [warp_input.grad]
    if hidden_shapes is not None:
        torch_grads += [array.grad for array in _flatten(torch_hidden)]
        warp_grads += [array.grad for array in _flatten(warp_hidden)]
    torch_grads += [parameter.grad for parameter in torch_module.parameters()]
    warp_grads += [parameter.grad for parameter in warp_module.parameters(as_array=True)]
    for torch_grad, warp_grad in zip(torch_grads, warp_grads, strict=True):
        scale = max(1.0, torch_grad.abs().max().item())
        utilities.check_arrays(torch_grad, warp_grad, flatten=True, rtol=rtol, atol=atol * scale)


def check_requires_grad(*, warp_module, device, inputs, requires_grad: bool):
    # move module to target device
    warp_module.to(device)
    # forward pass
    warp_outputs = warp_module(*inputs)
    # check the flag of the module, of its parameters and of its cached output arrays,
    # as well as the allocation of the corresponding gradient arrays
    assert warp_module.requires_grad == requires_grad
    for parameter in warp_module.parameters(as_array=False):
        assert parameter.requires_grad == requires_grad
        assert (parameter.data.grad is not None) == requires_grad
    for warp_output in _flatten(warp_outputs):
        assert warp_output.requires_grad == requires_grad
        assert (warp_output.grad is not None) == requires_grad


def check_initialize_parameters(*, module_type, module_kwargs: dict, device, inputs: list | None = None):
    def _create(initialize_parameters: bool):
        module = module_type(**module_kwargs, initialize_parameters=initialize_parameters).to(device)
        if inputs is not None:  # materialize the parameters of the lazily initialized modules
            module(*inputs)
        return module

    def _check_same_parameters(a, b):
        utilities.check_arrays(a.parameters(as_array=True), b.parameters(as_array=True), flatten=True, test="equal")

    # the parameters are drawn from the global NumPy random number generator, so seeding it makes them reproducible
    state = np.random.get_state()
    try:
        np.random.seed(0)
        reference = _create(initialize_parameters=True)
        np.random.seed(0)
        _check_same_parameters(reference, _create(initialize_parameters=True))
        # skipping the initialization draws nothing, so the next initialized module still gets the same parameters
        np.random.seed(0)
        _create(initialize_parameters=False)
        _check_same_parameters(reference, _create(initialize_parameters=True))
    finally:
        np.random.set_state(state)
