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


def check_forward_rnn_cell(
    *,
    warp_module,
    torch_module,
    device,
    dtype,
    shape,
    hidden_shape,
    cell_shape,
    rtol: float = 1e-02,
    atol: float = 1e-03,
):
    # move modules to target device
    warp_module.to(device)
    torch_module.to(device)
    # init parameters to same values
    utilities.init_parameters(torch_module.parameters())
    utilities.init_parameters(warp_module.parameters(as_array=True))
    # create inputs
    # - input
    array = utilities.sample_array(shape, dtype=dtype)
    torch_input = torch.tensor(array, device=device)
    warp_input = wp.array(array, device=device)
    # - hidden state
    array = utilities.sample_array(hidden_shape, dtype=dtype)
    torch_hidden = torch.tensor(array, device=device)
    warp_hidden = wp.array(array, device=device)
    # - cell state
    if cell_shape is not None:
        array = utilities.sample_array(cell_shape, dtype=dtype)
        torch_cell = torch.tensor(array, device=device)
        warp_cell = wp.array(array, device=device)
    # forward pass
    torch_outputs = []
    warp_outputs = []
    for i in range(shape[0]):  # sequence length
        if cell_shape is None:
            torch_hidden = torch_module(torch_input[i], torch_hidden)
            torch_outputs.append(torch_hidden)
            warp_hidden = wp.clone(warp_module(warp_input[i], warp_hidden))
            warp_outputs.append(warp_hidden)
        else:
            torch_hidden, torch_cell = torch_module(torch_input[i], (torch_hidden, torch_cell))
            torch_outputs.append((torch_hidden, torch_cell))
            warp_output = warp_module(warp_input[i], (warp_hidden, warp_cell))
            warp_hidden, warp_cell = wp.clone(warp_output[0]), wp.clone(warp_output[1])
            warp_outputs.append((warp_hidden, warp_cell))
    # check outputs
    for warp_output, torch_output in zip(warp_outputs, torch_outputs):
        if cell_shape is None:
            utilities.check_arrays(torch_output, warp_output, rtol=rtol, atol=atol)
        else:
            utilities.check_arrays(torch_output[0], warp_output[0], rtol=rtol, atol=atol)
            utilities.check_arrays(torch_output[1], warp_output[1], rtol=rtol, atol=atol)


def check_gradients(*, warp_module, torch_module, device, dtype, shape, rtol: float = 1e-02, atol: float = 1e-03):
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
    torch_loss = torch_output.sum()
    torch_loss.backward()
    # - warp
    tape = wp.Tape()
    loss = wp.zeros((1,), dtype=wp.float32, requires_grad=True, device=device)
    with tape:
        warp_output = warp_module(warp_input)
        wp.launch(
            {1: _loss_1d, 2: _loss_2d, 3: _loss_3d, 4: _loss_4d}[len(warp_output.shape)],
            dim=warp_output.shape,
            inputs=[warp_output],
            outputs=[loss],
            device=device,
        )
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
    for warp_output in warp_outputs if isinstance(warp_outputs, tuple) else (warp_outputs,):
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
