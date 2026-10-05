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

import torch

import numpy as np
import warp as wp

from ... import utilities


class TorchLambda(torch.nn.Module):
    """PyTorch module wrapping a function, to be used as reference for functions without a PyTorch module."""

    def __init__(self, function):
        super().__init__()
        self.function = function

    def forward(self, input):
        return self.function(input)


def check_forward(*, warp_activation, torch_activation, device, dtype, ndim, atol: float = 1e-03):
    # move activations to target device
    warp_activation.to(device)
    torch_activation.to(device)
    # create inputs
    array = utilities.sample_array(shape=[10] * ndim, dtype=dtype)
    torch_input = torch.tensor(array, device=device)
    warp_input = wp.array(array, device=device)
    # forward pass
    warp_output = warp_activation(warp_input)
    torch_output = torch_activation(torch_input)
    # check outputs
    utilities.check_arrays(torch_output, warp_output, atol=atol)


def check_gradients(*, warp_activation, torch_activation, device, dtype, ndim):
    # move activations to target device
    warp_activation.to(device)
    torch_activation.to(device)
    # create inputs
    array = utilities.sample_array(shape=[10] * ndim, dtype=dtype)
    torch_input = torch.tensor(array, device=device, requires_grad=True)
    warp_input = wp.array(array, device=device, requires_grad=True)
    # forward pass
    torch_output = torch_activation(torch_input)
    tape = wp.Tape()
    with tape:
        warp_output = warp_activation(warp_input)
    # backward pass (with the same random upstream gradients)
    utilities.backward(tape, [torch_output], [warp_output])
    # check gradients
    utilities.check_arrays(torch_input.grad, warp_input.grad)


def check_requires_grad(*, warp_activation, device, ndim, requires_grad: bool):
    # move activation to target device
    warp_activation.to(device)
    # create inputs
    array = utilities.sample_array(shape=[10] * ndim)
    warp_input = wp.array(array, device=device, requires_grad=True)
    # forward pass
    warp_output = warp_activation(warp_input)
    # check the flag of the activation and of its cached output array
    assert warp_activation.requires_grad == requires_grad
    assert warp_output.requires_grad == requires_grad
    # check that the gradient array of the cached output array is allocated accordingly
    assert (warp_output.grad is not None) == requires_grad


def check_extreme_inputs(*, warp_activation, torch_activation, device, atol: float = 1e-6):
    # large-magnitude inputs (that overflow naive exp() implementations) and NaN (that must be propagated)
    array = np.array([-100.0, -20.0, 0.0, np.nan, 20.0, 100.0], dtype=np.float32)
    # move activations to target device
    warp_activation.to(device)
    torch_activation.to(device)
    # forward and backward passes (with distinct upstream gradients per element)
    weights = np.arange(1, array.size + 1, dtype=np.float32)
    # - torch
    torch_input = torch.tensor(array, device=device, requires_grad=True)
    torch_output = torch_activation(torch_input)
    (torch_output * torch.tensor(weights, device=device)).sum().backward()
    # - warp
    warp_input = wp.array(array, device=device, requires_grad=True)
    tape = wp.Tape()
    with tape:
        warp_output = warp_activation(warp_input)
    tape.backward(grads={warp_output: wp.array(weights, device=device)})
    # check outputs and gradients
    torch_output, torch_grad = torch_output.detach().cpu().numpy(), torch_input.grad.cpu().numpy()
    np.testing.assert_allclose(warp_output.numpy(), torch_output, atol=atol, equal_nan=True)
    np.testing.assert_allclose(warp_input.grad.numpy(), torch_grad, atol=atol, equal_nan=True)


def check_unsupported_input(*, warp_activation):
    warp_activation.to("cpu")
    # data type not supported by the activation, and unsupported number of dimensions
    for shape, dtype in [((2, 2), wp.int32), ((2,) * 4, wp.float32)]:
        with pytest.raises(TypeError, match="Unsupported input"):
            warp_activation(wp.zeros(shape, dtype=dtype, device="cpu"))
