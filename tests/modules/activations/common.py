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

import torch

import warp as wp

from ... import utilities


class TorchLambda(torch.nn.Module):
    """PyTorch module wrapping a function, to be used as reference for functions without a PyTorch module."""

    def __init__(self, function):
        super().__init__()
        self.function = function

    def forward(self, input):
        return self.function(input)


@wp.kernel
def _loss_1d(a: wp.array1d(dtype=float), loss: wp.array1d(dtype=float)):
    i = wp.tid()
    wp.atomic_add(loss, 0, a[i])


@wp.kernel
def _loss_2d(a: wp.array2d(dtype=float), loss: wp.array1d(dtype=float)):
    i, j = wp.tid()
    wp.atomic_add(loss, 0, a[i, j])


@wp.kernel
def _loss_3d(a: wp.array3d(dtype=float), loss: wp.array1d(dtype=float)):
    i, j, k = wp.tid()
    wp.atomic_add(loss, 0, a[i, j, k])


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
    # compute loss
    # - torch
    torch_output = torch_activation(torch_input)
    torch_loss = torch_output.sum()
    torch_loss.backward()
    # - warp
    tape = wp.Tape()
    loss = wp.zeros((1,), dtype=wp.float32, requires_grad=True, device=device)
    with tape:
        warp_output = warp_activation(warp_input)
        wp.launch(
            {1: _loss_1d, 2: _loss_2d, 3: _loss_3d}[ndim],
            dim=warp_output.shape,
            inputs=[warp_output],
            outputs=[loss],
            device=device,
        )
    tape.backward(loss)
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
