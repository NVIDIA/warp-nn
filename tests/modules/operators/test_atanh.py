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

import warp_nn.nn as nn

from ...utilities import is_device_available
from .common import check_forward, check_gradients, check_requires_grad, check_unsupported_input, check_values


# map the sampled inputs, in [-1, 1), to the operation's domain
_DOMAINS = (lambda x: 0.9 * x,)


# test-specific parameters
@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("dtype", [wp.float16, wp.float32, wp.float64])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_forward(capsys, device, dtype, ndim):
    if not is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    check_forward(
        module=nn.Atanh(), torch_function=torch.atanh, device=device, dtype=dtype, ndim=ndim, domains=_DOMAINS
    )


# test-specific parameters
@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_gradients(capsys, device, ndim):
    if not is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    check_gradients(module=nn.Atanh(), torch_function=torch.atanh, device=device, ndim=ndim, domains=_DOMAINS)


# module-specific parameters
@pytest.mark.parametrize("requires_grad", [True, False])
# test-specific parameters
@pytest.mark.parametrize("ndim", [2])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_requires_grad(capsys, device, ndim, requires_grad):
    if not is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    check_requires_grad(
        module=nn.Atanh(requires_grad=requires_grad), device=device, ndim=ndim, requires_grad=requires_grad
    )


def test_unsupported_input(capsys):
    check_unsupported_input(module=nn.Atanh())


# test-specific parameters
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_values(capsys, device):
    if not is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    # small inputs (accurate) and inputs close to the domain boundaries
    check_values(module=nn.Atanh(), values=[1e-8, -1e-8, 0.5, -0.999, 0.999], numpy_function=np.arctanh, device=device)


# test-specific parameters
@pytest.mark.parametrize("dtype", [wp.float16, wp.float32, wp.float64])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_signed_zero(capsys, device, dtype):
    if not is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    module = nn.Atanh().to(device)
    torch_input = torch.tensor([0.0, -0.0], dtype=torch.float64, requires_grad=True)
    warp_input = wp.array(torch_input.detach().numpy(), dtype=dtype, device=device, requires_grad=True)
    tape = wp.Tape()
    with tape:
        warp_output = module(warp_input)
    tape.backward(grads={warp_output: wp.ones_like(warp_output)})
    torch_output = torch.atanh(torch_input)
    torch_output.sum().backward()
    # the sign of zero is preserved, and the derivative at both zeros is 1
    np.testing.assert_array_equal(np.signbit(warp_output.numpy()), np.signbit(torch_output.detach().numpy()))
    np.testing.assert_array_equal(warp_input.grad.numpy(), torch_input.grad.numpy())
