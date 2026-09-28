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
from .common import check_forward, check_gradients, check_requires_grad, check_unsupported_input


# test-specific parameters
@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("dtype", [wp.float16, wp.float32, wp.float64])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_forward(capsys, device, dtype, ndim):
    if not is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    check_forward(
        warp_activation=nn.HardSwish(), torch_activation=torch.nn.Hardswish(), device=device, dtype=dtype, ndim=ndim
    )


# test-specific parameters
@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("dtype", [wp.float32])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_gradients(capsys, device, dtype, ndim):
    if not is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    check_gradients(
        warp_activation=nn.HardSwish(), torch_activation=torch.nn.Hardswish(), device=device, dtype=dtype, ndim=ndim
    )


# module-specific parameters
@pytest.mark.parametrize("requires_grad", [True, False])
# test-specific parameters
@pytest.mark.parametrize("ndim", [2])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_requires_grad(capsys, device, ndim, requires_grad):
    if not is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    check_requires_grad(
        warp_activation=nn.HardSwish(requires_grad=requires_grad), device=device, ndim=ndim, requires_grad=requires_grad
    )


# test-specific parameters
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_boundaries(capsys, device):
    if not is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    # the values and gradients at (and around) the boundaries of the saturated regions match PyTorch
    array = np.array([-np.inf, -3.5, -3.0, -2.5, 2.5, 3.0, 3.5], dtype=np.float32)
    torch_input = torch.tensor(array, requires_grad=True)
    torch_output = torch.nn.functional.hardswish(torch_input)
    torch_output.sum().backward()
    warp_input = wp.array(array, device=device, requires_grad=True)
    tape = wp.Tape()
    with tape:
        warp_output = nn.HardSwish().to(device)(warp_input)
    tape.backward(grads={warp_output: wp.ones_like(warp_output)})
    # PyTorch yields NaN for -inf (-inf * 0), while the saturated region yields 0
    np.testing.assert_allclose(warp_output.numpy()[1:], torch_output.detach().numpy()[1:], rtol=1e-6)
    assert warp_output.numpy()[0] == 0.0
    np.testing.assert_allclose(warp_input.grad.numpy(), torch_input.grad.numpy(), rtol=1e-6)


def test_unsupported_input(capsys):
    check_unsupported_input(warp_activation=nn.HardSwish())
