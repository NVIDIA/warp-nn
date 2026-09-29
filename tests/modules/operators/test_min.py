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
from .common import (
    check_forward,
    check_gradients,
    check_mismatched_inputs,
    check_requires_grad,
    check_unsupported_input,
)


@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("dtype", [wp.float16, wp.float32, wp.float64])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_forward(capsys, device, dtype, ndim):
    if not is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    check_forward(module=nn.Min(), torch_function=torch.minimum, device=device, dtype=dtype, ndim=ndim, num_inputs=2)


@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_gradients(capsys, device, ndim):
    if not is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    check_gradients(module=nn.Min(), torch_function=torch.minimum, device=device, ndim=ndim, num_inputs=2)


@pytest.mark.parametrize("requires_grad", [True, False])
@pytest.mark.parametrize("ndim", [2])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_requires_grad(capsys, device, ndim, requires_grad):
    if not is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    check_requires_grad(
        module=nn.Min(requires_grad=requires_grad), device=device, ndim=ndim, requires_grad=requires_grad, num_inputs=2
    )


@pytest.mark.parametrize("dtype", [wp.float16, wp.float32, wp.float64])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_special_values(capsys, device, dtype):
    if not is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    # NaN (propagated, with a unit gradient for both inputs) and ties (with the gradient split evenly)
    np_dtype = wp.dtype_to_numpy(dtype)
    a = np.array([np.nan, 1.0, np.nan, 1.0, -0.0, 0.0, np.inf, -np.inf, -5.0], dtype=np_dtype)
    b = np.array([1.0, np.nan, np.nan, 1.0, 0.0, -0.0, np.inf, -np.inf, 7.0], dtype=np_dtype)
    warp_a = wp.array(a, device=device, requires_grad=True)
    warp_b = wp.array(b, device=device, requires_grad=True)
    tape = wp.Tape()
    with tape:
        warp_output = nn.Min().to(device)(warp_a, warp_b)
    tape.backward(grads={warp_output: wp.ones_like(warp_output)})
    torch_a = torch.tensor(a, dtype=torch.float64, requires_grad=True)
    torch_b = torch.tensor(b, dtype=torch.float64, requires_grad=True)
    torch.minimum(torch_a, torch_b).sum().backward()
    np.testing.assert_array_equal(warp_output.numpy(), np.minimum(a, b))
    np.testing.assert_array_equal(warp_a.grad.numpy(), torch_a.grad.numpy())
    np.testing.assert_array_equal(warp_b.grad.numpy(), torch_b.grad.numpy())


def test_unsupported_input(capsys):
    check_unsupported_input(module=nn.Min(), num_inputs=2)


def test_mismatched_inputs(capsys):
    check_mismatched_inputs(module=nn.Min())
