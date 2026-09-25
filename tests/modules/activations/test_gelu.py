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
from .common import check_forward, check_gradients, check_requires_grad


@pytest.mark.parametrize("approximate", ["none", "tanh"])
@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("dtype", [wp.float16, wp.float32, wp.float64])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_forward(capsys, device, dtype, ndim, approximate):
    if not is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    check_forward(
        warp_activation=nn.GELU(approximate=approximate),
        torch_activation=torch.nn.GELU(approximate=approximate),
        device=device,
        dtype=dtype,
        ndim=ndim,
    )


@pytest.mark.parametrize("approximate", ["none", "tanh"])
@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("dtype", [wp.float32])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_gradients(capsys, device, dtype, ndim, approximate):
    if not is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    check_gradients(
        warp_activation=nn.GELU(approximate=approximate),
        torch_activation=torch.nn.GELU(approximate=approximate),
        device=device,
        dtype=dtype,
        ndim=ndim,
    )


@pytest.mark.parametrize("requires_grad", [True, False])
@pytest.mark.parametrize("ndim", [2])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_requires_grad(capsys, device, ndim, requires_grad):
    if not is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    check_requires_grad(
        warp_activation=nn.GELU(requires_grad=requires_grad), device=device, ndim=ndim, requires_grad=requires_grad
    )


def test_invalid_approximate(capsys):
    assert nn.GELU().approximate == "none"
    assert nn.GELU(approximate="tanh").approximate == "tanh"
    with pytest.raises(ValueError, match="erf"):
        nn.GELU(approximate="erf")


@pytest.mark.parametrize("approximate", ["none", "tanh"])
@pytest.mark.parametrize("dtype", [wp.float16, wp.float32, wp.float64])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_extreme_values(capsys, device, dtype, approximate):
    if not is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    # large inputs (whose cube overflows in half precision) and negative inputs (prone to cancellation)
    array = np.array([-60000, -2000, -30, -10.5, -9.5, -4, -3, 0, 3, 9.5, 10.5, 30, 2000, 60000])
    array = array.astype(wp.dtype_to_numpy(dtype))
    torch_input = torch.tensor(array, dtype=torch.float64, requires_grad=True)
    torch_output = torch.nn.functional.gelu(torch_input, approximate=approximate)
    torch_output.sum().backward()
    warp_input = wp.array(array, dtype=dtype, device=device, requires_grad=True)
    tape = wp.Tape()
    with tape:
        warp_output = nn.GELU(approximate=approximate).to(device)(warp_input)
    tape.backward(grads={warp_output: wp.ones_like(warp_output)})
    # half-precision tolerances include the quantization of subnormal intermediate values
    rtol, atol = (2e-3, 1e-6) if dtype == wp.float16 else (1e-6, 1e-12)
    np.testing.assert_allclose(warp_output.numpy(), torch_output.detach().numpy(), rtol=rtol, atol=atol)
    np.testing.assert_allclose(warp_input.grad.numpy(), torch_input.grad.numpy(), rtol=rtol, atol=atol)
