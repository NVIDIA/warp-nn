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

from ... import utilities
from ...utilities import is_device_available
from .common import check_forward, check_gradients, check_requires_grad


_MODULES = {"Softmax": (nn.Softmax, torch.nn.Softmax), "LogSoftmax": (nn.LogSoftmax, torch.nn.LogSoftmax)}

# (ndim, dim) for every valid (positive and negative) dimension
_DIMS = [(ndim, dim) for ndim in [1, 2, 3] for dim in range(-ndim, ndim)]


@pytest.mark.parametrize("name", list(_MODULES))
@pytest.mark.parametrize("ndim, dim", _DIMS)
@pytest.mark.parametrize("dtype", [wp.float16, wp.float32, wp.float64])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_forward(capsys, device, dtype, ndim, dim, name):
    if not is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    warp_module, torch_module = _MODULES[name]
    check_forward(
        warp_activation=warp_module(dim=dim),
        torch_activation=torch_module(dim=dim),
        device=device,
        dtype=dtype,
        ndim=ndim,
        atol=4e-3 if dtype == wp.float16 else 1e-3,  # Log-Softmax outputs are in [-4, -2]: 1 ULP is 2e-3
    )


@pytest.mark.parametrize("name", list(_MODULES))
@pytest.mark.parametrize("ndim, dim", _DIMS)
@pytest.mark.parametrize("dtype", [wp.float32])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_gradients(capsys, device, dtype, ndim, dim, name):
    if not is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    warp_module, torch_module = _MODULES[name]
    # the gradient of the sum of the outputs is zero for Softmax, so check the gradient of a weighted sum too
    array = utilities.sample_array(shape=[10] * ndim)
    weights = np.linspace(-1.0, 1.0, 10**ndim, dtype=np.float32).reshape([10] * ndim)
    # - torch
    torch_input = torch.tensor(array, requires_grad=True)
    (torch_module(dim=dim)(torch_input) * torch.tensor(weights)).sum().backward()
    # - warp
    warp_input = wp.array(array, device=device, requires_grad=True)
    tape = wp.Tape()
    with tape:
        warp_output = warp_module(dim=dim).to(device)(warp_input)
    tape.backward(grads={warp_output: wp.array(weights, device=device)})
    utilities.check_arrays(torch_input.grad, warp_input.grad)
    # the gradient of the sum of the outputs
    check_gradients(
        warp_activation=warp_module(dim=dim),
        torch_activation=torch_module(dim=dim),
        device=device,
        dtype=dtype,
        ndim=ndim,
    )


@pytest.mark.parametrize("name", list(_MODULES))
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_numerical_stability(capsys, device, name):
    if not is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    warp_module, torch_module = _MODULES[name]
    array = np.array([[1000.0, 1001.0, 1002.0], [-1000.0, -1001.0, -1002.0]], dtype=np.float32)
    warp_output = warp_module().to(device)(wp.array(array, device=device)).numpy()
    torch_output = torch_module(dim=-1)(torch.tensor(array)).numpy()
    assert np.all(np.isfinite(warp_output))
    np.testing.assert_allclose(warp_output, torch_output, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("name", list(_MODULES))
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_half_precision_accumulation(capsys, device, name):
    # the small terms of a large sum of exponentials would be lost if the sum were accumulated in half precision
    if not is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    warp_module, torch_module = _MODULES[name]
    array = utilities.sample_array(shape=[3, 4096], dtype=wp.float16)
    warp_output = warp_module().to(device)(wp.array(array, device=device)).numpy()
    torch_output = torch_module(dim=-1)(torch.tensor(array, dtype=torch.float64)).numpy()
    np.testing.assert_allclose(warp_output, torch_output, rtol=2e-3)


@pytest.mark.parametrize("name", list(_MODULES))
@pytest.mark.parametrize("requires_grad", [True, False])
@pytest.mark.parametrize("ndim", [2])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_requires_grad(capsys, device, ndim, requires_grad, name):
    if not is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    check_requires_grad(
        warp_activation=_MODULES[name][0](requires_grad=requires_grad),
        device=device,
        ndim=ndim,
        requires_grad=requires_grad,
    )


@pytest.mark.parametrize("name", list(_MODULES))
def test_invalid_inputs(capsys, name):
    module = _MODULES[name][0](dim=2).to("cpu")
    assert module.dim == 2
    # out-of-range dimension
    with pytest.raises(IndexError, match="out of range"):
        module(wp.zeros((2, 2), dtype=wp.float32, device="cpu"))
    # unsupported data type
    with pytest.raises(TypeError, match="int32"):
        module(wp.zeros((2, 2, 2), dtype=wp.int32, device="cpu"))


@pytest.mark.parametrize("name", list(_MODULES))
def test_empty_input(capsys, name):
    output = _MODULES[name][0](dim=1).to("cpu")(wp.zeros((3, 0), dtype=wp.float32, device="cpu"))
    assert output.shape == (3, 0)
