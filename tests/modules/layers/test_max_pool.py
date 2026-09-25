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
from .common import check_forward, check_gradients, check_requires_grad


_MODULES = {1: (nn.MaxPool1D, torch.nn.MaxPool1d), 2: (nn.MaxPool2D, torch.nn.MaxPool2d)}

_ARGUMENTS = [
    {"kernel_size": 2},
    {"kernel_size": 3, "stride": 1},
    {"kernel_size": 3, "stride": 2, "padding": 1},
    {"kernel_size": 2, "stride": 1, "dilation": 3},
    {"kernel_size": 3, "stride": 2, "padding": 1, "ceil_mode": True},
    {"kernel_size": 4, "stride": 3, "padding": 2, "dilation": 2, "ceil_mode": True},
]
_ARGUMENTS_2D = [
    {"kernel_size": (2, 3), "stride": (1, 2), "padding": (1, 0), "dilation": (2, 1)},
    {"kernel_size": (3, 1), "stride": (2, 1), "padding": (1, 0), "ceil_mode": True},
]
_CASES = [(1, kwargs) for kwargs in _ARGUMENTS] + [(2, kwargs) for kwargs in _ARGUMENTS + _ARGUMENTS_2D]


def _shape(spatial_dims):
    return [3, 4, 11] if spatial_dims == 1 else [3, 4, 11, 10]


@pytest.mark.parametrize("spatial_dims, kwargs", _CASES)
@pytest.mark.parametrize("dtype", [wp.float32])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_forward(capsys, device, dtype, spatial_dims, kwargs):
    if not utilities.is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    warp_module, torch_module = _MODULES[spatial_dims]
    check_forward(
        warp_module=warp_module(**kwargs),
        torch_module=torch_module(**kwargs),
        device=device,
        dtype=dtype,
        shape=_shape(spatial_dims),
    )


@pytest.mark.parametrize("spatial_dims, kwargs", _CASES)
@pytest.mark.parametrize("dtype", [wp.float32])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_gradients(capsys, device, dtype, spatial_dims, kwargs):
    if not utilities.is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    warp_module, torch_module = _MODULES[spatial_dims]
    check_gradients(
        warp_module=warp_module(**kwargs),
        torch_module=torch_module(**kwargs),
        device=device,
        dtype=dtype,
        shape=_shape(spatial_dims),
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_special_values(capsys, device):
    if not utilities.is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    array = np.array([[[-np.inf, -np.inf, 1.0, np.nan, 2.0, 3.0]]], dtype=np.float32)
    output = nn.MaxPool1D(2).to(device)(wp.array(array, device=device)).numpy()
    torch_output = torch.nn.MaxPool1d(2)(torch.tensor(array)).numpy()
    np.testing.assert_array_equal(output, torch_output)
    np.testing.assert_array_equal(output, [[[-np.inf, np.nan, 3.0]]])


@pytest.mark.parametrize("spatial_dims", [1, 2])
@pytest.mark.parametrize("requires_grad", [True, False])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_requires_grad(capsys, device, requires_grad, spatial_dims):
    if not utilities.is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    check_requires_grad(
        warp_module=_MODULES[spatial_dims][0](2, requires_grad=requires_grad),
        device=device,
        inputs=[wp.array(utilities.sample_array(_shape(spatial_dims)), device=device, requires_grad=True)],
        requires_grad=requires_grad,
    )


def test_arguments(capsys):
    max_pool = nn.MaxPool2D(3, padding=1)
    assert max_pool.kernel_size == (3, 3)
    assert max_pool.stride == (3, 3)  # the kernel size by default
    assert max_pool.padding == (1, 1)
    assert max_pool.dilation == (1, 1)
    assert not max_pool.ceil_mode
    for kwargs in [
        {"kernel_size": 0},
        {"kernel_size": 2, "stride": 0},
        {"kernel_size": 2, "dilation": 0},
        {"kernel_size": 2, "padding": -1},
        {"kernel_size": 3, "padding": 2},  # more than half of the kernel size
    ]:
        with pytest.raises(ValueError):
            nn.MaxPool1D(**kwargs)


@pytest.mark.parametrize("spatial_dims", [1, 2])
def test_invalid_input(capsys, spatial_dims):
    max_pool = _MODULES[spatial_dims][0](4).to("cpu")
    # unsupported number of dimensions
    with pytest.raises(ValueError, match="input array"):
        max_pool(wp.zeros([2] * (spatial_dims + 1), dtype=wp.float32, device="cpu"))
    # output too small
    with pytest.raises(ValueError, match="too small"):
        max_pool(wp.zeros([2, 2] + [3] * spatial_dims, dtype=wp.float32, device="cpu"))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_gradient_routing(capsys, device):
    if not utilities.is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    # windows: ties (the first maximum is selected), NaN (propagated) and -inf values
    array = np.array([[[1.0, 1.0, 2.0, 2.0, np.nan, 1.0, 1.0, np.nan, -np.inf, -np.inf]]], dtype=np.float32)
    input = wp.array(array, device=device, requires_grad=True)
    tape = wp.Tape()
    with tape:
        output = nn.MaxPool1D(2).to(device)(input)
    tape.backward(grads={output: wp.ones_like(output)})
    np.testing.assert_array_equal(output.numpy(), [[[1.0, 2.0, np.nan, np.nan, -np.inf]]])
    np.testing.assert_array_equal(input.grad.numpy(), [[[1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0]]])


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_empty_window(capsys, device):
    if not utilities.is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    # the (only) window covers the positions -1 and 2 of a single-element input, which are both out of bounds
    input = wp.array([[[5.0]]], dtype=wp.float32, device=device, requires_grad=True)
    tape = wp.Tape()
    with tape:
        output = nn.MaxPool1D(2, padding=1, dilation=3, ceil_mode=True).to(device)(input)
    tape.backward(grads={output: wp.ones_like(output)})
    np.testing.assert_array_equal(output.numpy(), [[[-np.inf]]])
    np.testing.assert_array_equal(input.grad.numpy(), [[[0.0]]])
