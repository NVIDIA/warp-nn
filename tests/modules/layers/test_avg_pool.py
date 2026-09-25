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

import warp as wp

import warp_nn.nn as nn

from ... import utilities
from .common import check_forward, check_gradients, check_requires_grad


_MODULES = {1: (nn.AvgPool1D, torch.nn.AvgPool1d), 2: (nn.AvgPool2D, torch.nn.AvgPool2d)}

_ARGUMENTS = [
    {"kernel_size": 2},
    {"kernel_size": 3, "stride": 1},
    {"kernel_size": 3, "stride": 2, "padding": 1},
    {"kernel_size": 3, "stride": 2, "padding": 1, "count_include_pad": False},
    {"kernel_size": 3, "stride": 2, "padding": 1, "ceil_mode": True},
    {"kernel_size": 4, "stride": 3, "padding": 2, "ceil_mode": True},
    {"kernel_size": 4, "stride": 3, "padding": 2, "ceil_mode": True, "count_include_pad": False},
]
_ARGUMENTS_2D = [
    {"kernel_size": (2, 3), "stride": (1, 2), "padding": (1, 0)},
    {"kernel_size": (3, 1), "stride": (2, 1), "padding": (1, 0), "ceil_mode": True, "count_include_pad": False},
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
    avg_pool = nn.AvgPool1D(3, stride=2)
    assert (avg_pool.kernel_size, avg_pool.stride, avg_pool.padding) == ((3,), (2,), (0,))
    assert not avg_pool.ceil_mode and avg_pool.count_include_pad
    for kwargs in [
        {"kernel_size": 0},
        {"kernel_size": 2, "stride": 0},
        {"kernel_size": 2, "padding": -1},
        {"kernel_size": (3, 3), "padding": (1, 2)},  # more than half of the kernel size
    ]:
        with pytest.raises(ValueError):
            nn.AvgPool2D(**kwargs)


@pytest.mark.parametrize("spatial_dims", [1, 2])
def test_invalid_input(capsys, spatial_dims):
    avg_pool = _MODULES[spatial_dims][0](4).to("cpu")
    # unsupported number of dimensions
    with pytest.raises(ValueError, match="input array"):
        avg_pool(wp.zeros([2] * (spatial_dims + 1), dtype=wp.float32, device="cpu"))
    # output too small
    with pytest.raises(ValueError, match="too small"):
        avg_pool(wp.zeros([2, 2] + [3] * spatial_dims, dtype=wp.float32, device="cpu"))
