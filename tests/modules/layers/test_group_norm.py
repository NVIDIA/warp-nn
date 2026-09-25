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
from .common import check_forward, check_gradients, check_initialize_parameters, check_requires_grad


_SHAPES = [(5, 6), (4, 6, 7), (3, 6, 4, 5)]


@pytest.mark.parametrize("affine", [True, False])
@pytest.mark.parametrize("num_groups", [1, 2, 3, 6])
@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("dtype", [wp.float32])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_forward(capsys, device, dtype, shape, num_groups, affine):
    if not utilities.is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    if num_groups == shape[1] and len(shape) == 2:
        pytest.skip("Normalizing single values is ill-conditioned")
    check_forward(
        warp_module=nn.GroupNorm(num_groups, shape[1], eps=1e-3, affine=affine),
        torch_module=torch.nn.GroupNorm(num_groups, shape[1], eps=1e-3, affine=affine),
        device=device,
        dtype=dtype,
        shape=shape,
    )


@pytest.mark.parametrize("affine", [True, False])
@pytest.mark.parametrize("num_groups", [1, 2, 3, 6])
@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("dtype", [wp.float32])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_gradients(capsys, device, dtype, shape, num_groups, affine):
    if not utilities.is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    if num_groups == shape[1] and len(shape) == 2:
        pytest.skip("Normalizing single values is ill-conditioned")
    check_gradients(
        warp_module=nn.GroupNorm(num_groups, shape[1], affine=affine),
        torch_module=torch.nn.GroupNorm(num_groups, shape[1], affine=affine),
        device=device,
        dtype=dtype,
        shape=shape,
        weighted=True,
    )


@pytest.mark.parametrize("requires_grad", [True, False])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_requires_grad(capsys, device, requires_grad):
    if not utilities.is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    check_requires_grad(
        warp_module=nn.GroupNorm(2, 4, requires_grad=requires_grad),
        device=device,
        inputs=[wp.array(utilities.sample_array([2, 4, 3]), device=device, requires_grad=True)],
        requires_grad=requires_grad,
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_initialize_parameters(capsys, device):
    if not utilities.is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    check_initialize_parameters(
        module_type=nn.GroupNorm, module_kwargs={"num_groups": 2, "num_channels": 4}, device=device
    )
    # the default/initial values match PyTorch
    utilities.check_arrays(
        list(torch.nn.GroupNorm(2, 4).parameters()), nn.GroupNorm(2, 4).to(device).parameters(), test="equal"
    )


def test_invalid_arguments(capsys):
    with pytest.raises(ValueError, match="divisible"):
        nn.GroupNorm(4, 6)
    with pytest.raises(ValueError, match="divisible"):
        nn.GroupNorm(0, 6)
    group_norm = nn.GroupNorm(2, 6).to("cpu")
    assert (group_norm.num_groups, group_norm.num_channels) == (2, 6)
    with pytest.raises(ValueError, match="shape"):
        group_norm(wp.zeros((2, 4, 3), dtype=wp.float32, device="cpu"))
    with pytest.raises(ValueError, match="shape"):
        group_norm(wp.zeros((6,), dtype=wp.float32, device="cpu"))
