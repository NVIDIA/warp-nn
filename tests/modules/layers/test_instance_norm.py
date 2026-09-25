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


_TORCH_MODULES = {3: torch.nn.InstanceNorm1d, 4: torch.nn.InstanceNorm2d}
_SHAPES = [(4, 6, 7), (3, 5, 4, 5), (1, 2, 16)]


@pytest.mark.parametrize("affine", [True, False])
@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("dtype", [wp.float32])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_forward(capsys, device, dtype, shape, affine):
    if not utilities.is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    check_forward(
        warp_module=nn.InstanceNorm(shape[1], eps=1e-3, affine=affine),
        torch_module=_TORCH_MODULES[len(shape)](shape[1], eps=1e-3, affine=affine),
        device=device,
        dtype=dtype,
        shape=shape,
    )


@pytest.mark.parametrize("affine", [True, False])
@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("dtype", [wp.float32])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_gradients(capsys, device, dtype, shape, affine):
    if not utilities.is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    check_gradients(
        warp_module=nn.InstanceNorm(shape[1], affine=affine),
        torch_module=_TORCH_MODULES[len(shape)](shape[1], affine=affine),
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
        warp_module=nn.InstanceNorm(4, affine=True, requires_grad=requires_grad),
        device=device,
        inputs=[wp.array(utilities.sample_array([2, 4, 3]), device=device, requires_grad=True)],
        requires_grad=requires_grad,
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_initialize_parameters(capsys, device):
    if not utilities.is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    check_initialize_parameters(
        module_type=nn.InstanceNorm, module_kwargs={"num_features": 4, "affine": True}, device=device
    )
    assert not nn.InstanceNorm(4).parameters()  # not affine by default


def test_invalid_input(capsys):
    instance_norm = nn.InstanceNorm(4).to("cpu")
    assert instance_norm.num_features == 4
    with pytest.raises(ValueError, match="at least 3 dimensions"):
        instance_norm(wp.zeros((2, 4), dtype=wp.float32, device="cpu"))
    with pytest.raises(ValueError, match="shape"):
        instance_norm(wp.zeros((2, 3, 5), dtype=wp.float32, device="cpu"))
