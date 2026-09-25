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

from ...utilities import is_device_available
from .common import check_forward, check_gradients, check_requires_grad


@pytest.mark.parametrize("alpha", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("dtype", [wp.float16, wp.float32, wp.float64])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_forward(capsys, device, dtype, ndim, alpha):
    if not is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    check_forward(
        warp_activation=nn.CELU(alpha=alpha),
        torch_activation=torch.nn.CELU(alpha=alpha),
        device=device,
        dtype=dtype,
        ndim=ndim,
    )


@pytest.mark.parametrize("alpha", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("dtype", [wp.float32])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_gradients(capsys, device, dtype, ndim, alpha):
    if not is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    check_gradients(
        warp_activation=nn.CELU(alpha=alpha),
        torch_activation=torch.nn.CELU(alpha=alpha),
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
        warp_activation=nn.CELU(requires_grad=requires_grad), device=device, ndim=ndim, requires_grad=requires_grad
    )


def test_invalid_alpha(capsys):
    assert nn.CELU(alpha=0.5).alpha == 0.5
    with pytest.raises(ValueError, match="non-zero"):
        nn.CELU(alpha=0.0)
