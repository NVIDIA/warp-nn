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
from .common import TorchLambda, check_forward, check_gradients, check_requires_grad


def _torch_swish(alpha):
    if alpha == 1.0:
        return torch.nn.SiLU()
    return TorchLambda(lambda x: x * torch.sigmoid(alpha * x))


@pytest.mark.parametrize("alpha", [1.0, 2.5, -0.5])
@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("dtype", [wp.float16, wp.float32, wp.float64])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_forward(capsys, device, dtype, ndim, alpha):
    if not is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    check_forward(
        warp_activation=nn.Swish(alpha=alpha),
        torch_activation=_torch_swish(alpha),
        device=device,
        dtype=dtype,
        ndim=ndim,
    )


@pytest.mark.parametrize("alpha", [1.0, 2.5, -0.5])
@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("dtype", [wp.float32])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_gradients(capsys, device, dtype, ndim, alpha):
    if not is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    check_gradients(
        warp_activation=nn.Swish(alpha=alpha),
        torch_activation=_torch_swish(alpha),
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
    swish = nn.Swish(requires_grad=requires_grad)
    assert swish.alpha == 1.0
    check_requires_grad(warp_activation=swish, device=device, ndim=ndim, requires_grad=requires_grad)
