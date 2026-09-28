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
from .common import check_forward, check_gradients, check_requires_grad, check_unsupported_input, check_values


# map the sampled inputs, in [-1, 1), to the operation's domain
_DOMAINS = (lambda x: x + 2.5,)


# test-specific parameters
@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("dtype", [wp.float16, wp.float32, wp.float64])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_forward(capsys, device, dtype, ndim):
    if not is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    check_forward(
        module=nn.Acosh(), torch_function=torch.acosh, device=device, dtype=dtype, ndim=ndim, domains=_DOMAINS
    )


# test-specific parameters
@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_gradients(capsys, device, ndim):
    if not is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    check_gradients(module=nn.Acosh(), torch_function=torch.acosh, device=device, ndim=ndim, domains=_DOMAINS)


# module-specific parameters
@pytest.mark.parametrize("requires_grad", [True, False])
# test-specific parameters
@pytest.mark.parametrize("ndim", [2])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_requires_grad(capsys, device, ndim, requires_grad):
    if not is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    check_requires_grad(
        module=nn.Acosh(requires_grad=requires_grad), device=device, ndim=ndim, requires_grad=requires_grad
    )


def test_unsupported_input(capsys):
    check_unsupported_input(module=nn.Acosh())


# test-specific parameters
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_values(capsys, device):
    if not is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    # large-magnitude inputs (without overflowing x^2)
    check_values(module=nn.Acosh(), values=[1.5, 1e20, 1e30], numpy_function=np.arccosh, device=device)
