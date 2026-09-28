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
from .common import check_forward, check_gradients


# test-specific parameters
@pytest.mark.parametrize("ndim", [1, 2, 3, 4])
@pytest.mark.parametrize("dtype", [wp.float16, wp.float32, wp.float64])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_forward(capsys, device, dtype, ndim):
    if not utilities.is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    check_forward(
        warp_module=nn.Identity(),
        torch_module=torch.nn.Identity(),
        device=device,
        dtype=dtype,
        shape=(10, 11, 12, 13)[:ndim],
    )


# test-specific parameters
@pytest.mark.parametrize("ndim", [1, 2, 3, 4])
@pytest.mark.parametrize("dtype", [wp.float16, wp.float32, wp.float64])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_gradients(capsys, device, dtype, ndim):
    if not utilities.is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    check_gradients(
        warp_module=nn.Identity(),
        torch_module=torch.nn.Identity(),
        device=device,
        dtype=dtype,
        shape=(10, 11, 12, 13)[:ndim],
    )
