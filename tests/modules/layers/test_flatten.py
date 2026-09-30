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


def generate_indices(ndim):
    # generate valid start_dim, end_dim pairs between [-rank, rank-1]
    pairs = [(i, j) for i in range(ndim) for j in range(i, ndim)]
    # extend to include negative indices
    indices = []
    for item in pairs:
        indices.append((item[0], item[1]))
        indices.append((item[0], item[1] - ndim))
        indices.append((item[0] - ndim, item[1]))
        indices.append((item[0] - ndim, item[1] - ndim))
    return indices


# test-specific parameters
@pytest.mark.parametrize("ndim", [1, 2, 3, 4])
@pytest.mark.parametrize("dtype", [wp.float16, wp.float32, wp.float64])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_forward(capsys, device, dtype, ndim):
    if not utilities.is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    for start_dim, end_dim in generate_indices(ndim):
        check_forward(
            warp_module=nn.Flatten(start_dim=start_dim, end_dim=end_dim),
            torch_module=torch.nn.Flatten(start_dim=start_dim, end_dim=end_dim),
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
    for start_dim, end_dim in generate_indices(ndim):
        check_gradients(
            warp_module=nn.Flatten(start_dim=start_dim, end_dim=end_dim),
            torch_module=torch.nn.Flatten(start_dim=start_dim, end_dim=end_dim),
            device=device,
            dtype=dtype,
            shape=(10, 11, 12, 13)[:ndim],
        )


def test_invalid_input(capsys):
    # the start dimension comes after the end dimension (once resolved for the number of dimensions of the input)
    with pytest.raises(IndexError, match="cannot come after"):
        nn.Flatten(start_dim=2, end_dim=1)(wp.zeros((2, 3, 4), dtype=wp.float32, device="cpu"))
    with pytest.raises(IndexError, match="cannot come after"):
        nn.Flatten(start_dim=-1, end_dim=0)(wp.zeros((2, 3, 4), dtype=wp.float32, device="cpu"))
