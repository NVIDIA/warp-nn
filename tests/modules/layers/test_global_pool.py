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


_MODULES = {
    ("GlobalAvgPool", 3): (nn.GlobalAvgPool, torch.nn.AdaptiveAvgPool1d),
    ("GlobalAvgPool", 4): (nn.GlobalAvgPool, torch.nn.AdaptiveAvgPool2d),
    ("GlobalMaxPool", 3): (nn.GlobalMaxPool, torch.nn.AdaptiveMaxPool1d),
    ("GlobalMaxPool", 4): (nn.GlobalMaxPool, torch.nn.AdaptiveMaxPool2d),
}
_SHAPES = [(3, 4, 17), (2, 5, 1), (3, 4, 7, 9), (1, 2, 1, 1)]


@pytest.mark.parametrize("name", ["GlobalAvgPool", "GlobalMaxPool"])
@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("dtype", [wp.float32])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_forward(capsys, device, dtype, shape, name):
    if not utilities.is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    warp_module, torch_module = _MODULES[(name, len(shape))]
    check_forward(warp_module=warp_module(), torch_module=torch_module(1), device=device, dtype=dtype, shape=shape)


@pytest.mark.parametrize("name", ["GlobalAvgPool", "GlobalMaxPool"])
@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("dtype", [wp.float32])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_gradients(capsys, device, dtype, shape, name):
    if not utilities.is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    warp_module, torch_module = _MODULES[(name, len(shape))]
    check_gradients(warp_module=warp_module(), torch_module=torch_module(1), device=device, dtype=dtype, shape=shape)


@pytest.mark.parametrize("name", ["GlobalAvgPool", "GlobalMaxPool"])
@pytest.mark.parametrize("requires_grad", [True, False])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_requires_grad(capsys, device, requires_grad, name):
    if not utilities.is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    check_requires_grad(
        warp_module=_MODULES[(name, 4)][0](requires_grad=requires_grad),
        device=device,
        inputs=[wp.array(utilities.sample_array([2, 3, 4, 5]), device=device, requires_grad=True)],
        requires_grad=requires_grad,
    )


@pytest.mark.parametrize("name", ["GlobalAvgPool", "GlobalMaxPool"])
def test_invalid_input(capsys, name):
    module = _MODULES[(name, 4)][0]().to("cpu")
    for shape in [(2,), (2, 3)]:
        with pytest.raises(ValueError, match="input array"):
            module(wp.zeros(shape, dtype=wp.float32, device="cpu"))
