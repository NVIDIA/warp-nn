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
from .common import check_forward, check_gradients, check_initialize_parameters, check_requires_grad


# (input shape, normalized shape)
_SHAPES = [((7, 16), 16), ((3, 5, 16), (16,)), ((4, 5, 6), (5, 6)), ((2, 3, 4, 5), (3, 4, 5)), ((33,), 33)]
# (elementwise_affine, bias)
_AFFINE = [(True, True), (True, False), (False, False)]


@pytest.mark.parametrize("elementwise_affine, bias", _AFFINE)
@pytest.mark.parametrize("shape, normalized_shape", _SHAPES)
@pytest.mark.parametrize("dtype", [wp.float32])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_forward(capsys, device, dtype, shape, normalized_shape, elementwise_affine, bias):
    if not utilities.is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    kwargs = {"eps": 1e-3, "elementwise_affine": elementwise_affine, "bias": bias}
    check_forward(
        warp_module=nn.LayerNorm(normalized_shape, **kwargs),
        torch_module=torch.nn.LayerNorm(normalized_shape, **kwargs),
        device=device,
        dtype=dtype,
        shape=shape,
    )


@pytest.mark.parametrize("elementwise_affine, bias", _AFFINE)
@pytest.mark.parametrize("shape, normalized_shape", _SHAPES)
@pytest.mark.parametrize("dtype", [wp.float32])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_gradients(capsys, device, dtype, shape, normalized_shape, elementwise_affine, bias):
    if not utilities.is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    kwargs = {"elementwise_affine": elementwise_affine, "bias": bias}
    check_gradients(
        warp_module=nn.LayerNorm(normalized_shape, **kwargs),
        torch_module=torch.nn.LayerNorm(normalized_shape, **kwargs),
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
        warp_module=nn.LayerNorm(8, requires_grad=requires_grad),
        device=device,
        inputs=[wp.array(utilities.sample_array([2, 8]), device=device, requires_grad=True)],
        requires_grad=requires_grad,
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_initialize_parameters(capsys, device):
    if not utilities.is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    check_initialize_parameters(module_type=nn.LayerNorm, module_kwargs={"normalized_shape": (3, 4)}, device=device)
    # the default/initial values match PyTorch
    utilities.check_arrays(
        list(torch.nn.LayerNorm((3, 4)).parameters()),
        nn.LayerNorm((3, 4)).to(device).parameters(),
        test="equal",
    )


def test_invalid_input(capsys):
    layer_norm = nn.LayerNorm((4, 5)).to("cpu")
    assert layer_norm.normalized_shape == (4, 5)
    with pytest.raises(ValueError, match="normalized shape"):
        layer_norm(wp.zeros((4, 5, 4), dtype=wp.float32, device="cpu"))
    with pytest.raises(ValueError, match="normalized shape"):
        layer_norm(wp.zeros((5,), dtype=wp.float32, device="cpu"))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_precision(capsys, device):
    if not utilities.is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    # many values per row, with a large mean compared to their deviation
    array = (np.random.rand(3, 256, 256) + 10.0).astype(np.float32)
    warp_output = nn.LayerNorm((256, 256)).to(device)(wp.array(array, device=device))
    torch_output = torch.nn.LayerNorm((256, 256), dtype=torch.float64)(torch.tensor(array, dtype=torch.float64))
    np.testing.assert_allclose(warp_output.numpy(), torch_output.detach().numpy(), atol=1e-4)
