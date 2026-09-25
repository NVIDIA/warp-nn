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
from ..activations.common import TorchLambda, check_forward, check_gradients, check_requires_grad


# (min_val, max_val), including unbounded intervals and an empty one (all the values are set to max_val)
_BOUNDS = [(-0.5, 0.25), (None, 0.25), (-0.5, None), (None, None), (0.5, -0.25)]


def _torch_clip(min_val, max_val):
    return TorchLambda(lambda x: torch.clamp(x, min_val, max_val) if (min_val, max_val) != (None, None) else x)


@pytest.mark.parametrize("min_val, max_val", _BOUNDS)
@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("dtype", [wp.float16, wp.float32, wp.float64])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_forward(capsys, device, dtype, ndim, min_val, max_val):
    if not is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    clip = nn.Clip(min_val, max_val)
    assert (clip.min_val, clip.max_val) == (min_val, max_val)
    check_forward(
        warp_activation=clip,
        torch_activation=_torch_clip(min_val, max_val),
        device=device,
        dtype=dtype,
        ndim=ndim,
    )


@pytest.mark.parametrize("min_val, max_val", _BOUNDS)
@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("dtype", [wp.float32])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_gradients(capsys, device, dtype, ndim, min_val, max_val):
    if not is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    check_gradients(
        warp_activation=nn.Clip(min_val, max_val),
        torch_activation=_torch_clip(min_val, max_val),
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
        warp_activation=nn.Clip(-0.5, 0.5, requires_grad=requires_grad),
        device=device,
        ndim=ndim,
        requires_grad=requires_grad,
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_special_values(capsys, device):
    if not is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    clip = nn.Clip(-1.0, 1.0).to(device)
    output = clip(wp.array([np.nan, -np.inf, np.inf, 0.5], dtype=wp.float32, device=device)).numpy()
    np.testing.assert_array_equal(output, [np.nan, -1.0, 1.0, 0.5])


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_boundary_gradients(capsys, device):
    if not is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    # the gradient is zero at the bounds, as in PyTorch
    array = np.array([-1.5, -1.0, 0.0, 2.0, 2.5], dtype=np.float32)
    torch_input = torch.tensor(array, requires_grad=True)
    torch.clamp(torch_input, -1.0, 2.0).sum().backward()
    warp_input = wp.array(array, device=device, requires_grad=True)
    tape = wp.Tape()
    with tape:
        warp_output = nn.Clip(-1.0, 2.0).to(device)(warp_input)
    tape.backward(grads={warp_output: wp.ones_like(warp_output)})
    np.testing.assert_array_equal(warp_input.grad.numpy(), torch_input.grad.numpy())
    np.testing.assert_array_equal(warp_input.grad.numpy(), [0.0, 0.0, 1.0, 0.0, 0.0])


def test_numpy_scalar_arguments(capsys):
    # the bounds are converted to Python floats, since NumPy scalars cannot be embedded in the kernels
    clip = nn.Clip(np.float32(-0.5), np.float64(0.5)).to("cpu")
    assert type(clip.min_val) is float and type(clip.max_val) is float
    output = clip(wp.array([-1.0, 0.0, 1.0], dtype=wp.float32, device="cpu"))
    np.testing.assert_array_equal(output.numpy(), [-0.5, 0.0, 0.5])
