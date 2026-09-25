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
from .common import check_forward, check_gradients, check_requires_grad


# (threshold, value), where the last one is equivalent to the ONNX ThresholdedRelu operator
_PARAMETERS = [(0.1, 20.0), (-0.5, -2.0), (0.25, 0.0)]


@pytest.mark.parametrize("threshold, value", _PARAMETERS)
@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("dtype", [wp.float16, wp.float32, wp.float64])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_forward(capsys, device, dtype, ndim, threshold, value):
    if not is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    check_forward(
        warp_activation=nn.Threshold(threshold, value),
        torch_activation=torch.nn.Threshold(threshold, value),
        device=device,
        dtype=dtype,
        ndim=ndim,
    )


@pytest.mark.parametrize("threshold, value", _PARAMETERS)
@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("dtype", [wp.float32])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_gradients(capsys, device, dtype, ndim, threshold, value):
    if not is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    check_gradients(
        warp_activation=nn.Threshold(threshold, value),
        torch_activation=torch.nn.Threshold(threshold, value),
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
        warp_activation=nn.Threshold(0.1, 20.0, requires_grad=requires_grad),
        device=device,
        ndim=ndim,
        requires_grad=requires_grad,
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_nan_propagation(capsys, device):
    if not is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    threshold = nn.Threshold(0.5, -1.0).to(device)
    assert (threshold.threshold, threshold.value) == (0.5, -1.0)
    output = threshold(wp.array([np.nan, 0.5, 0.75], dtype=wp.float32, device=device)).numpy()
    np.testing.assert_array_equal(output, [np.nan, -1.0, 0.75])
