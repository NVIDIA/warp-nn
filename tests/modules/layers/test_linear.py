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

import hypothesis
import hypothesis.strategies as st
import pytest

import torch

import warp as wp

import warp_nn.nn as nn

from ... import utilities
from .common import check_forward, check_gradients, check_initialize_parameters, check_requires_grad


@hypothesis.given(
    batch_size=st.integers(min_value=1, max_value=100),
    in_features=st.integers(min_value=1, max_value=100),
    out_features=st.integers(min_value=1, max_value=100),
)
@hypothesis.settings(
    suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture],
    deadline=None,
    max_examples=15,
    phases=[hypothesis.Phase.explicit, hypothesis.Phase.reuse, hypothesis.Phase.generate],
)
# module-specific parameters
@pytest.mark.parametrize("bias", [True, False])
# test-specific parameters
@pytest.mark.parametrize("ndim", [2])
@pytest.mark.parametrize("dtype", [wp.float32])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_forward(capsys, device, dtype, ndim, bias, batch_size, in_features, out_features):
    if not utilities.is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    check_forward(
        warp_module=nn.Linear(in_features=in_features, out_features=out_features, bias=bias),
        torch_module=torch.nn.Linear(in_features=in_features, out_features=out_features, bias=bias),
        device=device,
        dtype=dtype,
        shape=[batch_size, in_features],
    )


@hypothesis.given(
    batch_size=st.integers(min_value=1, max_value=100),
    in_features=st.integers(min_value=1, max_value=100),
    out_features=st.integers(min_value=1, max_value=100),
)
@hypothesis.settings(
    suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture],
    deadline=None,
    max_examples=15,
    phases=[hypothesis.Phase.explicit, hypothesis.Phase.reuse, hypothesis.Phase.generate],
)
# module-specific parameters
@pytest.mark.parametrize("bias", [True, False])
# test-specific parameters
@pytest.mark.parametrize("ndim", [2])
@pytest.mark.parametrize("dtype", [wp.float32])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_gradients(capsys, device, dtype, ndim, bias, batch_size, in_features, out_features):
    if not utilities.is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    check_gradients(
        warp_module=nn.Linear(in_features=in_features, out_features=out_features, bias=bias),
        torch_module=torch.nn.Linear(in_features=in_features, out_features=out_features, bias=bias),
        device=device,
        dtype=dtype,
        shape=[batch_size, in_features],
    )


# module-specific parameters
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("requires_grad", [True, False])
# test-specific parameters
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_requires_grad(capsys, device, requires_grad, bias):
    if not utilities.is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    check_requires_grad(
        warp_module=nn.Linear(in_features=8, out_features=4, bias=bias, requires_grad=requires_grad),
        device=device,
        inputs=[wp.array(utilities.sample_array([2, 8]), device=device, requires_grad=True)],
        requires_grad=requires_grad,
    )
    check_requires_grad(
        warp_module=nn.LazyLinear(out_features=4, bias=bias, requires_grad=requires_grad),
        device=device,
        inputs=[wp.array(utilities.sample_array([2, 8]), device=device, requires_grad=True)],
        requires_grad=requires_grad,
    )


# module-specific parameters
@pytest.mark.parametrize("bias", [True, False])
# test-specific parameters
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_initialize_parameters(capsys, device, bias):
    if not utilities.is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    check_initialize_parameters(
        module_type=nn.Linear,
        module_kwargs={"in_features": 8, "out_features": 4, "bias": bias},
        device=device,
    )
    check_initialize_parameters(
        module_type=nn.LazyLinear,
        module_kwargs={"out_features": 4, "bias": bias},
        device=device,
        inputs=[wp.array(utilities.sample_array([2, 8]), device=device)],
    )
