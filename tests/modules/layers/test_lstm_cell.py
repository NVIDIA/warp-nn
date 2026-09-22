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
from .common import check_forward_rnn_cell, check_initialize_parameters, check_requires_grad


@hypothesis.given(
    batch_size=st.integers(min_value=1, max_value=100),
    input_size=st.integers(min_value=1, max_value=100),
    hidden_size=st.integers(min_value=1, max_value=100),
    sequence_length=st.integers(min_value=1, max_value=100),
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
def test_forward(capsys, device, dtype, ndim, bias, batch_size, input_size, hidden_size, sequence_length):
    if not utilities.is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")

    check_forward_rnn_cell(
        warp_module=nn.LSTMCell(input_size=input_size, hidden_size=hidden_size, bias=bias),
        torch_module=torch.nn.LSTMCell(input_size=input_size, hidden_size=hidden_size, bias=bias),
        device=device,
        dtype=dtype,
        shape=[sequence_length, batch_size, input_size],
        hidden_shape=[batch_size, hidden_size],
        cell_shape=[batch_size, hidden_size],
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
        warp_module=nn.LSTMCell(input_size=8, hidden_size=4, bias=bias, requires_grad=requires_grad),
        device=device,
        inputs=[
            wp.array(utilities.sample_array([2, 8]), device=device, requires_grad=True),
            (
                wp.array(utilities.sample_array([2, 4]), device=device, requires_grad=True),
                wp.array(utilities.sample_array([2, 4]), device=device, requires_grad=True),
            ),
        ],
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
        module_type=nn.LSTMCell,
        module_kwargs={"input_size": 8, "hidden_size": 4, "bias": bias},
        device=device,
    )
