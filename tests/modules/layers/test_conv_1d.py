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


@pytest.fixture(autouse=True)
def _disable_cudnn_tf32():
    # cuDNN uses TF32 for convolutions on CUDA by default, which loses precision relative to
    # the warp reference implementation; disable it so outputs/gradients match within tolerance
    previous = torch.backends.cudnn.allow_tf32
    torch.backends.cudnn.allow_tf32 = False
    yield
    torch.backends.cudnn.allow_tf32 = previous


@hypothesis.given(
    batch_size=st.integers(min_value=1, max_value=100),
    in_channels=st.sampled_from(list(range(30, 100, 3))),
    out_channels=st.sampled_from(list(range(30, 100, 3))),
    in_signal_length=st.integers(min_value=10, max_value=100),
)
@hypothesis.settings(
    suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture],
    deadline=None,
    max_examples=15,
    phases=[hypothesis.Phase.explicit, hypothesis.Phase.reuse, hypothesis.Phase.generate],
)
# module-specific parameters
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("groups", [1, 3])
@pytest.mark.parametrize("dilation", [1, 3])
@pytest.mark.parametrize("padding", [0, 2])
@pytest.mark.parametrize("stride", [1, 3])
@pytest.mark.parametrize("kernel_size", [1, 3])
# test-specific parameters
@pytest.mark.parametrize("dtype", [wp.float32])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_forward(
    capsys,
    device,
    dtype,
    kernel_size,
    stride,
    padding,
    dilation,
    groups,
    bias,
    batch_size,
    in_channels,
    out_channels,
    in_signal_length,
):
    if not utilities.is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    # reduce sizes for faster tests on CPU
    if device == "cpu":
        batch_size = min(batch_size, 10)
        in_channels = min(in_channels - 15, 48)  # must be divisible by groups
        out_channels = min(out_channels - 15, 48)  # must be divisible by groups
        in_signal_length = min(in_signal_length, 50)
    check_forward(
        warp_module=nn.Conv1D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        ),
        torch_module=torch.nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        ),
        device=device,
        dtype=dtype,
        shape=[batch_size, in_channels, in_signal_length],
    )


@hypothesis.given(
    batch_size=st.integers(min_value=1, max_value=100),
    in_channels=st.sampled_from(list(range(30, 100, 3))),
    out_channels=st.sampled_from(list(range(30, 100, 3))),
    in_signal_length=st.integers(min_value=10, max_value=100),
)
@hypothesis.settings(
    suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture],
    deadline=None,
    max_examples=15,
    phases=[hypothesis.Phase.explicit, hypothesis.Phase.reuse, hypothesis.Phase.generate],
)
# module-specific parameters
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("groups", [1, 3])
@pytest.mark.parametrize("dilation", [1, 3])
@pytest.mark.parametrize("padding", [0, 2])
@pytest.mark.parametrize("stride", [1, 3])
@pytest.mark.parametrize("kernel_size", [1, 3])
# test-specific parameters
@pytest.mark.parametrize("dtype", [wp.float32])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_gradients(
    capsys,
    device,
    dtype,
    kernel_size,
    stride,
    padding,
    dilation,
    groups,
    bias,
    batch_size,
    in_channels,
    out_channels,
    in_signal_length,
):
    if not utilities.is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    # reduce sizes for faster tests on CPU
    if device == "cpu":
        batch_size = min(batch_size, 10)
        in_channels = min(in_channels - 15, 48)  # must be divisible by groups
        out_channels = min(out_channels - 15, 48)  # must be divisible by groups
        in_signal_length = min(in_signal_length, 50)

    check_gradients(
        warp_module=nn.Conv1D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        ),
        torch_module=torch.nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        ),
        device=device,
        dtype=dtype,
        shape=[batch_size, in_channels, in_signal_length],
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
        warp_module=nn.Conv1D(in_channels=3, out_channels=6, kernel_size=3, bias=bias, requires_grad=requires_grad),
        device=device,
        inputs=[wp.array(utilities.sample_array([2, 3, 10]), device=device, requires_grad=True)],
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
        module_type=nn.Conv1D,
        module_kwargs={"in_channels": 3, "out_channels": 6, "kernel_size": 3, "bias": bias},
        device=device,
    )
