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

import contextlib
import torch

import numpy as np
import warp as wp

import warp_nn.nn as nn
from warp_nn.utils import kernel_config

from ... import utilities
from .common import check_forward_rnn, check_gradients_rnn, check_initialize_parameters, check_requires_grad


@pytest.fixture(autouse=True)
def _disable_cudnn_and_tf32():
    # cuDNN's RNN implementation is much less precise than Warp's implementation (even without TF32),
    # so it is disabled to use PyTorch's native CUDA implementation as reference, whose matrix multiplications
    # must not use TF32 either
    previous = torch.backends.cudnn.enabled, torch.backends.cuda.matmul.allow_tf32
    torch.backends.cudnn.enabled, torch.backends.cuda.matmul.allow_tf32 = False, False
    yield
    torch.backends.cudnn.enabled, torch.backends.cuda.matmul.allow_tf32 = previous


@hypothesis.given(
    batch_size=st.integers(min_value=1, max_value=16),
    sequence_length=st.integers(min_value=1, max_value=8),
    input_size=st.integers(min_value=1, max_value=48),
    hidden_size=st.integers(min_value=1, max_value=48),
    num_layers=st.integers(min_value=1, max_value=3),
)
@hypothesis.settings(
    suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture],
    deadline=None,
    max_examples=10,
    phases=[hypothesis.Phase.explicit, hypothesis.Phase.reuse, hypothesis.Phase.generate],
)
# module-specific parameters
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("bidirectional", [False, True])
# test-specific parameters
@pytest.mark.parametrize("initial_hidden", [True, False])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_forward(
    capsys,
    device,
    initial_hidden,
    bidirectional,
    bias,
    batch_size,
    sequence_length,
    input_size,
    hidden_size,
    num_layers,
):
    if not utilities.is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    num_directions = 2 if bidirectional else 1
    kwargs = {"num_layers": num_layers, "bidirectional": bidirectional, "bias": bias}
    check_forward_rnn(
        warp_module=nn.LSTM(input_size, hidden_size, **kwargs),
        torch_module=torch.nn.LSTM(input_size, hidden_size, batch_first=True, **kwargs),
        device=device,
        shape=[batch_size, sequence_length, input_size],
        hidden_shapes=[[num_directions * num_layers, batch_size, hidden_size]] * 2 if initial_hidden else None,
    )


@hypothesis.given(
    batch_size=st.integers(min_value=1, max_value=16),
    sequence_length=st.integers(min_value=1, max_value=8),
    input_size=st.integers(min_value=1, max_value=48),
    hidden_size=st.integers(min_value=1, max_value=48),
    num_layers=st.integers(min_value=1, max_value=3),
)
@hypothesis.settings(
    suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture],
    deadline=None,
    max_examples=10,
    phases=[hypothesis.Phase.explicit, hypothesis.Phase.reuse, hypothesis.Phase.generate],
)
# module-specific parameters
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("bidirectional", [False, True])
# test-specific parameters
@pytest.mark.parametrize("initial_hidden", [True, False])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_gradients(
    capsys,
    device,
    initial_hidden,
    bidirectional,
    bias,
    batch_size,
    sequence_length,
    input_size,
    hidden_size,
    num_layers,
):
    if not utilities.is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    num_directions = 2 if bidirectional else 1
    kwargs = {"num_layers": num_layers, "bidirectional": bidirectional, "bias": bias}
    # the backward kernel of the LSTM cell requires more shared memory than available on some CUDA devices
    # with the default (32, 32) tile shape
    with kernel_config(tile_2d=(16, 16)) if device == "cuda" else contextlib.nullcontext():
        warp_module = nn.LSTM(input_size, hidden_size, **kwargs)
    check_gradients_rnn(
        warp_module=warp_module,
        torch_module=torch.nn.LSTM(input_size, hidden_size, batch_first=True, **kwargs),
        device=device,
        shape=[batch_size, sequence_length, input_size],
        hidden_shapes=[[num_directions * num_layers, batch_size, hidden_size]] * 2 if initial_hidden else None,
    )


# module-specific parameters
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("bidirectional", [False, True])
@pytest.mark.parametrize("requires_grad", [True, False])
# test-specific parameters
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_requires_grad(capsys, device, requires_grad, bidirectional, bias):
    if not utilities.is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    num_directions = 2 if bidirectional else 1
    check_requires_grad(
        warp_module=nn.LSTM(8, 4, num_layers=2, bidirectional=bidirectional, bias=bias, requires_grad=requires_grad),
        device=device,
        inputs=[
            wp.array(utilities.sample_array([2, 3, 8]), device=device, requires_grad=True),
            (
                wp.array(utilities.sample_array([num_directions * 2, 2, 4]), device=device, requires_grad=True),
                wp.array(utilities.sample_array([num_directions * 2, 2, 4]), device=device, requires_grad=True),
            ),
        ],
        requires_grad=requires_grad,
    )


# module-specific parameters
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("bidirectional", [False, True])
# test-specific parameters
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_initialize_parameters(capsys, device, bidirectional, bias):
    if not utilities.is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    check_initialize_parameters(
        module_type=nn.LSTM,
        module_kwargs={
            "input_size": 8,
            "hidden_size": 4,
            "num_layers": 2,
            "bidirectional": bidirectional,
            "bias": bias,
        },
        device=device,
    )


def test_invalid_arguments(capsys):
    with pytest.raises(ValueError, match="number of layers"):
        nn.LSTM(8, 4, num_layers=0)


def test_invalid_input(capsys):
    module = nn.LSTM(8, 4, num_layers=2, bidirectional=True)
    # input: not 3D, and wrong number of features
    with pytest.raises(ValueError, match="input array"):
        module(wp.zeros((3, 8)))
    with pytest.raises(ValueError, match="input array"):
        module(wp.zeros((2, 3, 7)))
    # hidden and cell states: wrong shape (number of layers times directions, batch size or hidden size)
    for shape in [(2, 2, 4), (4, 3, 4), (4, 2, 5)]:
        with pytest.raises(ValueError, match="hidden state array with shape"):
            module(wp.zeros((2, 3, 8)), (wp.zeros(shape), wp.zeros((4, 2, 4))))
        with pytest.raises(ValueError, match="cell state array with shape"):
            module(wp.zeros((2, 3, 8)), (wp.zeros((4, 2, 4)), wp.zeros(shape)))
    # hidden and cell states: not a tuple of two arrays (a single array, a list, or a tuple of another length)
    state = wp.zeros((4, 2, 4))
    for hidden in [state, [state, state], (state,), (state, state, state)]:
        with pytest.raises(ValueError, match="tuple"):
            module(wp.zeros((2, 3, 8)), hidden)


def test_graph_capture(capsys):
    if not utilities.is_device_available("cuda"):
        pytest.skip("Device 'cuda' is not available")
    module = nn.LSTM(8, 4, num_layers=2, bidirectional=True).to("cuda")
    input = wp.array(utilities.sample_array([2, 3, 8]), device="cuda")
    hidden = (
        wp.array(utilities.sample_array([4, 2, 4]), device="cuda"),
        wp.array(utilities.sample_array([4, 2, 4]), device="cuda"),
    )
    outputs = module(input, hidden)  # allocate the cached arrays before capturing
    outputs = [outputs[0], *outputs[1]]
    expected = [output.numpy() for output in outputs]
    for output in outputs:
        output.zero_()
    with wp.ScopedCapture(device="cuda") as capture:
        module(input, hidden)
    wp.capture_launch(capture.graph)
    for output, array in zip(outputs, expected):
        np.testing.assert_array_equal(output.numpy(), array)
