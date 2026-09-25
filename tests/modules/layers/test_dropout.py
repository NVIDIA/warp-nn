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

import numpy as np
import warp as wp

import warp_nn.nn as nn

from ... import utilities
from .common import check_requires_grad


def _positive_input(shape, *, dtype=wp.float32, device, requires_grad=False):
    # strictly positive values, so that the zeroed outputs identify the dropped values
    array = (np.random.rand(*shape) + 0.5).astype(utilities.parse_dtype("numpy", dtype))
    return wp.array(array, dtype=dtype, device=device, requires_grad=requires_grad)


@pytest.mark.parametrize("p", [0.2, 0.5, 0.9])
@pytest.mark.parametrize("shape", [(100_000,), (100, 1000), (10, 100, 100), (10, 10, 10, 100)])
@pytest.mark.parametrize("dtype", [wp.float16, wp.float32, wp.float64])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_forward(capsys, device, dtype, shape, p):
    if not utilities.is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    dropout = nn.Dropout(p).to(device)
    assert dropout.p == p
    input = _positive_input(shape, dtype=dtype, device=device)
    output = dropout(input)
    assert output.shape == input.shape and output.dtype == dtype
    # the values are dropped with probability p, and the kept values are scaled
    input, output = input.numpy(), output.numpy()
    kept = output != 0
    assert abs(1.0 - kept.mean() - p) < 0.01
    np.testing.assert_allclose(output[kept], input[kept] / (1.0 - p), rtol=1e-3 if dtype == wp.float16 else 1e-6)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_gradients(capsys, device):
    if not utilities.is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    dropout = nn.Dropout(0.3).to(device)
    input = _positive_input((50, 40), device=device, requires_grad=True)
    tape = wp.Tape()
    with tape:
        output = dropout(input)
    # the seed has already changed, so the backward pass must use the stored mask of the forward pass
    expected = output.numpy() / input.numpy()
    tape.backward(grads={output: wp.ones_like(output)})
    np.testing.assert_allclose(input.grad.numpy(), expected, rtol=1e-6)
    assert set(np.unique(input.grad.numpy())) == {0.0, np.float32(1.0 / 0.7)}


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_different_masks(capsys, device):
    if not utilities.is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    dropout = nn.Dropout(0.5).to(device)
    input = _positive_input((1000,), device=device)
    # copy the outputs, since the output array is reused (and CPU arrays are exposed without copying)
    first = dropout(input).numpy().copy()
    second = dropout(input).numpy().copy()
    assert not np.array_equal(first != 0, second != 0)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_reproducibility(capsys, device):
    if not utilities.is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    # the random number generator is seeded from NumPy's global random number generator
    input = _positive_input((1000,), device=device)
    state = np.random.get_state()
    try:
        outputs = []
        for _ in range(2):
            np.random.seed(0)
            dropout = nn.Dropout(0.5).to(device)
            outputs.append([dropout(input).numpy().copy() for _ in range(3)])
    finally:
        np.random.set_state(state)
    np.testing.assert_array_equal(outputs[0], outputs[1])
    assert not np.array_equal(outputs[0][0], outputs[0][1])


def test_graph_capture(capsys):
    if not utilities.is_device_available("cuda"):
        pytest.skip("Device 'cuda' is not available")
    dropout = nn.Dropout(0.5).to("cuda")
    input = _positive_input((1000,), device="cuda")
    output = dropout(input)  # allocate the cached arrays before capturing
    with wp.ScopedCapture(device="cuda") as capture:
        dropout(input)
    # each replay draws a different mask
    masks = []
    for _ in range(3):
        wp.capture_launch(capture.graph)
        masks.append(output.numpy() != 0)
    assert not np.array_equal(masks[0], masks[1]) and not np.array_equal(masks[1], masks[2])


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_identity(capsys, device):
    if not utilities.is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    input = _positive_input((10, 10), device=device)
    # evaluation mode
    dropout = nn.Dropout(0.5).to(device)
    assert dropout.eval() is dropout and not dropout.training
    assert dropout(input) is input
    assert dropout.train() is dropout and dropout.training
    assert dropout(input) is not input
    # zero probability
    assert nn.Dropout(0.0).to(device)(input) is input


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_all_dropped(capsys, device):
    if not utilities.is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    output = nn.Dropout(1.0).to(device)(_positive_input((10, 10), device=device))
    np.testing.assert_array_equal(output.numpy(), np.zeros((10, 10)))


def test_move_to_device(capsys):
    if not utilities.is_device_available("cuda"):
        pytest.skip("Device 'cuda' is not available")
    dropout = nn.Dropout(0.5)
    for device in ["cpu", "cuda", "cpu"]:
        output = dropout.to(device)(_positive_input((100,), device=device))
        assert output.device == wp.get_device(device)
        assert np.any(output.numpy() == 0) and np.any(output.numpy() != 0)


@pytest.mark.parametrize("requires_grad", [True, False])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_requires_grad(capsys, device, requires_grad):
    if not utilities.is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    check_requires_grad(
        warp_module=nn.Dropout(0.5, requires_grad=requires_grad),
        device=device,
        inputs=[_positive_input((2, 8), device=device, requires_grad=True)],
        requires_grad=requires_grad,
    )


def test_invalid_arguments(capsys):
    with pytest.raises(ValueError, match="interval"):
        nn.Dropout(-0.1)
    with pytest.raises(ValueError, match="interval"):
        nn.Dropout(1.5)
    with pytest.raises(TypeError, match="int32"):
        nn.Dropout(0.5).to("cpu")(wp.zeros((2, 2), dtype=wp.int32, device="cpu"))
