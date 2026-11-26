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
from warp_nn.utils import cache

from ... import utilities
from ...utilities import is_device_available
from .common import INT_DTYPES


# permutations (or None, to reverse the dimensions) for 1D to 4D arrays
_DIMS = {
    1: [None, (0,), (-1,)],
    2: [None, (0, 1), (1, 0), (-1, 0)],
    3: [None, (0, 1, 2), (2, 0, 1), (1, 0, 2), (0, -1, 1)],
    4: [None, (0, 2, 3, 1), (3, 1, 0, 2), (-1, -2, -3, -4)],
}
_CASES = [(ndim, dims) for ndim, all_dims in _DIMS.items() for dims in all_dims]
_SHAPES = {1: (7,), 2: (5, 7), 3: (3, 5, 7), 4: (2, 3, 5, 7)}


@pytest.mark.parametrize(("ndim", "dims"), _CASES)
@pytest.mark.parametrize("dtype", [wp.float16, wp.float32, wp.float64])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_forward(capsys, device, dtype, ndim, dims):
    if not is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    module = nn.Transpose(dims).to(device)
    array = utilities.sample_array(shape=_SHAPES[ndim], dtype=dtype)
    warp_output = module(wp.array(array, device=device))
    # the transposition copies the values, so they must be exact
    assert warp_output.dtype == dtype
    assert warp_output.is_contiguous
    # NumPy reverses the dimensions if `axes` is None
    np.testing.assert_array_equal(warp_output.numpy(), np.transpose(array, dims))


@pytest.mark.parametrize(("ndim", "dims"), _CASES)
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_gradients(capsys, device, ndim, dims):
    if not is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    module = nn.Transpose(dims).to(device)
    array = utilities.sample_array(shape=_SHAPES[ndim])
    torch_input = torch.tensor(array, dtype=torch.float64, requires_grad=True)
    warp_input = wp.array(array, device=device, requires_grad=True)
    # forward pass
    torch_dims = tuple(reversed(range(ndim))) if dims is None else dims
    torch_output = torch_input.permute(torch_dims)
    tape = wp.Tape()
    with tape:
        warp_output = module(warp_input)
    # backward pass (with the same random upstream gradients)
    utilities.backward(tape, [torch_output], [warp_output])
    utilities.check_arrays(torch_input.grad, warp_input.grad)


@pytest.mark.parametrize("dtype", [*INT_DTYPES, wp.bool])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_non_differentiable_forward(capsys, device, dtype):
    if not is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    module = nn.Transpose((2, 0, 1)).to(device)
    np_dtype = wp.dtype_to_numpy(dtype)
    rng = np.random.default_rng(20261125)
    if dtype == wp.bool:
        array = rng.integers(0, 2, size=(3, 4, 5)).astype(np_dtype)
    else:
        info = np.iinfo(np_dtype)
        array = rng.integers(info.min, info.max, size=(3, 4, 5), dtype=np_dtype, endpoint=True)
    warp_output = module(wp.array(array, dtype=dtype, device=device))
    # integer and Boolean outputs are not differentiable
    assert warp_output.dtype == dtype
    assert not warp_output.requires_grad
    np.testing.assert_array_equal(warp_output.numpy(), np.transpose(array, (2, 0, 1)))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_non_contiguous_input(capsys, device):
    if not is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    module = nn.Transpose((1, 2, 0)).to(device)
    array = utilities.sample_array(shape=(4, 6, 8))
    torch_input = torch.tensor(array, dtype=torch.float64, requires_grad=True)
    warp_input = wp.array(array, device=device, requires_grad=True)
    # strided view (every other element of the last two dimensions)
    torch_output = torch_input[:, ::2, ::2].permute(1, 2, 0)
    tape = wp.Tape()
    with tape:
        warp_output = module(warp_input[:, ::2, ::2])
    assert warp_output.is_contiguous
    np.testing.assert_array_equal(warp_output.numpy(), np.transpose(array[:, ::2, ::2], (1, 2, 0)))
    utilities.backward(tape, [torch_output], [warp_output])
    utilities.check_arrays(torch_input.grad, warp_input.grad)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_large_input(capsys, device):
    if not is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    # many CUDA blocks, and dimensions that are not multiples of the block size
    module = nn.Transpose().to(device)
    array = utilities.sample_array(shape=(257, 513))
    np.testing.assert_array_equal(module(wp.array(array, device=device)).numpy(), array.T)


@pytest.mark.parametrize("requires_grad", [True, False])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_requires_grad(capsys, device, requires_grad):
    if not is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    module = nn.Transpose(requires_grad=requires_grad).to(device)
    warp_output = module(wp.array(utilities.sample_array(shape=(2, 3)), device=device, requires_grad=True))
    # check the flag of the module and of its cached output array
    assert module.requires_grad == requires_grad
    assert warp_output.requires_grad == requires_grad
    assert (warp_output.grad is not None) == requires_grad


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_cached_output(capsys, device):
    if not is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    module = nn.Transpose((1, 0)).to(device)
    input = wp.array(utilities.sample_array(shape=(2, 3)), device=device)
    # the output array (if the elements are copied) is reused for the same output shape (e.g. to replay CUDA graphs)
    output = module(input)
    assert output.ptr != input.ptr
    assert module(input).ptr == output.ptr
    assert module(wp.array(utilities.sample_array(shape=(3, 2)), device=device)).ptr != output.ptr


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_disabled_cache(capsys, device):
    if not is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    module = nn.Transpose().to(device)
    array = utilities.sample_array(shape=(2, 3))
    input = wp.array(array, device=device)
    cache.set_enabled(False)
    try:
        # a new output array is allocated at every call
        output = module(input)
        assert module(input).ptr != output.ptr
        # so the output of a (square) transposition can be transposed back
        square = wp.array(utilities.sample_array(shape=(4, 4)), device=device)
        np.testing.assert_array_equal(module(module(square)).numpy(), square.numpy())
    finally:
        cache.set_enabled(True)
    np.testing.assert_array_equal(output.numpy(), array.T)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_empty_input(capsys, device):
    if not is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    module = nn.Transpose((2, 0, 1)).to(device)
    output = module(wp.zeros((0, 3, 4), dtype=wp.float32, device=device))
    assert output.shape == (4, 0, 3)
    assert output.size == 0


def test_dims(capsys):
    assert nn.Transpose().dims is None
    assert nn.Transpose([2, 0, -2]).dims == (2, 0, -2)


def test_shared_memory(capsys):
    module = nn.Transpose().to("cpu")
    output = module(wp.array(utilities.sample_array(shape=(3, 3)), device="cpu"))
    # transposing the output array in place would overwrite elements before reading them
    for input in [output, output.transpose()]:
        with pytest.raises(ValueError, match="shares memory with the output array"):
            module(input)


def test_unsupported_input(capsys):
    module = nn.Transpose().to("cpu")
    with pytest.raises(TypeError, match="Unsupported data type vec3f"):
        module(wp.zeros((2, 2), dtype=wp.vec3, device="cpu"))


def test_invalid_dims(capsys):
    # duplicated values
    with pytest.raises(ValueError, match="contain duplicated values"):
        nn.Transpose((0, 0))
    # wrong number of dimensions, out of range, or duplicated after resolving negative values
    for dims in [(0, 1), (0, 1, 2, 3), (0, 1, 3), (0, -4, 2), (0, 1, -3)]:
        with pytest.raises(ValueError, match="not a permutation of the dimensions of 3D arrays"):
            nn.Transpose(dims).to("cpu")(wp.zeros((2, 3, 4), dtype=wp.float32, device="cpu"))


def test_cuda_graph_capture(capsys):
    if not is_device_available("cuda"):
        pytest.skip("Device 'cuda' is not available")
    module = nn.Transpose((2, 0, 1)).to("cuda")
    array = utilities.sample_array(shape=(3, 4, 5))
    input = wp.array(array, device="cuda")
    # allocate the cached output array before the capture
    output = module(input)
    with wp.ScopedCapture(device="cuda") as capture:
        assert module(input).ptr == output.ptr
    # the replayed graph reads the current values of the input array
    input.assign(2.0 * array)
    wp.capture_launch(capture.graph)
    np.testing.assert_array_equal(output.numpy(), np.transpose(2.0 * array, (2, 0, 1)))


def test_input_on_another_device(capsys):
    if not is_device_available("cuda"):
        pytest.skip("Device 'cuda' is not available")
    # the copy kernel would be launched on the input array's device, with an output array on another device
    for module_device, input_device in [("cuda", "cpu"), ("cpu", "cuda")]:
        module = nn.Transpose().to(module_device)
        with pytest.raises(ValueError, match="The input array is on device"):
            module(wp.zeros((2, 3), dtype=wp.float32, device=input_device))


@pytest.mark.parametrize(
    ("shape", "dims"),
    [
        ((3, 4), (0, 1)),  # identity
        ((3, 1, 4), (1, 0, 2)),  # only a dimension of size 1 is moved
        ((1, 3, 1, 4), (2, 1, 3, 0)),  # only dimensions of size 1 are moved
        ((7,), None),  # 1D arrays are never permuted
        ((0, 3, 4), (2, 0, 1)),  # empty array
    ],
)
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_view_output(capsys, device, shape, dims):
    if not is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    module = nn.Transpose(dims, requires_grad=False).to(device)
    array = utilities.sample_array(shape=shape)
    input = wp.array(array, device=device, requires_grad=True)
    tape = wp.Tape()
    with tape:
        output = module(input)
    # the output array is a (contiguous) view of the input array, which no kernel is launched for
    assert len(tape.launches) == 0
    assert output.is_contiguous
    np.testing.assert_array_equal(output.numpy(), np.transpose(array, dims))
    if not input.size:
        return
    assert output.ptr == input.ptr
    # it shares the gradients of the input array (regardless of the module's requires_grad)
    assert output.requires_grad
    weights = utilities.sample_array(shape=output.shape)
    output.grad.assign(weights)
    # the inverse permutation maps the output gradients back to the input dimensions
    inverse = None if dims is None else np.argsort(dims)
    np.testing.assert_array_equal(input.grad.numpy(), np.transpose(weights, inverse))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_view_output_requires_contiguous_input(capsys, device):
    if not is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    module = nn.Transpose((1, 0, 2)).to(device)
    array = utilities.sample_array(shape=(3, 1, 8))
    input = wp.array(array, device=device)
    # a strided view is copied, even if the permutation only moves a dimension of size 1
    output = module(input[:, :, ::2])
    assert output.ptr != input.ptr
    assert output.is_contiguous
    np.testing.assert_array_equal(output.numpy(), np.transpose(array[:, :, ::2], (1, 0, 2)))
