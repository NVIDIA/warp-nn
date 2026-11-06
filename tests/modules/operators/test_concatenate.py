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
from ...utilities import is_device_available
from .common import INT_DTYPES


def _sample_inputs(*, ndim, dim, dtype=wp.float32, sizes=(3, 1, 5)):
    # input arrays with different sizes (given by `sizes`) in the concatenated dimension
    shapes = [[10] * ndim for _ in sizes]
    for shape, size in zip(shapes, sizes):
        shape[dim] = size
    return [utilities.sample_array(shape=shape, dtype=dtype) for shape in shapes]


@pytest.mark.parametrize("dim", [0, 1, 2, -1])
@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("dtype", [wp.float16, wp.float32, wp.float64])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_forward(capsys, device, dtype, ndim, dim):
    if not is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    if not -ndim <= dim < ndim:
        pytest.skip(f"Dimension {dim} is out of range for {ndim}D arrays")
    module = nn.Concatenate(dim).to(device)
    arrays = _sample_inputs(ndim=ndim, dim=dim, dtype=dtype)
    warp_output = module(*[wp.array(array, device=device) for array in arrays])
    # the concatenation copies the values, so they must be exact
    assert warp_output.dtype == dtype
    np.testing.assert_array_equal(warp_output.numpy(), np.concatenate(arrays, axis=dim))


@pytest.mark.parametrize("dim", [0, 1, 2, -1])
@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_gradients(capsys, device, ndim, dim):
    if not is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    if not -ndim <= dim < ndim:
        pytest.skip(f"Dimension {dim} is out of range for {ndim}D arrays")
    module = nn.Concatenate(dim).to(device)
    arrays = _sample_inputs(ndim=ndim, dim=dim)
    torch_inputs = [torch.tensor(array, dtype=torch.float64, requires_grad=True) for array in arrays]
    warp_inputs = [wp.array(array, device=device, requires_grad=True) for array in arrays]
    # forward pass
    torch_output = torch.cat(torch_inputs, dim=dim)
    tape = wp.Tape()
    with tape:
        warp_output = module(*warp_inputs)
    # backward pass (with the same random upstream gradients)
    utilities.backward(tape, [torch_output], [warp_output])
    # check gradients (with respect to all the inputs)
    for torch_input, warp_input in zip(torch_inputs, warp_inputs):
        utilities.check_arrays(torch_input.grad, warp_input.grad)


@pytest.mark.parametrize("requires_grad", [True, False])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_requires_grad(capsys, device, requires_grad):
    if not is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    module = nn.Concatenate(1, requires_grad=requires_grad).to(device)
    arrays = _sample_inputs(ndim=2, dim=1)
    warp_output = module(*[wp.array(array, device=device, requires_grad=True) for array in arrays])
    # check the flag of the module and of its cached output array
    assert module.requires_grad == requires_grad
    assert warp_output.requires_grad == requires_grad
    # check that the gradient array of the cached output array is allocated accordingly
    assert (warp_output.grad is not None) == requires_grad


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_cached_output(capsys, device):
    if not is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    module = nn.Concatenate(0).to(device)
    arrays = [wp.array(array, device=device) for array in _sample_inputs(ndim=2, dim=0)]
    # the output array is reused for the same output shape (e.g. to replay CUDA graphs)
    output = module(*arrays)
    assert module(*arrays).ptr == output.ptr
    assert module(*arrays[:2]).ptr != output.ptr


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_empty_inputs(capsys, device):
    if not is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    module = nn.Concatenate(1).to(device)
    # input array with no elements in the concatenated dimension
    arrays = _sample_inputs(ndim=2, dim=1, sizes=(3, 0, 2))
    warp_output = module(*[wp.array(array, device=device) for array in arrays])
    np.testing.assert_array_equal(warp_output.numpy(), np.concatenate(arrays, axis=1))
    # input arrays with no elements outside the concatenated dimension
    arrays = [np.zeros((0, 3), np.float32), np.zeros((0, 2), np.float32)]
    assert module(*[wp.array(array, device=device) for array in arrays]).shape == (0, 5)


def test_single_input(capsys):
    module = nn.Concatenate(0).to("cpu")
    (array,) = _sample_inputs(ndim=2, dim=0, sizes=(3,))
    input = wp.array(array, device="cpu")
    warp_output = module(input)
    # the input array is copied
    assert warp_output.ptr != input.ptr
    np.testing.assert_array_equal(warp_output.numpy(), array)


def test_dim(capsys):
    assert nn.Concatenate().dim == 0
    assert nn.Concatenate(-1).dim == -1


@pytest.mark.parametrize("dtype", INT_DTYPES)
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_integer_forward(capsys, device, dtype):
    if not is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    module = nn.Concatenate(1).to(device)
    np_dtype = wp.dtype_to_numpy(dtype)
    info = np.iinfo(np_dtype)
    rng = np.random.default_rng(20261106)
    arrays = [rng.integers(info.min, info.max, size=(4, size), dtype=np_dtype, endpoint=True) for size in (3, 2)]
    warp_output = module(*[wp.array(array, dtype=dtype, device=device) for array in arrays])
    # integer outputs are not differentiable
    assert warp_output.dtype == dtype
    assert not warp_output.requires_grad
    np.testing.assert_array_equal(warp_output.numpy(), np.concatenate(arrays, axis=1))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_non_contiguous_inputs(capsys, device):
    if not is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    module = nn.Concatenate(0).to(device)
    arrays = _sample_inputs(ndim=2, dim=0, sizes=(6, 4))
    torch_inputs = [torch.tensor(array, dtype=torch.float64, requires_grad=True) for array in arrays]
    warp_inputs = [wp.array(array, device=device, requires_grad=True) for array in arrays]
    # strided views (every other column) of the input arrays
    torch_output = torch.cat([input[:, ::2] for input in torch_inputs], dim=0)
    tape = wp.Tape()
    with tape:
        warp_output = module(*[input[:, ::2] for input in warp_inputs])
    np.testing.assert_array_equal(warp_output.numpy(), np.concatenate([array[:, ::2] for array in arrays], axis=0))
    utilities.backward(tape, [torch_output], [warp_output])
    for torch_input, warp_input in zip(torch_inputs, warp_inputs):
        utilities.check_arrays(torch_input.grad, warp_input.grad)


def test_unsupported_input(capsys):
    module = nn.Concatenate().to("cpu")
    # data type not supported by the operation, and unsupported number of dimensions
    for shape, dtype in [((2, 2), wp.vec3), ((2,) * 4, wp.float32)]:
        with pytest.raises(TypeError, match="Unsupported input"):
            module(*[wp.zeros(shape, dtype=dtype, device="cpu") for _ in range(2)])


def test_invalid_inputs(capsys):
    module = nn.Concatenate(1).to("cpu")
    # no input arrays
    with pytest.raises(ValueError, match="At least one input array is required"):
        module()
    # dimension out of range
    with pytest.raises(ValueError, match="dimension 1 is out of range for 1D input arrays"):
        module(wp.zeros(2, dtype=wp.float32, device="cpu"))
    # different shapes outside the concatenated dimension, or different number of dimensions
    for shape in [(3, 2), (2, 2, 1)]:
        with pytest.raises(ValueError, match="same shape, except in dimension 1"):
            module(wp.zeros((2, 2), dtype=wp.float32, device="cpu"), wp.zeros(shape, dtype=wp.float32, device="cpu"))
    # different data types
    with pytest.raises(TypeError, match="same data type"):
        module(wp.zeros((2, 2), dtype=wp.float32, device="cpu"), wp.zeros((2, 2), dtype=wp.float64, device="cpu"))
