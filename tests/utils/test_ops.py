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

from typing import Any

import pytest

import numpy as np
import warp as wp

from warp_nn.utils import contiguous, overload_kernels

from .. import utilities


@wp.kernel(module="unique")
def _copy_1d(input: wp.array1d[Any], output: wp.array1d[Any]):
    i = wp.tid()
    output[i] = input[i]


@wp.kernel(module="unique")
def _copy_2d(input: wp.array2d[Any], output: wp.array2d[Any]):
    i, j = wp.tid()
    output[i, j] = input[i, j]


@wp.kernel(module="unique")
def _add_1d(a: wp.array1d[Any], b: wp.array1d[Any], output: wp.array1d[Any]):
    i = wp.tid()
    output[i] = a[i] + b[i]


def test_overload_kernels_default(capsys):
    kernels = overload_kernels(kernels=[_copy_1d, _copy_2d])
    assert set(kernels) == {(ndim, dtype) for ndim in [1, 2] for dtype in [wp.float16, wp.float32, wp.float64]}
    # launch an overload to check its argument types
    array = np.arange(6, dtype=np.float64).reshape(2, 3)
    input = wp.array(array, dtype=wp.float64, device="cpu")
    output = wp.zeros_like(input)
    wp.launch(kernels[(2, wp.float64)], dim=array.shape, inputs=[input], outputs=[output], device="cpu")
    np.testing.assert_array_equal(output.numpy(), array)


@pytest.mark.parametrize("dtype", [wp.int32, wp.float32], ids=lambda dtype: dtype.__name__)
def test_overload_kernels_num_arrays(capsys, dtype):
    kernels = overload_kernels(kernels=[_add_1d], dtypes=[dtype], num_arrays=3)
    assert set(kernels) == {(1, dtype)}
    # launch the overload to check its argument types
    array = np.arange(5).astype(wp.dtype_to_numpy(dtype))
    a = wp.array(array, dtype=dtype, device="cpu")
    b = wp.array(array, dtype=dtype, device="cpu")
    output = wp.zeros_like(a)
    wp.launch(kernels[(1, dtype)], dim=array.shape, inputs=[a, b], outputs=[output], device="cpu")
    np.testing.assert_array_equal(output.numpy(), 2 * array)


@pytest.mark.parametrize("ndim", [1, 2, 3, 4])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_contiguous(capsys, device, ndim):
    if not utilities.is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    base_array = np.random.rand(*[4] * (ndim - 1), 8).astype(np.float32)
    base = wp.array(base_array, device=device, requires_grad=True)
    # contiguous arrays are returned as is
    assert contiguous(base) is base
    # non-contiguous arrays (every other element of the last dimension) are copied,
    # and the gradient of the copy is propagated to them
    view = base[(slice(None),) * (ndim - 1) + (slice(None, None, 2),)]
    assert not view.is_contiguous
    tape = wp.Tape()
    with tape:
        output = contiguous(view)
    assert output.is_contiguous and output.requires_grad
    np.testing.assert_array_equal(output.numpy(), base_array[..., ::2])
    weights = np.random.rand(*output.shape).astype(np.float32)
    tape.backward(grads={output: wp.array(weights, device=device)})
    expected = np.zeros_like(base_array)
    expected[..., ::2] = weights
    np.testing.assert_array_equal(base.grad.numpy(), expected)
