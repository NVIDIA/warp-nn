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

from ... import utilities


INT_DTYPES = [wp.int8, wp.int16, wp.int32, wp.int64, wp.uint8, wp.uint16, wp.uint32, wp.uint64]


def _identity(x):
    return x


def _sample_inputs(*, ndim, dtype, domains):
    # map the sampled inputs, in [-1, 1), to the operation's domain
    np_dtype = utilities.parse_dtype("numpy", dtype)
    return [domain(utilities.sample_array(shape=[10] * ndim)).astype(np_dtype) for domain in domains]


def _sample_integer_inputs(*, ndim, dtype, num_inputs):
    # cover the full range of the integer type
    np_dtype = wp.dtype_to_numpy(dtype)
    info = np.iinfo(np_dtype)
    rng = np.random.default_rng(20260925)
    return [
        rng.integers(info.min, info.max, size=[10] * ndim, dtype=np_dtype, endpoint=True) for _ in range(num_inputs)
    ]


def check_forward(*, module, torch_function, device, dtype, ndim, num_inputs=1, domains=None):
    # domains: one function per input, mapping the sampled inputs to the operation's domain
    domains = domains or [_identity] * num_inputs
    module.to(device)
    arrays = _sample_inputs(ndim=ndim, dtype=dtype, domains=domains)
    # forward pass (compute the torch reference in double precision)
    warp_output = module(*[wp.array(array, device=device) for array in arrays])
    torch_output = torch_function(*[torch.tensor(array, dtype=torch.float64) for array in arrays])
    # check outputs
    assert warp_output.dtype == dtype
    utilities.check_arrays(torch_output, warp_output, atol=1e-2 if dtype == wp.float16 else 1e-3)


def check_gradients(*, module, torch_function, device, ndim, num_inputs=1, domains=None):
    # domains: one function per input, mapping the sampled inputs to the operation's domain
    domains = domains or [_identity] * num_inputs
    module.to(device)
    arrays = _sample_inputs(ndim=ndim, dtype=wp.float32, domains=domains)
    torch_inputs = [torch.tensor(array, dtype=torch.float64, requires_grad=True) for array in arrays]
    warp_inputs = [wp.array(array, device=device, requires_grad=True) for array in arrays]
    # forward pass
    torch_output = torch_function(*torch_inputs)
    tape = wp.Tape()
    with tape:
        warp_output = module(*warp_inputs)
    # backward pass (with the same random upstream gradients)
    utilities.backward(tape, [torch_output], [warp_output])
    # check gradients (with respect to all the inputs)
    for torch_input, warp_input in zip(torch_inputs, warp_inputs):
        utilities.check_arrays(torch_input.grad, warp_input.grad)


def check_integer_forward(*, module, numpy_function, device, dtype, ndim, num_inputs=1):
    module.to(device)
    arrays = _sample_integer_inputs(ndim=ndim, dtype=dtype, num_inputs=num_inputs)
    # forward pass
    warp_output = module(*[wp.array(array, dtype=dtype, device=device) for array in arrays])
    # check outputs (integer operations are not differentiable)
    assert not module.requires_grad
    assert not warp_output.requires_grad
    assert warp_output.dtype == dtype
    np.testing.assert_array_equal(warp_output.numpy(), numpy_function(*arrays))


def check_requires_grad(*, module, device, ndim, requires_grad, num_inputs=1):
    module.to(device)
    arrays = _sample_inputs(ndim=ndim, dtype=wp.float32, domains=[_identity] * num_inputs)
    # forward pass
    warp_output = module(*[wp.array(array, device=device, requires_grad=True) for array in arrays])
    # check the flag of the module and of its cached output array
    assert module.requires_grad == requires_grad
    assert warp_output.requires_grad == requires_grad
    # check that the gradient array of the cached output array is allocated accordingly
    assert (warp_output.grad is not None) == requires_grad


def check_unsupported_input(*, module, dtype=wp.float32, num_inputs=1):
    # dtype: a data type supported by the operation
    module.to("cpu")
    unsupported_dtype = wp.int32 if wp.types.type_is_float(dtype) else wp.float32
    # data type not supported by the operation, and unsupported number of dimensions
    for shape, input_dtype in [((2, 2), unsupported_dtype), ((2,) * 4, dtype)]:
        with pytest.raises(TypeError, match="Unsupported input"):
            module(*[wp.zeros(shape, dtype=input_dtype, device="cpu") for _ in range(num_inputs)])


def check_mismatched_inputs(*, module, dtype=wp.float32):
    module.to("cpu")
    other_dtype = next(other for other in (wp.float32, wp.float64, wp.int32, wp.int64) if other != dtype)
    # different shapes
    with pytest.raises(ValueError, match="same shape"):
        module(wp.zeros((2, 2), dtype=dtype, device="cpu"), wp.zeros((2, 1), dtype=dtype, device="cpu"))
    # different data types
    with pytest.raises(TypeError, match="same data type"):
        module(wp.zeros((2, 2), dtype=dtype, device="cpu"), wp.zeros((2, 2), dtype=other_dtype, device="cpu"))


def check_values(*, module, values, numpy_function, device, rtol=1e-6):
    # specific (e.g. special, or large-magnitude) input values, compared in single precision
    module.to(device)
    array = np.array(values, dtype=np.float32)
    warp_output = module(wp.array(array, device=device)).numpy()
    np.testing.assert_allclose(warp_output, numpy_function(array), rtol=rtol, equal_nan=True)
