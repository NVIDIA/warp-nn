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


_TORCH_OPERATIONS = {
    nn.BinaryOpKind.Add: torch.add,
    nn.BinaryOpKind.Div: torch.div,
    nn.BinaryOpKind.Mul: torch.mul,
    nn.BinaryOpKind.Pow: torch.pow,
    nn.BinaryOpKind.PRelu: lambda x, slope: torch.where(x < 0, slope * x, x),
    nn.BinaryOpKind.Sub: torch.sub,
}

_NUMPY_OPERATIONS = {
    nn.BinaryOpKind.BitwiseAnd: np.bitwise_and,
    nn.BinaryOpKind.BitwiseOr: np.bitwise_or,
    nn.BinaryOpKind.BitwiseXor: np.bitwise_xor,
}

# map the sampled inputs, in [-1, 1), to the operation's domain
_DOMAINS = {
    nn.BinaryOpKind.Div: (lambda a: a, lambda b: b + 1.5),
    nn.BinaryOpKind.Pow: (lambda a: a + 1.5, lambda b: 2.0 * b),
}

_FLOAT_OPERATIONS = list(_TORCH_OPERATIONS)
_UNSUPPORTED_OPERATIONS = [op for op in nn.BinaryOpKind if op not in _TORCH_OPERATIONS and op not in _NUMPY_OPERATIONS]


def _sample_inputs(op_kind, *, ndim, dtype):
    np_dtype = utilities.parse_dtype("numpy", dtype)
    domain_a, domain_b = _DOMAINS.get(op_kind, (lambda a: a, lambda b: b))
    a = domain_a(utilities.sample_array(shape=[10] * ndim)).astype(np_dtype)
    b = domain_b(utilities.sample_array(shape=[10] * ndim)).astype(np_dtype)
    return a, b


# module-specific parameters
@pytest.mark.parametrize("op_kind", _FLOAT_OPERATIONS, ids=lambda op: op.name)
# test-specific parameters
@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("dtype", [wp.float16, wp.float32, wp.float64])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_forward(capsys, device, dtype, ndim, op_kind):
    if not is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    binary_op = nn.BinaryOp(op_kind).to(device)
    assert binary_op.op_kind == op_kind
    # create inputs
    a, b = _sample_inputs(op_kind, ndim=ndim, dtype=dtype)
    # forward pass (compute the torch reference in double precision)
    warp_output = binary_op(wp.array(a, device=device), wp.array(b, device=device))
    torch_output = _TORCH_OPERATIONS[op_kind](
        torch.tensor(a, dtype=torch.float64), torch.tensor(b, dtype=torch.float64)
    )
    # check outputs
    assert warp_output.dtype == dtype
    utilities.check_arrays(torch_output, warp_output, atol=1e-2 if dtype == wp.float16 else 1e-3)


# module-specific parameters
@pytest.mark.parametrize("op_kind", _FLOAT_OPERATIONS, ids=lambda op: op.name)
# test-specific parameters
@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("dtype", [wp.float32])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_gradients(capsys, device, dtype, ndim, op_kind):
    if not is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    binary_op = nn.BinaryOp(op_kind).to(device)
    # create inputs
    a, b = _sample_inputs(op_kind, ndim=ndim, dtype=dtype)
    torch_a = torch.tensor(a, dtype=torch.float64, requires_grad=True)
    torch_b = torch.tensor(b, dtype=torch.float64, requires_grad=True)
    warp_a = wp.array(a, device=device, requires_grad=True)
    warp_b = wp.array(b, device=device, requires_grad=True)
    # backward pass (with the gradient of the sum of the outputs)
    # - torch
    _TORCH_OPERATIONS[op_kind](torch_a, torch_b).sum().backward()
    # - warp
    tape = wp.Tape()
    with tape:
        warp_output = binary_op(warp_a, warp_b)
    tape.backward(grads={warp_output: wp.ones_like(warp_output)})
    # check gradients (with respect to both inputs)
    utilities.check_arrays(torch_a.grad, warp_a.grad)
    utilities.check_arrays(torch_b.grad, warp_b.grad)


# module-specific parameters
@pytest.mark.parametrize("op_kind", list(_NUMPY_OPERATIONS), ids=lambda op: op.name)
# test-specific parameters
@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize(
    "dtype",
    [wp.int8, wp.int16, wp.int32, wp.int64, wp.uint8, wp.uint16, wp.uint32, wp.uint64],
    ids=lambda dtype: dtype.__name__,
)
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_bitwise(capsys, device, dtype, ndim, op_kind):
    if not is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    binary_op = nn.BinaryOp(op_kind).to(device)
    # create inputs (covering the full range of the integer type)
    np_dtype = wp.dtype_to_numpy(dtype)
    info = np.iinfo(np_dtype)
    rng = np.random.default_rng(20260925)
    a = rng.integers(info.min, info.max, size=[10] * ndim, dtype=np_dtype, endpoint=True)
    b = rng.integers(info.min, info.max, size=[10] * ndim, dtype=np_dtype, endpoint=True)
    # forward pass
    warp_output = binary_op(wp.array(a, dtype=dtype, device=device), wp.array(b, dtype=dtype, device=device))
    # check outputs (integer outputs don't require gradients, even if the module does)
    assert binary_op.requires_grad
    assert not warp_output.requires_grad
    np.testing.assert_array_equal(warp_output.numpy(), _NUMPY_OPERATIONS[op_kind](a, b))


# module-specific parameters
@pytest.mark.parametrize("requires_grad", [True, False])
# test-specific parameters
@pytest.mark.parametrize("ndim", [2])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_requires_grad(capsys, device, ndim, requires_grad):
    if not is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    binary_op = nn.BinaryOp(nn.BinaryOpKind.Mul, requires_grad=requires_grad).to(device)
    # create inputs
    a, b = _sample_inputs(nn.BinaryOpKind.Mul, ndim=ndim, dtype=wp.float32)
    # forward pass
    warp_output = binary_op(wp.array(a, device=device, requires_grad=True), wp.array(b, device=device))
    # check the flag of the module and of its cached output array
    assert binary_op.requires_grad == requires_grad
    assert warp_output.requires_grad == requires_grad
    assert (warp_output.grad is not None) == requires_grad


# module-specific parameters
@pytest.mark.parametrize("op_kind", _UNSUPPORTED_OPERATIONS, ids=lambda op: op.name)
def test_unsupported_operation(capsys, op_kind):
    with pytest.raises(NotImplementedError, match=op_kind.name):
        nn.BinaryOp(op_kind)


# module-specific parameters
@pytest.mark.parametrize(
    "op_kind, dtypes, shapes, exception",
    [
        (nn.BinaryOpKind.Add, (wp.int32, wp.int32), ((2, 2), (2, 2)), TypeError),  # integer arrays for a float op
        (
            nn.BinaryOpKind.BitwiseAnd,
            (wp.float32, wp.float32),
            ((2, 2), (2, 2)),
            TypeError,
        ),  # float arrays for an int op
        (nn.BinaryOpKind.Add, (wp.float32, wp.float32), ((2,) * 4, (2,) * 4), TypeError),  # unsupported dimensions
        (nn.BinaryOpKind.Add, (wp.float32, wp.float64), ((2, 2), (2, 2)), TypeError),  # different data types
        (nn.BinaryOpKind.Add, (wp.float32, wp.float32), ((2, 2), (2, 1)), ValueError),  # different shapes
    ],
    ids=["int-for-float-op", "float-for-int-op", "4d", "mismatched-dtypes", "mismatched-shapes"],
)
def test_unsupported_input(capsys, op_kind, dtypes, shapes, exception):
    binary_op = nn.BinaryOp(op_kind).to("cpu")
    a = wp.zeros(shapes[0], dtype=dtypes[0], device="cpu")
    b = wp.zeros(shapes[1], dtype=dtypes[1], device="cpu")
    with pytest.raises(exception, match=op_kind.name):
        binary_op(a, b)
