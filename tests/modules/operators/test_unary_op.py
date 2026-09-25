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
from ..activations.common import check_requires_grad


_TORCH_OPERATIONS = {
    nn.UnaryOpKind.Abs: torch.abs,
    nn.UnaryOpKind.Acos: torch.acos,
    nn.UnaryOpKind.Acosh: torch.acosh,
    nn.UnaryOpKind.Asin: torch.asin,
    nn.UnaryOpKind.Asinh: torch.asinh,
    nn.UnaryOpKind.Atan: torch.atan,
    nn.UnaryOpKind.Atanh: torch.atanh,
    nn.UnaryOpKind.Ceil: torch.ceil,
    nn.UnaryOpKind.Cos: torch.cos,
    nn.UnaryOpKind.Cosh: torch.cosh,
    nn.UnaryOpKind.Erf: torch.erf,
    nn.UnaryOpKind.Exp: torch.exp,
    nn.UnaryOpKind.Floor: torch.floor,
    nn.UnaryOpKind.HardSwish: torch.nn.functional.hardswish,
    nn.UnaryOpKind.Identity: lambda x: x,
    nn.UnaryOpKind.Log: torch.log,
    nn.UnaryOpKind.Mish: torch.nn.functional.mish,
    nn.UnaryOpKind.Neg: torch.neg,
    nn.UnaryOpKind.Reciprocal: torch.reciprocal,
    nn.UnaryOpKind.Relu: torch.relu,
    nn.UnaryOpKind.Round: torch.round,
    nn.UnaryOpKind.Sigmoid: torch.sigmoid,
    nn.UnaryOpKind.Sign: torch.sign,
    nn.UnaryOpKind.Sin: torch.sin,
    nn.UnaryOpKind.Sinh: torch.sinh,
    nn.UnaryOpKind.Softplus: torch.nn.functional.softplus,
    nn.UnaryOpKind.Softsign: torch.nn.functional.softsign,
    nn.UnaryOpKind.Sqrt: torch.sqrt,
    nn.UnaryOpKind.Tan: torch.tan,
    nn.UnaryOpKind.Tanh: torch.tanh,
}

# map the sampled inputs, in [-1, 1), to the operation's domain (and to cover its numerically-stable branches)
_DOMAINS = {
    nn.UnaryOpKind.Acos: lambda x: 0.9 * x,
    nn.UnaryOpKind.Acosh: lambda x: x + 2.5,
    nn.UnaryOpKind.Asin: lambda x: 0.9 * x,
    nn.UnaryOpKind.Asinh: lambda x: 3.0 * x,
    nn.UnaryOpKind.Atanh: lambda x: 0.9 * x,
    nn.UnaryOpKind.HardSwish: lambda x: 5.0 * x,  # cover both saturated regions
    nn.UnaryOpKind.Log: lambda x: x + 1.5,
    nn.UnaryOpKind.Mish: lambda x: 5.0 * x,
    nn.UnaryOpKind.Reciprocal: lambda x: x + 1.5,
    nn.UnaryOpKind.Sqrt: lambda x: x + 1.5,
}

_FLOAT_OPERATIONS = list(_TORCH_OPERATIONS)


def _sample_input(op_kind, *, ndim, dtype):
    array = utilities.sample_array(shape=[10] * ndim)
    return _DOMAINS.get(op_kind, lambda x: x)(array).astype(utilities.parse_dtype("numpy", dtype))


# module-specific parameters
@pytest.mark.parametrize("op_kind", _FLOAT_OPERATIONS, ids=lambda op: op.name)
# test-specific parameters
@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("dtype", [wp.float16, wp.float32, wp.float64])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_forward(capsys, device, dtype, ndim, op_kind):
    if not is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    unary_op = nn.UnaryOp(op_kind).to(device)
    assert unary_op.op_kind == op_kind
    # create inputs
    array = _sample_input(op_kind, ndim=ndim, dtype=dtype)
    # forward pass (compute the torch reference in double precision)
    warp_output = unary_op(wp.array(array, device=device))
    torch_output = _TORCH_OPERATIONS[op_kind](torch.tensor(array, dtype=torch.float64))
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
    unary_op = nn.UnaryOp(op_kind).to(device)
    # create inputs
    array = _sample_input(op_kind, ndim=ndim, dtype=dtype)
    torch_input = torch.tensor(array, dtype=torch.float64, requires_grad=True)
    warp_input = wp.array(array, device=device, requires_grad=True)
    # backward pass (with the gradient of the sum of the outputs)
    # - torch
    _TORCH_OPERATIONS[op_kind](torch_input).sum().backward()
    # - warp
    tape = wp.Tape()
    with tape:
        warp_output = unary_op(warp_input)
    tape.backward(grads={warp_output: wp.ones_like(warp_output)})
    # check gradients
    utilities.check_arrays(torch_input.grad, warp_input.grad)


# test-specific parameters
@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize(
    "dtype",
    [wp.int8, wp.int16, wp.int32, wp.int64, wp.uint8, wp.uint16, wp.uint32, wp.uint64],
    ids=lambda dtype: dtype.__name__,
)
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_bitwise_not(capsys, device, dtype, ndim):
    if not is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    unary_op = nn.UnaryOp(nn.UnaryOpKind.BitwiseNot).to(device)
    # create inputs (covering the full range of the integer type)
    np_dtype = wp.dtype_to_numpy(dtype)
    info = np.iinfo(np_dtype)
    array = np.random.default_rng(20260925).integers(
        info.min, info.max, size=[10] * ndim, dtype=np_dtype, endpoint=True
    )
    # forward pass
    warp_output = unary_op(wp.array(array, dtype=dtype, device=device))
    # check outputs (integer outputs don't require gradients, even if the module does)
    assert unary_op.requires_grad
    assert not warp_output.requires_grad
    np.testing.assert_array_equal(warp_output.numpy(), np.invert(array))


# module-specific parameters
@pytest.mark.parametrize("requires_grad", [True, False])
# test-specific parameters
@pytest.mark.parametrize("ndim", [2])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_requires_grad(capsys, device, ndim, requires_grad):
    if not is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    check_requires_grad(
        warp_activation=nn.UnaryOp(nn.UnaryOpKind.Exp, requires_grad=requires_grad),
        device=device,
        ndim=ndim,
        requires_grad=requires_grad,
    )


# module-specific parameters
@pytest.mark.parametrize("op_kind", [nn.UnaryOpKind.IsNaN, nn.UnaryOpKind.Not], ids=lambda op: op.name)
def test_unsupported_operation(capsys, op_kind):
    with pytest.raises(NotImplementedError, match=op_kind.name):
        nn.UnaryOp(op_kind)


# module-specific parameters
@pytest.mark.parametrize(
    "op_kind, dtype, ndim",
    [
        (nn.UnaryOpKind.Exp, wp.int32, 2),  # integer array for a floating-point operation
        (nn.UnaryOpKind.BitwiseNot, wp.float32, 2),  # floating-point array for an integer operation
        (nn.UnaryOpKind.Exp, wp.float32, 4),  # unsupported number of dimensions
    ],
    ids=["int-for-float-op", "float-for-int-op", "4d"],
)
def test_unsupported_input(capsys, op_kind, dtype, ndim):
    unary_op = nn.UnaryOp(op_kind).to("cpu")
    with pytest.raises(TypeError, match=op_kind.name):
        unary_op(wp.zeros([2] * ndim, dtype=dtype, device="cpu"))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_hard_swish_boundaries(capsys, device):
    if not is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    # the values and gradients at (and around) the boundaries of the saturated regions match PyTorch
    array = np.array([-np.inf, -3.5, -3.0, -2.5, 2.5, 3.0, 3.5], dtype=np.float32)
    torch_input = torch.tensor(array, requires_grad=True)
    torch_output = torch.nn.functional.hardswish(torch_input)
    torch_output.sum().backward()
    warp_input = wp.array(array, device=device, requires_grad=True)
    tape = wp.Tape()
    with tape:
        warp_output = nn.UnaryOp(nn.UnaryOpKind.HardSwish).to(device)(warp_input)
    tape.backward(grads={warp_output: wp.ones_like(warp_output)})
    # PyTorch yields NaN for -inf (-inf * 0), while the saturated region yields 0
    np.testing.assert_allclose(warp_output.numpy()[1:], torch_output.detach().numpy()[1:], rtol=1e-6)
    assert warp_output.numpy()[0] == 0.0
    np.testing.assert_allclose(warp_input.grad.numpy(), torch_input.grad.numpy(), rtol=1e-6)
