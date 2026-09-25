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

from enum import Enum, auto

import warp as wp

from warp_nn.modules.module import Module
from warp_nn.utils import KernelConfig, get_kernel_config, overload_kernels, resolve_dim


class UnaryOpKind(Enum):
    Abs = auto()
    Acos = auto()
    Acosh = auto()
    Asin = auto()
    Asinh = auto()
    Atan = auto()
    Atanh = auto()
    BitwiseNot = auto()
    Ceil = auto()
    Cos = auto()
    Cosh = auto()
    Erf = auto()
    Exp = auto()
    Floor = auto()
    HardSwish = auto()
    Identity = auto()
    IsNaN = auto()  # unsupported: Warp tiles do not support bool elements
    Log = auto()
    Mish = auto()
    Neg = auto()
    Not = auto()  # unsupported: Warp tiles do not support bool elements
    Reciprocal = auto()
    Relu = auto()
    Round = auto()
    Sigmoid = auto()
    Sign = auto()
    Sin = auto()
    Sinh = auto()
    Softplus = auto()
    Softsign = auto()
    Sqrt = auto()
    Tan = auto()
    Tanh = auto()


# Every operation is a user function (even if it only calls a builtin), since tile_map can't differentiate
# builtins whose adjoint requires the forward output (e.g. wp.exp).


@wp.func
def _log1p(x: Any):
    # log(1 + x), accurate for small |x| (Goldberg's trick, since Warp has no log1p). x must be finite
    w = x.dtype(1.0) + x
    if w == x.dtype(1.0):
        return x
    return x * (wp.log(w) / (w - x.dtype(1.0)))


@wp.func
def _abs(x: Any):
    return wp.abs(x)


@wp.func
def _acos(x: Any):
    return wp.acos(x)


@wp.func
def _acosh(x: Any):
    # log(x + sqrt(x^2 - 1)), factored to avoid overflowing x^2
    return wp.log(x) + wp.log(x.dtype(1.0) + wp.sqrt(x.dtype(1.0) - x.dtype(1.0) / (x * x)))


@wp.func
def _asin(x: Any):
    return wp.asin(x)


@wp.func
def _asinh(x: Any):
    # sign(x) * log(|x| + sqrt(x^2 + 1)), factored for |x| > 1 to avoid overflowing x^2
    a = wp.abs(x)
    if a > x.dtype(1.0):
        return wp.sign(x) * (wp.log(a) + wp.log(x.dtype(1.0) + wp.sqrt(x.dtype(1.0) + x.dtype(1.0) / (a * a))))
    # sign(x) * log1p(|x| + x^2 / (1 + sqrt(x^2 + 1))), accurate for small |x|
    return wp.sign(x) * _log1p(a + a * a / (x.dtype(1.0) + wp.sqrt(a * a + x.dtype(1.0))))


@wp.func
def _atan(x: Any):
    return wp.atan(x)


@wp.func
def _atanh(x: Any):
    a = wp.abs(x)
    if a < x.dtype(0.5):
        # sign(x) * 0.5 * log1p(2|x| / (1 - |x|)), accurate for small |x|
        return wp.sign(x) * x.dtype(0.5) * _log1p(x.dtype(2.0) * a / (x.dtype(1.0) - a))
    return x.dtype(0.5) * wp.log((x.dtype(1.0) + x) / (x.dtype(1.0) - x))


@wp.func
def _bitwise_not(x: Any):
    return wp.invert(x)


@wp.func
def _ceil(x: Any):
    return wp.ceil(x)


@wp.func
def _cos(x: Any):
    return wp.cos(x)


@wp.func
def _cosh(x: Any):
    return wp.cosh(x)


@wp.func
def _erf(x: Any):
    return wp.erf(x)


@wp.func
def _exp(x: Any):
    return wp.exp(x)


@wp.func
def _floor(x: Any):
    return wp.floor(x)


@wp.func
def _hard_swish(x: Any):
    # x * clamp(x / 6 + 1 / 2, 0, 1), with the saturation boundaries matching PyTorch's gradient
    if x <= x.dtype(-3.0):
        return x.dtype(0.0)
    if x >= x.dtype(3.0):
        return x
    return x * (x / x.dtype(6.0) + x.dtype(0.5))


@wp.func
def _identity(x: Any):
    return x


@wp.func
def _log(x: Any):
    return wp.log(x)


@wp.func
def _mish(x: Any):
    return x * wp.tanh(_softplus(x))


@wp.func
def _neg(x: Any):
    return -x


@wp.func
def _reciprocal(x: Any):
    return x.dtype(1.0) / x


@wp.func
def _relu(x: Any):
    if x <= x.dtype(0.0):  # false for NaN, which is then propagated
        return x.dtype(0.0)
    return x


@wp.func
def _round(x: Any):
    return wp.rint(x)  # ONNX rounds halfway cases to even (unlike wp.round)


@wp.func
def _sigmoid(x: Any):
    # keep the exponent non-positive, since an overflowing exp() yields NaN gradients
    if x < x.dtype(0.0):
        e = wp.exp(x)
        return e / (x.dtype(1.0) + e)
    return x.dtype(1.0) / (x.dtype(1.0) + wp.exp(-x))


@wp.func
def _sign(x: Any):
    # wp.sign returns 1 for 0
    if x > x.dtype(0.0):
        return x.dtype(1.0)
    if x < x.dtype(0.0):
        return x.dtype(-1.0)
    return x * x.dtype(0.0)  # zero, or NaN for NaN inputs


@wp.func
def _sin(x: Any):
    return wp.sin(x)


@wp.func
def _sinh(x: Any):
    return wp.sinh(x)


@wp.func
def _softplus(x: Any):
    # log1p(exp(x)), rewritten for positive x to avoid overflowing exp()
    if x > x.dtype(0.0):
        return x + _log1p(wp.exp(-x))
    return _log1p(wp.exp(x))


@wp.func
def _softsign(x: Any):
    return x / (x.dtype(1.0) + wp.abs(x))


@wp.func
def _sqrt(x: Any):
    return wp.sqrt(x)


@wp.func
def _tan(x: Any):
    return wp.tan(x)


@wp.func
def _tanh(x: Any):
    return wp.tanh(x)


_FLOAT_DTYPES = [wp.float16, wp.float32, wp.float64]
_INT_DTYPES = [wp.int8, wp.int16, wp.int32, wp.int64, wp.uint8, wp.uint16, wp.uint32, wp.uint64]

_OPERATIONS = {
    UnaryOpKind.Abs: (_abs, _FLOAT_DTYPES),
    UnaryOpKind.Acos: (_acos, _FLOAT_DTYPES),
    UnaryOpKind.Acosh: (_acosh, _FLOAT_DTYPES),  # no Warp builtin
    UnaryOpKind.Asin: (_asin, _FLOAT_DTYPES),
    UnaryOpKind.Asinh: (_asinh, _FLOAT_DTYPES),  # no Warp builtin
    UnaryOpKind.Atan: (_atan, _FLOAT_DTYPES),
    UnaryOpKind.Atanh: (_atanh, _FLOAT_DTYPES),  # no Warp builtin
    UnaryOpKind.BitwiseNot: (_bitwise_not, _INT_DTYPES),
    UnaryOpKind.Ceil: (_ceil, _FLOAT_DTYPES),
    UnaryOpKind.Cos: (_cos, _FLOAT_DTYPES),
    UnaryOpKind.Cosh: (_cosh, _FLOAT_DTYPES),
    UnaryOpKind.Erf: (_erf, _FLOAT_DTYPES),
    UnaryOpKind.Exp: (_exp, _FLOAT_DTYPES),
    UnaryOpKind.Floor: (_floor, _FLOAT_DTYPES),
    UnaryOpKind.HardSwish: (_hard_swish, _FLOAT_DTYPES),
    UnaryOpKind.Identity: (_identity, _FLOAT_DTYPES),
    UnaryOpKind.Log: (_log, _FLOAT_DTYPES),
    UnaryOpKind.Mish: (_mish, _FLOAT_DTYPES),
    UnaryOpKind.Neg: (_neg, _FLOAT_DTYPES),
    UnaryOpKind.Reciprocal: (_reciprocal, _FLOAT_DTYPES),
    UnaryOpKind.Relu: (_relu, _FLOAT_DTYPES),
    UnaryOpKind.Round: (_round, _FLOAT_DTYPES),
    UnaryOpKind.Sigmoid: (_sigmoid, _FLOAT_DTYPES),
    UnaryOpKind.Sign: (_sign, _FLOAT_DTYPES),
    UnaryOpKind.Sin: (_sin, _FLOAT_DTYPES),
    UnaryOpKind.Sinh: (_sinh, _FLOAT_DTYPES),
    UnaryOpKind.Softplus: (_softplus, _FLOAT_DTYPES),
    UnaryOpKind.Softsign: (_softsign, _FLOAT_DTYPES),
    UnaryOpKind.Sqrt: (_sqrt, _FLOAT_DTYPES),
    UnaryOpKind.Tan: (_tan, _FLOAT_DTYPES),
    UnaryOpKind.Tanh: (_tanh, _FLOAT_DTYPES),
}


def _create_kernels(config: KernelConfig, *, op_kind: UnaryOpKind):
    function, dtypes = _OPERATIONS[op_kind]
    # integer operations are not differentiable, and CUDA lacks the 8/16-bit atomics their adjoints would require
    enable_backward = all(wp.types.type_is_float(dtype) for dtype in dtypes)
    # unique modules: each operation is compiled independently, and only for the launched dimensionality.
    # wp.static(function) makes Warp hash the function (and the functions it calls) to specialize the kernels

    @wp.kernel(enable_backward=enable_backward, module="unique")
    def kernel_1d(input: wp.array1d[Any], output: wp.array1d[Any]):
        i = wp.tid()
        shape = (wp.static(config.tile_1d[0]),)
        offset = (i * wp.static(config.tile_1d[0]),)
        tile = wp.tile_map(wp.static(function), wp.tile_load(input, shape=shape, offset=offset))
        wp.tile_store(output, tile, offset=offset)

    @wp.kernel(enable_backward=enable_backward, module="unique")
    def kernel_2d(input: wp.array2d[Any], output: wp.array2d[Any]):
        i, j = wp.tid()
        shape = (wp.static(config.tile_2d[0]), wp.static(config.tile_2d[1]))
        offset = (i * wp.static(config.tile_2d[0]), j * wp.static(config.tile_2d[1]))
        tile = wp.tile_map(wp.static(function), wp.tile_load(input, shape=shape, offset=offset))
        wp.tile_store(output, tile, offset=offset)

    @wp.kernel(enable_backward=enable_backward, module="unique")
    def kernel_3d(input: wp.array3d[Any], output: wp.array3d[Any]):
        i, j, k = wp.tid()
        shape = (wp.static(config.tile_3d[0]), wp.static(config.tile_3d[1]), wp.static(config.tile_3d[2]))
        offset = (i * wp.static(config.tile_3d[0]), j * wp.static(config.tile_3d[1]), k * wp.static(config.tile_3d[2]))
        tile = wp.tile_map(wp.static(function), wp.tile_load(input, shape=shape, offset=offset))
        wp.tile_store(output, tile, offset=offset)

    return overload_kernels(kernels=[kernel_1d, kernel_2d, kernel_3d], dtypes=dtypes)


class UnaryOp(Module):
    def __init__(self, op_kind: UnaryOpKind, *, requires_grad: bool = True) -> None:
        r"""Element-wise unary operation.

        This class applies an (attribute-free) unary operation element-wise to the input array.
        ``BitwiseNot`` operates on integer arrays, while all other operations operate on floating-point arrays.

        .. note::

            ``IsNaN`` and ``Not`` are not supported, since Warp tiles do not support boolean elements.

        :param op_kind: The unary operation to apply.
        :param requires_grad: Whether the cached output arrays of the module require gradients.
            It only applies to floating-point outputs.

        :raises NotImplementedError: If the unary operation is not supported.
        """
        super().__init__(requires_grad=requires_grad)
        if op_kind not in _OPERATIONS:
            raise NotImplementedError(
                f"Unary operation '{op_kind.name}' is not supported (Warp tiles do not support bool elements)"
            )
        self._op_kind = op_kind
        # runtime variables
        self._cache = {}
        self._config = get_kernel_config()
        self._kernels = _create_kernels(self._config, op_kind=op_kind)

    @property
    def op_kind(self) -> UnaryOpKind:
        """The unary operation applied by the module."""
        return self._op_kind

    def __call__(self, input: wp.array) -> wp.array:
        """Forward pass of the unary operation.

        :param input: The input array, with up to 3 dimensions.

        :return: The output array, with same shape as the input array.

        :raises TypeError: If the input array's data type or number of dimensions is not supported by the operation.
        """
        dtype = input.dtype
        shape = tuple(input.shape)
        key = (shape, dtype)
        differentiable = wp.types.type_is_float(dtype)  # integer operations are not differentiable
        # get kernel
        try:
            kernel = self._kernels[(len(shape), dtype)]
        except KeyError:
            supported = ", ".join(t.__name__ for t in _OPERATIONS[self._op_kind][1])
            raise TypeError(
                f"Unary operation '{self._op_kind.name}' does not support {len(shape)}D arrays of "
                f"{dtype.__name__} (supported: 1D to 3D arrays of {supported})"
            ) from None
        # cache output
        if key not in self._cache:
            requires_grad = self.requires_grad and differentiable
            self._cache[key] = wp.empty(shape, dtype=dtype, device=self.device, requires_grad=requires_grad)
        output = self._cache[key]
        # launch kernel
        wp.launch_tiled(
            kernel,
            dim=resolve_dim(config=self._config, shape=shape, tiled=True),
            inputs=[input],
            outputs=[output],
            device=self.device,
            block_dim=self._config.block_dim,
            record_tape=differentiable,
        )
        return output
