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


class BinaryOpKind(Enum):
    Add = auto()
    And = auto()  # unsupported: Warp tiles do not support bool elements
    BitwiseAnd = auto()
    BitwiseOr = auto()
    BitwiseXor = auto()
    Div = auto()
    Equal = auto()  # unsupported: Warp tiles do not support bool elements
    Greater = auto()  # unsupported: Warp tiles do not support bool elements
    GreaterOrEqual = auto()  # unsupported: Warp tiles do not support bool elements
    Less = auto()  # unsupported: Warp tiles do not support bool elements
    LessOrEqual = auto()  # unsupported: Warp tiles do not support bool elements
    Mul = auto()
    Or = auto()  # unsupported: Warp tiles do not support bool elements
    Pow = auto()
    PRelu = auto()
    Sub = auto()
    Xor = auto()  # unsupported: Warp tiles do not support bool elements


# Every operation is a user function (even if it only calls a builtin), since tile_map can't differentiate
# builtins whose adjoint requires the forward output (e.g. wp.pow).


@wp.func
def _add(a: Any, b: Any):
    return a + b


@wp.func
def _bitwise_and(a: Any, b: Any):
    return a & b


@wp.func
def _bitwise_or(a: Any, b: Any):
    return a | b


@wp.func
def _bitwise_xor(a: Any, b: Any):
    return a ^ b


@wp.func
def _div(a: Any, b: Any):
    return a / b


@wp.func
def _mul(a: Any, b: Any):
    return a * b


@wp.func
def _pow(a: Any, b: Any):
    return wp.pow(a, b)


@wp.func
def _prelu(a: Any, b: Any):
    # a: input, b: slope
    if a < a.dtype(0.0):
        return b * a
    return a


@wp.func
def _sub(a: Any, b: Any):
    return a - b


_FLOAT_DTYPES = [wp.float16, wp.float32, wp.float64]
_INT_DTYPES = [wp.int8, wp.int16, wp.int32, wp.int64, wp.uint8, wp.uint16, wp.uint32, wp.uint64]

_OPERATIONS = {
    BinaryOpKind.Add: (_add, _FLOAT_DTYPES),
    BinaryOpKind.BitwiseAnd: (_bitwise_and, _INT_DTYPES),
    BinaryOpKind.BitwiseOr: (_bitwise_or, _INT_DTYPES),
    BinaryOpKind.BitwiseXor: (_bitwise_xor, _INT_DTYPES),
    BinaryOpKind.Div: (_div, _FLOAT_DTYPES),
    BinaryOpKind.Mul: (_mul, _FLOAT_DTYPES),
    BinaryOpKind.Pow: (_pow, _FLOAT_DTYPES),
    BinaryOpKind.PRelu: (_prelu, _FLOAT_DTYPES),
    BinaryOpKind.Sub: (_sub, _FLOAT_DTYPES),
}


def _create_kernels(config: KernelConfig, *, op_kind: BinaryOpKind):
    function, dtypes = _OPERATIONS[op_kind]
    # integer operations are not differentiable, and CUDA lacks the 8/16-bit atomics their adjoints would require
    enable_backward = all(wp.types.type_is_float(dtype) for dtype in dtypes)
    # unique modules: each operation is compiled independently, and only for the launched dimensionality.
    # wp.static(function) makes Warp hash the function (and the functions it calls) to specialize the kernels

    @wp.kernel(enable_backward=enable_backward, module="unique")
    def kernel_1d(a: wp.array1d[Any], b: wp.array1d[Any], output: wp.array1d[Any]):
        i = wp.tid()
        shape = (wp.static(config.tile_1d[0]),)
        offset = (i * wp.static(config.tile_1d[0]),)
        tile = wp.tile_map(
            wp.static(function),
            wp.tile_load(a, shape=shape, offset=offset),
            wp.tile_load(b, shape=shape, offset=offset),
        )
        wp.tile_store(output, tile, offset=offset)

    @wp.kernel(enable_backward=enable_backward, module="unique")
    def kernel_2d(a: wp.array2d[Any], b: wp.array2d[Any], output: wp.array2d[Any]):
        i, j = wp.tid()
        shape = (wp.static(config.tile_2d[0]), wp.static(config.tile_2d[1]))
        offset = (i * wp.static(config.tile_2d[0]), j * wp.static(config.tile_2d[1]))
        tile = wp.tile_map(
            wp.static(function),
            wp.tile_load(a, shape=shape, offset=offset),
            wp.tile_load(b, shape=shape, offset=offset),
        )
        wp.tile_store(output, tile, offset=offset)

    @wp.kernel(enable_backward=enable_backward, module="unique")
    def kernel_3d(a: wp.array3d[Any], b: wp.array3d[Any], output: wp.array3d[Any]):
        i, j, k = wp.tid()
        shape = (wp.static(config.tile_3d[0]), wp.static(config.tile_3d[1]), wp.static(config.tile_3d[2]))
        offset = (i * wp.static(config.tile_3d[0]), j * wp.static(config.tile_3d[1]), k * wp.static(config.tile_3d[2]))
        tile = wp.tile_map(
            wp.static(function),
            wp.tile_load(a, shape=shape, offset=offset),
            wp.tile_load(b, shape=shape, offset=offset),
        )
        wp.tile_store(output, tile, offset=offset)

    return overload_kernels(kernels=[kernel_1d, kernel_2d, kernel_3d], dtypes=dtypes, num_arrays=3)


class BinaryOp(Module):
    def __init__(self, op_kind: BinaryOpKind, *, requires_grad: bool = True) -> None:
        r"""Element-wise binary operation.

        This class applies an (attribute-free) binary operation element-wise to two input arrays.
        ``BitwiseAnd``, ``BitwiseOr`` and ``BitwiseXor`` operate on integer arrays,
        while all other operations operate on floating-point arrays.

        .. note::

            ``And``, ``Equal``, ``Greater``, ``GreaterOrEqual``, ``Less``, ``LessOrEqual``, ``Or`` and ``Xor``
            are not supported, since Warp tiles do not support boolean elements.

        .. note::

            Broadcasting is not supported: both input arrays must have the same shape and data type.

        :param op_kind: The binary operation to apply.
        :param requires_grad: Whether the cached output arrays of the module require gradients.
            It only applies to floating-point outputs.

        :raises NotImplementedError: If the binary operation is not supported.
        """
        super().__init__(requires_grad=requires_grad)
        if op_kind not in _OPERATIONS:
            raise NotImplementedError(
                f"Binary operation '{op_kind.name}' is not supported (Warp tiles do not support bool elements)"
            )
        self._op_kind = op_kind
        # runtime variables
        self._cache = {}
        self._config = get_kernel_config()
        self._kernels = _create_kernels(self._config, op_kind=op_kind)

    @property
    def op_kind(self) -> BinaryOpKind:
        """The binary operation applied by the module."""
        return self._op_kind

    def __call__(self, a: wp.array, b: wp.array) -> wp.array:
        """Forward pass of the binary operation.

        :param a: The first input array, with up to 3 dimensions (the input ``X`` for ``Pow`` and ``PRelu``).
        :param b: The second input array, with the same shape and data type as the first input array
            (the exponent ``Y`` for ``Pow``, and the ``slope`` for ``PRelu``).

        :return: The output array, with same shape as the input arrays.

        :raises ValueError: If the input arrays' shapes differ.
        :raises TypeError: If the input arrays' data types differ, or if their data type or number of dimensions
            is not supported by the operation.
        """
        dtype = a.dtype
        shape = tuple(a.shape)
        if tuple(b.shape) != shape:
            raise ValueError(
                f"Binary operation '{self._op_kind.name}' requires input arrays of the same shape "
                f"(got {shape} and {tuple(b.shape)})"
            )
        if b.dtype != dtype:
            raise TypeError(
                f"Binary operation '{self._op_kind.name}' requires input arrays of the same data type "
                f"(got {dtype.__name__} and {b.dtype.__name__})"
            )
        key = (shape, dtype)
        differentiable = wp.types.type_is_float(dtype)  # integer operations are not differentiable
        # get kernel
        try:
            kernel = self._kernels[(len(shape), dtype)]
        except KeyError:
            supported = ", ".join(t.__name__ for t in _OPERATIONS[self._op_kind][1])
            raise TypeError(
                f"Binary operation '{self._op_kind.name}' does not support {len(shape)}D arrays of "
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
            inputs=[a, b],
            outputs=[output],
            device=self.device,
            block_dim=self._config.block_dim,
            record_tape=differentiable,
        )
        return output
