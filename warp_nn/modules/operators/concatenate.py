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

import warp as wp

from warp_nn.modules._common import INT_DTYPES, create_unary_kernels
from warp_nn.modules.module import Module
from warp_nn.utils import resolve_dim


def _create_function():
    @wp.func
    def function(x: Any):
        return x

    return function


class Concatenate(Module):
    def __init__(self, dim: int = 0, *, requires_grad: bool = True) -> None:
        r"""Concatenation (Concatenate) operation.

        This class concatenates arrays along an existing dimension ``dim``:

        .. math::

            \text{Concatenate}(x_1, \ldots, x_n) = \left[ x_1, \ldots, x_n \right]_{\text{dim}}

        Each input array is copied into its slice of the output array by a kernel launch, so the gradients of
        the output array are propagated to the (floating-point) input arrays. Integer arrays are not differentiable.

        :param dim: The dimension along which the arrays are concatenated.
            Negative values count from the last dimension.
        :param requires_grad: Whether the cached (floating-point) output arrays of the module require gradients.
        """
        super().__init__(requires_grad=requires_grad)
        self._dim = int(dim)
        # runtime variables
        # the integer kernels are created separately, since they are not differentiable
        function = _create_function()
        self._kernels = create_unary_kernels(config=self.config, function=function)
        self._kernels |= create_unary_kernels(config=self.config, function=function, dtypes=INT_DTYPES)

    @property
    def dim(self) -> int:
        """The dimension along which the arrays are concatenated."""
        return self._dim

    def __call__(self, *inputs: wp.array) -> wp.array:
        """Forward pass of the operation.

        :param inputs: The input arrays, with up to 3 dimensions (floating-point or integer). They must have the same number of dimensions
            and data type, and the same shape except in the dimension ``dim``.

        :return: The output array, with the shape of the input arrays except in the dimension ``dim``,
            whose size is the sum of the input arrays' sizes in that dimension.

        :raises ValueError: If no input arrays are given, if ``dim`` is out of range, or if the input arrays'
            shapes differ outside the dimension ``dim``.
        :raises TypeError: If the input arrays' data types differ, or if their data type or number of dimensions
            is not supported.
        """
        if not inputs:
            raise ValueError("At least one input array is required")
        dtype = inputs[0].dtype
        shapes = tuple(tuple(input.shape) for input in inputs)
        shape = shapes[0]
        ndim = len(shape)
        if not -ndim <= self._dim < ndim:
            raise ValueError(f"The dimension {self._dim} is out of range for {ndim}D input arrays")
        dim = self._dim % ndim
        for input, other in zip(inputs[1:], shapes[1:]):
            if len(other) != ndim or other[:dim] + other[dim + 1 :] != shape[:dim] + shape[dim + 1 :]:
                raise ValueError(
                    f"The input arrays must have the same shape, except in dimension {dim} (got {shape} and {other})"
                )
            if input.dtype != dtype:
                raise TypeError(
                    f"The input arrays must have the same data type (got {dtype.__name__} and {input.dtype.__name__})"
                )
        output_shape = shape[:dim] + (sum(other[dim] for other in shapes),) + shape[dim + 1 :]
        key = (output_shape, dtype)
        kernel = self._kernels[(ndim, dtype)]
        # cache output
        output = self.cache.get(
            key,
            lambda: wp.empty(
                output_shape,
                dtype=dtype,
                device=self.device,
                requires_grad=self.requires_grad and wp.types.type_is_float(dtype),
            ),
        )
        # launch kernels (one per input array, which is copied into its slice of the output array)
        offset = 0
        for input, input_shape in zip(inputs, shapes):
            size = input_shape[dim]
            # an empty slice of the output array cannot be indexed
            if input.size:
                index = [slice(None)] * ndim
                index[dim] = slice(offset, offset + size)
                wp.launch_tiled(
                    kernel,
                    dim=resolve_dim(config=self.config, shape=input_shape, tiled=True),
                    inputs=[input],
                    outputs=[output[tuple(index)]],
                    device=self.device,
                    block_dim=self.config.block_dim,
                )
            offset += size
        return output
