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

import math

import warp as wp

from warp_nn.modules.module import Module
from warp_nn.utils import contiguous


# The input is viewed as a 3D array (outer, dim, inner) and reduced along its second dimension.
# The maximum only shifts the exponent (for numerical stability): since softmax is shift-invariant, it is excluded
# from the tape and treated as a constant, which yields the exact gradient.
# The maximum and the sum of exponentials (and the computations involving them) use the accumulation data type,
# which is single precision for half-precision inputs, to avoid losing the small terms of the sum.


@wp.kernel(grid_stride=False, enable_backward=False)
def _max_kernel(input: wp.array3d[Any], maximum: wp.array3d[Any]):
    i, k = wp.tid()
    value = input[i, 0, k]
    for j in range(1, input.shape[1]):
        value = wp.max(value, input[i, j, k])
    maximum[i, 0, k] = maximum.dtype(value)


@wp.kernel(grid_stride=False)
def _sum_exp_kernel(input: wp.array3d[Any], maximum: wp.array3d[Any], total: wp.array3d[Any]):
    i, k = wp.tid()
    m = maximum[i, 0, k]
    value = m.dtype(0.0)
    for j in range(input.shape[1]):
        value += wp.exp(m.dtype(input[i, j, k]) - m)
    total[i, 0, k] = value


# data type -> accumulation data type
ACCUMULATION_DTYPES = {wp.float16: wp.float32, wp.float32: wp.float32, wp.float64: wp.float64}


def overload_kernels(kernel: wp.Kernel, *, num_accumulation_arrays: int, has_output: bool) -> dict[type, wp.Kernel]:
    # arguments: input array, accumulation arrays and (optionally) output array
    return {
        dtype: wp.overload(
            kernel,
            [wp.array3d(dtype=dtype)]
            + [wp.array3d(dtype=accumulation_dtype)] * num_accumulation_arrays
            + ([wp.array3d(dtype=dtype)] if has_output else []),
        )
        for dtype, accumulation_dtype in ACCUMULATION_DTYPES.items()
    }


_MAX_KERNELS = overload_kernels(_max_kernel, num_accumulation_arrays=1, has_output=False)
_SUM_EXP_KERNELS = overload_kernels(_sum_exp_kernel, num_accumulation_arrays=2, has_output=False)


class SoftmaxBase(Module):
    """Shared implementation for Softmax-like activations that reduce along a single dimension.

    Subclasses provide ``output_kernels``, keyed by the entries of ``ACCUMULATION_DTYPES`` and built with
    ``overload_kernels(..., has_output=True)``, with each kernel taking ``(input, maximum, total)`` and writing
    to an ``output`` array.
    """

    def __init__(self, *, dim: int, output_kernels: dict[type, wp.Kernel], requires_grad: bool) -> None:
        super().__init__(requires_grad=requires_grad)
        self._dim = dim
        # runtime variables
        self._output_kernels = output_kernels

    @property
    def dim(self) -> int:
        """The dimension along which the function is computed."""
        return self._dim

    def __call__(self, input: wp.array) -> wp.array:
        """Forward pass of the activation function.

        :param input: The input array.

        :return: The output array, with same shape as the input array.

        :raises IndexError: If the dimension is out of range for the input array.
        :raises TypeError: If the input array's data type is not supported.
        """
        dtype = input.dtype
        shape = tuple(input.shape)
        ndim = len(shape)
        if not -ndim <= self._dim < ndim:
            raise IndexError(f"Dimension {self._dim} is out of range for a {ndim}D input array")
        if dtype not in ACCUMULATION_DTYPES:
            supported = ", ".join(t.__name__ for t in ACCUMULATION_DTYPES)
            raise TypeError(f"Unsupported data type {dtype.__name__} (supported: {supported})")
        dim = self._dim % ndim
        view_shape = (math.prod(shape[:dim]), shape[dim], math.prod(shape[dim + 1 :]))
        key = (shape, dtype)
        # cache output and intermediate arrays
        if key not in self._cache:
            output = wp.empty(shape, dtype=dtype, device=self.device, requires_grad=self.requires_grad)
            reduced_shape = (view_shape[0], 1, view_shape[2])
            accumulation_dtype = ACCUMULATION_DTYPES[dtype]
            maximum = wp.empty(reduced_shape, dtype=accumulation_dtype, device=self.device)
            total = wp.empty(
                reduced_shape, dtype=accumulation_dtype, device=self.device, requires_grad=self.requires_grad
            )
            self._cache[key] = (output, output.reshape(view_shape), maximum, total)
        output, output_view, maximum, total = self._cache[key]
        if output.size == 0:
            return output
        # launch kernels
        input_view = contiguous(input).reshape(view_shape)
        reduced_dim = (view_shape[0], view_shape[2])
        wp.launch(
            _MAX_KERNELS[dtype],
            dim=reduced_dim,
            inputs=[input_view],
            outputs=[maximum],
            device=self.device,
            record_tape=False,
        )
        wp.launch(
            _SUM_EXP_KERNELS[dtype],
            dim=reduced_dim,
            inputs=[input_view, maximum],
            outputs=[total],
            device=self.device,
        )
        wp.launch(
            self._output_kernels[dtype],
            dim=view_shape,
            inputs=[input_view, maximum, total],
            outputs=[output_view],
            device=self.device,
        )
        return output
