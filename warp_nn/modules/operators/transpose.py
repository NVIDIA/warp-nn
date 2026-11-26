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

from collections.abc import Sequence

import warp as wp

from warp_nn.modules._impl.kernel_factory import FLOAT_DTYPES, INT_DTYPES
from warp_nn.modules.module import Module
from warp_nn.utils import copy, overlap


_SUPPORTED_DTYPES = (*FLOAT_DTYPES, *INT_DTYPES, wp.bool)


class Transpose(Module):
    def __init__(self, dims: Sequence[int] | None = None, *, requires_grad: bool = True) -> None:
        r"""Transposition (Transpose) operation.

        This class permutes the dimensions of an array, so that the dimension ``i`` of the output array is the
        dimension ``dims[i]`` of the input array:

        .. math::

            \text{Transpose}(x)_{j_0, \ldots, j_{n-1}} = x_{i_0, \ldots, i_{n-1}}
            \quad \text{with} \quad i_{\text{dims}[k]} = j_k

        The transposed (strided) view of the input array is copied into a contiguous output array by a kernel launch,
        so the gradients of the output array are propagated to the (floating-point) input array. Integer and Boolean
        arrays are not differentiable.

        If the input array is contiguous and the permutation keeps the order of its dimensions larger than 1 (e.g. it
        only moves dimensions of size 1), or the input array is empty, its elements are already in the output order:
        the output array is then a reshaped view of the input array (no data is copied, and no kernel is launched),
        which shares its memory and gradients (and whether it requires gradients) with the input array.

        :param dims: The permutation of the input array's dimensions. Negative values count from the last dimension.
            If not given, the order of the dimensions is reversed.
        :param requires_grad: Whether the cached (floating-point) output arrays of the module require gradients.
            It does not apply to the output arrays that are views of the input array.

        :raises ValueError: If ``dims`` contains duplicated values.
        """
        super().__init__(requires_grad=requires_grad)
        self._dims = None if dims is None else tuple(int(dim) for dim in dims)
        if self._dims is not None and len(set(self._dims)) != len(self._dims):
            raise ValueError(f"The dimensions {self._dims} contain duplicated values")

    @property
    def dims(self) -> tuple[int, ...] | None:
        """The permutation of the input array's dimensions (``None`` if the order of the dimensions is reversed)."""
        return self._dims

    def _resolve_dims(self, ndim: int) -> tuple[int, ...]:
        if self._dims is None:
            return tuple(range(ndim - 1, -1, -1))
        dims = tuple(dim % ndim for dim in self._dims)
        if not all(-ndim <= dim < ndim for dim in self._dims) or sorted(dims) != list(range(ndim)):
            raise ValueError(f"The dimensions {self._dims} are not a permutation of the dimensions of {ndim}D arrays")
        return dims

    def __call__(self, input: wp.array) -> wp.array:
        """Forward pass of the operation.

        :param input: The input array, with up to 4 dimensions (floating-point, integer or Boolean).
            It can be non-contiguous (e.g. a strided view), but it must not share memory with an output array
            of the module (e.g. the output of a previous call).

        :return: The (contiguous) output array, whose dimension ``i`` is the dimension ``dims[i]`` of the input array.
            It is a view of the input array if no data has to be moved, in which case it requires gradients
            if (and only if) the input array does, regardless of the module's ``requires_grad``.

        :raises ValueError: If the input array is not on the module's device, if ``dims`` is not a permutation of the
            input array's dimensions, or if the input array shares memory with the output array.
        :raises TypeError: If the input array's data type is not supported.
        """
        # the copy kernel is launched on the input array's device
        if input.device != self.device:
            raise ValueError(f"The input array is on device {input.device}, expected {self.device}")
        dtype = input.dtype
        shape = input.shape
        if dtype not in _SUPPORTED_DTYPES:
            supported = ", ".join(t.__name__ for t in _SUPPORTED_DTYPES)
            raise TypeError(f"Unsupported data type {dtype.__name__} (supported: {supported})")
        dims = self._resolve_dims(len(shape))
        output_shape = tuple(shape[dim] for dim in dims)
        # the elements are already in the output order (the dimensions of size 1 do not affect it)
        moved = [dim for dim in dims if shape[dim] != 1]
        if input.is_contiguous and (input.size == 0 or moved == sorted(moved)):
            return input.reshape(output_shape)
        # cache output
        output = self.cache.get(
            (output_shape, dtype),
            lambda: wp.empty(
                output_shape,
                dtype=dtype,
                device=self.device,
                requires_grad=self.requires_grad and wp.types.type_is_float(dtype),
            ),
        )
        if not output.size:
            return output
        # the elements would be overwritten before being read
        if overlap(input, output):
            raise ValueError("The input array shares memory with the output array of the module")
        # copy the transposed view (launch kernel)
        return copy(input.transpose(dims), output)
