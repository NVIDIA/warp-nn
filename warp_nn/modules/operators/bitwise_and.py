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

from warp_nn.modules._common import INT_DTYPES, create_binary_kernels
from warp_nn.modules.module import Module
from warp_nn.utils import get_kernel_config, resolve_dim


def _create_function():
    @wp.func
    def function(a: Any, b: Any):
        return a & b

    return function


class BitwiseAnd(Module):
    def __init__(self) -> None:
        r"""Bitwise AND (BitwiseAnd) operation.

        This class computes the element-wise bitwise AND of two integer arrays:

        .. math::

            \text{BitwiseAnd}(a, b) = a \mathbin{\&} b

        Integer operations are not differentiable, so the output arrays never require gradients.
        """
        super().__init__(requires_grad=False)
        # runtime variables
        self._cache = {}
        self._config = get_kernel_config()
        self._kernels = create_binary_kernels(config=self._config, function=_create_function(), dtypes=INT_DTYPES)

    def __call__(self, a: wp.array, b: wp.array) -> wp.array:
        """Forward pass of the operation.

        :param a: The first input array, with up to 3 dimensions.
        :param b: The second input array, with the same shape and data type as ``a``.

        :return: The output array, with same shape as the input arrays.

        :raises ValueError: If the input arrays' shapes differ.
        :raises TypeError: If the input arrays' data types differ, or if their data type or number of dimensions
            is not supported.
        """
        dtype = a.dtype
        shape = tuple(a.shape)
        if tuple(b.shape) != shape:
            raise ValueError(f"The input arrays must have the same shape (got {shape} and {tuple(b.shape)})")
        if b.dtype != dtype:
            raise TypeError(
                f"The input arrays must have the same data type (got {dtype.__name__} and {b.dtype.__name__})"
            )
        key = (shape, dtype)
        kernel = self._kernels[(len(shape), dtype)]
        # cache output
        if key not in self._cache:
            self._cache[key] = wp.empty(shape, dtype=dtype, device=self.device)
        output = self._cache[key]
        # launch kernel
        wp.launch_tiled(
            kernel,
            dim=resolve_dim(config=self._config, shape=shape, tiled=True),
            inputs=[a, b],
            outputs=[output],
            device=self.device,
            block_dim=self._config.block_dim,
            record_tape=False,
        )
        return output
