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

from warp_nn.modules._common import create_binary_kernels
from warp_nn.modules.module import Module
from warp_nn.utils import get_kernel_config, resolve_dim


def _create_function():
    @wp.func
    def function(x: Any, slope: Any):
        if x < x.dtype(0.0):
            return slope * x
        return x

    return function


class PReLU(Module):
    def __init__(self, *, requires_grad: bool = True) -> None:
        r"""Parametric ReLU (PReLU) operation.

        This class computes the element-wise Parametric Rectified Linear Unit (PReLU) function:

        .. math::

            \text{PReLU}(x, \text{slope}) = \begin{cases}
                x, & \text{ if } x \geq 0\\
                \text{slope} \, x, & \text{ if } x < 0
            \end{cases}

        Unlike PyTorch's ``torch.nn.PReLU`` layer (with a learnable slope parameter), the slope is an input array
        with the same shape as the input array.

        :param requires_grad: Whether the cached output arrays of the module require gradients.
        """
        super().__init__(requires_grad=requires_grad)
        # runtime variables
        self._cache = {}
        self._config = get_kernel_config()
        self._kernels = create_binary_kernels(config=self._config, function=_create_function())

    def __call__(self, x: wp.array, slope: wp.array) -> wp.array:
        """Forward pass of the operation.

        :param x: The input array, with up to 3 dimensions.
        :param slope: The (element-wise) slope array for negative inputs, with the same shape and data type as ``x``.

        :return: The output array, with same shape as the input arrays.

        :raises ValueError: If the input arrays' shapes differ.
        :raises TypeError: If the input arrays' data types differ, or if their data type or number of dimensions
            is not supported.
        """
        dtype = x.dtype
        shape = tuple(x.shape)
        if tuple(slope.shape) != shape:
            raise ValueError(f"The input arrays must have the same shape (got {shape} and {tuple(slope.shape)})")
        if slope.dtype != dtype:
            raise TypeError(
                f"The input arrays must have the same data type (got {dtype.__name__} and {slope.dtype.__name__})"
            )
        key = (shape, dtype)
        kernel = self._kernels[(len(shape), dtype)]
        # cache output
        if key not in self._cache:
            self._cache[key] = wp.empty(shape, dtype=dtype, device=self.device, requires_grad=self.requires_grad)
        output = self._cache[key]
        # launch kernel
        wp.launch_tiled(
            kernel,
            dim=resolve_dim(config=self._config, shape=shape, tiled=True),
            inputs=[x, slope],
            outputs=[output],
            device=self.device,
            block_dim=self._config.block_dim,
        )
        return output
