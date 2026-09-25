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

from warp_nn.modules.module import Module
from warp_nn.utils import KernelConfig, get_kernel_config, overload_kernels, resolve_dim


def _create_kernels(config: KernelConfig, *, lambd: float, bias: float):
    @wp.func
    def activation(x: Any):
        if x < x.dtype(wp.static(-lambd)):
            return x + x.dtype(wp.static(bias))
        if x > x.dtype(wp.static(lambd)):
            return x - x.dtype(wp.static(bias))
        return x * x.dtype(0.0)  # zero, or NaN for NaN inputs

    @wp.kernel
    def kernel_1d(input: wp.array1d[Any], output: wp.array1d[Any]):
        i = wp.tid()
        shape = (wp.static(config.tile_1d[0]),)
        offset = (i * wp.static(config.tile_1d[0]),)
        tile = wp.tile_map(wp.static(activation), wp.tile_load(input, shape=shape, offset=offset))
        wp.tile_store(output, tile, offset=offset)

    @wp.kernel
    def kernel_2d(input: wp.array2d[Any], output: wp.array2d[Any]):
        i, j = wp.tid()
        shape = (wp.static(config.tile_2d[0]), wp.static(config.tile_2d[1]))
        offset = (i * wp.static(config.tile_2d[0]), j * wp.static(config.tile_2d[1]))
        tile = wp.tile_map(wp.static(activation), wp.tile_load(input, shape=shape, offset=offset))
        wp.tile_store(output, tile, offset=offset)

    @wp.kernel
    def kernel_3d(input: wp.array3d[Any], output: wp.array3d[Any]):
        i, j, k = wp.tid()
        shape = (wp.static(config.tile_3d[0]), wp.static(config.tile_3d[1]), wp.static(config.tile_3d[2]))
        offset = (i * wp.static(config.tile_3d[0]), j * wp.static(config.tile_3d[1]), k * wp.static(config.tile_3d[2]))
        tile = wp.tile_map(wp.static(activation), wp.tile_load(input, shape=shape, offset=offset))
        wp.tile_store(output, tile, offset=offset)

    return overload_kernels(kernels=[kernel_1d, kernel_2d, kernel_3d])


class Shrink(Module):
    def __init__(self, *, lambd: float = 0.5, bias: float = 0.0, requires_grad: bool = True) -> None:
        r"""Shrink activation function.

        This class computes the element-wise Shrink activation function:

        .. math::

            \text{Shrink}(x) = \begin{cases}
                x + \text{bias}, & \text{ if } x < -\lambda\\
                x - \text{bias}, & \text{ if } x > \lambda\\
                0, & \text{ otherwise}
            \end{cases}

        where

        .. math::

            \lambda = \text{lambd}

        PyTorch's ``torch.nn.Hardshrink(lambd)`` and ``torch.nn.Softshrink(lambd)`` are equivalent to
        ``Shrink(lambd=lambd, bias=0)`` and ``Shrink(lambd=lambd, bias=lambd)``, respectively.

        :param lambd: The lambda value (threshold) for the Shrink function.
        :param bias: The bias value subtracted from (added to) the values above (below) the threshold.
        :param requires_grad: Whether the cached output arrays of the module require gradients.
        """
        super().__init__(requires_grad=requires_grad)
        # the values are converted to Python floats, since they are embedded in the kernels
        self._lambd = float(lambd)
        self._bias = float(bias)
        # runtime variables
        self._cache = {}
        self._config = get_kernel_config()
        self._kernels = _create_kernels(self._config, lambd=self._lambd, bias=self._bias)

    @property
    def lambd(self) -> float:
        """The lambda value (threshold) for the Shrink function."""
        return self._lambd

    @property
    def bias(self) -> float:
        """The bias value subtracted from (added to) the values above (below) the threshold."""
        return self._bias

    def __call__(self, input: wp.array) -> wp.array:
        """Forward pass of the activation function.

        :param input: The input array, with up to 3 dimensions.

        :return: The output array, with same shape as the input array.
        """
        dtype = input.dtype
        shape = tuple(input.shape)
        key = (shape, dtype)
        # cache output
        if key not in self._cache:
            self._cache[key] = wp.empty(shape, dtype=dtype, device=self.device, requires_grad=self.requires_grad)
        output = self._cache[key]
        kernel = self._kernels[(len(shape), dtype)]
        # launch kernel
        wp.launch_tiled(
            kernel,
            dim=resolve_dim(config=self._config, shape=shape, tiled=True),
            inputs=[input],
            outputs=[output],
            device=self.device,
            block_dim=self._config.block_dim,
        )
        return output
