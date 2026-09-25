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

from typing import Any, Literal

import warp as wp

from warp_nn.modules.module import Module
from warp_nn.utils import KernelConfig, get_kernel_config, overload_kernels, resolve_dim


@wp.func
def _gelu(x: Any):
    # 0.5 * x * (1 + erf(x / sqrt(2))), rewritten with erfc to avoid the cancellation for negative x
    return x.dtype(0.5) * x * wp.erfc(x * x.dtype(-0.7071067811865476))


@wp.func
def _gelu_tanh(x: Any):
    # 0.5 * x * (1 + tanh(z)) with z = sqrt(2 / pi) * (x + 0.044715 * x^3), which saturates (to x or 0) for |x| > 10.
    # Saturating explicitly avoids overflowing x^3 (in half precision), which yields NaN gradients
    if x > x.dtype(10.0):
        return x
    if x < x.dtype(-10.0):
        return x * x.dtype(0.0)
    # 0.5 * (1 + tanh(z)) = sigmoid(2 z), computed without cancellation (nor overflowing exp()) for negative z
    z = x.dtype(1.5957691216057308) * (x + x.dtype(0.044715) * x * x * x)
    if z < x.dtype(0.0):
        e = wp.exp(z)
        return x * e / (x.dtype(1.0) + e)
    return x / (x.dtype(1.0) + wp.exp(-z))


_FUNCTIONS = {"none": _gelu, "tanh": _gelu_tanh}


def _create_kernels(config: KernelConfig, *, approximate: str):
    activation = _FUNCTIONS[approximate]

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


class GELU(Module):
    def __init__(self, *, approximate: Literal["none", "tanh"] = "none", requires_grad: bool = True) -> None:
        r"""Gaussian Error Linear Unit (GELU) activation function.

        This class computes the element-wise GELU activation function:

        .. math::

            \text{GELU}(x) = x \, \Phi(x) = \frac{x}{2} \left(1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right)

        where :math:`\Phi(x)` is the cumulative distribution function of the standard normal distribution.
        If ``approximate`` is ``"tanh"``, the function is estimated with:

        .. math::

            \text{GELU}(x) \approx \frac{x}{2} \left(1 + \tanh\left(\sqrt{2 / \pi} \, (x + 0.044715 \, x^3)\right)\right)

        :param approximate: The approximation algorithm to use: ``"none"`` (exact computation) or ``"tanh"``.
        :param requires_grad: Whether the cached output arrays of the module require gradients.

        :raises ValueError: If the approximation algorithm is not supported.
        """
        super().__init__(requires_grad=requires_grad)
        if approximate not in _FUNCTIONS:
            raise ValueError(f"Unsupported GELU approximation '{approximate}' (supported: 'none', 'tanh')")
        self._approximate = approximate
        # runtime variables
        self._cache = {}
        self._config = get_kernel_config()
        self._kernels = _create_kernels(self._config, approximate=self._approximate)

    @property
    def approximate(self) -> str:
        """The approximation algorithm used by the GELU function."""
        return self._approximate

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
