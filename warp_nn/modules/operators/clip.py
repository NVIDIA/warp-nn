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

from __future__ import annotations

from typing import Any

import warp as wp

from warp_nn.modules.module import Module
from warp_nn.utils import KernelConfig, get_kernel_config, overload_kernels, resolve_dim


def _create_kernels(config: KernelConfig, *, min_val: float, max_val: float):
    @wp.func
    def clip(x: Any):
        # min(max_val, max(x, min_val)), where NaN inputs are propagated.
        # The inclusive comparisons zero the gradient at the bounds (as PyTorch does)
        y = x
        if y <= x.dtype(wp.static(min_val)):
            y = x.dtype(wp.static(min_val))
        if y >= x.dtype(wp.static(max_val)):
            y = x.dtype(wp.static(max_val))
        return y

    @wp.kernel
    def kernel_1d(input: wp.array1d[Any], output: wp.array1d[Any]):
        i = wp.tid()
        shape = (wp.static(config.tile_1d[0]),)
        offset = (i * wp.static(config.tile_1d[0]),)
        tile = wp.tile_map(wp.static(clip), wp.tile_load(input, shape=shape, offset=offset))
        wp.tile_store(output, tile, offset=offset)

    @wp.kernel
    def kernel_2d(input: wp.array2d[Any], output: wp.array2d[Any]):
        i, j = wp.tid()
        shape = (wp.static(config.tile_2d[0]), wp.static(config.tile_2d[1]))
        offset = (i * wp.static(config.tile_2d[0]), j * wp.static(config.tile_2d[1]))
        tile = wp.tile_map(wp.static(clip), wp.tile_load(input, shape=shape, offset=offset))
        wp.tile_store(output, tile, offset=offset)

    @wp.kernel
    def kernel_3d(input: wp.array3d[Any], output: wp.array3d[Any]):
        i, j, k = wp.tid()
        shape = (wp.static(config.tile_3d[0]), wp.static(config.tile_3d[1]), wp.static(config.tile_3d[2]))
        offset = (i * wp.static(config.tile_3d[0]), j * wp.static(config.tile_3d[1]), k * wp.static(config.tile_3d[2]))
        tile = wp.tile_map(wp.static(clip), wp.tile_load(input, shape=shape, offset=offset))
        wp.tile_store(output, tile, offset=offset)

    return overload_kernels(kernels=[kernel_1d, kernel_2d, kernel_3d])


class Clip(Module):
    def __init__(
        self, min_val: float | None = None, max_val: float | None = None, *, requires_grad: bool = True
    ) -> None:
        r"""Element-wise clipping (clamping) of the input values into the interval [``min_val``, ``max_val``].

        .. math::

            \text{Clip}(x) = \min(\text{max\_val}, \max(x, \text{min\_val}))

        When ``min_val`` is greater than ``max_val``, all the values are set to ``max_val``.

        PyTorch's ``torch.nn.Hardtanh(min_val, max_val)`` is equivalent to ``Clip(min_val, max_val)``.

        :param min_val: The lower bound of the interval. If None, the values are not bounded below.
        :param max_val: The upper bound of the interval. If None, the values are not bounded above.
        :param requires_grad: Whether the cached output arrays of the module require gradients.
        """
        super().__init__(requires_grad=requires_grad)
        # the values are converted to Python floats, since they are embedded in the kernels
        self._min_val = None if min_val is None else float(min_val)
        self._max_val = None if max_val is None else float(max_val)
        # runtime variables
        self._cache = {}
        self._config = get_kernel_config()
        self._kernels = _create_kernels(
            self._config,
            min_val=float("-inf") if min_val is None else self._min_val,
            max_val=float("inf") if max_val is None else self._max_val,
        )

    @property
    def min_val(self) -> float | None:
        """The lower bound of the interval."""
        return self._min_val

    @property
    def max_val(self) -> float | None:
        """The upper bound of the interval."""
        return self._max_val

    def __call__(self, input: wp.array) -> wp.array:
        """Forward pass of the operation.

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
