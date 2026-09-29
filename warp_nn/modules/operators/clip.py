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

from warp_nn.modules._common import create_unary_kernels
from warp_nn.modules.module import Module
from warp_nn.utils import get_kernel_config, resolve_dim


def _create_function(*, min_val: float, max_val: float):
    @wp.func
    def function(x: Any):
        # min(max_val, max(x, min_val)), where NaN inputs are propagated.
        # The inclusive comparisons zero the gradient at the bounds (as PyTorch does)
        y = x
        if y <= x.dtype(wp.static(min_val)):
            y = x.dtype(wp.static(min_val))
        if y >= x.dtype(wp.static(max_val)):
            y = x.dtype(wp.static(max_val))
        return y

    return function


class Clip(Module):
    def __init__(
        self, min_val: float | None = None, max_val: float | None = None, *, requires_grad: bool = True
    ) -> None:
        r"""Element-wise clipping (clamping) of the input values into the interval [``min_val``, ``max_val``].

        .. math::

            \text{Clip}(x) = \min(\text{max\_val}, \max(x, \text{min\_val}))

        When ``min_val`` is greater than ``max_val``, all the values are set to ``max_val``.

        PyTorch's ``Hardtanh(min_val, max_val)`` is equivalent to ``Clip(min_val, max_val)``.

        :param min_val: The lower bound of the interval. If None, the values are not bounded below.
        :param max_val: The upper bound of the interval. If None, the values are not bounded above.
        :param requires_grad: Whether the cached output arrays of the module require gradients.
        """
        super().__init__(requires_grad=requires_grad)
        # the values are converted to Python floats, since they are embedded in the kernels
        self._min_val = None if min_val is None else float(min_val)
        self._max_val = None if max_val is None else float(max_val)
        # runtime variables
        self._config = get_kernel_config()
        self._kernels = create_unary_kernels(
            config=self._config,
            function=_create_function(
                min_val=float("-inf") if min_val is None else self._min_val,
                max_val=float("inf") if max_val is None else self._max_val,
            ),
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

        :raises TypeError: If the input array's data type or number of dimensions is not supported.
        """
        dtype = input.dtype
        shape = tuple(input.shape)
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
            inputs=[input],
            outputs=[output],
            device=self.device,
            block_dim=self._config.block_dim,
        )
        return output
