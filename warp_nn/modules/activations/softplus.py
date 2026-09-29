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

from warp_nn.modules._common import create_unary_kernels, log1p
from warp_nn.modules.module import Module
from warp_nn.utils import get_kernel_config, resolve_dim


def _create_function(*, beta: float, threshold: float):
    @wp.func
    def function(x: Any):
        b = x.dtype(wp.static(beta))
        bx = b * x
        # revert to the linear function for numerical stability
        if bx > x.dtype(wp.static(threshold)):
            return x
        # log1p(exp(beta * x)) / beta, rewritten for positive beta * x to avoid overflowing exp()
        if bx > x.dtype(0.0):
            return x + log1p(wp.exp(-bx)) / b
        return log1p(wp.exp(bx)) / b

    return function


class Softplus(Module):
    def __init__(self, *, beta: float = 1.0, threshold: float = 20.0, requires_grad: bool = True) -> None:
        r"""Softplus activation function.

        This class computes the element-wise Softplus activation function:

        .. math::

            \text{Softplus}(x) = \frac{1}{\beta} \log(1 + e^{\beta x})

        For numerical stability, the implementation reverts to the linear function when
        :math:`\beta \, x > \text{threshold}`.

        :param beta: The beta value for the Softplus function. It must be non-zero.
        :param threshold: When :math:`\beta \, x` exceeds this value, the output reverts to the linear function.
        :param requires_grad: Whether the cached output arrays of the module require gradients.

        :raises ValueError: If ``beta`` is zero.
        """
        super().__init__(requires_grad=requires_grad)
        if beta == 0.0:
            raise ValueError("The beta value for the Softplus function must be non-zero")
        self._beta = float(beta)
        self._threshold = float(threshold)
        # runtime variables
        self._config = get_kernel_config()
        self._kernels = create_unary_kernels(
            config=self._config, function=_create_function(beta=self._beta, threshold=self._threshold)
        )

    @property
    def beta(self):
        """The beta value for the Softplus function."""
        return self._beta

    @property
    def threshold(self):
        """The threshold value above which the Softplus function reverts to the linear function."""
        return self._threshold

    def __call__(self, input: wp.array) -> wp.array:
        """Forward pass of the activation function.

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
