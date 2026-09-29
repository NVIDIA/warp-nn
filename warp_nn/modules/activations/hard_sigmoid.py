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

from warp_nn.modules._common import create_unary_kernels
from warp_nn.modules.module import Module
from warp_nn.utils import get_kernel_config, resolve_dim


def _create_function(*, alpha: float, beta: float):
    @wp.func
    def function(x: Any):
        y = x.dtype(wp.static(alpha)) * x + x.dtype(wp.static(beta))
        if y <= x.dtype(0.0):
            return x.dtype(0.0)
        if y >= x.dtype(1.0):
            return x.dtype(1.0)
        return y

    return function


class HardSigmoid(Module):
    def __init__(self, *, alpha: float = 1.0 / 6.0, beta: float = 0.5, requires_grad: bool = True) -> None:
        r"""Hard Sigmoid activation function.

        This class computes the element-wise Hard Sigmoid activation function:

        .. math::

            \text{HardSigmoid}(x) = \max(0, \min(1, \alpha \, x + \beta))

        .. note::

            The default values match PyTorch's ``Hardsigmoid``
            (the default ``alpha`` value of the ONNX ``HardSigmoid`` operator is 0.2).

        :param alpha: The slope of the linear region.
        :param beta: The offset of the linear region.
        :param requires_grad: Whether the cached output arrays of the module require gradients.
        """
        super().__init__(requires_grad=requires_grad)
        self._alpha = float(alpha)
        self._beta = float(beta)
        # runtime variables
        self._config = get_kernel_config()
        self._kernels = create_unary_kernels(
            config=self._config, function=_create_function(alpha=self._alpha, beta=self._beta)
        )

    @property
    def alpha(self) -> float:
        """The slope of the linear region."""
        return self._alpha

    @property
    def beta(self) -> float:
        """The offset of the linear region."""
        return self._beta

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
