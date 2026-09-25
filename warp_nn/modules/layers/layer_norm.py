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

import math

import warp as wp

from warp_nn.initializers import ones, zeros
from warp_nn.modules.module import Module
from warp_nn.modules.parameter import Parameter
from warp_nn.utils import contiguous

from ._normalization import Moments, normalize


class LayerNorm(Module):
    def __init__(
        self,
        normalized_shape: int | tuple[int, ...],
        *,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        initialize_parameters: bool = True,
        requires_grad: bool = True,
    ):
        r"""Apply Layer Normalization over the trailing dimensions of the input.

        .. math::

            \text{LayerNorm}(x) = \frac{x - \text{E}[x]}{\sqrt{\text{Var}[x] + \epsilon}} \, \gamma + \beta

        The mean and the (biased) variance are computed over the last ``len(normalized_shape)`` dimensions.

        |hr|

        Learnable parameters:

        .. list-table::
            :header-rows: 1

            * -
              - Name
              - Shape
              - Description
            * - :math:`\gamma`
              - ``weight``
              - ``normalized_shape``
              - Weights. Only if ``elementwise_affine`` is true
            * - :math:`\beta`
              - ``bias``
              - ``normalized_shape``
              - Bias. Only if ``elementwise_affine`` and ``bias`` are true

        The weights are initialized to ones and the bias to zeros.

        |hr|

        :param normalized_shape: The shape of the trailing dimensions to normalize over.
        :param eps: The value added to the variance for numerical stability.
        :param elementwise_affine: Whether to include learnable per-element affine parameters.
        :param bias: Whether to include a bias term (only if ``elementwise_affine`` is true).
        :param initialize_parameters: Whether to initialize the parameters with their default/initial values.
            If false, the parameters are left as uninitialized memory, and reading them (e.g. during a forward pass)
            before loading their values (e.g. from a state dictionary) is undefined behavior.
        :param requires_grad: Whether the parameters and the cached output arrays of the module require gradients.
        """
        super().__init__(requires_grad=requires_grad)
        self.normalized_shape = (normalized_shape,) if isinstance(normalized_shape, int) else tuple(normalized_shape)
        self.eps = eps
        # create/register parameters
        self.weight = None
        self.bias = None
        if elementwise_affine:
            self.weight = self.register_parameter(
                "weight",
                Parameter(
                    wp.empty(shape=self.normalized_shape, dtype=wp.float32, device=self.device),
                    requires_grad=self.requires_grad,
                ),
            )
            if bias:
                self.bias = self.register_parameter(
                    "bias",
                    Parameter(
                        wp.empty(shape=self.normalized_shape, dtype=wp.float32, device=self.device),
                        requires_grad=self.requires_grad,
                    ),
                )
        # set default/initial values
        if initialize_parameters:
            self._initialize_parameters()
        # runtime variables
        self._cache = {}

    def _initialize_parameters(self):
        if self.weight:
            ones(self.weight.data)
        if self.bias:
            zeros(self.bias.data)

    def __call__(self, input: wp.array) -> wp.array:
        """Forward pass of the module.

        :param input: The input array, with shape ``(*, *normalized_shape)``.

        :return: The output array, with same shape as the input array.

        :raises ValueError: If the trailing dimensions of the input array do not match the normalized shape.
        """
        shape = tuple(input.shape)
        if shape[len(shape) - len(self.normalized_shape) :] != self.normalized_shape:
            raise ValueError(
                f"The trailing dimensions of the input array {shape} must match the normalized shape "
                f"{self.normalized_shape}"
            )
        features = math.prod(self.normalized_shape)
        rows = math.prod(shape) // features
        # cache output and statistics
        if shape not in self._cache:
            output = wp.empty(shape, dtype=wp.float32, device=self.device, requires_grad=self.requires_grad)
            moments = Moments(rows, features, centered=True, device=self.device, requires_grad=self.requires_grad)
            self._cache[shape] = (output, moments)
        output, moments = self._cache[shape]
        # launch kernels
        input = contiguous(input)
        moments.compute(input.reshape((1, rows, features)), device=self.device)
        normalize(
            input.reshape((rows, features, 1)),
            moments.mean,
            moments.var,
            weight=self.weight.data if self.weight else None,
            bias=self.bias.data if self.bias else None,
            eps=self.eps,
            batch_stride=1,
            group_size=features,
            output=output.reshape((rows, features, 1)),
            device=self.device,
        )
        return output
