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

import math

import warp as wp

from warp_nn.initializers import ones, zeros
from warp_nn.modules.module import Module
from warp_nn.modules.parameter import Parameter
from warp_nn.utils import contiguous

from ._normalization import Moments, normalize


class GroupNorm(Module):
    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        *,
        eps: float = 1e-5,
        affine: bool = True,
        initialize_parameters: bool = True,
        requires_grad: bool = True,
    ):
        r"""Apply Group Normalization over the channels of the input.

        .. math::

            \text{GroupNorm}(x) = \frac{x - \text{E}[x]}{\sqrt{\text{Var}[x] + \epsilon}} \, \gamma + \beta

        The channels are separated into ``num_groups`` groups, each containing ``num_channels / num_groups``
        consecutive channels. The mean and the (biased) variance are computed separately for each sample
        over the channels (and all the remaining dimensions) of each group.

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
              - ``(num_channels,)``
              - Per-channel weights. Only if ``affine`` is true
            * - :math:`\beta`
              - ``bias``
              - ``(num_channels,)``
              - Per-channel bias. Only if ``affine`` is true

        The weights are initialized to ones and the bias to zeros.

        |hr|

        :param num_groups: The number of groups to separate the channels into.
        :param num_channels: The number of channels expected in the input.
            It must be divisible by ``num_groups``.
        :param eps: The value added to the variance for numerical stability.
        :param affine: Whether to include learnable per-channel affine parameters.
        :param initialize_parameters: Whether to initialize the parameters with their default/initial values.
            If false, the parameters are left as uninitialized memory, and reading them (e.g. during a forward pass)
            before loading their values (e.g. from a state dictionary) is undefined behavior.
        :param requires_grad: Whether the parameters and the cached output arrays of the module require gradients.

        :raises ValueError: If ``num_channels`` is not divisible by ``num_groups``.
        """
        super().__init__(requires_grad=requires_grad)
        if num_groups <= 0 or num_channels % num_groups:
            raise ValueError(f"num_channels ({num_channels}) must be divisible by num_groups ({num_groups})")
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        # create/register parameters
        self.weight = None
        self.bias = None
        if affine:
            self.weight = self.register_parameter(
                "weight",
                Parameter(
                    wp.empty(shape=(num_channels,), dtype=wp.float32, device=self.device),
                    requires_grad=self.requires_grad,
                ),
            )
            self.bias = self.register_parameter(
                "bias",
                Parameter(
                    wp.empty(shape=(num_channels,), dtype=wp.float32, device=self.device),
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

        :param input: The input array, with shape ``(batch_size, num_channels, *)``.

        :return: The output array, with same shape as the input array.

        :raises ValueError: If the input array's shape is not supported.
        """
        shape = tuple(input.shape)
        if len(shape) < 2 or shape[1] != self.num_channels:
            raise ValueError(
                f"Expected an input array with shape (batch_size, {self.num_channels}, *), got shape {shape}"
            )
        batch_size = shape[0]
        features = math.prod(shape[2:])
        rows = batch_size * self.num_groups
        group_size = self.num_channels // self.num_groups
        # cache output and statistics
        if shape not in self._cache:
            output = wp.empty(shape, dtype=wp.float32, device=self.device, requires_grad=self.requires_grad)
            moments = Moments(
                rows, group_size * features, centered=True, device=self.device, requires_grad=self.requires_grad
            )
            self._cache[shape] = (output, moments)
        output, moments = self._cache[shape]
        # launch kernels
        input = contiguous(input)
        moments.compute(input.reshape((1, rows, group_size * features)), device=self.device)
        view_shape = (batch_size, self.num_channels, features)
        normalize(
            input.reshape(view_shape),
            moments.mean,
            moments.var,
            weight=self.weight.data if self.weight else None,
            bias=self.bias.data if self.bias else None,
            eps=self.eps,
            batch_stride=self.num_groups,
            group_size=group_size,
            output=output.reshape(view_shape),
            device=self.device,
        )
        return output
