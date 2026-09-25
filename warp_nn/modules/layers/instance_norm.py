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

import warp as wp

from .group_norm import GroupNorm


class InstanceNorm(GroupNorm):
    def __init__(
        self,
        num_features: int,
        *,
        eps: float = 1e-5,
        affine: bool = False,
        initialize_parameters: bool = True,
        requires_grad: bool = True,
    ):
        r"""Apply Instance Normalization over the spatial dimensions of the input.

        .. math::

            \text{InstanceNorm}(x) = \frac{x - \text{E}[x]}{\sqrt{\text{Var}[x] + \epsilon}} \, \gamma + \beta

        The mean and the (biased) variance are computed separately for each sample and channel
        over the remaining (spatial) dimensions. It is equivalent to a group normalization
        with as many groups as channels.

        .. note::

            Unlike PyTorch's ``torch.nn.InstanceNorm*d``, running statistics are not supported:
            the statistics are always computed from the input data.

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
              - ``(num_features,)``
              - Per-channel weights. Only if ``affine`` is true
            * - :math:`\beta`
              - ``bias``
              - ``(num_features,)``
              - Per-channel bias. Only if ``affine`` is true

        The weights are initialized to ones and the bias to zeros.

        |hr|

        :param num_features: The number of channels expected in the input.
        :param eps: The value added to the variance for numerical stability.
        :param affine: Whether to include learnable per-channel affine parameters.
        :param initialize_parameters: Whether to initialize the parameters with their default/initial values.
            If false, the parameters are left as uninitialized memory, and reading them (e.g. during a forward pass)
            before loading their values (e.g. from a state dictionary) is undefined behavior.
        :param requires_grad: Whether the parameters and the cached output arrays of the module require gradients.
        """
        super().__init__(
            num_features,
            num_features,
            eps=eps,
            affine=affine,
            initialize_parameters=initialize_parameters,
            requires_grad=requires_grad,
        )
        self.num_features = num_features

    def __call__(self, input: wp.array) -> wp.array:
        """Forward pass of the module.

        :param input: The input array, with shape ``(batch_size, num_features, *)``, with at least 3 dimensions.

        :return: The output array, with same shape as the input array.

        :raises ValueError: If the input array's shape is not supported.
        """
        if input.ndim < 3:
            raise ValueError(
                f"Expected an input array with shape (batch_size, {self.num_features}, *) "
                f"with at least 3 dimensions, got shape {tuple(input.shape)}"
            )
        return super().__call__(input)
