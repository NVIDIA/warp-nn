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
from warp_nn.modules.buffer import Buffer
from warp_nn.modules.module import Module
from warp_nn.modules.parameter import Parameter
from warp_nn.utils import contiguous

from ._normalization import Moments, normalize, update_running_stats_kernel


class BatchNorm(Module):
    def __init__(
        self,
        num_features: int,
        *,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        initialize_parameters: bool = True,
        requires_grad: bool = True,
    ):
        r"""Apply Batch Normalization over the channels of the input.

        .. math::

            \text{BatchNorm}(x) = \frac{x - \text{E}[x]}{\sqrt{\text{Var}[x] + \epsilon}} \, \gamma + \beta

        The mean and the (biased) variance are computed separately for each channel
        over the batch (and all the remaining dimensions).

        In training mode (see :py:meth:`~warp_nn.modules.module.Module.train`), the statistics are computed
        from the input data and, if ``track_running_stats`` is true, their running estimates are updated
        (using the unbiased variance) as follows:

        .. math::

            \hat{x}_{\text{running}} = (1 - \text{momentum}) \, \hat{x}_{\text{running}} + \text{momentum} \, x_t

        In evaluation mode, the running estimates are used for normalization if ``track_running_stats`` is true.
        Otherwise, the statistics are computed from the input data.

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

        Buffers (only if ``track_running_stats`` is true):

        .. list-table::
            :header-rows: 1

            * - Name
              - Shape
              - Description
            * - ``running_mean``
              - ``(num_features,)``
              - Running mean, initialized to zeros
            * - ``running_var``
              - ``(num_features,)``
              - Running (unbiased) variance, initialized to ones

        |hr|

        :param num_features: The number of channels expected in the input.
        :param eps: The value added to the variance for numerical stability.
        :param momentum: The weight of the current statistics in the running estimates update.
        :param affine: Whether to include learnable per-channel affine parameters.
        :param track_running_stats: Whether to track running estimates of the mean and variance.
        :param initialize_parameters: Whether to initialize the parameters with their default/initial values.
            If false, the parameters are left as uninitialized memory, and reading them (e.g. during a forward pass)
            before loading their values (e.g. from a state dictionary) is undefined behavior.
            The buffers are always initialized.
        :param requires_grad: Whether the parameters and the cached output arrays of the module require gradients.
        """
        super().__init__(requires_grad=requires_grad)
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.track_running_stats = track_running_stats
        # create/register parameters
        self.weight = None
        self.bias = None
        if affine:
            self.weight = self.register_parameter(
                "weight",
                Parameter(
                    wp.empty(shape=(num_features,), dtype=wp.float32, device=self.device),
                    requires_grad=self.requires_grad,
                ),
            )
            self.bias = self.register_parameter(
                "bias",
                Parameter(
                    wp.empty(shape=(num_features,), dtype=wp.float32, device=self.device),
                    requires_grad=self.requires_grad,
                ),
            )
        # create/register buffers
        self.running_mean = None
        self.running_var = None
        if track_running_stats:
            self.running_mean = self.register_buffer(
                "running_mean", Buffer(wp.zeros(num_features, dtype=wp.float32, device=self.device))
            )
            self.running_var = self.register_buffer(
                "running_var", Buffer(wp.ones(num_features, dtype=wp.float32, device=self.device))
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

        :param input: The input array, with shape ``(batch_size, num_features, *)``.

        :return: The output array, with same shape as the input array.

        :raises ValueError: If the input array's shape is not supported,
            or if there is only one value per channel in training mode.
        """
        shape = tuple(input.shape)
        if len(shape) < 2 or shape[1] != self.num_features:
            raise ValueError(
                f"Expected an input array with shape (batch_size, {self.num_features}, *), got shape {shape}"
            )
        view_shape = (shape[0], self.num_features, math.prod(shape[2:]))
        count = view_shape[0] * view_shape[2]  # number of values per channel
        if self.training and count <= 1:
            raise ValueError(f"Expected more than 1 value per channel when training, got input shape {shape}")
        # cache output (and batch statistics, only allocated when used)
        if shape not in self._cache:
            output = wp.empty(shape, dtype=wp.float32, device=self.device, requires_grad=self.requires_grad)
            self._cache[shape] = (output, None)
        output, moments = self._cache[shape]
        input_view = contiguous(input).reshape(view_shape)
        # compute (and track) the batch statistics, or use the running estimates
        if self.training or not self.track_running_stats:
            if moments is None:
                moments = Moments(
                    self.num_features, count, centered=True, device=self.device, requires_grad=self.requires_grad
                )
                self._cache[shape] = (output, moments)
            moments.compute(input_view, device=self.device)
            mean, var = moments.mean, moments.var
            if self.training and self.track_running_stats:
                wp.launch(
                    update_running_stats_kernel,
                    dim=self.num_features,
                    inputs=[mean, var, self.momentum, count],
                    outputs=[self.running_mean.data, self.running_var.data],
                    device=self.device,
                    record_tape=False,
                )
        else:
            mean, var = self.running_mean.data, self.running_var.data
        # launch kernel
        normalize(
            input_view,
            mean,
            var,
            weight=self.weight.data if self.weight else None,
            bias=self.bias.data if self.bias else None,
            eps=self.eps,
            batch_stride=0,
            group_size=1,
            output=output.reshape(view_shape),
            device=self.device,
        )
        return output
