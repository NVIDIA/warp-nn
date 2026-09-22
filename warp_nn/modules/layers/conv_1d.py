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

from warp_nn.initializers import kaiming_uniform
from warp_nn.modules.module import Module
from warp_nn.modules.parameter import Parameter
from warp_nn.utils import KernelConfig, get_kernel_config, resolve_dim

from ._common import expand_tuple


def _create_kernels(
    config: KernelConfig,
    *,
    include_bias: bool,
    kernel_sizes: tuple[int],
    strides: tuple[int],
    paddings: tuple[int],
    dilations: tuple[int],
    groups: int,
):
    @wp.kernel
    def kernel(
        input: wp.array3d[float],  # (batch_size, in_channels, in_signal_length)
        weight: wp.array3d[float],  # shape: (out_channels, in_channels / groups, kernel_size)
        bias: wp.array2d[float],  # shape: (out_channels, 1)
        output: wp.array3d[float],  # (batch_size, out_channels, out_signal_length)
    ):
        i, j, k = wp.tid()  # batch_size, out_channels and out_signal_length indices
        in_channels = input.shape[1]
        out_channels = output.shape[1]
        in_signal_length = input.shape[2]
        in_channels_per_group = in_channels // wp.static(groups)
        group_index = j // (out_channels // wp.static(groups))
        sum = float(0.0)
        # iterate over input channels per group
        for in_channel_per_group in range(in_channels_per_group):
            in_channel_index = group_index * in_channels_per_group + in_channel_per_group
            # iterate over kernel positions
            for kernel_index in range(wp.static(kernel_sizes[0])):
                # compute input position with stride, padding, and dilation
                position = k * wp.static(strides[0]) - wp.static(paddings[0]) + kernel_index * wp.static(dilations[0])
                # check bounds (padding handled by skipping out-of-bounds positions)
                if position >= 0 and position < in_signal_length:
                    sum += input[i, in_channel_index, position] * weight[j, in_channel_per_group, kernel_index]
        # add bias
        if wp.static(include_bias):
            sum += bias[j, 0]
        output[i, j, k] = sum

    return kernel


class Conv1D(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        initialize_parameters: bool = True,
        requires_grad: bool = True,
    ):
        r"""Apply a 1D convolution.

        .. math::

            \text{Conv1D}(x) = TODO

        |hr|

        Learnable parameters:

        .. list-table::
            :header-rows: 1

            * -
              - Name
              - Shape
              - Description
            * - :math:`W`
              - ``weight``
              - ``(out_channels, in_channels / groups, kernel_size)``
              - Weights
            * - :math:`b`
              - ``bias``
              - ``(out_channels, 1)``
              - Bias. Only if ``bias`` is true

        The parameters are initialized from the uniform distribution :math:`u(-k, k)`
        where :math:`k = \sqrt{\frac{groups}{\text{in\_channels * kernel\_size}}}`

        |hr|

        :param in_channels: The number of input channels.
        :param out_channels: The number of output channels.
        :param kernel_size: The size of the kernel.
        :param stride: The stride of the convolution.
        :param padding: The padding of the convolution.
        :param dilation: The dilation of the convolution.
        :param groups: The number of groups.
            Both, the ``in_channels`` and the ``out_channels`` arguments must be divisible by ``groups``.
        :param bias: Whether to include a bias term.
        :param initialize_parameters: Whether to initialize the parameters with their default/initial values.
            If false, the parameters are left as uninitialized memory, and reading them (e.g. during a forward pass)
            before loading their values (e.g. from a state dictionary) is undefined behavior.
        :param requires_grad: Whether the parameters and the cached output arrays of the module require gradients.
        """
        super().__init__(requires_grad=requires_grad)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = expand_tuple(kernel_size, length=1)
        self.stride = expand_tuple(stride, length=1)
        self.padding = expand_tuple(padding, length=1)
        self.dilation = expand_tuple(dilation, length=1)
        self.groups = groups
        # validate arguments
        self._validate_arguments()
        # create/register parameters
        # - weight
        shape = (self.out_channels, self.in_channels // self.groups, self.kernel_size[0])
        self.weight = self.register_parameter(
            "weight",
            Parameter(wp.empty(shape=shape, dtype=wp.float32, device=self.device), requires_grad=self.requires_grad),
        )
        # - bias
        if bias:
            shape = (self.out_channels, 1)
            self.bias = self.register_parameter(
                "bias",
                Parameter(
                    wp.empty(shape=shape, dtype=wp.float32, device=self.device), requires_grad=self.requires_grad
                ),
            )
        else:
            self.bias = None
        # set default/initial values
        if initialize_parameters:
            self._initialize_parameters()
        # runtime variables
        self._cache = {}
        self._config = get_kernel_config()
        self._kernel = _create_kernels(
            self._config,
            include_bias=bias,
            kernel_sizes=self.kernel_size,
            strides=self.stride,
            paddings=self.padding,
            dilations=self.dilation,
            groups=self.groups,
        )

    def _validate_arguments(self):
        if self.in_channels % self.groups:
            raise ValueError(f"in_channels ({self.in_channels}) must be divisible by groups ({self.groups})")
        if self.out_channels % self.groups:
            raise ValueError(f"out_channels ({self.out_channels}) must be divisible by groups ({self.groups})")

    def _initialize_parameters(self):
        scale = self.groups / (self.in_channels * self.kernel_size[0])
        kaiming_uniform(self.weight.data, mode="scale", scale=scale)
        if self.bias:
            kaiming_uniform(self.bias.data, mode="scale", scale=scale)

    def __call__(self, input: wp.array) -> wp.array:
        r"""Forward pass of the module.

        :param input: The input array, with shape ``(batch_size, in_channels, in_signal_length)``.

        :return: The output array, with shape ``(batch_size, out_channels, out_signal_length)`` where:

        .. math::

            L_{out} =
                \left\lfloor
                    \frac{L_{in} + 2 \, \text{padding} - \text{dilation} \, (\text{kernel\_size} - 1) - 1}{\text{stride}}
                \right\rfloor + 1
        """
        dtype = input.dtype
        in_signal_length = input.shape[2]
        out_signal_length = int(
            (in_signal_length + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1)
            // self.stride[0]
            + 1
        )
        shape = (input.shape[0], self.out_channels, out_signal_length)
        key = (shape, dtype)
        # cache output
        if key not in self._cache:
            self._cache[key] = wp.empty(shape, dtype=dtype, device=self.device, requires_grad=self.requires_grad)
        output = self._cache[key]
        # launch kernel
        wp.launch(
            self._kernel,
            dim=resolve_dim(config=self._config, shape=shape, tiled=False),
            inputs=[input, self.weight.data, self.bias.data if self.bias else None],
            outputs=[output],
            device=self.device,
            block_dim=self._config.block_dim,
        )
        return output
