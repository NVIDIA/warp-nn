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

import warp as wp

from warp_nn.modules.module import Module
from warp_nn.utils import contiguous

from ._common import expand_tuple
from ._pooling import avg_pool_kernel, output_size, to_vec2i, validate_arguments


class _AvgPool(Module):
    def __init__(
        self,
        kernel_size: int | tuple[int, ...],
        *,
        stride: int | tuple[int, ...] | None,
        padding: int | tuple[int, ...],
        ceil_mode: bool,
        count_include_pad: bool,
        requires_grad: bool,
        spatial_dims: int,
    ):
        super().__init__(requires_grad=requires_grad)
        self.kernel_size = expand_tuple(kernel_size, length=spatial_dims)
        self.stride = self.kernel_size if stride is None else expand_tuple(stride, length=spatial_dims)
        self.padding = expand_tuple(padding, length=spatial_dims)
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        validate_arguments(
            kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=(1,) * spatial_dims
        )
        # runtime variables
        self._cache = {}

    def __call__(self, input: wp.array) -> wp.array:
        spatial_dims = len(self.kernel_size)
        shape = tuple(input.shape)
        if len(shape) != spatial_dims + 2:
            raise ValueError(
                f"Expected a {spatial_dims + 2}D input array with shape (batch_size, channels, *spatial), "
                f"got shape {shape}"
            )
        # 4D views of the input and output arrays (1D pooling uses a height of 1)
        view_prefix = (1,) * (2 - spatial_dims)
        # cache output
        if shape not in self._cache:
            output_spatial = tuple(
                output_size(size, kernel_size=k, stride=s, padding=p, dilation=1, ceil_mode=self.ceil_mode)
                for size, k, s, p in zip(shape[2:], self.kernel_size, self.stride, self.padding)
            )
            output = wp.empty(
                shape[:2] + output_spatial, dtype=wp.float32, device=self.device, requires_grad=self.requires_grad
            )
            self._cache[shape] = (output, output.reshape(shape[:2] + view_prefix + output_spatial))
        output, output_view = self._cache[shape]
        # launch kernel
        wp.launch(
            avg_pool_kernel,
            dim=output_view.shape,
            inputs=[
                contiguous(input).reshape(shape[:2] + view_prefix + shape[2:]),
                to_vec2i(self.kernel_size, fill=1),
                to_vec2i(self.stride, fill=1),
                to_vec2i(self.padding, fill=0),
                self.count_include_pad,
            ],
            outputs=[output_view],
            device=self.device,
        )
        return output


class AvgPool1D(_AvgPool):
    def __init__(
        self,
        kernel_size: int | tuple[int],
        *,
        stride: int | tuple[int] | None = None,
        padding: int | tuple[int] = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        requires_grad: bool = True,
    ):
        r"""Apply a 1D average pooling.

        The output value for each window is the average of the input values in the window.
        The input is implicitly padded with zeros on both sides.

        :param kernel_size: The size of the window.
        :param stride: The stride of the window. If None, it is equal to ``kernel_size``.
        :param padding: The implicit zero padding added on both sides. It must be at most half of the kernel size.
        :param ceil_mode: Whether to use ceil (instead of floor) to compute the output size.
            If true, the last window can extend beyond the (padded) input, as long as it starts inside the input
            or its left padding.
        :param count_include_pad: Whether to include the zero padding in the averaging calculation.
        :param requires_grad: Whether the cached output arrays of the module require gradients.

        :raises ValueError: If any of the arguments is not valid.
        """
        super().__init__(
            kernel_size,
            stride=stride,
            padding=padding,
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad,
            requires_grad=requires_grad,
            spatial_dims=1,
        )

    def __call__(self, input: wp.array) -> wp.array:
        r"""Forward pass of the module.

        :param input: The input array, with shape ``(batch_size, channels, in_length)``.

        :return: The output array, with shape ``(batch_size, channels, out_length)`` where:

        .. math::

            L_{out} =
                \left\lfloor
                    \frac{L_{in} + 2 \, \text{padding} - \text{kernel\_size}}{\text{stride}}
                \right\rfloor + 1

        (using ceil instead of floor if ``ceil_mode`` is true).

        :raises ValueError: If the input array's shape is not supported, or if the output size is too small.
        """
        return super().__call__(input)


class AvgPool2D(_AvgPool):
    def __init__(
        self,
        kernel_size: int | tuple[int, int],
        *,
        stride: int | tuple[int, int] | None = None,
        padding: int | tuple[int, int] = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        requires_grad: bool = True,
    ):
        r"""Apply a 2D average pooling.

        The output value for each window is the average of the input values in the window.
        The input is implicitly padded with zeros on all sides.

        :param kernel_size: The size of the window.
        :param stride: The stride of the window. If None, it is equal to ``kernel_size``.
        :param padding: The implicit zero padding added on all sides. It must be at most half of the kernel size.
        :param ceil_mode: Whether to use ceil (instead of floor) to compute the output size.
            If true, the last windows can extend beyond the (padded) input, as long as they start inside the input
            or its top/left padding.
        :param count_include_pad: Whether to include the zero padding in the averaging calculation.
        :param requires_grad: Whether the cached output arrays of the module require gradients.

        :raises ValueError: If any of the arguments is not valid.
        """
        super().__init__(
            kernel_size,
            stride=stride,
            padding=padding,
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad,
            requires_grad=requires_grad,
            spatial_dims=2,
        )

    def __call__(self, input: wp.array) -> wp.array:
        r"""Forward pass of the module.

        :param input: The input array, with shape ``(batch_size, channels, in_height, in_width)``.

        :return: The output array, with shape ``(batch_size, channels, out_height, out_width)`` where:

        .. math::

            H_{out} =
                \left\lfloor
                    \frac{H_{in} + 2 \, \text{padding}[0] - \text{kernel\_size}[0]}{\text{stride}[0]}
                \right\rfloor + 1

            W_{out} =
                \left\lfloor
                    \frac{W_{in} + 2 \, \text{padding}[1] - \text{kernel\_size}[1]}{\text{stride}[1]}
                \right\rfloor + 1

        (using ceil instead of floor if ``ceil_mode`` is true).

        :raises ValueError: If the input array's shape is not supported, or if the output size is too small.
        """
        return super().__call__(input)
