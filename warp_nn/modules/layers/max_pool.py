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
from ._pooling import gather_kernel, max_pool_indices_kernel, output_size, to_vec2i, validate_arguments


class _MaxPool(Module):
    def __init__(
        self,
        kernel_size: int | tuple[int, ...],
        *,
        stride: int | tuple[int, ...] | None,
        padding: int | tuple[int, ...],
        dilation: int | tuple[int, ...],
        ceil_mode: bool,
        requires_grad: bool,
        spatial_dims: int,
    ):
        super().__init__(requires_grad=requires_grad)
        self.kernel_size = expand_tuple(kernel_size, length=spatial_dims)
        self.stride = self.kernel_size if stride is None else expand_tuple(stride, length=spatial_dims)
        self.padding = expand_tuple(padding, length=spatial_dims)
        self.dilation = expand_tuple(dilation, length=spatial_dims)
        self.ceil_mode = ceil_mode
        validate_arguments(
            kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation
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
        # cache output and indices
        if shape not in self._cache:
            output_spatial = tuple(
                output_size(size, kernel_size=k, stride=s, padding=p, dilation=d, ceil_mode=self.ceil_mode)
                for size, k, s, p, d in zip(shape[2:], self.kernel_size, self.stride, self.padding, self.dilation)
            )
            output = wp.empty(
                shape[:2] + output_spatial, dtype=wp.float32, device=self.device, requires_grad=self.requires_grad
            )
            view_shape = shape[:2] + view_prefix + output_spatial
            indices = wp.empty(view_shape, dtype=wp.int32, device=self.device)
            self._cache[shape] = (output, output.reshape(view_shape), indices)
        output, output_view, indices = self._cache[shape]
        # launch kernels
        input_view = contiguous(input).reshape(shape[:2] + view_prefix + shape[2:])
        wp.launch(
            max_pool_indices_kernel,
            dim=indices.shape,
            inputs=[
                input_view,
                to_vec2i(self.kernel_size, fill=1),
                to_vec2i(self.stride, fill=1),
                to_vec2i(self.padding, fill=0),
                to_vec2i(self.dilation, fill=1),
            ],
            outputs=[indices],
            device=self.device,
            record_tape=False,
        )
        wp.launch(
            gather_kernel, dim=indices.shape, inputs=[input_view, indices], outputs=[output_view], device=self.device
        )
        return output


class MaxPool1D(_MaxPool):
    def __init__(
        self,
        kernel_size: int | tuple[int],
        *,
        stride: int | tuple[int] | None = None,
        padding: int | tuple[int] = 0,
        dilation: int | tuple[int] = 1,
        ceil_mode: bool = False,
        requires_grad: bool = True,
    ):
        r"""Apply a 1D max pooling.

        The output value for each window is the maximum of the input values in the window.
        The input is implicitly padded with negative infinity values on both sides.

        :param kernel_size: The size of the window.
        :param stride: The stride of the window. If None, it is equal to ``kernel_size``.
        :param padding: The implicit padding added on both sides. It must be at most half of the kernel size.
        :param dilation: The spacing between the elements in the window.
        :param ceil_mode: Whether to use ceil (instead of floor) to compute the output size.
            If true, the last window can extend beyond the (padded) input, as long as it starts inside the input
            or its left padding.
        :param requires_grad: Whether the cached output arrays of the module require gradients.

        :raises ValueError: If any of the arguments is not valid.
        """
        super().__init__(
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            ceil_mode=ceil_mode,
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
                    \frac{L_{in} + 2 \, \text{padding} - \text{dilation} \, (\text{kernel\_size} - 1) - 1}{\text{stride}}
                \right\rfloor + 1

        (using ceil instead of floor if ``ceil_mode`` is true).

        :raises ValueError: If the input array's shape is not supported, or if the output size is too small.
        """
        return super().__call__(input)


class MaxPool2D(_MaxPool):
    def __init__(
        self,
        kernel_size: int | tuple[int, int],
        *,
        stride: int | tuple[int, int] | None = None,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        ceil_mode: bool = False,
        requires_grad: bool = True,
    ):
        r"""Apply a 2D max pooling.

        The output value for each window is the maximum of the input values in the window.
        The input is implicitly padded with negative infinity values on all sides.

        :param kernel_size: The size of the window.
        :param stride: The stride of the window. If None, it is equal to ``kernel_size``.
        :param padding: The implicit padding added on all sides. It must be at most half of the kernel size.
        :param dilation: The spacing between the elements in the window.
        :param ceil_mode: Whether to use ceil (instead of floor) to compute the output size.
            If true, the last windows can extend beyond the (padded) input, as long as they start inside the input
            or its top/left padding.
        :param requires_grad: Whether the cached output arrays of the module require gradients.

        :raises ValueError: If any of the arguments is not valid.
        """
        super().__init__(
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            ceil_mode=ceil_mode,
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
                    \frac{H_{in} + 2 \, \text{padding}[0] - \text{dilation}[0] \, (\text{kernel\_size}[0] - 1) - 1}{\text{stride}[0]}
                \right\rfloor + 1

            W_{out} =
                \left\lfloor
                    \frac{W_{in} + 2 \, \text{padding}[1] - \text{dilation}[1] \, (\text{kernel\_size}[1] - 1) - 1}{\text{stride}[1]}
                \right\rfloor + 1

        (using ceil instead of floor if ``ceil_mode`` is true).

        :raises ValueError: If the input array's shape is not supported, or if the output size is too small.
        """
        return super().__call__(input)
