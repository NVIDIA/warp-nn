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

from warp_nn.modules.module import Module
from warp_nn.utils import contiguous

from ._pooling import avg_pool_kernel, gather_kernel, max_pool_indices_kernel


# The spatial dimensions are pooled with a single window over a (batch, channels, 1, spatial size) view of the input.


def _check_input(input: wp.array) -> tuple[tuple[int, ...], tuple[int, int, int, int], wp.vec2i]:
    shape = tuple(input.shape)
    if len(shape) not in (3, 4):
        raise ValueError(f"Expected a 3D or 4D input array with shape (batch_size, channels, *spatial), got {shape}")
    spatial_size = math.prod(shape[2:])
    return shape, (shape[0], shape[1], 1, spatial_size), wp.vec2i(1, spatial_size)


class GlobalAvgPool(Module):
    def __init__(self, *, requires_grad: bool = True):
        """Apply a global average pooling over the spatial dimensions of the input.

        The output value for each sample and channel is the average of all the input values
        in the spatial dimensions.

        :param requires_grad: Whether the cached output arrays of the module require gradients.
        """
        super().__init__(requires_grad=requires_grad)
        # runtime variables
        self._cache = {}

    def __call__(self, input: wp.array) -> wp.array:
        """Forward pass of the module.

        :param input: The input array, with shape ``(batch_size, channels, *spatial)``, with 1 or 2 spatial dimensions.

        :return: The output array, with shape ``(batch_size, channels, 1)`` or ``(batch_size, channels, 1, 1)``.

        :raises ValueError: If the input array's shape is not supported.
        """
        shape, view_shape, kernel_size = _check_input(input)
        # cache output
        if shape not in self._cache:
            output = wp.empty(
                shape[:2] + (1,) * (len(shape) - 2),
                dtype=wp.float32,
                device=self.device,
                requires_grad=self.requires_grad,
            )
            self._cache[shape] = (output, output.reshape(shape[:2] + (1, 1)))
        output, output_view = self._cache[shape]
        # launch kernel
        wp.launch(
            avg_pool_kernel,
            dim=output_view.shape,
            inputs=[contiguous(input).reshape(view_shape), kernel_size, wp.vec2i(1, 1), wp.vec2i(0, 0), True],
            outputs=[output_view],
            device=self.device,
        )
        return output


class GlobalMaxPool(Module):
    def __init__(self, *, requires_grad: bool = True):
        """Apply a global max pooling over the spatial dimensions of the input.

        The output value for each sample and channel is the maximum of all the input values
        in the spatial dimensions.

        :param requires_grad: Whether the cached output arrays of the module require gradients.
        """
        super().__init__(requires_grad=requires_grad)
        # runtime variables
        self._cache = {}

    def __call__(self, input: wp.array) -> wp.array:
        """Forward pass of the module.

        :param input: The input array, with shape ``(batch_size, channels, *spatial)``, with 1 or 2 spatial dimensions.

        :return: The output array, with shape ``(batch_size, channels, 1)`` or ``(batch_size, channels, 1, 1)``.

        :raises ValueError: If the input array's shape is not supported.
        """
        shape, view_shape, kernel_size = _check_input(input)
        # cache output and indices
        if shape not in self._cache:
            output = wp.empty(
                shape[:2] + (1,) * (len(shape) - 2),
                dtype=wp.float32,
                device=self.device,
                requires_grad=self.requires_grad,
            )
            indices = wp.empty(shape[:2] + (1, 1), dtype=wp.int32, device=self.device)
            self._cache[shape] = (output, output.reshape(shape[:2] + (1, 1)), indices)
        output, output_view, indices = self._cache[shape]
        # launch kernels
        input_view = contiguous(input).reshape(view_shape)
        wp.launch(
            max_pool_indices_kernel,
            dim=indices.shape,
            inputs=[input_view, kernel_size, wp.vec2i(1, 1), wp.vec2i(0, 0), wp.vec2i(1, 1)],
            outputs=[indices],
            device=self.device,
            record_tape=False,
        )
        wp.launch(
            gather_kernel, dim=indices.shape, inputs=[input_view, indices], outputs=[output_view], device=self.device
        )
        return output
