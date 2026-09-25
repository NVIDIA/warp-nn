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


# The statistics of each group of a 3D (outer, groups, inner) view of the input are computed over its (outer, inner)
# elements, which are split into chunks: the partial sums of the chunks are computed in parallel and then summed,
# which limits the accumulated rounding error. The accumulation loops are linear, so their adjoints are exact.

_CHUNK_SIZE = wp.constant(256)


@wp.kernel
def _partial_sums_kernel(input: wp.array3d[float], partial: wp.array2d[float]):
    b, k = wp.tid()
    inner = input.shape[2]
    start = k * _CHUNK_SIZE
    end = wp.min(start + _CHUNK_SIZE, input.shape[0] * inner)
    total = float(0.0)
    for j in range(start, end):
        total += input[j // inner, b, j % inner]
    partial[b, k] = total


@wp.kernel
def _partial_squares_kernel(
    input: wp.array3d[float], mean: wp.array1d[float], centered: bool, partial: wp.array2d[float]
):
    b, k = wp.tid()
    inner = input.shape[2]
    start = k * _CHUNK_SIZE
    end = wp.min(start + _CHUNK_SIZE, input.shape[0] * inner)
    # read the mean once, so that its adjoint is accumulated locally rather than atomically for each element
    m = float(0.0)
    if centered:
        m = mean[b]
    total = float(0.0)
    for j in range(start, end):
        d = input[j // inner, b, j % inner] - m
        total += d * d
    partial[b, k] = total


@wp.kernel
def _average_kernel(partial: wp.array2d[float], count: int, output: wp.array1d[float]):
    b = wp.tid()
    total = float(0.0)
    for k in range(partial.shape[1]):
        total += partial[b, k]
    output[b] = total / float(count)


class Moments:
    def __init__(self, groups: int, count: int, *, centered: bool, device: wp.Device, requires_grad: bool):
        """Mean and (biased) variance of each group of a 3D (outer, groups, inner) view of an array.

        If not centered, only the variance (i.e. the mean square, for RMS normalization) is computed.

        :param groups: The number of groups.
        :param count: The number of elements per group (i.e. outer * inner).
        :param centered: Whether to compute the mean and subtract it, or to assume a zero mean.
        :param device: The device on which to allocate the arrays.
        :param requires_grad: Whether the arrays require gradients.
        """
        chunks = max(1, math.ceil(count / _CHUNK_SIZE))
        self.count = count
        # separate partial sums for each pass, so that their gradients do not accumulate into each other
        self.mean = None
        self._partial_sums = None
        if centered:
            self.mean = wp.empty(groups, dtype=wp.float32, device=device, requires_grad=requires_grad)
            self._partial_sums = wp.empty(
                (groups, chunks), dtype=wp.float32, device=device, requires_grad=requires_grad
            )
        self.var = wp.empty(groups, dtype=wp.float32, device=device, requires_grad=requires_grad)
        self._partial_squares = wp.empty((groups, chunks), dtype=wp.float32, device=device, requires_grad=requires_grad)

    def compute(self, input: wp.array, *, device: wp.Device) -> None:
        """Compute the statistics.

        :param input: A 3D (outer, groups, inner) view of the input array.
        :param device: The device on which to launch the kernels.
        """
        groups, chunks = self._partial_squares.shape
        if self.mean is not None:
            wp.launch(
                _partial_sums_kernel, dim=(groups, chunks), inputs=[input], outputs=[self._partial_sums], device=device
            )
            wp.launch(
                _average_kernel, dim=groups, inputs=[self._partial_sums, self.count], outputs=[self.mean], device=device
            )
        wp.launch(
            _partial_squares_kernel,
            dim=(groups, chunks),
            inputs=[input, self.mean, self.mean is not None],
            outputs=[self._partial_squares],
            device=device,
        )
        wp.launch(
            _average_kernel, dim=groups, inputs=[self._partial_squares, self.count], outputs=[self.var], device=device
        )


@wp.kernel
def _normalize_kernel(
    input: wp.array3d[float],  # (batch, channels, features)
    mean: wp.array1d[float],
    var: wp.array1d[float],
    weight: wp.array1d[float],  # (channels,)
    bias: wp.array1d[float],  # (channels,)
    eps: float,
    batch_stride: int,
    group_size: int,
    centered: bool,
    use_weight: bool,
    use_bias: bool,
    output: wp.array3d[float],  # (batch, channels, features)
):
    # the statistics of the element (n, c, s) are at index n * batch_stride + c // group_size
    n, c, s = wp.tid()
    i = n * batch_stride + c // group_size
    y = input[n, c, s]
    if centered:
        y = y - mean[i]
    y = y / wp.sqrt(var[i] + eps)
    if use_weight:
        y = y * weight[c]
    if use_bias:
        y = y + bias[c]
    output[n, c, s] = y


@wp.kernel(enable_backward=False)
def update_running_stats_kernel(
    mean: wp.array1d[float],
    var: wp.array1d[float],
    momentum: float,
    count: int,
    running_mean: wp.array1d[float],
    running_var: wp.array1d[float],
):
    # exponential moving average of the mean and the (unbiased) variance
    c = wp.tid()
    unbiased_var = var[c] * float(count) / float(count - 1)
    running_mean[c] = (1.0 - momentum) * running_mean[c] + momentum * mean[c]
    running_var[c] = (1.0 - momentum) * running_var[c] + momentum * unbiased_var


def normalize(
    input: wp.array,
    mean: wp.array | None,
    var: wp.array,
    *,
    weight: wp.array | None,
    bias: wp.array | None,
    eps: float,
    batch_stride: int,
    group_size: int,
    output: wp.array,
    device: wp.Device,
) -> None:
    """Launch the normalization kernel on 3D (batch, channels, features) views of the input and output arrays.

    If the mean is None, the input is not centered (for RMS normalization).
    The weight and bias arrays (if any) are flattened to 1D (channels,) views.
    """
    wp.launch(
        _normalize_kernel,
        dim=input.shape,
        inputs=[
            input,
            mean,
            var,
            None if weight is None else weight.flatten(),
            None if bias is None else bias.flatten(),
            eps,
            batch_stride,
            group_size,
            mean is not None,
            weight is not None,
            bias is not None,
        ],
        outputs=[output],
        device=device,
    )
