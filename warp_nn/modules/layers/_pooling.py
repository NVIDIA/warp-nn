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


# All the kernels operate on 4D (batch, channels, height, width) arrays: 1D pooling uses a height of 1.
# Max pooling first finds the index of the maximum of each window (outside of the tape),
# and then gathers the maximum values, so that the gradient is routed to the selected elements only.


@wp.kernel(enable_backward=False)
def max_pool_indices_kernel(
    input: wp.array4d[float],
    kernel_size: wp.vec2i,
    stride: wp.vec2i,
    padding: wp.vec2i,
    dilation: wp.vec2i,
    indices: wp.array4d[wp.int32],  # flattened (height * width) index of the maximum, or -1 for empty windows
):
    n, c, oh, ow = wp.tid()
    height = input.shape[2]
    width = input.shape[3]
    maximum = float(0.0)
    index = int(-1)
    for kh in range(kernel_size[0]):
        h = oh * stride[0] - padding[0] + kh * dilation[0]
        if h >= 0 and h < height:
            for kw in range(kernel_size[1]):
                w = ow * stride[1] - padding[1] + kw * dilation[1]
                if w >= 0 and w < width:
                    value = input[n, c, h, w]
                    if index < 0 or value > maximum or wp.isnan(value):
                        maximum = value
                        index = h * width + w
    indices[n, c, oh, ow] = index


@wp.kernel
def gather_kernel(input: wp.array4d[float], indices: wp.array4d[wp.int32], output: wp.array4d[float]):
    n, c, oh, ow = wp.tid()
    index = indices[n, c, oh, ow]
    if index >= 0:
        output[n, c, oh, ow] = input[n, c, index // input.shape[3], index % input.shape[3]]
    else:
        output[n, c, oh, ow] = -wp.inf


@wp.kernel
def avg_pool_kernel(
    input: wp.array4d[float],
    kernel_size: wp.vec2i,
    stride: wp.vec2i,
    padding: wp.vec2i,
    count_include_pad: bool,
    output: wp.array4d[float],
):
    n, c, oh, ow = wp.tid()
    height = input.shape[2]
    width = input.shape[3]
    # window bounds, including the padding (but not the overhang of the windows added by the ceil mode)
    h0 = oh * stride[0] - padding[0]
    w0 = ow * stride[1] - padding[1]
    h1 = wp.min(h0 + kernel_size[0], height + padding[0])
    w1 = wp.min(w0 + kernel_size[1], width + padding[1])
    padded_count = (h1 - h0) * (w1 - w0)
    # window bounds, excluding the padding
    h0 = wp.max(h0, 0)
    w0 = wp.max(w0, 0)
    h1 = wp.min(h1, height)
    w1 = wp.min(w1, width)
    total = float(0.0)
    for h in range(h0, h1):
        for w in range(w0, w1):
            total += input[n, c, h, w]
    count = (h1 - h0) * (w1 - w0)
    if count_include_pad:
        count = padded_count
    if h0 < h1 and w0 < w1:
        output[n, c, oh, ow] = total / float(count)
    else:
        output[n, c, oh, ow] = 0.0


def to_vec2i(values: tuple[int, ...], *, fill: int) -> wp.vec2i:
    """Convert the (1D or 2D) arguments of a pooling operation to 2D, since 1D pooling uses a height of 1."""
    return wp.vec2i(*((fill,) * (2 - len(values)) + tuple(values)))


def output_size(size: int, *, kernel_size: int, stride: int, padding: int, dilation: int, ceil_mode: bool) -> int:
    """Compute the output size of a pooling operation along one dimension.

    :raises ValueError: If the output size is smaller than 1.
    """
    numerator = size + 2 * padding - dilation * (kernel_size - 1) - 1
    if ceil_mode:
        result = (numerator + stride - 1) // stride + 1
        # the last window must start inside the input or its left padding
        if (result - 1) * stride >= size + padding:
            result -= 1
    else:
        result = numerator // stride + 1
    if result < 1:
        raise ValueError(
            f"The output size ({result}) of the pooling operation is too small for an input of size {size}, "
            f"kernel size {kernel_size}, stride {stride}, padding {padding} and dilation {dilation}"
        )
    return result


def validate_arguments(
    *, kernel_size: tuple[int, ...], stride: tuple[int, ...], padding: tuple[int, ...], dilation: tuple[int, ...]
) -> None:
    """Validate the arguments of a pooling operation.

    :raises ValueError: If any of the arguments is not valid.
    """
    if any(k <= 0 for k in kernel_size):
        raise ValueError(f"The kernel size must be positive, got {kernel_size}")
    if any(s <= 0 for s in stride):
        raise ValueError(f"The stride must be positive, got {stride}")
    if any(d <= 0 for d in dilation):
        raise ValueError(f"The dilation must be positive, got {dilation}")
    if any(p < 0 or p > k // 2 for p, k in zip(padding, kernel_size)):
        raise ValueError(
            f"The padding {padding} must be non-negative and at most half of the kernel size {kernel_size}"
        )
