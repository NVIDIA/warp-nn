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

from typing import Any, Callable

import math

import warp as wp

from warp_nn.utils.config import KernelConfig


@wp.kernel
def _copy_1d(input: wp.array1d[Any], output: wp.array1d[Any]):
    i = wp.tid()
    output[i] = input[i]


@wp.kernel
def _copy_2d(input: wp.array2d[Any], output: wp.array2d[Any]):
    i, j = wp.tid()
    output[i, j] = input[i, j]


@wp.kernel
def _copy_3d(input: wp.array3d[Any], output: wp.array3d[Any]):
    i, j, k = wp.tid()
    output[i, j, k] = input[i, j, k]


@wp.kernel
def _copy_4d(input: wp.array4d[Any], output: wp.array4d[Any]):
    i, j, k, l = wp.tid()
    output[i, j, k, l] = input[i, j, k, l]


_COPY_KERNELS = {1: _copy_1d, 2: _copy_2d, 3: _copy_3d, 4: _copy_4d}


def contiguous(array: wp.array) -> wp.array:
    """Get a contiguous version of an array (e.g. to reshape it).

    Unlike ``wp.array.contiguous()``, the copy of a non-contiguous array is recorded on the active tape (if any),
    so that gradients are propagated to the original array.

    :param array: The array.

    :return: The array itself if it is contiguous, otherwise a contiguous copy of it.
    """
    if array.is_contiguous:
        return array
    output = wp.empty(array.shape, dtype=array.dtype, device=array.device, requires_grad=array.requires_grad)
    wp.launch(_COPY_KERNELS[array.ndim], dim=array.shape, inputs=[array], outputs=[output], device=array.device)
    return output


def resolve_dim(
    *, config: KernelConfig, shape: tuple[int, ...], tiled: bool, dimensions: int | None = None
) -> tuple[int, ...]:
    if tiled:
        ndim = len(shape)
        if ndim == 1:
            return (math.ceil(shape[0] / config.tile_1d[0]),)
        elif ndim == 2:
            grid = (
                math.ceil(shape[0] / config.tile_2d[0]),
                math.ceil(shape[1] / config.tile_2d[1]),
            )
            return grid if dimensions is None else grid[:dimensions]
        elif ndim == 3:
            grid = (
                math.ceil(shape[0] / config.tile_3d[0]),
                math.ceil(shape[1] / config.tile_3d[1]),
                math.ceil(shape[2] / config.tile_3d[2]),
            )
            return grid if dimensions is None else grid[:dimensions]
        elif ndim == 4:
            grid = (
                math.ceil(shape[0] / config.tile_4d[0]),
                math.ceil(shape[1] / config.tile_4d[1]),
                math.ceil(shape[2] / config.tile_4d[2]),
                math.ceil(shape[3] / config.tile_4d[3]),
            )
            return grid[:3] if dimensions is None else grid[:dimensions]  # tiled launch grid must be less than 4D
        else:
            raise ValueError(f"Unsupported number of dimensions: {ndim}")
    return shape


def overload_kernels(
    *,
    kernels: list[Callable],
    dtypes: tuple[type, ...] | list[type] = (wp.float16, wp.float32, wp.float64),
    num_arrays: int = 2,
) -> dict[tuple[int, type], wp.Kernel]:
    # kernels: one per dimensionality (1D, 2D, ...), whose arguments are `num_arrays` arrays of the same type
    _kernels = {}
    for i, kernel in enumerate(kernels):
        for dtype in dtypes:
            ndim = i + 1
            _kernels[(ndim, dtype)] = wp.overload(kernel, [wp.array(ndim=ndim, dtype=dtype)] * num_arrays)
    return _kernels
