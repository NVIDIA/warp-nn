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

from typing import Any, Sequence

import warp as wp

from warp_nn.utils import KernelConfig, overload_kernels


FLOAT_DTYPES = (wp.float16, wp.float32, wp.float64)
INT_DTYPES = (wp.int8, wp.int16, wp.int32, wp.int64, wp.uint8, wp.uint16, wp.uint32, wp.uint64)


# Element-wise operations are user functions (even if they only call a builtin), since tile_map can't differentiate
# builtins whose adjoint requires the forward output (e.g. wp.exp).
#
# The kernels are created in unique modules: each operation is compiled independently, and only for the launched
# dimensionality. wp.static(function) makes Warp hash the function (and the functions it calls, as well as the values
# captured by closure functions via wp.static) to specialize the kernels, whose source code is otherwise identical.
# Integer operations are not differentiable, and CUDA lacks the 8/16-bit atomics their adjoints would require,
# so their backward pass is disabled. Otherwise, it follows the global setting (wp.config.enable_backward).


@wp.func
def log1p(x: Any):
    # log(1 + x), accurate for small |x| (Goldberg's trick, since Warp has no log1p). x must be finite
    w = x.dtype(1.0) + x
    if w == x.dtype(1.0):
        return x
    return x * (wp.log(w) / (w - x.dtype(1.0)))


class Kernels(dict):
    """Kernels indexed by ``(number of dimensions, data type)``, raising TypeError for unsupported inputs."""

    def __missing__(self, key: tuple[int, type]):
        ndim, dtype = key
        supported = ", ".join(dict.fromkeys(supported_dtype.__name__ for _, supported_dtype in self))
        raise TypeError(
            f"Unsupported input: {ndim}D arrays of {dtype.__name__} (supported: 1D to 3D arrays of {supported})"
        )


def create_unary_kernels(
    *, config: KernelConfig, function: wp.Function, dtypes: Sequence[type] = FLOAT_DTYPES
) -> Kernels:
    """Create the tiled kernels that apply an element-wise unary function: ``output = function(input)``.

    :param config: The kernel configuration (tile sizes).
    :param function: The Warp function, with signature ``(x: Any) -> Any``.
    :param dtypes: The supported data types. The kernels are differentiable (unless disabled globally)
        only if all of them are floating-point.

    :return: The kernels (with signature ``(input, output)``), indexed by ``(number of dimensions, data type)``
        for 1D to 3D arrays. Indexing unsupported inputs raises TypeError.
    """
    enable_backward = None if all(wp.types.type_is_float(dtype) for dtype in dtypes) else False

    @wp.kernel(enable_backward=enable_backward, module="unique")
    def kernel_1d(input: wp.array1d[Any], output: wp.array1d[Any]):
        i = wp.tid()
        shape = (wp.static(config.tile_1d[0]),)
        offset = (i * wp.static(config.tile_1d[0]),)
        tile = wp.tile_map(wp.static(function), wp.tile_load(input, shape=shape, offset=offset))
        wp.tile_store(output, tile, offset=offset)

    @wp.kernel(enable_backward=enable_backward, module="unique")
    def kernel_2d(input: wp.array2d[Any], output: wp.array2d[Any]):
        i, j = wp.tid()
        shape = (wp.static(config.tile_2d[0]), wp.static(config.tile_2d[1]))
        offset = (i * wp.static(config.tile_2d[0]), j * wp.static(config.tile_2d[1]))
        tile = wp.tile_map(wp.static(function), wp.tile_load(input, shape=shape, offset=offset))
        wp.tile_store(output, tile, offset=offset)

    @wp.kernel(enable_backward=enable_backward, module="unique")
    def kernel_3d(input: wp.array3d[Any], output: wp.array3d[Any]):
        i, j, k = wp.tid()
        shape = (wp.static(config.tile_3d[0]), wp.static(config.tile_3d[1]), wp.static(config.tile_3d[2]))
        offset = (i * wp.static(config.tile_3d[0]), j * wp.static(config.tile_3d[1]), k * wp.static(config.tile_3d[2]))
        tile = wp.tile_map(wp.static(function), wp.tile_load(input, shape=shape, offset=offset))
        wp.tile_store(output, tile, offset=offset)

    return Kernels(overload_kernels(kernels=[kernel_1d, kernel_2d, kernel_3d], dtypes=dtypes))


def create_binary_kernels(
    *, config: KernelConfig, function: wp.Function, dtypes: Sequence[type] = FLOAT_DTYPES
) -> Kernels:
    """Create the tiled kernels that apply an element-wise binary function: ``output = function(a, b)``.

    :param config: The kernel configuration (tile sizes).
    :param function: The Warp function, with signature ``(a: Any, b: Any) -> Any``.
    :param dtypes: The supported data types. The kernels are differentiable (unless disabled globally)
        only if all of them are floating-point.

    :return: The kernels (with signature ``(a, b, output)``), indexed by ``(number of dimensions, data type)``
        for 1D to 3D arrays. Indexing unsupported inputs raises TypeError.
    """
    enable_backward = None if all(wp.types.type_is_float(dtype) for dtype in dtypes) else False

    @wp.kernel(enable_backward=enable_backward, module="unique")
    def kernel_1d(a: wp.array1d[Any], b: wp.array1d[Any], output: wp.array1d[Any]):
        i = wp.tid()
        shape = (wp.static(config.tile_1d[0]),)
        offset = (i * wp.static(config.tile_1d[0]),)
        tile = wp.tile_map(
            wp.static(function),
            wp.tile_load(a, shape=shape, offset=offset),
            wp.tile_load(b, shape=shape, offset=offset),
        )
        wp.tile_store(output, tile, offset=offset)

    @wp.kernel(enable_backward=enable_backward, module="unique")
    def kernel_2d(a: wp.array2d[Any], b: wp.array2d[Any], output: wp.array2d[Any]):
        i, j = wp.tid()
        shape = (wp.static(config.tile_2d[0]), wp.static(config.tile_2d[1]))
        offset = (i * wp.static(config.tile_2d[0]), j * wp.static(config.tile_2d[1]))
        tile = wp.tile_map(
            wp.static(function),
            wp.tile_load(a, shape=shape, offset=offset),
            wp.tile_load(b, shape=shape, offset=offset),
        )
        wp.tile_store(output, tile, offset=offset)

    @wp.kernel(enable_backward=enable_backward, module="unique")
    def kernel_3d(a: wp.array3d[Any], b: wp.array3d[Any], output: wp.array3d[Any]):
        i, j, k = wp.tid()
        shape = (wp.static(config.tile_3d[0]), wp.static(config.tile_3d[1]), wp.static(config.tile_3d[2]))
        offset = (i * wp.static(config.tile_3d[0]), j * wp.static(config.tile_3d[1]), k * wp.static(config.tile_3d[2]))
        tile = wp.tile_map(
            wp.static(function),
            wp.tile_load(a, shape=shape, offset=offset),
            wp.tile_load(b, shape=shape, offset=offset),
        )
        wp.tile_store(output, tile, offset=offset)

    return Kernels(overload_kernels(kernels=[kernel_1d, kernel_2d, kernel_3d], dtypes=dtypes, num_arrays=3))
