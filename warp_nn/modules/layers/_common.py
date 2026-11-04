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

import dataclasses
import operator

import warp as wp

from warp_nn.utils import KernelConfig


def expand_tuple(value: int | tuple[int, ...], *, length: int) -> tuple[int, ...]:
    if isinstance(value, (list, tuple)):
        if len(value) != length:
            raise ValueError(f"Expected a tuple of length {length}, got length {len(value)}")
        return tuple(value)
    elif isinstance(value, int):
        return (value,) * length
    raise TypeError(f"Expected a tuple or int, got {type(value)}")


_MIN_TILE_COLUMNS = 8
_MIN_TILE_ROWS = 16


def fit_tile_2d(config: KernelConfig, features: int, batch_size: int, num_sms: int) -> KernelConfig:
    """Fit the 2D tile (and block dimension) to a layer's output features and batch size.

    The tile columns cover the output features, so a tile wider than them only computes out-of-bounds columns
    (e.g. a (16, 64) tile over 8 features wastes 7/8 of the work): the tile width is halved while it still covers the
    features (down to 8 columns) and its rows doubled, keeping its number of elements. The operand tiles of the matrix
    multiplications take shared memory in proportion to the tile's rows plus columns, so the tile is not narrowed
    beyond the configured tile's sum. Then, while the batch is too small to give every SM two blocks, the rows are
    halved (down to 16), and the block dimension is limited to the number of tile elements. A non-positive block
    dimension (the hardware maximum) is kept.
    """
    rows, columns = config.tile_2d
    while columns % 2 == 0 and columns // 2 >= max(features, _MIN_TILE_COLUMNS):
        columns //= 2
        if rows * 2 + columns <= sum(config.tile_2d):
            rows *= 2
    small_batch = False
    while rows % 2 == 0 and rows // 2 >= _MIN_TILE_ROWS and -(-batch_size // rows) < 2 * num_sms:
        rows //= 2
        small_batch = True
    block_dim = config.block_dim
    if block_dim > 0:
        if small_batch:
            block_dim = min(block_dim, rows * columns)
        else:
            elements_per_thread = max(1, config.tile_2d[0] * config.tile_2d[1] // block_dim)
            block_dim = min(block_dim, max(32, rows * columns // elements_per_thread))
    return dataclasses.replace(config, tile_2d=(rows, columns), block_dim=block_dim)


def tiled_kernel(module, features: int, batch_size: int):
    """Return the configuration and kernel of a tiled layer for a batch size (created once per fitted tile)."""
    config = fit_tile_2d(module._config, features, batch_size, wp.get_device(module.device).sm_count)
    key = (config.tile_2d, config.block_dim)
    if key not in module._kernels:
        module._kernels[key] = module._create_kernels(config)
    return config, module._kernels[key]


def reduction_chunk(config: KernelConfig, size: int) -> int:
    """Return the reduction (K) chunk of a tiled layer's matrix multiplications for a reduction size.

    It is the configured tile's chunk (its width, up to :data:`_MAX_RESIDUAL_DIM`), independent of the fitted tile,
    or a smaller power of two (down to 8) that still covers a small reduction size, which would be zero-padded.
    """
    return min(config.tile_2d[1], _MAX_RESIDUAL_DIM, max(8, 1 << (operator.index(size) - 1).bit_length()))


# Warp functions

# Maximum size of the reduction (K) chunk loaded per matrix-multiplication step. The operand tiles of each step have
# ``tile_dim * residual_dim`` elements, so tying the chunk to the tile width makes them (and the shared memory that
# the backward pass keeps for them, for each GEMM in a kernel) grow quadratically with the tile width.
_MAX_RESIDUAL_DIM = 16


def tile_gemm_2d(shape: tuple[int, int], residual_dim: int | None = None):
    (d0, d1) = shape
    residual_dim = min(d1, _MAX_RESIDUAL_DIM) if residual_dim is None else residual_dim

    @wp.func
    def function(
        A: wp.array2d[float],
        B: wp.array2d[float],
        index: tuple[int, int],
    ):
        i, j = index[0], index[1]
        # compute iteration steps
        d = A.shape[1]
        count = d / residual_dim
        if d % residual_dim:
            count += 1
        # C += A @ B
        C = wp.tile_zeros(shape=(d0, d1), dtype=A.dtype)
        for k in range(count):
            a = wp.tile_load(A, shape=(d0, residual_dim), offset=(i * d0, k * residual_dim))
            b = wp.tile_load(B, shape=(residual_dim, d1), offset=(k * residual_dim, j * d1))
            wp.tile_matmul(a, b, C)
        return C

    return function


def tile_transposed_gemm_2d(shape: tuple[int, int], residual_dim: int | None = None):
    (d0, d1) = shape
    residual_dim = min(d1, _MAX_RESIDUAL_DIM) if residual_dim is None else residual_dim

    @wp.func
    def function(
        A: wp.array2d[float],
        B: wp.array2d[float],
        index: tuple[int, int],
    ):
        i, j = index[0], index[1]
        # compute iteration steps
        d = A.shape[1]
        count = d / residual_dim
        if d % residual_dim:
            count += 1
        # C += A @ B
        C = wp.tile_zeros(shape=(d1, d0), dtype=A.dtype)
        for k in range(count):
            a = wp.tile_load(A, shape=(d1, residual_dim), offset=(j * d1, k * residual_dim))
            b = wp.tile_load(B, shape=(d0, residual_dim), offset=(i * d0, k * residual_dim))
            wp.tile_matmul(a, wp.tile_transpose(b), C)
        return C

    return function


def tile_dual_gemm_2d(shape: tuple[int, int], residual_dim: int | None = None):
    """Same as :func:`tile_gemm_2d`, but accumulating two GEMMs (``A1 @ B1`` and ``A2 @ B2``).

    Accumulating both products in a single tile (instead of summing two tiles) saves one tile expression (and its
    shared memory) per call, which matters for kernels with many gates.
    """
    (d0, d1) = shape
    residual_dim = min(d1, _MAX_RESIDUAL_DIM) if residual_dim is None else residual_dim

    @wp.func
    def function(
        A1: wp.array2d[float],
        B1: wp.array2d[float],
        A2: wp.array2d[float],
        B2: wp.array2d[float],
        index: tuple[int, int],
    ):
        i, j = index[0], index[1]
        # the products share a single loop (consecutive loops with tile_matmul segfault in the backward pass),
        # but each one only iterates over its own reduction size (instead of multiplying zero-padded tiles)
        count1 = A1.shape[1] / residual_dim
        if A1.shape[1] % residual_dim:
            count1 += 1
        count2 = A2.shape[1] / residual_dim
        if A2.shape[1] % residual_dim:
            count2 += 1
        C = wp.tile_zeros(shape=(d0, d1), dtype=A1.dtype)
        for k in range(wp.max(count1, count2)):
            if k < count1:
                a1 = wp.tile_load(A1, shape=(d0, residual_dim), offset=(i * d0, k * residual_dim))
                b1 = wp.tile_load(B1, shape=(residual_dim, d1), offset=(k * residual_dim, j * d1))
                wp.tile_matmul(a1, b1, C)
            if k < count2:
                a2 = wp.tile_load(A2, shape=(d0, residual_dim), offset=(i * d0, k * residual_dim))
                b2 = wp.tile_load(B2, shape=(residual_dim, d1), offset=(k * residual_dim, j * d1))
                wp.tile_matmul(a2, b2, C)
        return C

    return function


def tile_transposed_dual_gemm_2d(shape: tuple[int, int], residual_dim: int | None = None):
    """Same as :func:`tile_transposed_gemm_2d`, but accumulating two GEMMs (``A1 @ B1^T`` and ``A2 @ B2^T``).

    Accumulating both products in a single tile (instead of summing two tiles) saves one tile expression (and its
    shared memory) per call, which matters for kernels with many gates.
    """
    (d0, d1) = shape
    residual_dim = min(d1, _MAX_RESIDUAL_DIM) if residual_dim is None else residual_dim

    @wp.func
    def function(
        A1: wp.array2d[float],
        B1: wp.array2d[float],
        A2: wp.array2d[float],
        B2: wp.array2d[float],
        index: tuple[int, int],
    ):
        i, j = index[0], index[1]
        # the products share a single loop (consecutive loops with tile_matmul segfault in the backward pass),
        # but each one only iterates over its own reduction size (instead of multiplying zero-padded tiles)
        count1 = A1.shape[1] / residual_dim
        if A1.shape[1] % residual_dim:
            count1 += 1
        count2 = A2.shape[1] / residual_dim
        if A2.shape[1] % residual_dim:
            count2 += 1
        C = wp.tile_zeros(shape=(d1, d0), dtype=A1.dtype)
        for k in range(wp.max(count1, count2)):
            if k < count1:
                a1 = wp.tile_load(A1, shape=(d1, residual_dim), offset=(j * d1, k * residual_dim))
                b1 = wp.tile_load(B1, shape=(d0, residual_dim), offset=(i * d0, k * residual_dim))
                wp.tile_matmul(a1, wp.tile_transpose(b1), C)
            if k < count2:
                a2 = wp.tile_load(A2, shape=(d1, residual_dim), offset=(j * d1, k * residual_dim))
                b2 = wp.tile_load(B2, shape=(d0, residual_dim), offset=(i * d0, k * residual_dim))
                wp.tile_matmul(a2, wp.tile_transpose(b2), C)
        return C

    return function
