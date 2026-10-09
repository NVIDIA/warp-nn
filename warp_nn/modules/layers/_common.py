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


def expand_tuple(value: int | tuple[int, ...], *, length: int) -> tuple[int, ...]:
    if isinstance(value, (list, tuple)):
        if len(value) != length:
            raise ValueError(f"Expected a tuple of length {length}, got length {len(value)}")
        return tuple(value)
    elif isinstance(value, int):
        return (value,) * length
    raise TypeError(f"Expected a tuple or int, got {type(value)}")


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
        # the products share a single loop (consecutive loops with tile_matmul segfault in the backward
        # pass); the loads of the shorter operand pair beyond its reduction size are zero-padded
        d = wp.max(A1.shape[1], A2.shape[1])
        count = d / residual_dim
        if d % residual_dim:
            count += 1
        C = wp.tile_zeros(shape=(d0, d1), dtype=A1.dtype)
        for k in range(count):
            a1 = wp.tile_load(A1, shape=(d0, residual_dim), offset=(i * d0, k * residual_dim))
            b1 = wp.tile_load(B1, shape=(residual_dim, d1), offset=(k * residual_dim, j * d1))
            wp.tile_matmul(a1, b1, C)
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
        # the products share a single loop (consecutive loops with tile_matmul segfault in the backward
        # pass); the loads of the shorter operand pair beyond its reduction size are zero-padded
        d = wp.max(A1.shape[1], A2.shape[1])
        count = d / residual_dim
        if d % residual_dim:
            count += 1
        C = wp.tile_zeros(shape=(d1, d0), dtype=A1.dtype)
        for k in range(count):
            a1 = wp.tile_load(A1, shape=(d1, residual_dim), offset=(j * d1, k * residual_dim))
            b1 = wp.tile_load(B1, shape=(d0, residual_dim), offset=(i * d0, k * residual_dim))
            wp.tile_matmul(a1, wp.tile_transpose(b1), C)
            a2 = wp.tile_load(A2, shape=(d1, residual_dim), offset=(j * d1, k * residual_dim))
            b2 = wp.tile_load(B2, shape=(d0, residual_dim), offset=(i * d0, k * residual_dim))
            wp.tile_matmul(a2, wp.tile_transpose(b2), C)
        return C

    return function
