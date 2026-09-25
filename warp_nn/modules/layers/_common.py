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


def tile_gemm_2d(shape: tuple[int, int], residual_dim: int | None = None):
    (d0, d1) = shape
    residual_dim = d1 if residual_dim is None else residual_dim

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
    residual_dim = d1 if residual_dim is None else residual_dim

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
