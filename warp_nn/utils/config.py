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

from typing import Generator

import contextlib
import dataclasses
import threading


_context = threading.local()  # thread-local storage to handle nested contexts and concurrent access

# [start-default-config]
_BLOCK_DIM = 256
_TILE_1D = (1024,)  # 1024 elements
_TILE_2D = (16, 64)  # 1024 elements
_TILE_3D = (4, 8, 32)  # 1024 elements
_TILE_4D = (2, 4, 8, 16)  # 1024 elements
# [end-default-config]


@dataclasses.dataclass(kw_only=True, frozen=True)
class KernelConfig:
    """Configuration for Warp kernels generation."""

    block_dim: int
    """Maximum number of CUDA thread blocks to use.

    It only has an effect for CUDA kernel launches. If negative or zero, the maximum hardware value will be used.
    """

    tile_1d: tuple[int]
    """Shape when operating with 1D tiles."""

    tile_2d: tuple[int, int]
    """Shape when operating with 2D tiles."""

    tile_3d: tuple[int, int, int]
    """Shape when operating with 3D tiles."""

    tile_4d: tuple[int, int, int, int]
    """Shape when operating with 4D tiles."""


@contextlib.contextmanager
def kernel_config(
    *,
    block_dim: int | None = None,
    tile_1d: tuple[int] | None = None,
    tile_2d: tuple[int, int] | None = None,
    tile_3d: tuple[int, int, int] | None = None,
    tile_4d: tuple[int, int, int, int] | None = None,
) -> Generator[None, None, None]:
    """Context manager that sets a thread-local configuration values.

    Unspecified (``None``) values are inherited from the enclosing context (or the default values).
    """
    # store previous context values
    previous_block_dim = getattr(_context, "block_dim", None)
    previous_tile_1d = getattr(_context, "tile_1d", None)
    previous_tile_2d = getattr(_context, "tile_2d", None)
    previous_tile_3d = getattr(_context, "tile_3d", None)
    previous_tile_4d = getattr(_context, "tile_4d", None)
    # set new context values (keeping the previous ones for the unspecified values)
    try:
        _context.block_dim = previous_block_dim if block_dim is None else block_dim
        _context.tile_1d = previous_tile_1d if tile_1d is None else tile_1d
        _context.tile_2d = previous_tile_2d if tile_2d is None else tile_2d
        _context.tile_3d = previous_tile_3d if tile_3d is None else tile_3d
        _context.tile_4d = previous_tile_4d if tile_4d is None else tile_4d
        yield
    # remove context value or restore previous one if it exists
    finally:
        _context.block_dim = previous_block_dim
        _context.tile_1d = previous_tile_1d
        _context.tile_2d = previous_tile_2d
        _context.tile_3d = previous_tile_3d
        _context.tile_4d = previous_tile_4d


def get_kernel_config() -> KernelConfig:
    """Get the current configuration."""
    # context values are None outside of any context (after exiting the outermost one)
    return KernelConfig(
        block_dim=_BLOCK_DIM if (value := getattr(_context, "block_dim", None)) is None else value,
        tile_1d=_TILE_1D if (value := getattr(_context, "tile_1d", None)) is None else value,
        tile_2d=_TILE_2D if (value := getattr(_context, "tile_2d", None)) is None else value,
        tile_3d=_TILE_3D if (value := getattr(_context, "tile_3d", None)) is None else value,
        tile_4d=_TILE_4D if (value := getattr(_context, "tile_4d", None)) is None else value,
    )
