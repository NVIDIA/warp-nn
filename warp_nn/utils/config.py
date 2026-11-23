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

from typing import ClassVar, Generator

import contextlib
import dataclasses
import functools
import math
import operator
import threading

import warp as wp


_context = threading.local()  # thread-local storage to handle nested contexts and concurrent access

# [start-default-config]
_BLOCK_DIM = 256
_TILE_1D = (1024,)  # 1024 elements
_TILE_2D = (16, 64)  # 1024 elements
_TILE_3D = (4, 8, 32)  # 1024 elements
_TILE_4D = (2, 4, 8, 16)  # 1024 elements
_ADAPTIVE = True  # adapt the tiles (and block dimension) to the launched arrays
# [end-default-config]

_WARP_SIZE = 32


def _is_recording() -> bool:
    # whether a wp.Tape is recording the launches (for differentiation). The recording tape is process-wide in Warp
    # (not per thread): a tape recording on a thread affects the launches of the other threads too. Warp has no public
    # API for it: it is set on the (private) runtime by wp.Tape.__enter__/__exit__ (Warp >= 1.15), and the runtime is
    # None before Warp is initialized
    return getattr(wp._src.context.runtime, "tape", None) is not None


def _adapt_tile(
    tile: tuple[int, ...],
    block_dim: int,
    shape: tuple[int, ...],
    num_sms: int,
    *,
    policy: AdaptationPolicy,
    backward: bool = False,
) -> tuple[tuple[int, ...], int]:
    """Adapt a tile (and the block dimension) to the shape of the array it is launched on.

    The adapted tile never has more elements than the configured tile, although single dimensions may grow beyond
    their configured size (e.g. more rows for fewer columns):

    1. **Narrow**: each dimension steps down (see below) while the smaller size still covers the array.
    2. **Grow**: the dimensions that do not cover the array are doubled (last first) while the tile has no more
       elements than the configured tile (and, if limited by the policy, no larger sum of dimensions). Skipped for
       launches whose backward kernels run too, if the policy does not grow them (see
       :attr:`AdaptationPolicy.grow_for_backward`).
    3. **Shrink**: while the grid has fewer blocks than ``policy.blocks_per_sm * num_sms`` (but some, i.e. the array
       is not empty), the dimensions step down (leading ones first), down to ``policy.min_elements``.
    4. **Block dimension**: the configured number of elements per thread (rounded up to a warp), up to the configured
       block dimension, or one thread per tile element if the grid still has fewer blocks than
       ``policy.thread_per_element_blocks_per_sm * num_sms``. Policies with ``thread_per_element_after_shrink`` also
       give one thread per element after shrinking the tile. The block dimension is then raised (never lowered) to the
       next number of whole warps, up to the configured block dimension, that divides the tile elements, if any, so that
       every thread gets the same number of elements. Policies that limit inference threads by columns use a lower block
       dimension cap once the grid meets the block target. Warp may compile a separate kernel for each block dimension
       of a tile.

    A dimension steps down the ladder of its minimum size (see :meth:`AdaptationPolicy.min_tile`) times a power of two,
    to the next smaller size on it (e.g. 40 -> 32 -> 16 -> 8 for a minimum size of 8): the half, for a power of two
    times the minimum size. The sizes reached by stepping down are therefore multiples of the minimum size (e.g. the
    tensor-core MMA shape), and few (one per power of two), so that few configurations are compiled. A configured size
    below its minimum size is its own minimum size.

    :param tile: The configured tile.
    :param block_dim: The configured block dimension. A non-positive one (the hardware maximum) is kept.
    :param shape: The launch shape (the array covered by the tiles), with as many dimensions as the tile.
    :param num_sms: The number of SMs of the device. Zero (e.g. for CPU devices) skips the shrink step and the one
        thread per tile element of small grids, which only serve to keep the SMs busy.
    :param policy: The adaptation policy of the operation (e.g. :attr:`AdaptationPolicy.ELEMENTWISE` or
        :attr:`AdaptationPolicy.MATMUL`).
    :param backward: Whether the backward kernels run on the adapted tile too (the launch is recorded for
        differentiation).

    :return: The adapted tile and block dimension.

    :raises ValueError: If the tile has non-positive dimensions, or not as many dimensions as the launch shape.
    """
    if min(tile) <= 0:
        raise ValueError(f"The tile dimensions must be positive (got {tile})")
    if len(tile) != len(shape):
        raise ValueError(f"The tile {tile} and the launch shape {shape} must have the same number of dimensions")
    ndim = len(tile)
    minimum = [min(low, size) for low, size in zip(policy.min_tile(ndim), tile)]
    max_elements, max_dims_sum = math.prod(tile), sum(tile)
    # 1. narrow to the first ladder size that covers each extent, capped by the configured size.
    adapted = [
        size if extent >= size else min(size, low << operator.index((max(1, extent) - 1) // low).bit_length())
        for size, low, extent in zip(tile, minimum, shape)
    ]
    elements = math.prod(adapted)

    # 2. grow (unless the backward kernels run on the tile too, and the policy does not grow them).
    if not (backward and not policy.grow_for_backward):
        dims_sum = sum(adapted)
        for dim in reversed(range(ndim)):
            while adapted[dim] < shape[dim] and 2 * elements <= max_elements:
                if policy.limit_dims_sum and dims_sum + adapted[dim] > max_dims_sum:
                    break
                dims_sum += adapted[dim]
                adapted[dim] *= 2
                elements *= 2

    # 3. shrink (an empty array has no blocks to provide).
    elements_before_shrink = elements
    grid = [-(-extent // size) for extent, size in zip(shape, adapted)]
    blocks = math.prod(grid)
    target_blocks = policy.blocks_per_sm * num_sms
    for dim, low in enumerate(minimum):
        if not 0 < blocks < target_blocks:
            break
        while 0 < blocks < target_blocks:
            size = adapted[dim]
            steps = (size - 1) // low
            if not steps:
                break
            smaller = low << (operator.index(steps).bit_length() - 1)
            reduced_elements = elements // size * smaller
            if reduced_elements < policy.min_elements:
                break
            count = -(-shape[dim] // smaller)
            blocks = blocks // grid[dim] * count
            grid[dim] = count
            adapted[dim] = smaller
            elements = reduced_elements
    # 4. block dimension
    if block_dim > 0:
        if policy.limit_inference_threads_by_columns and not backward and blocks >= target_blocks > 0:
            # Across a full grid, use fewer warps per block and let each warp process more tile rows.
            column_warps = -(-adapted[-1] // _WARP_SIZE)
            block_dim = min(block_dim, column_warps * _WARP_SIZE)
        few_blocks = 0 < blocks < policy.thread_per_element_blocks_per_sm * num_sms
        shrank = elements < elements_before_shrink
        one_thread_per_element = few_blocks or (policy.thread_per_element_after_shrink and shrank)
        elements_per_thread = 1 if one_thread_per_element else max(1, max_elements // block_dim)
        elements_per_warp = elements_per_thread * _WARP_SIZE
        threads = min(block_dim, -(-elements // elements_per_warp) * _WARP_SIZE)
        # more threads (whole warps, up to the configured block dimension) if they divide the tile elements evenly:
        # every thread then gets the same number of elements, and Warp skips the per-element bounds checks of the
        # register tiles of the adapted tile shape (whose size is then a multiple of the block dimension). The threads
        # are only raised, never lowered, so that the block keeps its parallelism
        if elements % threads:
            threads = next(
                (n for n in range(threads + _WARP_SIZE, block_dim + 1, _WARP_SIZE) if elements % n == 0), threads
            )
        block_dim = threads
    return tuple(adapted), block_dim


@dataclasses.dataclass(frozen=True)
class AdaptationPolicy:
    """Rules under which the kernel configuration (tile and block dimension) of an operation is adapted to its
    launch shapes (see :py:meth:`KernelConfig.adapt`).

    Use the predefined policies (e.g. :py:attr:`AdaptationPolicy.MATMUL`), or derive a policy from them
    (e.g. with :py:func:`dataclasses.replace`).
    """

    ELEMENTWISE: ClassVar[AdaptationPolicy]
    """Element-wise operations: every dimension can be narrowed to the array (they load and store their tiles element
    by element, so a tile row narrower than a vector load costs nothing), and shrunk for occupancy (leading ones
    first). Tiles shrunk to supply more blocks get one thread per element, up to the configured block dimension."""

    MATMUL: ClassVar[AdaptationPolicy]
    """Matrix multiplications, whose tile is ``(batch rows, output features)``: the tile keeps at least 16 rows and
    8 columns (the tensor-core MMA shape), and the rows plus columns (which set the shared memory of the operand
    tiles) are limited to the configured tile's. Small batches shrink the columns too once the rows are at their
    minimum: reloading the few input rows for each narrower column tile costs less than leaving SMs idle. The grid
    targets 4 blocks per SM, while one thread per tile element is kept for grids below 2 blocks per SM. The tile is
    not grown (its rows are kept at the configured 16 by default) for launches recorded for differentiation: their
    backward kernels were fastest with 16 rows (Linear, RNN, GRU and LSTM layers of up to 64 output features,
    batches of 64 to 32768, on an sm_89 GPU)."""

    LINEAR: ClassVar[AdaptationPolicy]
    """Linear layers use the matrix multiplication tile (see :py:attr:`MATMUL`), with one warp per warp-sized span of
    output columns when the inference grid meets the block target. This gives each warp more rows to process per
    block."""

    min_sizes: tuple[int, ...] = ()
    """Minimum sizes of the last tile dimensions. The other dimensions have a minimum size of 1."""

    limit_dims_sum: bool = False
    """Whether the sum of the tile dimensions is limited to the configured tile's too (besides its elements)."""

    blocks_per_sm: int = 2
    """Number of blocks per SM that the grid should provide to keep the GPU busy (the target of the shrink step)."""

    thread_per_element_blocks_per_sm: int = 2
    """Number of blocks per SM below which the block dimension gives one thread per tile element (at most
    :py:attr:`blocks_per_sm`, since the shrink step stops at that target, so a larger value gives one thread per
    element to every grid below it).

    It is independent of :py:attr:`blocks_per_sm`: a grid can be worth shrinking the tile for, while having enough
    blocks for the configured number of elements per thread.
    """

    thread_per_element_after_shrink: bool = False
    """Whether to give one thread per tile element after shrinking the tile to provide more blocks."""

    min_elements: int = 128
    """Minimum number of elements of a tile shrunk to provide more blocks."""

    grow_for_backward: bool = True
    """Whether to grow the tile (see :py:meth:`KernelConfig.adapt`) for launches whose backward kernels run too.

    The backward kernels (the adjoints of the forward kernels) run on the tile of the forward kernels, and may favor a
    different tile than the forward pass alone.
    """

    limit_inference_threads_by_columns: bool = False
    """Whether to cap the block at one warp per warp-sized span of tile columns for a well-filled inference grid."""

    def __hash__(self) -> int:
        return self._hash

    @functools.cached_property
    def _hash(self) -> int:
        # computed once (cached_property bypasses the frozen __setattr__): policies key the adapted configurations of
        # KernelConfig.adapt, which are looked up at every launch
        return hash(dataclasses.astuple(self))

    def min_tile(self, ndim: int) -> list[int]:
        """Get the minimum size of each dimension of a tile.

        The minimum sizes of the policy apply to the last dimensions of the tile (the leading ones, if the tile has
        fewer dimensions, are ignored), and the other dimensions have a minimum size of 1.

        :param ndim: The number of dimensions of the tile.

        :return: The minimum size of each dimension.
        """
        return ([1] * ndim + list(self.min_sizes))[len(self.min_sizes) :]


AdaptationPolicy.ELEMENTWISE = AdaptationPolicy(thread_per_element_after_shrink=True)
AdaptationPolicy.MATMUL = AdaptationPolicy(
    min_sizes=(16, 8), limit_dims_sum=True, blocks_per_sm=4, grow_for_backward=False
)
AdaptationPolicy.LINEAR = dataclasses.replace(AdaptationPolicy.MATMUL, limit_inference_threads_by_columns=True)


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

    adaptive: bool = _ADAPTIVE
    """Whether to adapt the tiles (and block dimension) to the launched arrays.

    If true, the modules that support it narrow their tiles to the arrays they process (e.g. to the output features
    of a narrow layer), and, on CUDA devices, use smaller tiles when the arrays are too small to keep every SM busy.
    An adapted tile has no more elements than the configured tile (although single dimensions may grow, e.g. more rows
    for fewer columns), and the adapted block dimension is at most the configured one. The modules create a kernel for
    each adapted tile, on the first launch that needs it. If false, the configured tiles and block dimension are used
    as they are.
    """

    def __hash__(self) -> int:
        return self._hash

    @functools.cached_property
    def _hash(self) -> int:
        # computed once (cached_property bypasses the frozen __setattr__): configurations key the kernels of the
        # modules, which are looked up at every launch
        return hash((self.block_dim, self.tile_1d, self.tile_2d, self.tile_3d, self.tile_4d, self.adaptive))

    @functools.cached_property
    def _adapted(self) -> dict[tuple, KernelConfig]:
        # (launch shape, device alias, policy, backward) -> adapted configuration
        return {}

    def adapt(
        self,
        shape: tuple[int, ...],
        device: wp.Device,
        *,
        policy: AdaptationPolicy = AdaptationPolicy.ELEMENTWISE,
        requires_grad: bool = False,
    ) -> KernelConfig:
        """Get the configuration adapted to a launch shape on a device.

        The tile with the launch shape's number of dimensions, and the block dimension, are sized for the launch shape
        (see :ref:`how a tile is adapted <how-a-tile-is-adapted>`). The resulting tile never has more elements than
        the configured tile, although single dimensions may grow beyond their configured size (e.g. more rows for fewer
        columns), and the resulting block dimension is at most the configured one.

        On CPU devices, which have no SMs to keep busy, the tile is never shrunk for more blocks.

        This configuration is returned unchanged if :py:attr:`adaptive` is false, if it has no tile with the launch
        shape's number of dimensions, or if the sizing yields the configured values. The result is cached per
        configuration instance, so that adapting it to a known launch shape is a lookup: keep the configuration (e.g.
        read it once, as the modules do when they are created) rather than getting a new one for each launch (e.g.
        with :py:func:`get_kernel_config`).

        The backward kernels run on the resulting tile too if ``requires_grad`` is true and a :class:`warp.Tape` is
        recording the launch, so that the forward-only launches of a module that requires gradients (e.g. inference)
        get the same tile as a module that does not. Policies that grow their tiles for the backward kernels too
        (see :py:attr:`AdaptationPolicy.grow_for_backward`) ignore it.

        :param shape: The launch shape (the array covered by the tiles), as a tuple.
        :param device: The device on which the kernels are launched.
        :param policy: The adaptation policy of the operation (e.g. :py:attr:`AdaptationPolicy.ELEMENTWISE` or
            :py:attr:`AdaptationPolicy.MATMUL`).
        :param requires_grad: Whether the launched kernels require gradients (e.g. the module requires gradients).

        :return: The adapted configuration.

        :raises ValueError: If the configured tile with the launch shape's number of dimensions has non-positive
            dimensions.
        """
        backward = requires_grad and not policy.grow_for_backward and _is_recording()
        key = (shape, device.alias, policy, backward)
        try:
            return self._adapted[key]
        except KeyError:
            pass
        # adapt the tile with the launch shape's number of dimensions (if any), keeping this configuration if unchanged
        adapted = self
        field = f"tile_{len(shape)}d"
        if self.adaptive and hasattr(self, field):
            configured_tile = getattr(self, field)
            num_sms = device.sm_count if device.is_cuda else 0  # no SMs to keep busy: never shrink the tile
            tile, block_dim = _adapt_tile(
                configured_tile, self.block_dim, shape, num_sms, policy=policy, backward=backward
            )
            if tile != configured_tile or block_dim != self.block_dim:
                adapted = dataclasses.replace(self, **{field: tile}, block_dim=block_dim)
        self._adapted[key] = adapted
        return adapted


@contextlib.contextmanager
def kernel_config(
    *,
    block_dim: int | None = None,
    tile_1d: tuple[int] | None = None,
    tile_2d: tuple[int, int] | None = None,
    tile_3d: tuple[int, int, int] | None = None,
    tile_4d: tuple[int, int, int, int] | None = None,
    adaptive: bool | None = None,
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
    previous_adaptive = getattr(_context, "adaptive", None)
    # set new context values (keeping the previous ones for the unspecified values)
    # - tiles are stored as tuples (e.g. if given as lists), so that the configurations are hashable
    try:
        _context.block_dim = previous_block_dim if block_dim is None else block_dim
        _context.tile_1d = previous_tile_1d if tile_1d is None else tuple(tile_1d)
        _context.tile_2d = previous_tile_2d if tile_2d is None else tuple(tile_2d)
        _context.tile_3d = previous_tile_3d if tile_3d is None else tuple(tile_3d)
        _context.tile_4d = previous_tile_4d if tile_4d is None else tuple(tile_4d)
        _context.adaptive = previous_adaptive if adaptive is None else adaptive
        yield
    # remove context value or restore previous one if it exists
    finally:
        _context.block_dim = previous_block_dim
        _context.tile_1d = previous_tile_1d
        _context.tile_2d = previous_tile_2d
        _context.tile_3d = previous_tile_3d
        _context.tile_4d = previous_tile_4d
        _context.adaptive = previous_adaptive


def get_kernel_config() -> KernelConfig:
    """Get the current configuration."""
    # context values are None outside of any context (after exiting the outermost one)
    return KernelConfig(
        block_dim=_BLOCK_DIM if (value := getattr(_context, "block_dim", None)) is None else value,
        tile_1d=_TILE_1D if (value := getattr(_context, "tile_1d", None)) is None else value,
        tile_2d=_TILE_2D if (value := getattr(_context, "tile_2d", None)) is None else value,
        tile_3d=_TILE_3D if (value := getattr(_context, "tile_3d", None)) is None else value,
        tile_4d=_TILE_4D if (value := getattr(_context, "tile_4d", None)) is None else value,
        adaptive=_ADAPTIVE if (value := getattr(_context, "adaptive", None)) is None else value,
    )
