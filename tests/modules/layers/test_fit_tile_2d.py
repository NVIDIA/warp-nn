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

import pytest

import numpy as np
import warp as wp

import warp_nn.nn as nn
from warp_nn.modules.layers._common import fit_tile_2d, reduction_chunk
from warp_nn.utils import get_kernel_config, kernel_config


NUM_SMS = 142


@pytest.mark.parametrize(
    "tile_2d, block_dim", [((16, 64), 256), ((16, 16), 256), ((32, 32), 256), ((32, 64), 256), ((16, 48), 128)]
)
@pytest.mark.parametrize("features", [1, 8, 16, 17, 32, 33, 64, 100])
@pytest.mark.parametrize("batch_size", [1, 50, 4096, 49152])
def test_fit_tile_2d(tile_2d, block_dim, features, batch_size):
    with kernel_config(tile_2d=tile_2d, block_dim=block_dim):
        config = fit_tile_2d(get_kernel_config(), features, batch_size, NUM_SMS)
    rows, columns = config.tile_2d
    # never a larger tile, more threads, or more operand memory (rows + columns) than configured
    assert rows * columns <= tile_2d[0] * tile_2d[1]
    assert config.block_dim <= block_dim
    assert rows + columns <= sum(tile_2d)
    # the tile still covers the features (down to 8 columns), unless the configured tile is narrower
    assert columns >= min(tile_2d[1], max(features, 8))


def test_fit_tile_2d_narrow_layer():
    # an 8-feature layer under the default (16, 64) tile: 8 columns instead of 64 padded ones, with the same number
    # of elements per thread at a large batch, and fewer rows when the batch cannot give every SM two blocks
    with kernel_config(tile_2d=(16, 64), block_dim=256):
        config = get_kernel_config()
        large = fit_tile_2d(config, 8, 49152, NUM_SMS)
        small = fit_tile_2d(config, 8, 50, NUM_SMS)
    assert (large.tile_2d, large.block_dim) == ((64, 8), 128)
    assert small.tile_2d == (16, 8)


def test_fit_tile_2d_wide_layer():
    # layers at least as wide as the tile keep the configured tile and block dimension
    with kernel_config(tile_2d=(16, 64), block_dim=256):
        config = get_kernel_config()
        for batch_size in (1, 4096, 49152):
            fitted = fit_tile_2d(config, 64, batch_size, NUM_SMS)
            assert (fitted.tile_2d, fitted.block_dim) == ((16, 64), 256)


@pytest.mark.parametrize("tile_2d", [(16, 64), (256, 8), (64, 4), (16, 16)])
@pytest.mark.parametrize("size", [1, 2, 8, 9, 16, 19, 64, 256])
def test_reduction_chunk(tile_2d, size):
    with kernel_config(tile_2d=tile_2d):
        chunk = reduction_chunk(get_kernel_config(), size)
    # never a larger chunk (operand memory) than the configured tile's, i.e. its width up to 16 ...
    assert chunk <= min(tile_2d[1], 16)
    # ... nor more chunks (iterations) than the configured tile's
    assert chunk >= min(tile_2d[1], 16, size)


def test_default_block_dim():
    # a non-positive block dimension (the hardware maximum) is kept, and the layers still run
    with kernel_config(block_dim=0):
        assert fit_tile_2d(get_kernel_config(), 8, 50, NUM_SMS).block_dim == 0
        linear = nn.Linear(3, 8)
    output = linear(wp.array(np.ones((5, 3), dtype=np.float32), device=linear.device))
    assert output.shape == (5, 8)


@pytest.mark.parametrize("module", [lambda: nn.Linear(np.prod([16]), 8), lambda: nn.LSTM(np.int64(32), 8)])
def test_numpy_integer_sizes(module):
    # sizes computed with NumPy (e.g. flattened feature counts) are accepted, as before
    module()
