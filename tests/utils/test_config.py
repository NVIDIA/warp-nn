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

from warp_nn.utils import get_kernel_config, kernel_config


def test_kernel_config(capsys):
    default = get_kernel_config()
    with kernel_config(tile_2d=(16, 16)):
        # unspecified values keep their default values
        config = get_kernel_config()
        assert config.tile_2d == (16, 16)
        assert config.block_dim == default.block_dim and config.tile_1d == default.tile_1d
        with kernel_config(block_dim=64):
            # unspecified values are inherited from the enclosing context
            config = get_kernel_config()
            assert config.block_dim == 64 and config.tile_2d == (16, 16)
        assert get_kernel_config() == dataclasses.replace(default, tile_2d=(16, 16))
    # exiting the outermost context restores the default values
    assert get_kernel_config() == default
