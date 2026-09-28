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

from ...utilities import is_device_available
from .common import INT_DTYPES, check_integer_forward, check_unsupported_input


# test-specific parameters
@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("dtype", INT_DTYPES, ids=lambda dtype: dtype.__name__)
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_forward(capsys, device, dtype, ndim):
    if not is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    check_integer_forward(module=nn.BitwiseNot(), numpy_function=np.invert, device=device, dtype=dtype, ndim=ndim)


def test_unsupported_input(capsys):
    check_unsupported_input(module=nn.BitwiseNot(), dtype=wp.int32)
