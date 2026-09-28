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

import torch

import numpy as np
import warp as wp

import warp_nn.nn as nn


# activations whose arguments are embedded in the kernels (created with NumPy scalars), and their PyTorch references
_MODULES = {
    "CELU": (lambda: nn.CELU(alpha=np.float32(0.5)), torch.nn.CELU(alpha=0.5)),
    "HardSigmoid": (
        lambda: nn.HardSigmoid(alpha=np.float64(1.0), beta=np.float32(0.25)),
        lambda x: torch.clamp(x + 0.25, 0.0, 1.0),
    ),
    "Shrink": (lambda: nn.Shrink(lambd=np.float32(0.5), bias=np.float64(0.5)), torch.nn.Softshrink(lambd=0.5)),
    "Swish": (lambda: nn.Swish(alpha=np.float32(2.0)), lambda x: x * torch.sigmoid(2.0 * x)),
    "Threshold": (lambda: nn.Threshold(np.float32(0.25), np.int64(3)), torch.nn.Threshold(0.25, 3.0)),
}


@pytest.mark.parametrize("name", list(_MODULES))
def test_numpy_scalar_arguments(capsys, name):
    # NumPy scalars cannot be embedded in the kernels, so they must be converted to Python floats
    create_warp_module, torch_module = _MODULES[name]
    array = np.linspace(-1.0, 1.0, 9, dtype=np.float32)
    output = create_warp_module().to("cpu")(wp.array(array, device="cpu")).numpy()
    np.testing.assert_allclose(output, torch_module(torch.tensor(array)).numpy(), atol=1e-6)
