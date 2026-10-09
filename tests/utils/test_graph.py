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

from warp_nn.utils import ScopedCapture

from ..utilities import is_device_available


@wp.kernel
def _increment(x: wp.array1d[float]):
    i = wp.tid()
    x[i] += 1.0


def test_disabled(capsys):
    x = wp.zeros(4, dtype=float, device="cpu")
    with ScopedCapture(device="cpu", enabled=False) as capture:
        wp.launch(_increment, dim=x.shape, inputs=[x], device="cpu")
    # the launch runs eagerly and no graph is captured
    assert capture.graph is None
    np.testing.assert_array_equal(x.numpy(), np.ones(4))


@pytest.mark.parametrize("capture_mode", list(wp.CaptureMode))
def test_capture(capsys, capture_mode):
    if not is_device_available("cuda"):
        pytest.skip("Device 'cuda' is not available")
    x = wp.zeros(4, dtype=float, device="cuda")
    with ScopedCapture(device="cuda", force_module_load=True, capture_mode=capture_mode) as capture:
        wp.launch(_increment, dim=x.shape, inputs=[x], device="cuda")
    assert capture.graph is not None
    # the captured launch only runs on replay
    np.testing.assert_array_equal(x.numpy(), np.zeros(4))
    wp.capture_launch(capture.graph)
    wp.capture_launch(capture.graph)
    np.testing.assert_array_equal(x.numpy(), np.full(4, 2.0))
