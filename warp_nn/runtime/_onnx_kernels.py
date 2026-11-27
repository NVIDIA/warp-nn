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

"""Differentiable layout conversion kernels between the ONNX recurrent layouts and the Warp-NN recurrent modules.

The modules run sequence-first (as the ONNX layout), so only the reverse direction and the interleaving of the
bidirectional output sequence need a conversion.
"""

import warp as wp


@wp.kernel(grid_stride=False)
def reverse_sequence(input: wp.array3d[wp.float32], output: wp.array3d[wp.float32]):
    """Reverse the order of the time steps of a sequence-first sequence ``(seq_length, batch_size, features)``."""
    t, b, i = wp.tid()
    output[input.shape[0] - 1 - t, b, i] = input[t, b, i]


@wp.kernel(grid_stride=False)
def to_onnx_sequence(input: wp.array3d[wp.float32], reverse: int, output: wp.array4d[wp.float32]):
    """Convert an output sequence ``(seq_length, batch_size, num_directions * hidden_size)`` into the ONNX layout
    ``(seq_length, num_directions, batch_size, hidden_size)``, optionally reversing the order of the time steps."""
    t, d, b, h = wp.tid()
    if reverse != 0:
        output[t, d, b, h] = input[output.shape[0] - 1 - t, b, d * output.shape[3] + h]
    else:
        output[t, d, b, h] = input[t, b, d * output.shape[3] + h]
