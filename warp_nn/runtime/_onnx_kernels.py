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

"""Differentiable layout conversion kernels between the ONNX recurrent layouts and the Warp-NN recurrent modules."""

import warp as wp


@wp.kernel(grid_stride=False)
def sequence_first_to_batch_first(input: wp.array3d[wp.float32], reverse: int, output: wp.array3d[wp.float32]):
    """Convert an input sequence ``(seq_length, batch_size, features)`` into ``(batch_size, seq_length, features)``,
    optionally reversing the order of the time steps."""
    t, b, i = wp.tid()
    if reverse != 0:
        output[b, input.shape[0] - 1 - t, i] = input[t, b, i]
    else:
        output[b, t, i] = input[t, b, i]


@wp.kernel(grid_stride=False)
def batch_first_to_onnx_sequence(input: wp.array3d[wp.float32], reverse: int, output: wp.array4d[wp.float32]):
    """Convert an output sequence ``(batch_size, seq_length, num_directions * hidden_size)`` into the ONNX layout
    ``(seq_length, num_directions, batch_size, hidden_size)``, optionally reversing the order of the time steps."""
    t, d, b, h = wp.tid()
    if reverse != 0:
        output[t, d, b, h] = input[b, output.shape[0] - 1 - t, d * output.shape[3] + h]
    else:
        output[t, d, b, h] = input[b, t, d * output.shape[3] + h]
